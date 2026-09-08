from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

try:
    from scipy.signal import butter, detrend, filtfilt, iirnotch, resample as scipy_resample
except Exception:
    butter = None
    detrend = None
    filtfilt = None
    iirnotch = None
    scipy_resample = None


MODALITY_TO_STREAM = {
    "eeg": "EEG",
    "emg": "EMG",
    "imu": "IMU",
}

STREAM_TO_MODALITY = {v: k for k, v in MODALITY_TO_STREAM.items()}


class ConvEncoder1D(nn.Module):
    def __init__(self, in_channels: int, dropout: float = 0.3, feat_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(64, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x).squeeze(-1)
        z = self.dropout(z)
        z = self.proj(z)
        return z


class MultiModalNet(nn.Module):
    def __init__(
        self,
        input_channels: dict[str, int],
        modalities: list[str],
        num_classes: int,
        dropout: float = 0.3,
        feat_dim: int = 64,
        fusion_mode: str = "concat",
    ):
        super().__init__()
        self.modalities = list(modalities)
        self.fusion_mode = str(fusion_mode or "concat").lower()
        if self.fusion_mode not in {"concat", "late", "gated"}:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

        self.encoders = nn.ModuleDict(
            {
                modality: ConvEncoder1D(
                    in_channels=input_channels[modality],
                    dropout=dropout,
                    feat_dim=feat_dim,
                )
                for modality in self.modalities
            }
        )

        fusion_dim = feat_dim * len(self.modalities)

        if self.fusion_mode == "concat":
            # Original architecture: concatenate all modality features, then classify.
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes),
            )
        else:
            # Late/gated architectures: each modality predicts its own logits.
            # This prevents a weak modality from directly contaminating the feature
            # space of the stronger modalities.
            self.modality_heads = nn.ModuleDict(
                {
                    modality: nn.Sequential(
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(feat_dim, num_classes),
                    )
                    for modality in self.modalities
                }
            )

            if self.fusion_mode == "late":
                # Global learned reliability weights, shared across trials.
                self.global_logit_weights = nn.Parameter(torch.zeros(len(self.modalities)))
            else:
                # Trial-specific reliability weights predicted from all modality features.
                gate_hidden = max(32, 16 * len(self.modalities))
                self.gate = nn.Sequential(
                    nn.Linear(fusion_dim, gate_hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(gate_hidden, len(self.modalities)),
                )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        modality_scales: dict[str, float] | None = None,
    ) -> torch.Tensor:
        modality_scales = modality_scales or {}

        feats = []
        for modality in self.modalities:
            feat = self.encoders[modality](batch[modality])
            scale = float(modality_scales.get(modality, 1.0))
            feats.append(feat * scale)

        if self.fusion_mode == "concat":
            fused = torch.cat(feats, dim=1)
            return self.classifier(fused)

        logits_by_modality = [
            self.modality_heads[modality](feat)
            for modality, feat in zip(self.modalities, feats)
        ]
        logits_stack = torch.stack(logits_by_modality, dim=1)  # B, M, C

        if self.fusion_mode == "late":
            weights = torch.softmax(self.global_logit_weights, dim=0).view(1, -1, 1)
        else:
            fused = torch.cat(feats, dim=1)
            weights = torch.softmax(self.gate(fused), dim=1).unsqueeze(-1)  # B, M, 1

        return torch.sum(logits_stack * weights, dim=1)

    def current_fusion_weights(self) -> dict[str, float] | None:
        """Return global weights for late fusion; gated fusion is trial-specific."""
        if self.fusion_mode != "late":
            return None
        with torch.no_grad():
            w = torch.softmax(self.global_logit_weights.detach().cpu(), dim=0).numpy()
        return {m: float(wi) for m, wi in zip(self.modalities, w)}


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _infer_feat_dim(state_dict: dict[str, torch.Tensor], modalities: list[str]) -> int:
    for modality in modalities:
        key = f"encoders.{modality}.proj.weight"
        if key in state_dict:
            return int(state_dict[key].shape[0])

    cls_key = "classifier.0.weight"
    if cls_key in state_dict and len(modalities) > 0:
        fusion_dim = int(state_dict[cls_key].shape[1])
        return fusion_dim // len(modalities)

    return 64


def _resample_time_axis(data_tc: np.ndarray, target_size: int) -> np.ndarray:
    data_tc = np.asarray(data_tc, dtype=np.float32)

    if data_tc.ndim != 2:
        raise ValueError(f"Expected 2D time-by-channel array, got shape {data_tc.shape}")

    if data_tc.shape[0] < 2:
        raise ValueError(f"Need at least 2 time samples before resampling, got {data_tc.shape[0]}")

    if data_tc.shape[0] == target_size:
        return data_tc.astype(np.float32, copy=False)

    if scipy_resample is not None:
        return scipy_resample(data_tc, target_size, axis=0).astype(np.float32)

    old_x = np.linspace(0.0, 1.0, num=data_tc.shape[0], endpoint=True)
    new_x = np.linspace(0.0, 1.0, num=target_size, endpoint=True)

    out = np.empty((target_size, data_tc.shape[1]), dtype=np.float32)
    for ch in range(data_tc.shape[1]):
        out[:, ch] = np.interp(new_x, old_x, data_tc[:, ch]).astype(np.float32)
    return out


def _match_channel_count(data_tc: np.ndarray, expected_channels: int) -> np.ndarray:
    actual_channels = int(data_tc.shape[1])

    if actual_channels == expected_channels:
        return data_tc

    if actual_channels > expected_channels:
        return data_tc[:, :expected_channels]

    pad = np.zeros((data_tc.shape[0], expected_channels - actual_channels), dtype=data_tc.dtype)
    return np.concatenate([data_tc, pad], axis=1)


def _normalization_arrays(norm_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(norm_cfg["mean"], dtype=np.float32)
    std = np.asarray(norm_cfg["std"], dtype=np.float32)

    if mean.ndim == 1:
        mean = mean[:, None]
    if std.ndim == 1:
        std = std[:, None]

    std = std.copy()
    std[std < 1e-6] = 1.0
    return mean, std


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        out = float(value)
        if not np.isfinite(out):
            return default
        return out
    except Exception:
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    try:
        return bool(value)
    except Exception:
        return default


def _safe_str(value: Any, default: str) -> str:
    try:
        if value is None:
            return default
        return str(value)
    except Exception:
        return default


def _parse_eeg_bands_from_meta(meta: dict[str, Any], fallback_low: float, fallback_high: float) -> list[tuple[float, float]]:
    raw_bands = meta.get("eeg_bands")
    bands: list[tuple[float, float]] = []

    if isinstance(raw_bands, (list, tuple)):
        for item in raw_bands:
            try:
                low = float(item[0])
                high = float(item[1])
            except Exception:
                continue
            if np.isfinite(low) and np.isfinite(high) and 0 < low < high:
                bands.append((low, high))

    if not bands:
        band_string = _safe_str(meta.get("eeg_bands_string"), "")
        if not band_string:
            band_string = _safe_str(meta.get("eeg_bands"), "")
        for part in band_string.split(","):
            part = part.strip()
            if "-" not in part:
                continue
            left, right = part.split("-", 1)
            try:
                low = float(left.strip())
                high = float(right.strip())
            except ValueError:
                continue
            if np.isfinite(low) and np.isfinite(high) and 0 < low < high:
                bands.append((low, high))

    if not bands:
        bands = [(float(fallback_low), float(fallback_high))]
    return bands


def _can_filter(data_tc: np.ndarray, min_samples: int = 32) -> bool:
    return (
        butter is not None
        and filtfilt is not None
        and data_tc.ndim == 2
        and data_tc.shape[0] >= min_samples
    )


def _notch_filter(data_tc: np.ndarray, fs_hz: float, notch_hz: float) -> np.ndarray:
    if iirnotch is None or filtfilt is None or data_tc.shape[0] < 32:
        return data_tc
    nyq = 0.5 * fs_hz
    if nyq <= 0 or notch_hz <= 0 or notch_hz >= 0.98 * nyq:
        return data_tc
    try:
        b, a = iirnotch(notch_hz / nyq, Q=30)
        return filtfilt(b, a, data_tc, axis=0).astype(np.float32)
    except Exception:
        return data_tc


def _bandpass_filter(data_tc: np.ndarray, fs_hz: float, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    if not _can_filter(data_tc):
        return data_tc
    nyq = 0.5 * fs_hz
    if nyq <= 0:
        return data_tc
    low = max(float(low_hz) / nyq, 1e-6)
    high = min(float(high_hz) / nyq, 0.999999)
    if low <= 0 or high <= 0 or low >= high:
        return data_tc
    try:
        b, a = butter(order, [low, high], btype="bandpass")
        return filtfilt(b, a, data_tc, axis=0).astype(np.float32)
    except Exception:
        return data_tc


def _lowpass_filter(data_tc: np.ndarray, fs_hz: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    if not _can_filter(data_tc):
        return data_tc
    nyq = 0.5 * fs_hz
    if nyq <= 0:
        return data_tc
    cutoff = min(float(cutoff_hz) / nyq, 0.999999)
    if cutoff <= 0:
        return data_tc
    try:
        b, a = butter(order, cutoff, btype="low")
        return filtfilt(b, a, data_tc, axis=0).astype(np.float32)
    except Exception:
        return data_tc


def _common_average_reference(data_tc: np.ndarray) -> np.ndarray:
    if data_tc.ndim != 2 or data_tc.shape[1] < 2:
        return data_tc
    return (data_tc - data_tc.mean(axis=1, keepdims=True)).astype(np.float32)


def _baseline_correct_epoch(data_tc: np.ndarray, window_pre_s: float | None, window_post_s: float | None) -> np.ndarray:
    x = np.asarray(data_tc, dtype=np.float32)
    if window_pre_s is None or window_post_s is None:
        return x
    duration = float(window_pre_s) + float(window_post_s)
    if duration <= 0 or float(window_pre_s) <= 0 or x.shape[0] < 2:
        return x
    baseline_n = int(round(x.shape[0] * (float(window_pre_s) / duration)))
    baseline_n = max(1, min(baseline_n, x.shape[0] - 1))
    baseline = x[:baseline_n].mean(axis=0, keepdims=True)
    return (x - baseline).astype(np.float32)


def _add_temporal_delta(data_tc: np.ndarray) -> np.ndarray:
    x = np.asarray(data_tc, dtype=np.float32)
    if x.shape[0] < 2:
        delta = np.zeros_like(x)
    else:
        delta = np.diff(x, axis=0, prepend=x[:1]).astype(np.float32)
    return np.concatenate([x, delta], axis=1).astype(np.float32)


def _select_imu_axes(data_tc: np.ndarray, raw_channels_per_stream: int, imu_use_axes: str) -> np.ndarray:
    x = np.asarray(data_tc, dtype=np.float32)
    if imu_use_axes == "all":
        return x
    if imu_use_axes != "accgyro":
        return x

    raw_channels_per_stream = int(raw_channels_per_stream)
    if raw_channels_per_stream <= 0:
        return x

    # Online windows usually arrive as concatenated full 9-axis streams.
    # For accgyro, keep the first six channels of each full stream.
    if x.shape[1] % raw_channels_per_stream == 0:
        n_streams = x.shape[1] // raw_channels_per_stream
        keep = min(6, raw_channels_per_stream)
        chunks = []
        for i in range(n_streams):
            start = i * raw_channels_per_stream
            chunks.append(x[:, start:start + keep])
        return np.hstack(chunks).astype(np.float32) if chunks else x

    # If already reduced, leave unchanged.
    if x.shape[1] % 6 == 0:
        return x

    return x[:, :min(6, x.shape[1])].astype(np.float32)


@dataclass
class PredictionResult:
    predicted_id: int
    predicted_label: str
    confidence: float
    probabilities: list[float]
    top_predictions: list[dict[str, float | int | str]]


class BundleDecoderRuntime:
    def __init__(
        self,
        bundle_path: Path,
        device: torch.device,
        model: MultiModalNet,
        bundle_meta: dict[str, Any],
    ):
        self.bundle_path = Path(bundle_path)
        self.device = device
        self.model = model.eval()
        self.bundle_meta = bundle_meta

        self.required_modalities: list[str] = list(bundle_meta["required_modalities"])
        self.input_channels: dict[str, int] = {
            k: int(v) for k, v in bundle_meta["input_channels"].items()
        }
        self.target_size: dict[str, int] = {
            k: int(v) for k, v in bundle_meta["target_size"].items()
        }
        self.normalization: dict[str, dict[str, Any]] = dict(bundle_meta["normalization"])
        self.preprocessing_meta: dict[str, Any] = dict(
            bundle_meta.get("preprocessing_meta")
            or bundle_meta.get("preprocessing")
            or {}
        )

        raw_id_to_label = bundle_meta.get("id_to_label", {})
        self.id_to_label = {int(k): str(v) for k, v in raw_id_to_label.items()}

    @classmethod
    def load(cls, bundle_path: str | Path, device_arg: str = "auto") -> "BundleDecoderRuntime":
        bundle_path = Path(bundle_path).resolve()
        if not bundle_path.exists():
            raise FileNotFoundError(f"Decoder bundle not found: {bundle_path}")

        device = get_device(device_arg)
        payload = torch.load(bundle_path, map_location=device)

        if "state_dict" not in payload or "bundle" not in payload:
            raise RuntimeError(f"Invalid decoder bundle format: {bundle_path}")

        state_dict = payload["state_dict"]
        bundle_meta = payload["bundle"]

        modalities = list(bundle_meta["required_modalities"])
        input_channels = {k: int(v) for k, v in bundle_meta["input_channels"].items()}
        num_classes = int(bundle_meta["num_classes"])
        feat_dim = _infer_feat_dim(state_dict, modalities)

        fusion_mode = str(bundle_meta.get("fusion_mode", "concat") or "concat")
        dropout = float(bundle_meta.get("dropout", 0.3) or 0.3)

        model = MultiModalNet(
            input_channels=input_channels,
            modalities=modalities,
            num_classes=num_classes,
            dropout=dropout,
            feat_dim=feat_dim,
            fusion_mode=fusion_mode,
        ).to(device)

        model.load_state_dict(state_dict)
        model.eval()

        return cls(
            bundle_path=bundle_path,
            device=device,
            model=model,
            bundle_meta=bundle_meta,
        )

    def required_stream_modalities(self) -> list[str]:
        return [MODALITY_TO_STREAM[m] for m in self.required_modalities]

    def _window_duration_s(self) -> float | None:
        start = _safe_float(self.preprocessing_meta.get("window_start_s"), None)
        end = _safe_float(self.preprocessing_meta.get("window_end_s"), None)
        if start is not None and end is not None and end > start:
            return float(end - start)
        pre = _safe_float(self.preprocessing_meta.get("window_pre_s"), None)
        post = _safe_float(self.preprocessing_meta.get("window_post_s"), None)
        if pre is not None and post is not None and pre + post > 0:
            return float(pre + post)
        return None

    def _window_pre_post(self) -> tuple[float | None, float | None]:
        pre = _safe_float(self.preprocessing_meta.get("window_pre_s"), None)
        post = _safe_float(self.preprocessing_meta.get("window_post_s"), None)
        if pre is not None and post is not None:
            return pre, post
        start = _safe_float(self.preprocessing_meta.get("window_start_s"), None)
        end = _safe_float(self.preprocessing_meta.get("window_end_s"), None)
        if start is not None and end is not None:
            return abs(min(start, 0.0)), max(end, 0.0)
        return None, None

    def _estimate_fs_hz(self, modality: str, data_tc: np.ndarray, fs_hz: float | None) -> float:
        if fs_hz is not None and fs_hz > 0:
            return float(fs_hz)

        meta_key = f"{modality}_fs_hz"
        meta_fs = _safe_float(self.preprocessing_meta.get(meta_key), None)
        if meta_fs is not None and meta_fs > 0:
            return float(meta_fs)

        duration = self._window_duration_s()
        if duration is not None and duration > 0 and data_tc.shape[0] > 1:
            return float(data_tc.shape[0] / duration)

        return 256.0

    def _apply_training_preprocessing(
        self,
        modality: str,
        data_tc: np.ndarray,
        fs_hz: float | None = None,
    ) -> np.ndarray:
        x = np.asarray(data_tc, dtype=np.float32)
        fs = self._estimate_fs_hz(modality, x, fs_hz)
        notch = _safe_float(self.preprocessing_meta.get("notch_hz"), 50.0) or 50.0
        window_pre, window_post = self._window_pre_post()

        if modality == "eeg":
            # Match the raw EEG channel count before EEG transforms.
            raw_ch = int(_safe_float(self.preprocessing_meta.get("eeg_channels_requested"), self.input_channels.get("eeg", x.shape[1])) or x.shape[1])
            x = _match_channel_count(x, raw_ch)
            low = _safe_float(self.preprocessing_meta.get("eeg_low_hz"), 1.0) or 1.0
            high = _safe_float(self.preprocessing_meta.get("eeg_high_hz"), 40.0) or 40.0
            eeg_mode = _safe_str(self.preprocessing_meta.get("eeg_mode"), "raw")
            use_car = _safe_bool(self.preprocessing_meta.get("eeg_car"), False)

            if eeg_mode == "multiband":
                bands = _parse_eeg_bands_from_meta(self.preprocessing_meta, fallback_low=low, fallback_high=high)
                notched = _notch_filter(x, fs, notch)
                band_arrays = []
                for band_low, band_high in bands:
                    band = _bandpass_filter(notched, fs, band_low, band_high)
                    if use_car:
                        band = _common_average_reference(band)
                    band_arrays.append(band.astype(np.float32))
                x = np.concatenate(band_arrays, axis=1).astype(np.float32) if band_arrays else notched
            else:
                x = _notch_filter(x, fs, notch)
                x = _bandpass_filter(x, fs, low, high)
                if use_car:
                    x = _common_average_reference(x)

            if _safe_bool(self.preprocessing_meta.get("eeg_baseline_correct"), False):
                x = _baseline_correct_epoch(x, window_pre, window_post)
            return x.astype(np.float32)

        if modality == "emg":
            # EMG modes may change channel count, so match raw channel count first,
            # then create envelope/raw+envelope representation.
            raw_ch = int(_safe_float(self.preprocessing_meta.get("emg_channels_requested"), x.shape[1]) or x.shape[1])
            x = _match_channel_count(x, raw_ch)
            low = _safe_float(self.preprocessing_meta.get("emg_low_hz"), 20.0) or 20.0
            high = _safe_float(self.preprocessing_meta.get("emg_high_hz"), 100.0) or 100.0
            emg_mode = _safe_str(self.preprocessing_meta.get("emg_mode"), "raw")
            envelope_low = _safe_float(self.preprocessing_meta.get("emg_envelope_lowpass_hz"), 8.0) or 8.0

            raw = _bandpass_filter(x, fs, low, high)
            raw = _notch_filter(raw, fs, notch)

            if emg_mode == "raw":
                x = raw
            elif emg_mode == "envelope":
                x = _lowpass_filter(np.abs(raw), fs, envelope_low)
            elif emg_mode == "raw_envelope":
                envelope = _lowpass_filter(np.abs(raw), fs, envelope_low)
                x = np.concatenate([raw, envelope], axis=1).astype(np.float32)
            else:
                x = raw

            if _safe_bool(self.preprocessing_meta.get("emg_baseline_correct"), False):
                x = _baseline_correct_epoch(x, window_pre, window_post)
            return x.astype(np.float32)

        if modality == "imu":
            cutoff = _safe_float(self.preprocessing_meta.get("imu_lowpass_hz"), 15.0) or 15.0
            imu_use_axes = _safe_str(self.preprocessing_meta.get("imu_use_axes"), "all")
            imu_mode = _safe_str(self.preprocessing_meta.get("imu_mode"), "raw")
            raw_per_stream = int(_safe_float(self.preprocessing_meta.get("imu_channels_per_stream_requested"), 9.0) or 9)

            x = _select_imu_axes(x, raw_channels_per_stream=raw_per_stream, imu_use_axes=imu_use_axes)

            if detrend is not None and x.shape[0] >= 2:
                try:
                    x = detrend(x, axis=0).astype(np.float32)
                except Exception:
                    pass

            x = _lowpass_filter(x, fs, cutoff)

            if imu_mode in {"basecorr", "basecorr_delta"}:
                x = _baseline_correct_epoch(x, window_pre, window_post)
            if imu_mode in {"raw_delta", "basecorr_delta"}:
                x = _add_temporal_delta(x)

            return x.astype(np.float32)

        return x.astype(np.float32)

    def prepare_modality_sample(
        self,
        modality: str,
        data_tc: np.ndarray,
        fs_hz: float | None = None,
    ) -> np.ndarray:
        if modality not in self.required_modalities:
            raise ValueError(f"Modality '{modality}' is not required by this bundle.")

        data_tc = np.asarray(data_tc, dtype=np.float32)
        if data_tc.ndim != 2:
            raise ValueError(f"{modality}: expected 2D time-by-channel array, got {data_tc.shape}")
        if data_tc.shape[0] < 2:
            raise ValueError(f"{modality}: not enough samples in trial window ({data_tc.shape[0]})")

        expected_channels = self.input_channels[modality]
        target_size = self.target_size[modality]

        # Important: preprocessing happens before final channel matching because some
        # recipes change channel count, e.g. EMG raw+envelope or IMU delta.
        data_tc = self._apply_training_preprocessing(modality, data_tc, fs_hz=fs_hz)
        data_tc = _match_channel_count(data_tc, expected_channels)
        data_tc = _resample_time_axis(data_tc, target_size)

        sample_ct = data_tc.T.astype(np.float32)

        mean, std = _normalization_arrays(self.normalization[modality])
        sample_ct = ((sample_ct - mean) / std).astype(np.float32)

        return sample_ct

    def build_batch(
        self,
        modality_windows: dict[str, np.ndarray],
        modality_sample_rates: Optional[dict[str, float]] = None,
    ) -> dict[str, torch.Tensor]:
        batch: dict[str, torch.Tensor] = {}
        modality_sample_rates = modality_sample_rates or {}

        for modality in self.required_modalities:
            if modality not in modality_windows:
                raise ValueError(f"Missing modality window for '{modality}'")

            sample_ct = self.prepare_modality_sample(
                modality,
                modality_windows[modality],
                fs_hz=modality_sample_rates.get(modality),
            )
            batch[modality] = torch.tensor(
                sample_ct[None, :, :],
                dtype=torch.float32,
                device=self.device,
            )

        return batch

    def predict(
        self,
        modality_windows: dict[str, np.ndarray],
        top_k: int = 3,
        modality_scales: Optional[dict[str, float]] = None,
        modality_sample_rates: Optional[dict[str, float]] = None,
    ) -> PredictionResult:
        batch = self.build_batch(modality_windows, modality_sample_rates=modality_sample_rates)

        with torch.no_grad():
            logits = self.model(batch, modality_scales=modality_scales)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        pred_id = int(np.argmax(probs))
        pred_label = self.id_to_label.get(pred_id, str(pred_id))
        confidence = float(probs[pred_id])

        sorted_ids = np.argsort(probs)[::-1][:top_k]
        top_predictions = [
            {
                "class_id": int(idx),
                "label": self.id_to_label.get(int(idx), str(int(idx))),
                "probability": float(probs[idx]),
            }
            for idx in sorted_ids
        ]

        return PredictionResult(
            predicted_id=pred_id,
            predicted_label=pred_label,
            confidence=confidence,
            probabilities=[float(x) for x in probs.tolist()],
            top_predictions=top_predictions,
        )
