from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.signal import butter, detrend, filtfilt, iirnotch, resample

from train_export_decoder import MultiModalNet


DEFAULT_BRAINPRINT_STATS = {
    "mean_between_subject_cosine": 0.16015970347793174,
    "mean_within_subject_cosine": 0.2688431900633652,
}


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def notch_filter(data: np.ndarray, fs: float, freq: float = 50.0) -> np.ndarray:
    nyq = 0.5 * fs
    if nyq <= 0 or freq >= nyq:
        return data
    w0 = freq / nyq
    b, a = iirnotch(w0, Q=30)
    return filtfilt(b, a, data, axis=0)


def butter_bandpass_filter(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    nyq = 0.5 * fs
    if nyq <= 0:
        return data

    low = max(lowcut / nyq, 1e-6)
    high = min(highcut / nyq, 0.999999)

    if low >= high:
        return data

    b, a = butter(order, [low, high], btype="bandpass")
    return filtfilt(b, a, data, axis=0)


def low_pass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    if nyq <= 0:
        return data

    norm_cut = min(cutoff / nyq, 0.999999)
    if norm_cut <= 0:
        return data

    b, a = butter(order, norm_cut, btype="low")
    return filtfilt(b, a, data, axis=0)


def resample_tc_to_ct(x_tc: np.ndarray, target_size: int) -> np.ndarray:
    if x_tc.ndim != 2:
        raise ValueError(f"Expected 2D array [T, C], got shape {x_tc.shape}")
    if x_tc.shape[0] < 2:
        raise ValueError(f"Need at least 2 samples to resample, got {x_tc.shape[0]}")
    x_tc = resample(x_tc, target_size, axis=0).astype(np.float32)
    return np.transpose(x_tc, (1, 0))  # C, T


def normalize_ct(x_ct: np.ndarray, norm_cfg: dict) -> np.ndarray:
    mean = np.asarray(norm_cfg["mean"], dtype=np.float32)
    std = np.asarray(norm_cfg["std"], dtype=np.float32)

    if mean.ndim == 1:
        mean = mean[:, None]
    if std.ndim == 1:
        std = std[:, None]

    std = std.copy()
    std[std < 1e-6] = 1.0

    return ((x_ct - mean) / std).astype(np.float32)


def load_decoder_bundle(bundle_path: Path, device: torch.device):
    payload = torch.load(bundle_path, map_location=device)

    bundle = payload["bundle"]
    state_dict = payload["state_dict"]

    model = MultiModalNet(
        input_channels={k: int(v) for k, v in bundle["input_channels"].items()},
        modalities=list(bundle["required_modalities"]),
        num_classes=int(bundle["num_classes"]),
        dropout=0.0,
    ).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, bundle


def load_trial_samples(trial_dir: Path, summary: dict) -> np.ndarray:
    path = trial_dir / summary["samples_file"]
    if not path.exists():
        raise FileNotFoundError(f"Trial samples file not found: {path}")

    data = np.load(path)
    data = np.asarray(data, dtype=np.float32)

    if data.ndim == 1:
        data = data[:, None]

    return data


def prepare_eeg(samples_tc: np.ndarray, fs: float, target_size: int, expected_channels: int) -> np.ndarray:
    if samples_tc.shape[1] < expected_channels:
        raise ValueError(
            f"EEG trial has {samples_tc.shape[1]} channels but bundle expects {expected_channels}"
        )

    x = np.asarray(samples_tc[:, :expected_channels], dtype=np.float32)
    x = notch_filter(x, fs, 50.0)
    x = butter_bandpass_filter(x, 1.0, 40.0, fs)
    return resample_tc_to_ct(x, target_size)


def prepare_emg(samples_tc: np.ndarray, fs: float, target_size: int, expected_channels: int) -> np.ndarray:
    if samples_tc.shape[1] < expected_channels:
        raise ValueError(
            f"EMG trial has {samples_tc.shape[1]} channels but bundle expects {expected_channels}"
        )

    x = np.asarray(samples_tc[:, :expected_channels], dtype=np.float32)
    x = butter_bandpass_filter(x, 20.0, 100.0, fs)
    x = notch_filter(x, fs, 50.0)
    return resample_tc_to_ct(x, target_size)


def prepare_imu(
    imu_summaries: list[dict],
    trial_dir: Path,
    target_size: int,
    expected_channels: int,
) -> np.ndarray:
    if not imu_summaries:
        raise ValueError("No IMU summaries were provided for IMU preprocessing.")

    processed = []
    for summary in sorted(imu_summaries, key=lambda s: s["name"]):
        samples_tc = load_trial_samples(trial_dir, summary)
        fs = float(summary.get("nominal_srate", 256.0) or 256.0)

        x = np.asarray(samples_tc, dtype=np.float32)
        x = detrend(x, axis=0)
        x = low_pass_filter(x, 15.0, fs)
        processed.append(x)

    min_len = min(x.shape[0] for x in processed)
    if min_len < 2:
        raise ValueError("IMU streams do not contain enough samples.")

    processed = [x[:min_len] for x in processed]
    merged = np.hstack(processed)

    if merged.shape[1] < expected_channels:
        raise ValueError(
            f"IMU trial has {merged.shape[1]} merged channels but bundle expects {expected_channels}"
        )

    if merged.shape[1] > expected_channels:
        merged = merged[:, :expected_channels]

    return resample_tc_to_ct(merged, target_size)


def prepare_trial_batch(
    trial_dir: Path,
    stream_summaries: list[dict],
    bundle: dict,
) -> dict[str, torch.Tensor]:
    grouped: dict[str, list[dict]] = {"EEG": [], "EMG": [], "IMU": []}
    for summary in stream_summaries:
        modality = str(summary.get("modality", "")).upper()
        if modality in grouped:
            grouped[modality].append(summary)

    batch: dict[str, torch.Tensor] = {}

    required_modalities = list(bundle["required_modalities"])
    input_channels = {k: int(v) for k, v in bundle["input_channels"].items()}
    target_size = {k: int(v) for k, v in bundle["target_size"].items()}

    for modality in required_modalities:
        if modality == "eeg":
            if not grouped["EEG"]:
                raise ValueError("Bundle requires EEG but no EEG stream was saved for this trial.")

            summary = sorted(grouped["EEG"], key=lambda s: s["name"])[0]
            samples_tc = load_trial_samples(trial_dir, summary)
            fs = float(summary.get("nominal_srate", 256.0) or 256.0)

            x_ct = prepare_eeg(
                samples_tc=samples_tc,
                fs=fs,
                target_size=target_size["eeg"],
                expected_channels=input_channels["eeg"],
            )
            x_ct = normalize_ct(x_ct, bundle["normalization"]["eeg"])
            batch["eeg"] = torch.tensor(x_ct[None, ...], dtype=torch.float32)

        elif modality == "emg":
            if not grouped["EMG"]:
                raise ValueError("Bundle requires EMG but no EMG stream was saved for this trial.")

            summary = sorted(grouped["EMG"], key=lambda s: s["name"])[0]
            samples_tc = load_trial_samples(trial_dir, summary)
            fs = float(summary.get("nominal_srate", 256.0) or 256.0)

            x_ct = prepare_emg(
                samples_tc=samples_tc,
                fs=fs,
                target_size=target_size["emg"],
                expected_channels=input_channels["emg"],
            )
            x_ct = normalize_ct(x_ct, bundle["normalization"]["emg"])
            batch["emg"] = torch.tensor(x_ct[None, ...], dtype=torch.float32)

        elif modality == "imu":
            if not grouped["IMU"]:
                raise ValueError("Bundle requires IMU but no IMU streams were saved for this trial.")

            x_ct = prepare_imu(
                imu_summaries=grouped["IMU"],
                trial_dir=trial_dir,
                target_size=target_size["imu"],
                expected_channels=input_channels["imu"],
            )
            x_ct = normalize_ct(x_ct, bundle["normalization"]["imu"])
            batch["imu"] = torch.tensor(x_ct[None, ...], dtype=torch.float32)

        else:
            raise ValueError(f"Unsupported modality in bundle: {modality}")

    return batch


def predict_trial_from_bundle(
    bundle_path: Path,
    trial_dir: Path,
    stream_summaries: list[dict],
    device_arg: str = "cpu",
    modality_scales: dict[str, float] | None = None,
) -> dict:
    device = get_device(device_arg)
    model, bundle = load_decoder_bundle(bundle_path, device)

    batch = prepare_trial_batch(
        trial_dir=trial_dir,
        stream_summaries=stream_summaries,
        bundle=bundle,
    )
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        logits = model(batch, modality_scales=modality_scales)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    pred_id = int(np.argmax(probs))
    pred_conf = float(probs[pred_id])

    id_to_label = bundle["id_to_label"]
    pred_label = id_to_label.get(str(pred_id), str(pred_id))

    probabilities = {
        id_to_label.get(str(i), str(i)): float(probs[i])
        for i in range(len(probs))
    }

    return {
        "predicted_word": pred_label,
        "confidence": pred_conf,
        "predicted_id": pred_id,
        "probabilities": probabilities,
        "required_modalities": bundle["required_modalities"],
        "modality_scales": modality_scales or {},
        "bundle_model_key": bundle["model_key"],
    }


def load_brainprint_reference_stats(summary_json_path: Optional[Path]) -> dict:
    if summary_json_path is not None and summary_json_path.exists():
        try:
            with open(summary_json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return {
                "mean_between_subject_cosine": float(
                    payload.get(
                        "mean_between_subject_cosine",
                        DEFAULT_BRAINPRINT_STATS["mean_between_subject_cosine"],
                    )
                ),
                "mean_within_subject_cosine": float(
                    payload.get(
                        "mean_within_subject_cosine",
                        DEFAULT_BRAINPRINT_STATS["mean_within_subject_cosine"],
                    )
                ),
            }
        except Exception:
            pass

    return dict(DEFAULT_BRAINPRINT_STATS)


def compute_eeg_scale_from_comparison_json(
    compare_json_path: Optional[Path],
    summary_json_path: Optional[Path] = None,
    top_k: int = 3,
    min_scale: float = 0.65,
    max_scale: float = 1.00,
) -> tuple[float, dict]:
    info = {
        "enabled": False,
        "eeg_scale": 1.0,
        "top_k": top_k,
        "top_k_mean_cosine": None,
        "reference_between_cosine": None,
        "reference_within_cosine": None,
        "normalised_score": None,
        "status": "no_comparison_json",
    }

    if compare_json_path is None or not compare_json_path.exists():
        return 1.0, info

    try:
        with open(compare_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        info["status"] = f"failed_to_read_compare_json: {exc}"
        return 1.0, info

    ranked = payload.get("ranked_results", [])
    cosines = [
        float(row["cosine_similarity"])
        for row in ranked[:top_k]
        if "cosine_similarity" in row
    ]

    if not cosines:
        info["status"] = "no_ranked_results"
        return 1.0, info

    stats = load_brainprint_reference_stats(summary_json_path)
    between = float(stats["mean_between_subject_cosine"])
    within = float(stats["mean_within_subject_cosine"])

    # Conservative ceiling so the EEG gate does not saturate too easily.
    ceiling = max(within, between + 0.20)

    score = float(np.mean(cosines))
    norm = float(np.clip((score - between) / max(1e-6, ceiling - between), 0.0, 1.0))
    eeg_scale = float(min_scale + norm * (max_scale - min_scale))

    info = {
        "enabled": True,
        "eeg_scale": eeg_scale,
        "top_k": top_k,
        "top_k_mean_cosine": score,
        "reference_between_cosine": between,
        "reference_within_cosine": within,
        "reference_ceiling_cosine": ceiling,
        "normalised_score": norm,
        "status": "ok",
        "compare_json_path": str(compare_json_path),
        "summary_json_path": str(summary_json_path) if summary_json_path is not None else None,
    }

    return eeg_scale, info