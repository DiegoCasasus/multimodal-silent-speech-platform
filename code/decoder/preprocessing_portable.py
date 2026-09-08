from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pyxdf
from scipy.signal import butter, detrend, filtfilt, iirnotch, resample


LABEL_MAP = {
    "1 (ONE)": 0,
    "2 (TWO)": 1,
    "3 (THREE)": 2,
    "4 (FOUR)": 3,
    "5 (FIVE)": 4,
    "REJOIN": 5,
    "ASSET": 6,
    "ABORT": 7,
    "FORMATION": 8,
    "REFUEL": 9,
}

ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

OLD_NAME_RE = re.compile(
    r"exp_sub_(?P<sub>[^_]+)_ses_(?P<ses>[^_]+)_bl_(?P<block>[^.]+)\.xdf$",
    re.IGNORECASE,
)

NEW_NAME_RE = re.compile(
    r"exp_sub_(?P<sub>[^_]+)_ses_(?P<ses>[^_]+)_run_(?P<run>[^_]+)_mode_(?P<mode>[^_]+)_bl_(?P<block>[^.]+)\.xdf$",
    re.IGNORECASE,
)

CUE_START_RE = re.compile(r"^Cue_Start:\s*(.+)$")
SILENT_GO_RE = re.compile(r"^SILENT_GO:(.+)$")
IMAGINED_GO_RE = re.compile(r"^IMAGINED_GO:(.+)$")


@dataclass
class FileMeta:
    path: str
    participant_id: str
    session_id: str
    run_id: str | None
    experiment_mode: str | None
    block: str


@dataclass
class SessionOutput:
    source_xdf: str
    output_npz: str
    output_json: str
    participant_id: str
    session_id: str
    run_id: str | None
    experiment_mode: str | None
    block: str
    n_trials: int
    eeg_shape: list[int] | None
    emg_shape: list[int] | None
    imu_shape: list[int] | None
    label_counts: dict[str, int]
    preprocessing: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recordings-root", required=True, help="Root folder containing XDF recordings")
    parser.add_argument("--processed-root", required=True, help="Output folder for processed npz/json files")
    parser.add_argument(
        "--target-task",
        choices=["silent", "imagined", "calibration", "all"],
        default="silent",
        help="Which block/task to process",
    )
    parser.add_argument("--recursive", action="store_true", help="Scan subfolders recursively")
    parser.add_argument("--window-pre", type=float, default=1.0, help="Seconds before event")
    parser.add_argument("--window-post", type=float, default=2.0, help="Seconds after event")
    parser.add_argument(
        "--cue-delay",
        type=float,
        default=0.0,
        help="Fallback delay for old Cue_Start markers. Use 0.0 when Cue_Start already marks the green/GO onset.",
    )
    parser.add_argument("--target-size", type=int, default=256, help="Resampled samples per trial")
    parser.add_argument("--eeg-channels", type=int, default=32)
    parser.add_argument("--emg-channels", type=int, default=5)
    parser.add_argument("--imu-channels-per-stream", type=int, default=9)
    parser.add_argument("--notch-hz", type=float, default=50.0)
    parser.add_argument("--eeg-low", type=float, default=1.0)
    parser.add_argument("--eeg-high", type=float, default=40.0)
    parser.add_argument("--emg-low", type=float, default=20.0)
    parser.add_argument("--emg-high", type=float, default=100.0)
    parser.add_argument("--imu-lowpass", type=float, default=15.0)

    # v3 modality-specific preprocessing options. All are trial-level and online reproducible.
    parser.add_argument("--eeg-car", action="store_true", help="Apply EEG common average reference after filtering")
    parser.add_argument("--eeg-baseline-correct", action="store_true", help="Subtract pre-GO EEG baseline from each epoch")
    parser.add_argument(
        "--eeg-mode",
        choices=["raw", "multiband"],
        default="raw",
        help="EEG representation saved to the processed dataset. multiband stacks several filtered EEG copies as extra channels.",
    )
    parser.add_argument(
        "--eeg-bands",
        default="4-8,8-13,13-30,1-40",
        help="Comma-separated EEG bands used when --eeg-mode multiband, e.g. '4-8,8-13,13-30,1-40'.",
    )
    parser.add_argument(
        "--emg-mode",
        choices=["raw", "envelope", "raw_envelope"],
        default="raw",
        help="EMG representation saved to the processed dataset",
    )
    parser.add_argument("--emg-envelope-lowpass", type=float, default=8.0, help="Low-pass cutoff for rectified EMG envelope")
    parser.add_argument("--emg-baseline-correct", action="store_true", help="Subtract pre-GO EMG baseline from each epoch")
    parser.add_argument(
        "--imu-use-axes",
        choices=["all", "accgyro"],
        default="all",
        help="For 9-axis IMU streams, keep all axes or only accelerometer+gyroscope axes",
    )
    parser.add_argument(
        "--imu-mode",
        choices=["raw", "basecorr", "raw_delta", "basecorr_delta"],
        default="raw",
        help="IMU epoch representation. Delta modes append first temporal derivative channels.",
    )

    parser.add_argument("--allow-missing-emg", action="store_true")
    parser.add_argument("--allow-missing-imu", action="store_true")
    parser.add_argument("--save-index-json", action="store_true")
    return parser.parse_args()


def normalize_name(value: str | None) -> str:
    return (value or "").strip().lower()


def safe_first(info_field: Any, default: str = "") -> str:
    if isinstance(info_field, list) and info_field:
        return str(info_field[0])
    return default


def parse_filename_meta(path: Path) -> FileMeta | None:
    match = NEW_NAME_RE.match(path.name)
    if match:
        gd = match.groupdict()
        return FileMeta(
            path=str(path),
            participant_id=gd["sub"],
            session_id=gd["ses"],
            run_id=gd["run"],
            experiment_mode=gd["mode"],
            block=gd["block"],
        )

    match = OLD_NAME_RE.match(path.name)
    if match:
        gd = match.groupdict()
        return FileMeta(
            path=str(path),
            participant_id=gd["sub"],
            session_id=gd["ses"],
            run_id=None,
            experiment_mode=None,
            block=gd["block"],
        )

    return None


def iter_xdf_files(root: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.xdf" if recursive else "*.xdf"
    return sorted(root.glob(pattern))


def _safe_filter_data(data: np.ndarray) -> np.ndarray:
    return np.asarray(data, dtype=float)


def butter_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4) -> np.ndarray:
    data = _safe_filter_data(data)
    nyq = 0.5 * float(fs)
    if nyq <= 0:
        return data
    low = max(float(lowcut) / nyq, 1e-6)
    high = min(float(highcut) / nyq, 0.999999)
    if low <= 0 or high <= 0 or low >= high or data.shape[0] < 32:
        return data
    b, a = butter(order, [low, high], btype="bandpass")
    return filtfilt(b, a, data, axis=0)


def low_pass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    data = _safe_filter_data(data)
    nyq = 0.5 * float(fs)
    if nyq <= 0:
        return data
    cutoff_norm = min(float(cutoff) / nyq, 0.999999)
    if cutoff_norm <= 0 or data.shape[0] < 32:
        return data
    b, a = butter(order, cutoff_norm, btype="low")
    return filtfilt(b, a, data, axis=0)


def notch_filter(data: np.ndarray, fs: float, freq: float = 50.0) -> np.ndarray:
    data = _safe_filter_data(data)
    nyq = 0.5 * float(fs)
    if nyq <= 0 or freq <= 0 or freq >= 0.98 * nyq or data.shape[0] < 32:
        return data
    w0 = float(freq) / nyq
    b, a = iirnotch(w0, Q=30)
    return filtfilt(b, a, data, axis=0)


def common_average_reference(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        return data
    return data - data.mean(axis=1, keepdims=True)


def parse_eeg_bands(eeg_bands: str, fallback_low: float, fallback_high: float) -> list[tuple[float, float]]:
    bands: list[tuple[float, float]] = []
    for item in str(eeg_bands or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "-" not in item:
            continue
        left, right = item.split("-", 1)
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


def eeg_bands_to_string(bands: list[tuple[float, float]]) -> str:
    return ",".join(f"{low:g}-{high:g}" for low, high in bands)


def select_imu_axes(data: np.ndarray, raw_channels_per_stream: int, imu_use_axes: str) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    if imu_use_axes == "all":
        return data

    if imu_use_axes != "accgyro":
        raise ValueError(f"Unsupported imu_use_axes: {imu_use_axes}")

    raw_channels_per_stream = int(raw_channels_per_stream)
    if raw_channels_per_stream <= 0:
        return data

    # If the input still contains full 9-axis streams, keep first 6 channels per stream.
    if data.shape[1] % raw_channels_per_stream == 0:
        n_streams = data.shape[1] // raw_channels_per_stream
        chunks = []
        keep = min(6, raw_channels_per_stream)
        for i in range(n_streams):
            start = i * raw_channels_per_stream
            chunks.append(data[:, start:start + keep])
        return np.hstack(chunks) if chunks else data

    # If already reduced, leave unchanged.
    if data.shape[1] % 6 == 0:
        return data

    # Last-resort fallback: keep the first six channels if present.
    return data[:, :min(6, data.shape[1])]


def add_temporal_delta(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    if data.shape[0] < 2:
        delta = np.zeros_like(data)
    else:
        delta = np.diff(data, axis=0, prepend=data[:1])
    return np.concatenate([data, delta], axis=1)


def baseline_correct_epoch(epoch: np.ndarray, window_pre: float, window_post: float) -> np.ndarray:
    epoch = np.asarray(epoch, dtype=float)
    duration = float(window_pre) + float(window_post)
    if duration <= 0 or window_pre <= 0 or epoch.shape[0] < 2:
        return epoch
    baseline_n = int(round(epoch.shape[0] * (float(window_pre) / duration)))
    baseline_n = max(1, min(baseline_n, epoch.shape[0] - 1))
    baseline = epoch[:baseline_n].mean(axis=0, keepdims=True)
    return epoch - baseline


def get_stream_name(stream: dict) -> str:
    return safe_first(stream.get("info", {}).get("name"), "")


def get_stream_type(stream: dict) -> str:
    return safe_first(stream.get("info", {}).get("type"), "")


def get_stream_fs(stream: dict, default: float = 256.0) -> float:
    try:
        fs = float(safe_first(stream.get("info", {}).get("nominal_srate"), "0"))
        return fs if fs > 0 else default
    except Exception:
        return default


def stream_marker_values(stream: dict) -> list[str]:
    values = []
    for row in stream.get("time_series", []):
        if isinstance(row, (list, tuple, np.ndarray)) and len(row) > 0:
            values.append(str(row[0]).strip())
        else:
            values.append(str(row).strip())
    return values


def stream_has_task_markers(stream: dict) -> bool:
    for label_text in stream_marker_values(stream):
        if (
            CUE_START_RE.match(label_text)
            or SILENT_GO_RE.match(label_text)
            or IMAGINED_GO_RE.match(label_text)
        ):
            return True
    return False


def select_best_marker_stream(streams: list[dict]) -> dict | None:
    marker_candidates = []

    for stream in streams:
        name = normalize_name(get_stream_name(stream))
        stream_type = normalize_name(get_stream_type(stream))
        if stream_type == "markers" or "marker" in name:
            marker_candidates.append(stream)

    if not marker_candidates:
        return None

    for stream in marker_candidates:
        if normalize_name(get_stream_name(stream)) == "psychopy_markers":
            return stream

    for stream in marker_candidates:
        if stream_has_task_markers(stream):
            return stream

    return marker_candidates[0]


def load_session(path: Path) -> dict[str, Any]:
    streams, _ = pyxdf.load_xdf(str(path))

    eeg_stream = None
    emg_stream = None
    imu_streams: dict[str, dict] = {}

    for stream in streams:
        name = normalize_name(get_stream_name(stream))
        stream_type = normalize_name(get_stream_type(stream))

        if eeg_stream is None and (
            stream_type == "eeg"
            or name.endswith("_eeg")
            or "speech_eeg" in name
            or name == "e32_speech_eeg"
        ):
            eeg_stream = stream
            continue

        if emg_stream is None and (
            stream_type in {"emg", "exg"}
            or "emg-imu_exg" in name
            or ("exg" in name and "imu" in name)
        ):
            emg_stream = stream
            continue

        if "daux" in name or stream_type == "imu":
            imu_streams[get_stream_name(stream)] = stream

    marker_stream = select_best_marker_stream(streams)

    return {
        "eeg": eeg_stream,
        "emg": emg_stream,
        "imu": imu_streams,
        "markers": marker_stream,
    }


def preprocess_eeg(
    stream: dict,
    eeg_channels: int,
    notch_hz: float,
    low_hz: float,
    high_hz: float,
    eeg_car: bool,
    eeg_mode: str,
    eeg_bands: str,
) -> dict[str, Any]:
    data = np.asarray(stream["time_series"], dtype=float)[:, :eeg_channels]
    fs = get_stream_fs(stream)

    eeg_mode = str(eeg_mode or "raw")
    if eeg_mode == "raw":
        out = notch_filter(data, fs, notch_hz)
        out = butter_bandpass_filter(out, low_hz, high_hz, fs)
        if eeg_car:
            out = common_average_reference(out)
        used_bands = [(float(low_hz), float(high_hz))]

    elif eeg_mode == "multiband":
        used_bands = parse_eeg_bands(eeg_bands, fallback_low=low_hz, fallback_high=high_hz)
        notched = notch_filter(data, fs, notch_hz)
        band_arrays = []
        for band_low, band_high in used_bands:
            band = butter_bandpass_filter(notched, band_low, band_high, fs)
            if eeg_car:
                band = common_average_reference(band)
            band_arrays.append(band)
        out = np.concatenate(band_arrays, axis=1) if band_arrays else notched

    else:
        raise ValueError(f"Unsupported eeg_mode: {eeg_mode}")

    return {
        "time_series": out,
        "time_stamps": np.asarray(stream["time_stamps"], dtype=float),
        "fs": fs,
        "eeg_mode": eeg_mode,
        "eeg_bands": used_bands,
    }


def preprocess_emg(
    stream: dict,
    emg_channels: int,
    notch_hz: float,
    low_hz: float,
    high_hz: float,
    emg_mode: str,
    envelope_lowpass_hz: float,
) -> dict[str, Any]:
    data = np.asarray(stream["time_series"], dtype=float)[:, :emg_channels]
    fs = get_stream_fs(stream)

    raw = butter_bandpass_filter(data, low_hz, high_hz, fs)
    raw = notch_filter(raw, fs, notch_hz)

    if emg_mode == "raw":
        out = raw
    elif emg_mode == "envelope":
        out = low_pass_filter(np.abs(raw), envelope_lowpass_hz, fs)
    elif emg_mode == "raw_envelope":
        envelope = low_pass_filter(np.abs(raw), envelope_lowpass_hz, fs)
        out = np.concatenate([raw, envelope], axis=1)
    else:
        raise ValueError(f"Unsupported emg_mode: {emg_mode}")

    return {"time_series": out, "time_stamps": np.asarray(stream["time_stamps"], dtype=float), "fs": fs}


def preprocess_imu(
    imu_streams: dict[str, dict],
    imu_channels_per_stream: int,
    cutoff_hz: float,
    imu_use_axes: str,
) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for name, stream in imu_streams.items():
        data = np.asarray(stream["time_series"], dtype=float)[:, :imu_channels_per_stream]
        fs = get_stream_fs(stream)
        data = select_imu_axes(data, raw_channels_per_stream=imu_channels_per_stream, imu_use_axes=imu_use_axes)
        data = detrend(data, axis=0)
        data = low_pass_filter(data, cutoff_hz, fs)
        cleaned[name] = {
            "time_series": data,
            "time_stamps": np.asarray(stream["time_stamps"], dtype=float),
            "fs": fs,
        }
    return cleaned


def apply_epoch_transforms(
    epoch: np.ndarray,
    modality: str,
    window_pre: float,
    window_post: float,
    eeg_baseline_correct: bool,
    emg_baseline_correct: bool,
    imu_mode: str,
) -> np.ndarray:
    out = np.asarray(epoch, dtype=float)

    if modality == "eeg":
        if eeg_baseline_correct:
            out = baseline_correct_epoch(out, window_pre, window_post)
        return out

    if modality == "emg":
        if emg_baseline_correct:
            out = baseline_correct_epoch(out, window_pre, window_post)
        return out

    if modality == "imu":
        if imu_mode in {"basecorr", "basecorr_delta"}:
            out = baseline_correct_epoch(out, window_pre, window_post)
        if imu_mode in {"raw_delta", "basecorr_delta"}:
            out = add_temporal_delta(out)
        return out

    return out


def slice_by_time(stream: dict[str, Any], t_start: float, t_end: float) -> np.ndarray | None:
    ts = np.asarray(stream["time_stamps"], dtype=float)
    data = np.asarray(stream["time_series"], dtype=float)
    mask = (ts >= t_start) & (ts <= t_end)
    if not np.any(mask):
        return None
    clipped = data[mask]
    return clipped if clipped.shape[0] >= 2 else None


def slice_imu_epoch(imu_streams: dict[str, Any], t_start: float, t_end: float) -> np.ndarray | None:
    if not imu_streams:
        return None

    arrays = []
    for name in sorted(imu_streams.keys()):
        clipped = slice_by_time(imu_streams[name], t_start, t_end)
        if clipped is None or clipped.shape[0] < 2:
            return None
        arrays.append(clipped)

    min_len = min(arr.shape[0] for arr in arrays)
    arrays = [arr[:min_len] for arr in arrays]
    return np.hstack(arrays)


def resample_epochs(epochs: list[np.ndarray], target_size: int) -> np.ndarray:
    return np.asarray([resample(epoch, target_size, axis=0) for epoch in epochs], dtype=np.float32)


def parse_events(marker_stream: dict, target_task: str, cue_delay: float) -> list[dict[str, Any]]:
    explicit_events: list[dict[str, Any]] = []
    fallback_events: list[dict[str, Any]] = []

    marker_values = marker_stream["time_series"]
    marker_times = np.asarray(marker_stream["time_stamps"], dtype=float)

    for marker_val, ts in zip(marker_values, marker_times):
        if isinstance(marker_val, (list, tuple, np.ndarray)) and len(marker_val) > 0:
            label_text = str(marker_val[0]).strip()
        else:
            label_text = str(marker_val).strip()

        silent_match = SILENT_GO_RE.match(label_text)
        if target_task in {"silent", "all"} and silent_match:
            word = silent_match.group(1).strip()
            if word in LABEL_MAP:
                explicit_events.append({"word": word, "label_id": LABEL_MAP[word], "event_time": float(ts), "marker_type": "SILENT_GO"})
            continue

        imagined_match = IMAGINED_GO_RE.match(label_text)
        if target_task in {"imagined", "all"} and imagined_match:
            word = imagined_match.group(1).strip()
            if word in LABEL_MAP:
                explicit_events.append({"word": word, "label_id": LABEL_MAP[word], "event_time": float(ts), "marker_type": "IMAGINED_GO"})
            continue

        old_match = CUE_START_RE.match(label_text)
        if old_match:
            word = old_match.group(1).strip()
            if word in LABEL_MAP:
                fallback_events.append({"word": word, "label_id": LABEL_MAP[word], "event_time": float(ts + cue_delay), "marker_type": "Cue_Start"})

    if explicit_events:
        return explicit_events
    return fallback_events


def build_preprocessing_tag(
    window_pre: float,
    window_post: float,
    target_size: int,
    eeg_low: float,
    eeg_high: float,
    eeg_mode: str,
    eeg_bands_string: str,
    emg_low: float,
    emg_high: float,
    imu_lowpass: float,
    eeg_car: bool,
    eeg_baseline_correct: bool,
    emg_mode: str,
    emg_baseline_correct: bool,
    imu_use_axes: str,
    imu_mode: str,
) -> str:
    parts = [
        f"w{(-window_pre):+.2f}_{window_post:+.2f}",
        f"n{target_size}",
        f"eeg{eeg_low:.1f}-{eeg_high:.1f}",
        f"eeg{eeg_mode}",
        f"bands{eeg_bands_string.replace(',', '+')}",
        "car" if eeg_car else "nocAR",
        "eegbase" if eeg_baseline_correct else "noeegbase",
        f"emg{emg_low:.1f}-{emg_high:.1f}",
        f"emg{emg_mode}",
        "emgbase" if emg_baseline_correct else "noemgbase",
        f"imuLP{imu_lowpass:.1f}",
        f"imu{imu_use_axes}",
        f"imu{imu_mode}",
    ]
    return "_".join(parts)


def process_xdf_file(
    xdf_path: Path,
    processed_root: Path,
    target_task: str,
    window_pre: float,
    window_post: float,
    cue_delay: float,
    target_size: int,
    eeg_channels: int,
    emg_channels: int,
    imu_channels_per_stream: int,
    notch_hz: float,
    eeg_low: float,
    eeg_high: float,
    emg_low: float,
    emg_high: float,
    imu_lowpass: float,
    eeg_car: bool,
    eeg_baseline_correct: bool,
    eeg_mode: str,
    eeg_bands: str,
    emg_mode: str,
    emg_envelope_lowpass: float,
    emg_baseline_correct: bool,
    imu_use_axes: str,
    imu_mode: str,
    allow_missing_emg: bool,
    allow_missing_imu: bool,
) -> SessionOutput | None:
    meta = parse_filename_meta(xdf_path)
    if meta is None:
        print(f"[skip] filename not recognized: {xdf_path.name}")
        return None

    if meta.experiment_mode in {"online_silent", "online_imagined"}:
        print(f"[skip] online experiment excluded from training: {xdf_path.name}")
        return None

    if target_task != "all" and meta.block.lower() != target_task.lower():
        return None

    session = load_session(xdf_path)

    if session["markers"] is None:
        print(f"[skip] no marker stream: {xdf_path.name}")
        return None

    if session["eeg"] is None:
        print(f"[skip] no EEG stream: {xdf_path.name}")
        return None

    eeg = preprocess_eeg(
        session["eeg"],
        eeg_channels,
        notch_hz,
        eeg_low,
        eeg_high,
        eeg_car=eeg_car,
        eeg_mode=eeg_mode,
        eeg_bands=eeg_bands,
    )

    emg = None
    if session["emg"] is not None:
        emg = preprocess_emg(
            session["emg"],
            emg_channels,
            notch_hz,
            emg_low,
            emg_high,
            emg_mode=emg_mode,
            envelope_lowpass_hz=emg_envelope_lowpass,
        )
    elif not allow_missing_emg:
        print(f"[skip] no EMG stream: {xdf_path.name}")
        return None

    imu = None
    if session["imu"]:
        imu = preprocess_imu(
            session["imu"],
            imu_channels_per_stream,
            imu_lowpass,
            imu_use_axes=imu_use_axes,
        )
    elif not allow_missing_imu:
        print(f"[skip] no IMU streams: {xdf_path.name}")
        return None

    events = parse_events(session["markers"], target_task=target_task, cue_delay=cue_delay)
    if not events:
        print(f"[skip] no usable events: {xdf_path.name}")
        return None

    eeg_epochs: list[np.ndarray] = []
    emg_epochs: list[np.ndarray] = []
    imu_epochs: list[np.ndarray] = []
    labels: list[int] = []
    words: list[str] = []
    marker_types: list[str] = []

    for event in events:
        t0 = float(event["event_time"])
        t_start = t0 - window_pre
        t_end = t0 + window_post

        eeg_epoch = slice_by_time(eeg, t_start, t_end)
        if eeg_epoch is None:
            continue
        eeg_epoch = apply_epoch_transforms(eeg_epoch, "eeg", window_pre, window_post, eeg_baseline_correct, emg_baseline_correct, imu_mode)

        emg_epoch = None
        if emg is not None:
            emg_epoch = slice_by_time(emg, t_start, t_end)
            if emg_epoch is None:
                continue
            emg_epoch = apply_epoch_transforms(emg_epoch, "emg", window_pre, window_post, eeg_baseline_correct, emg_baseline_correct, imu_mode)

        imu_epoch = None
        if imu is not None:
            imu_epoch = slice_imu_epoch(imu, t_start, t_end)
            if imu_epoch is None:
                continue
            imu_epoch = apply_epoch_transforms(imu_epoch, "imu", window_pre, window_post, eeg_baseline_correct, emg_baseline_correct, imu_mode)

        eeg_epochs.append(eeg_epoch)
        if emg_epoch is not None:
            emg_epochs.append(emg_epoch)
        if imu_epoch is not None:
            imu_epochs.append(imu_epoch)

        labels.append(int(event["label_id"]))
        words.append(str(event["word"]))
        marker_types.append(str(event["marker_type"]))

    if not eeg_epochs:
        print(f"[skip] no valid epochs after slicing: {xdf_path.name}")
        return None

    eeg_arr = resample_epochs(eeg_epochs, target_size)

    emg_arr = None
    if emg is not None:
        emg_arr = resample_epochs(emg_epochs, target_size)

    imu_arr = None
    if imu is not None:
        imu_arr = resample_epochs(imu_epochs, target_size)

    participant_dir = processed_root / f"sub-{meta.participant_id}" / f"ses-{meta.session_id}"
    participant_dir.mkdir(parents=True, exist_ok=True)

    stem = xdf_path.stem
    npz_path = participant_dir / f"{stem}_preprocessed.npz"
    json_path = participant_dir / f"{stem}_preprocessed.json"

    imu_fs_by_stream = {}
    imu_stream_names: list[str] = []
    if imu is not None:
        imu_stream_names = sorted(imu.keys())
        imu_fs_by_stream = {name: float(stream_info["fs"]) for name, stream_info in imu.items()}

    eeg_bands_used: list[tuple[float, float]] = list(eeg.get("eeg_bands", [(float(eeg_low), float(eeg_high))]))
    eeg_bands_string = eeg_bands_to_string(eeg_bands_used)

    preprocessing_tag = build_preprocessing_tag(
        window_pre=window_pre,
        window_post=window_post,
        target_size=target_size,
        eeg_low=eeg_low,
        eeg_high=eeg_high,
        eeg_mode=eeg_mode,
        eeg_bands_string=eeg_bands_string,
        emg_low=emg_low,
        emg_high=emg_high,
        imu_lowpass=imu_lowpass,
        eeg_car=eeg_car,
        eeg_baseline_correct=eeg_baseline_correct,
        emg_mode=emg_mode,
        emg_baseline_correct=emg_baseline_correct,
        imu_use_axes=imu_use_axes,
        imu_mode=imu_mode,
    )

    preprocessing_meta: dict[str, Any] = {
        "preprocessing_version": "portable_v4_eeg_multiband_preprocessing",
        "preprocessing_tag": preprocessing_tag,
        "target_task": target_task,
        "window_pre_s": float(window_pre),
        "window_post_s": float(window_post),
        "window_start_s": float(-window_pre),
        "window_end_s": float(window_post),
        "target_size": int(target_size),
        "resample_size": int(target_size),
        "cue_delay_s": float(cue_delay),
        "notch_hz": float(notch_hz),
        "eeg_low_hz": float(eeg_low),
        "eeg_high_hz": float(eeg_high),
        "eeg_car": bool(eeg_car),
        "eeg_baseline_correct": bool(eeg_baseline_correct),
        "eeg_mode": str(eeg_mode),
        "eeg_bands": [[float(low), float(high)] for low, high in eeg_bands_used],
        "eeg_bands_string": eeg_bands_string,
        "emg_low_hz": float(emg_low),
        "emg_high_hz": float(emg_high),
        "emg_mode": str(emg_mode),
        "emg_envelope_lowpass_hz": float(emg_envelope_lowpass),
        "emg_baseline_correct": bool(emg_baseline_correct),
        "imu_lowpass_hz": float(imu_lowpass),
        "imu_use_axes": str(imu_use_axes),
        "imu_mode": str(imu_mode),
        "eeg_channels_requested": int(eeg_channels),
        "emg_channels_requested": int(emg_channels),
        "imu_channels_per_stream_requested": int(imu_channels_per_stream),
        "imu_stream_names": imu_stream_names,
        "eeg_fs_hz": float(eeg["fs"]),
        "emg_fs_hz": float(emg["fs"]) if emg is not None else None,
        "imu_fs_hz_by_stream": imu_fs_by_stream,
        "eeg_output_channels": int(eeg_arr.shape[2]),
        "emg_output_channels": int(emg_arr.shape[2]) if emg_arr is not None else None,
        "imu_output_channels": int(imu_arr.shape[2]) if imu_arr is not None else None,
        "online_compatibility": "trial-level online reproducible; filtering uses zero-phase filtfilt after each completed trial window",
    }

    save_payload = {
        "eeg": eeg_arr,
        "labels": np.asarray(labels, dtype=np.int64),
        "label_names": np.asarray(words, dtype=object),
        "marker_types": np.asarray(marker_types, dtype=object),
        "participant_id": np.asarray([meta.participant_id], dtype=object),
        "session_id": np.asarray([meta.session_id], dtype=object),
        "run_id": np.asarray([meta.run_id if meta.run_id is not None else ""], dtype=object),
        "experiment_mode": np.asarray([meta.experiment_mode if meta.experiment_mode is not None else ""], dtype=object),
        "block": np.asarray([meta.block], dtype=object),
        "source_xdf": np.asarray([str(xdf_path)], dtype=object),
        "preprocessing_json": np.asarray([json.dumps(preprocessing_meta, sort_keys=True)], dtype=object),
        "preprocessing_version": np.asarray([preprocessing_meta["preprocessing_version"]], dtype=object),
        "preprocessing_tag": np.asarray([preprocessing_meta["preprocessing_tag"]], dtype=object),
        "window_pre_s": np.asarray(preprocessing_meta["window_pre_s"], dtype=np.float32),
        "window_post_s": np.asarray(preprocessing_meta["window_post_s"], dtype=np.float32),
        "window_start_s": np.asarray(preprocessing_meta["window_start_s"], dtype=np.float32),
        "window_end_s": np.asarray(preprocessing_meta["window_end_s"], dtype=np.float32),
        "target_size": np.asarray(preprocessing_meta["target_size"], dtype=np.int64),
        "resample_size": np.asarray(preprocessing_meta["resample_size"], dtype=np.int64),
        "cue_delay_s": np.asarray(preprocessing_meta["cue_delay_s"], dtype=np.float32),
        "notch_hz": np.asarray(preprocessing_meta["notch_hz"], dtype=np.float32),
        "eeg_low_hz": np.asarray(preprocessing_meta["eeg_low_hz"], dtype=np.float32),
        "eeg_high_hz": np.asarray(preprocessing_meta["eeg_high_hz"], dtype=np.float32),
        "eeg_car": np.asarray(preprocessing_meta["eeg_car"], dtype=bool),
        "eeg_baseline_correct": np.asarray(preprocessing_meta["eeg_baseline_correct"], dtype=bool),
        "eeg_mode": np.asarray([preprocessing_meta["eeg_mode"]], dtype=object),
        "eeg_bands_json": np.asarray([json.dumps(preprocessing_meta["eeg_bands"])], dtype=object),
        "eeg_bands_string": np.asarray([preprocessing_meta["eeg_bands_string"]], dtype=object),
        "emg_low_hz": np.asarray(preprocessing_meta["emg_low_hz"], dtype=np.float32),
        "emg_high_hz": np.asarray(preprocessing_meta["emg_high_hz"], dtype=np.float32),
        "emg_mode": np.asarray([preprocessing_meta["emg_mode"]], dtype=object),
        "emg_envelope_lowpass_hz": np.asarray(preprocessing_meta["emg_envelope_lowpass_hz"], dtype=np.float32),
        "emg_baseline_correct": np.asarray(preprocessing_meta["emg_baseline_correct"], dtype=bool),
        "imu_lowpass_hz": np.asarray(preprocessing_meta["imu_lowpass_hz"], dtype=np.float32),
        "imu_use_axes": np.asarray([preprocessing_meta["imu_use_axes"]], dtype=object),
        "imu_mode": np.asarray([preprocessing_meta["imu_mode"]], dtype=object),
        "eeg_fs_hz": np.asarray(preprocessing_meta["eeg_fs_hz"], dtype=np.float32),
        "emg_fs_hz": np.asarray(preprocessing_meta["emg_fs_hz"] if preprocessing_meta["emg_fs_hz"] is not None else np.nan, dtype=np.float32),
    }

    if emg_arr is not None:
        save_payload["emg"] = emg_arr
    if imu_arr is not None:
        save_payload["imu"] = imu_arr

    np.savez_compressed(npz_path, **save_payload)

    counts: dict[str, int] = {}
    for label in words:
        counts[label] = counts.get(label, 0) + 1

    summary = SessionOutput(
        source_xdf=str(xdf_path),
        output_npz=str(npz_path),
        output_json=str(json_path),
        participant_id=meta.participant_id,
        session_id=meta.session_id,
        run_id=meta.run_id,
        experiment_mode=meta.experiment_mode,
        block=meta.block,
        n_trials=len(labels),
        eeg_shape=list(eeg_arr.shape),
        emg_shape=list(emg_arr.shape) if emg_arr is not None else None,
        imu_shape=list(imu_arr.shape) if imu_arr is not None else None,
        label_counts=counts,
        preprocessing=preprocessing_meta,
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    print(f"[ok] {xdf_path.name} -> {npz_path.name} | trials={len(labels)}")
    return summary


def main() -> int:
    args = parse_args()

    recordings_root = Path(args.recordings_root).resolve()
    processed_root = Path(args.processed_root).resolve()
    processed_root.mkdir(parents=True, exist_ok=True)

    if not recordings_root.exists():
        raise FileNotFoundError(f"Recordings root does not exist: {recordings_root}")

    xdf_files = iter_xdf_files(recordings_root, recursive=args.recursive)
    if not xdf_files:
        print("[info] no xdf files found")
        return 0

    summaries: list[SessionOutput] = []

    for xdf_path in xdf_files:
        result = process_xdf_file(
            xdf_path=xdf_path,
            processed_root=processed_root,
            target_task=args.target_task,
            window_pre=args.window_pre,
            window_post=args.window_post,
            cue_delay=args.cue_delay,
            target_size=args.target_size,
            eeg_channels=args.eeg_channels,
            emg_channels=args.emg_channels,
            imu_channels_per_stream=args.imu_channels_per_stream,
            notch_hz=args.notch_hz,
            eeg_low=args.eeg_low,
            eeg_high=args.eeg_high,
            emg_low=args.emg_low,
            emg_high=args.emg_high,
            imu_lowpass=args.imu_lowpass,
            eeg_car=args.eeg_car,
            eeg_baseline_correct=args.eeg_baseline_correct,
            eeg_mode=args.eeg_mode,
            eeg_bands=args.eeg_bands,
            emg_mode=args.emg_mode,
            emg_envelope_lowpass=args.emg_envelope_lowpass,
            emg_baseline_correct=args.emg_baseline_correct,
            imu_use_axes=args.imu_use_axes,
            imu_mode=args.imu_mode,
            allow_missing_emg=args.allow_missing_emg,
            allow_missing_imu=args.allow_missing_imu,
        )
        if result is not None:
            summaries.append(result)

    print(f"[done] processed {len(summaries)} file(s)")

    if args.save_index_json:
        index_path = processed_root / f"processed_index_{args.target_task}.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump([asdict(x) for x in summaries], f, indent=2)
        print(f"[done] wrote index: {index_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
