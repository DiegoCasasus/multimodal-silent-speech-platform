from __future__ import annotations

import argparse
import json
import os
import random
import re
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Project root is derived from this file, not from the current working directory.
# This keeps the decoder portable when it is launched by PsychoPy, a .bat file,
# an IDE, or a terminal opened in a different folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from model_runtime import BundleDecoderRuntime, STREAM_TO_MODALITY

try:
    from common.paths import resolve_project_path, to_project_relative
except Exception:
    resolve_project_path = None
    to_project_relative = None

import numpy as np
from pylsl import StreamInlet, resolve_byprop, resolve_streams

from decoder_runtime import compute_eeg_scale_from_comparison_json

SILENT_GO_RE = re.compile(r"^SILENT_GO:(.+)$")
SILENT_END_RE = re.compile(r"^SILENT_END:(.+)$")
IMAGINED_GO_RE = re.compile(r"^IMAGINED_GO:(.+)$")
IMAGINED_END_RE = re.compile(r"^IMAGINED_END:(.+)$")


DEFAULT_VOCABULARY = [
    "1 (ONE)",
    "2 (TWO)",
    "3 (THREE)",
    "4 (FOUR)",
    "5 (FIVE)",
    "REJOIN",
    "ASSET",
    "ABORT",
    "FORMATION",
    "REFUEL",
]


def get_bridge_project_root(bridge: dict) -> Path:
    """Return the project root used for resolving relative paths in bridge JSON.

    Current bridge files do not yet include project_root, so the safe default is
    the repository root derived from this file. Future portable bridge files can
    add project_root explicitly without breaking this runtime.
    """
    raw_root = bridge.get("project_root") or bridge.get("project_root_path")
    if raw_root:
        try:
            path = Path(str(raw_root)).expanduser()
            if path.is_absolute():
                return path.resolve()
            return (PROJECT_ROOT / path).resolve()
        except Exception:
            pass
    return PROJECT_ROOT


def resolve_runtime_path(
    value: str | os.PathLike[str] | Path | None,
    project_root: Path,
    *,
    label: str = "path",
    must_exist: bool = False,
) -> Path:
    """Resolve runtime paths without depending on the current working directory.

    Rules:
    - absolute paths are respected;
    - relative paths are resolved relative to project_root;
    - missing required paths produce clear errors.
    """
    if value is None or str(value).strip() == "":
        raise RuntimeError(f"Missing required {label}")

    if resolve_project_path is not None:
        path = resolve_project_path(value, project_root)
    else:
        raw = Path(str(value)).expanduser()
        path = raw.resolve() if raw.is_absolute() else (project_root / raw).resolve()

    if path is None:
        raise RuntimeError(f"Could not resolve {label}: {value}")

    if must_exist and not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")

    return path


def _looks_like_path_string(text: str) -> bool:
    """Return True for strings that are plausibly filesystem paths."""
    if not text:
        return False
    if text in {"ensemble", "none", "None", "null"}:
        return False
    if text.startswith(("~", ".", "/", "\\")):
        return True
    if re.match(r"^[A-Za-z]:[\\/]", text):
        return True
    if "/" in text or "\\" in text:
        return True
    return False


def portable_path_value(
    value: str | os.PathLike[str] | Path | None,
    project_root: Path,
) -> str | None:
    """Serialize project-internal paths as project-relative POSIX strings.

    This function is only used for JSON outputs. Runtime code still uses absolute
    Path objects after resolving bridge/config values.

    If a path points outside the project root, it is intentionally kept absolute
    because it is probably an external tool or user override.
    """
    if value is None:
        return None

    text = str(value).strip()
    if text == "":
        return ""

    if not _looks_like_path_string(text):
        return text

    if to_project_relative is not None:
        try:
            return to_project_relative(value, project_root)
        except Exception:
            pass

    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate

    try:
        return candidate.resolve().relative_to(project_root.resolve()).as_posix()
    except Exception:
        return str(candidate.resolve())


_PATH_LIKE_KEYS = {
    "bridge",
    "results_dir",
    "trial_dir",
    "bundle_path",
    "decoder_bundle_path",
    "model_bundle_path",
    "selected_bundle_path",
    "expected_xdf_path",
    "metadata_path",
    "manifest_path",
    "online_results_dir",
    "prepared_xdf_sidecar_path",
    "brainprint_compare_json",
    "brainprint_summary_json",
}


def make_json_portable(value, project_root: Path):
    """Recursively convert known path fields before writing JSON outputs.

    The conversion is conservative: normal strings such as labels, stream names,
    model keys, source IDs, and filenames are left untouched. Only known path-like
    keys and actual Path objects are converted.
    """
    if isinstance(value, Path):
        return portable_path_value(value, project_root)

    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            key_str = str(key)
            if (
                key_str in _PATH_LIKE_KEYS
                or key_str.endswith("_path")
                or key_str.endswith("_dir")
            ):
                if isinstance(item, (str, os.PathLike, Path)) or item is None:
                    out[key] = portable_path_value(item, project_root)
                else:
                    out[key] = make_json_portable(item, project_root)
            else:
                out[key] = make_json_portable(item, project_root)
        return out

    if isinstance(value, list):
        return [make_json_portable(item, project_root) for item in value]

    if isinstance(value, tuple):
        return [make_json_portable(item, project_root) for item in value]

    return value


def add_output_path_metadata(payload):
    """Add path-format metadata to decoder JSON outputs."""
    if isinstance(payload, dict):
        payload = dict(payload)
        payload.setdefault("path_format_version", "project_relative_v1")
        payload.setdefault("paths_are_relative_to", "project_root")
    return payload


def resolve_bundle_path(
    cli_bundle_path: str | None,
    bridge: dict,
    project_root: Path | None = None,
) -> Path:
    candidate_keys = [
        "decoder_bundle_path",
        "model_bundle_path",
        "bundle_path",
        "selected_bundle_path",
    ]
    root = project_root or get_bridge_project_root(bridge)

    if cli_bundle_path:
        return resolve_runtime_path(
            cli_bundle_path,
            root,
            label="decoder bundle path passed with --bundle-path",
            must_exist=True,
        )

    for key in candidate_keys:
        value = bridge.get(key)
        if value:
            path = resolve_runtime_path(
                value,
                root,
                label=f"bridge field {key}",
                must_exist=False,
            )
            if path.exists():
                return path

    raise RuntimeError(
        "Could not determine decoder bundle path. "
        "Pass --bundle-path explicitly or store it in the bridge JSON."
    )


def _safe_float(value, default=None):
    try:
        if value is None:
            return default
        out = float(value)
        if not np.isfinite(out):
            return default
        return out
    except Exception:
        return default


def resolve_model_window_offsets(bundle_meta: dict, args: argparse.Namespace) -> tuple[float, float, str]:
    """Return model inference window offsets relative to GO, in seconds.

    The first choice is the exported bundle preprocessing metadata. This keeps the online
    runtime aligned with the training epochs. CLI overrides are available for emergency
    debugging but should not be used for final experiments unless documented.
    """
    if args.model_window_start_s is not None and args.model_window_end_s is not None:
        start = float(args.model_window_start_s)
        end = float(args.model_window_end_s)
        if end <= start:
            raise ValueError("--model-window-end-s must be greater than --model-window-start-s")
        return start, end, "cli_override"

    meta = (
        bundle_meta.get("preprocessing_meta")
        or bundle_meta.get("preprocessing")
        or {}
    )

    start = _safe_float(meta.get("window_start_s"), None)
    end = _safe_float(meta.get("window_end_s"), None)
    if start is not None and end is not None and end > start:
        return float(start), float(end), "bundle.window_start_s/window_end_s"

    pre = _safe_float(meta.get("window_pre_s"), None)
    post = _safe_float(meta.get("window_post_s"), None)
    if pre is not None and post is not None and (pre + post) > 0:
        return -float(pre), float(post), "bundle.window_pre_s/window_post_s"

    # Legacy fallback. Current training preprocessing defaults are GO-1.0s to GO+2.0s.
    start = -float(args.pre_seconds)
    end = float(args.post_seconds)
    if end <= start:
        raise ValueError("Legacy fallback window is invalid; check --pre-seconds/--post-seconds")
    return start, end, "legacy_cli_fallback"




# -----------------------------------------------------------------------------
# Ensemble decoder support
# -----------------------------------------------------------------------------
def resolve_ensemble_config(bridge: dict) -> dict | None:
    """Return decoder_ensemble config from bridge JSON, if present."""
    cfg = bridge.get("decoder_ensemble") or bridge.get("ensemble_decoder")
    if not isinstance(cfg, dict):
        return None
    components = cfg.get("components", [])
    if not isinstance(components, list) or not components:
        return None
    return cfg


def prepare_ensemble_components(
    ensemble_config: dict,
    device_arg: str,
    project_root: Path | None = None,
) -> list[dict]:
    components: list[dict] = []
    root = project_root or PROJECT_ROOT

    for idx, raw in enumerate(ensemble_config.get("components", [])):
        if not isinstance(raw, dict):
            continue
        path_value = raw.get("bundle_path")
        if not path_value:
            raise RuntimeError(f"Ensemble component {idx} has no bundle_path")

        path = resolve_runtime_path(
            path_value,
            root,
            label=f"ensemble component {idx} bundle_path",
            must_exist=True,
        )

        runtime = BundleDecoderRuntime.load(path, device_arg=device_arg)
        try:
            weight = float(raw.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0

        components.append(
            {
                "name": str(raw.get("name") or path.stem),
                "bundle_path": path,
                "weight": weight,
                "runtime": runtime,
                "required_stream_modalities": runtime.required_stream_modalities(),
            }
        )

    if not components:
        raise RuntimeError("Ensemble config did not contain any valid components")

    total_weight = sum(max(0.0, float(c["weight"])) for c in components)
    if total_weight <= 0.0:
        raise RuntimeError("Ensemble component weights must sum to a positive value")

    for comp in components:
        comp["normalized_weight"] = max(0.0, float(comp["weight"])) / total_weight

    return components


def ensemble_required_stream_modalities(components: list[dict]) -> list[str]:
    found = set()
    for comp in components:
        found.update(comp.get("required_stream_modalities", []))
    order = ["EEG", "EMG", "IMU"]
    return [m for m in order if m in found]


def prediction_probabilities_by_label(raw_probs, runtime: BundleDecoderRuntime, component_name: str) -> dict[str, float]:
    """Convert BundleDecoderRuntime PredictionResult probabilities to {label: probability}.

    Current model_runtime.PredictionResult stores probabilities as a list ordered by
    class index. Some older helper code expected a dictionary. The ensemble needs
    a label-keyed dictionary so that component models can be averaged safely.
    """
    if isinstance(raw_probs, dict):
        out = {}
        for label, value in raw_probs.items():
            try:
                prob = float(value)
            except Exception:
                continue
            if np.isfinite(prob):
                out[str(label)] = prob
        if out:
            return out
        raise RuntimeError(f"Ensemble component {component_name} returned an empty probability dictionary")

    arr = np.asarray(raw_probs, dtype=float).reshape(-1)
    if arr.size == 0:
        raise RuntimeError(f"Ensemble component {component_name} returned no probabilities")

    out = {}
    id_to_label = getattr(runtime, "id_to_label", {}) or {}
    for idx, value in enumerate(arr):
        prob = float(value)
        if not np.isfinite(prob):
            prob = 0.0
        label = str(id_to_label.get(int(idx), str(int(idx))))
        out[label] = prob

    if not out:
        raise RuntimeError(f"Ensemble component {component_name} returned no usable probabilities")
    return out


_ENSEMBLE_RUNTIME_CACHE: dict[str, object] = {}


def predict_trial_from_ensemble(
    components: list[dict],
    trial_dir: Path,
    stream_summaries: list[dict],
    device_arg: str,
    modality_scales: dict[str, float] | None = None,
) -> dict:
    """Run all ensemble components and average output probabilities by label.

    The existing single-bundle helper predict_trial_from_bundle() uses the module-level
    decoder_runtime object. In ensemble mode there is no single global runtime, so this
    function temporarily assigns decoder_runtime to each component runtime before calling
    that already-tested helper.
    """
    global decoder_runtime

    try:
        BundleDecoderRuntime  # type: ignore[name-defined]
    except NameError:
        from model_runtime import BundleDecoderRuntime  # type: ignore[no-redef]

    weighted_probs: dict[str, float] = {}
    component_debugs: list[dict] = []

    previous_runtime = globals().get("decoder_runtime", None)

    try:
        for comp in components:
            bundle_path = Path(comp["bundle_path"])
            cache_key = str(bundle_path.resolve())

            runtime = _ENSEMBLE_RUNTIME_CACHE.get(cache_key)
            if runtime is None:
                runtime = BundleDecoderRuntime.load(bundle_path, device_arg=device_arg)
                _ENSEMBLE_RUNTIME_CACHE[cache_key] = runtime

            # predict_trial_from_bundle relies on this global, so set it for this component.
            decoder_runtime = runtime

            comp_scales = None
            required_modalities = [
                str(m).upper()
                for m in getattr(runtime, "required_modalities", comp.get("required_modalities", []))
            ]
            if modality_scales and "EEG" in required_modalities:
                comp_scales = modality_scales

            debug = predict_trial_from_bundle(
                bundle_path=bundle_path,
                trial_dir=trial_dir,
                stream_summaries=stream_summaries,
                device_arg=device_arg,
                modality_scales=comp_scales,
            )

            probs = debug.get("probabilities", {})
            if not isinstance(probs, dict) or not probs:
                raise RuntimeError(
                    f"Ensemble component {comp.get('name', bundle_path.name)} "
                    f"did not return probability dictionary. Returned keys: {list(debug.keys())}"
                )

            w = float(comp.get("normalized_weight", comp.get("weight", 1.0)))
            for label, prob in probs.items():
                weighted_probs[str(label)] = weighted_probs.get(str(label), 0.0) + w * float(prob)

            component_debugs.append(
                {
                    "name": comp.get("name", bundle_path.stem),
                    "bundle_path": str(bundle_path),
                    "weight": float(comp.get("weight", 1.0)),
                    "normalized_weight": w,
                    "predicted_word": debug.get("predicted_word"),
                    "confidence": debug.get("confidence"),
                    "probabilities": probs,
                    "required_modalities": required_modalities,
                    "modality_scales": debug.get("modality_scales", {}),
                    "bundle_model_key": debug.get("bundle_model_key"),
                    "backend": debug.get("backend", debug.get("model_backend")),
                }
            )

    finally:
        decoder_runtime = previous_runtime

    s = sum(float(v) for v in weighted_probs.values())
    if s <= 0:
        raise RuntimeError("Ensemble probabilities summed to zero.")

    weighted_probs = {k: float(v / s) for k, v in weighted_probs.items()}

    sorted_items = sorted(weighted_probs.items(), key=lambda kv: kv[1], reverse=True)
    pred_label, pred_conf = sorted_items[0]

    top_predictions = [
        {"rank": idx + 1, "label": label, "probability": float(prob)}
        for idx, (label, prob) in enumerate(sorted_items[:3])
    ]

    return {
        "backend": "weighted_probability_ensemble",
        "model_backend": "weighted_probability_ensemble",
        "predicted_word": pred_label,
        "confidence": float(pred_conf),
        "probabilities": weighted_probs,
        "top_predictions": top_predictions,
        "ensemble_components": component_debugs,
    }

# -----------------------------------------------------------------------------
# End ensemble decoder support
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Online collection compatibility with offline preprocessing
# -----------------------------------------------------------------------------
def _safe_int(value, default=None):
    try:
        if value is None:
            return default
        out = int(value)
        return out if out > 0 else default
    except Exception:
        return default


def _preprocessing_meta_from_bundle_meta(bundle_meta: dict) -> dict:
    if not isinstance(bundle_meta, dict):
        return {}
    meta = bundle_meta.get("preprocessing_meta") or bundle_meta.get("preprocessing") or {}
    return dict(meta) if isinstance(meta, dict) else {}


def build_online_collection_config(
    runtime_meta_for_status: dict,
    ensemble_components: list[dict] | None = None,
) -> dict:
    """Build the raw-stream collection rules required before model_runtime preprocessing.

    Most modality-specific channel selection can safely happen in BundleDecoderRuntime,
    because EEG and EMG each arrive as one stream and the runtime can slice the raw
    stream before applying training preprocessing.

    IMU is different: four independent LSL streams are concatenated in
    online_decoder_service before model_runtime sees the data. Therefore each IMU
    stream must be reduced to the same number of raw channels used offline BEFORE
    hstack. Otherwise 4 x 10 live channels becomes 40 raw channels, raw_delta becomes
    80 channels, and the runtime truncates 80 -> 72 with the wrong channel semantics.

    This config is derived from exported bundle metadata, not hardcoded local paths.
    """
    metas: list[dict] = []

    if isinstance(runtime_meta_for_status, dict):
        metas.append(runtime_meta_for_status)

    for comp in ensemble_components or []:
        runtime = comp.get("runtime") if isinstance(comp, dict) else None
        bundle_meta = getattr(runtime, "bundle_meta", None)
        if isinstance(bundle_meta, dict):
            metas.append(bundle_meta)

    imu_channels_values = []
    imu_stream_name_lists = []

    eeg_channels_values = []
    emg_channels_values = []

    for meta in metas:
        pre = _preprocessing_meta_from_bundle_meta(meta)

        imu_ch = _safe_int(pre.get("imu_channels_per_stream_requested"), None)
        if imu_ch is not None:
            imu_channels_values.append(imu_ch)

        names = pre.get("imu_stream_names")
        if isinstance(names, list) and names:
            imu_stream_name_lists.append([str(x) for x in names])

        eeg_ch = _safe_int(pre.get("eeg_channels_requested"), None)
        if eeg_ch is not None:
            eeg_channels_values.append(eeg_ch)

        emg_ch = _safe_int(pre.get("emg_channels_requested"), None)
        if emg_ch is not None:
            emg_channels_values.append(emg_ch)

    def _unique_or_none(values):
        values = [v for v in values if v is not None]
        if not values:
            return None
        unique = sorted(set(values))
        if len(unique) > 1:
            raise RuntimeError(
                f"Inconsistent exported preprocessing metadata across selected decoder components: {unique}"
            )
        return unique[0]

    imu_channels_per_stream = _unique_or_none(imu_channels_values)
    eeg_channels_requested = _unique_or_none(eeg_channels_values)
    emg_channels_requested = _unique_or_none(emg_channels_values)

    imu_stream_names = None
    if imu_stream_name_lists:
        first = imu_stream_name_lists[0]
        for names in imu_stream_name_lists[1:]:
            if names != first:
                raise RuntimeError(
                    "Inconsistent imu_stream_names across selected decoder components: "
                    f"{first} vs {names}"
                )
        imu_stream_names = first

    return {
        "imu_channels_per_stream_requested": imu_channels_per_stream,
        "imu_stream_names": imu_stream_names,
        "eeg_channels_requested": eeg_channels_requested,
        "emg_channels_requested": emg_channels_requested,
        "source": "exported_bundle_preprocessing_meta",
    }


def _sort_entries_by_expected_names(
    entries: list[tuple[str, np.ndarray, float]],
    expected_names: list[str] | None,
) -> list[tuple[str, np.ndarray, float]]:
    if not expected_names:
        return sorted(entries, key=lambda x: x[0])

    expected_rank = {name: idx for idx, name in enumerate(expected_names)}

    def _key(item):
        name = item[0]
        if name in expected_rank:
            return (0, expected_rank[name], name)
        return (1, len(expected_rank), name)

    return sorted(entries, key=_key)



def collect_modality_windows(
    collectors: list["RollingStreamBuffer"],
    t_start: float,
    t_end: float,
    required_modalities_upper: list[str],
    online_collection_config: dict | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    grouped: dict[str, list[tuple[str, np.ndarray, float]]] = {m: [] for m in required_modalities_upper}

    for collector in collectors:
        modality_upper = collector.spec.modality
        if modality_upper not in grouped:
            continue

        _, data = collector.extract_window(t_start, t_end)
        grouped[modality_upper].append((
            collector.spec.name,
            np.asarray(data, dtype=np.float32),
            float(collector.spec.nominal_srate or 0.0),
        ))

    modality_windows: dict[str, np.ndarray] = {}
    modality_sample_rates: dict[str, float] = {}

    for modality_upper in required_modalities_upper:
        entries = grouped.get(modality_upper, [])
        if not entries:
            raise RuntimeError(f"No stream data available for required modality: {modality_upper}")

        if modality_upper == "IMU":
            entries = _sort_entries_by_expected_names(
                entries,
                (online_collection_config or {}).get("imu_stream_names"),
            )
        else:
            entries = sorted(entries, key=lambda x: x[0])

        if modality_upper in {"EEG", "EMG"}:
            _, data_tc, fs_hz = entries[0]
            if data_tc.ndim != 2 or data_tc.shape[0] < 2:
                raise RuntimeError(
                    f"{modality_upper} trial window has insufficient samples: {data_tc.shape}"
                )
            modality = STREAM_TO_MODALITY[modality_upper]
            modality_windows[modality] = data_tc
            if fs_hz > 0:
                modality_sample_rates[modality] = fs_hz
            continue

        if modality_upper == "IMU":
            imu_arrays = []
            fs_values = []
            min_len = None

            imu_channels_per_stream = _safe_int(
                (online_collection_config or {}).get("imu_channels_per_stream_requested"),
                None,
            )

            for stream_name, data_tc, fs_hz in entries:
                if data_tc.ndim != 2 or data_tc.shape[0] < 2:
                    raise RuntimeError(
                        f"IMU trial window has insufficient samples in one stream: {data_tc.shape}"
                    )

                # Critical offline/online compatibility rule:
                # Bitbrain DIGITAL_AUX live streams can expose an additional 10th
                # counter/timestamp-like channel. Offline decoder bundles were exported
                # with imu_channels_per_stream_requested=9. Slice each stream before
                # concatenating streams, preserving the offline 4 x 9 -> 36 raw channel
                # layout. Do not concatenate 4 x 10 -> 40 and truncate after raw_delta.
                if imu_channels_per_stream is not None:
                    if data_tc.shape[1] < imu_channels_per_stream:
                        raise RuntimeError(
                            f"IMU stream {stream_name} has {data_tc.shape[1]} channels, "
                            f"but the selected decoder expects at least "
                            f"{imu_channels_per_stream} channels per IMU stream"
                        )
                    data_tc = data_tc[:, :imu_channels_per_stream]

                imu_arrays.append(data_tc)
                if fs_hz > 0:
                    fs_values.append(fs_hz)
                min_len = data_tc.shape[0] if min_len is None else min(min_len, data_tc.shape[0])

            if min_len is None or min_len < 2:
                raise RuntimeError("IMU trial window could not be aligned across streams.")

            imu_arrays = [arr[:min_len] for arr in imu_arrays]
            modality_windows["imu"] = np.hstack(imu_arrays).astype(np.float32)
            if fs_values:
                modality_sample_rates["imu"] = float(np.median(fs_values))
            continue

        raise RuntimeError(f"Unsupported modality during collection: {modality_upper}")

    return modality_windows, modality_sample_rates

def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = add_output_path_metadata(make_json_portable(payload, PROJECT_ROOT))
    text = json.dumps(payload, indent=2)

    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass

    for _ in range(10):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError:
            time.sleep(0.1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    try:
        if tmp_path.exists():
            tmp_path.unlink()
    except Exception:
        pass

def sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return safe[:120] if safe else "stream"


def parse_selected_mode(selected_mode: str | None, experiment_mode: str) -> list[str]:
    if experiment_mode == "online_imagined":
        return ["EEG"]

    if not selected_mode:
        return ["EEG"]

    required = []
    if "EEG" in selected_mode:
        required.append("EEG")
    if "EMG" in selected_mode:
        required.append("EMG")
    if "IMU" in selected_mode:
        required.append("IMU")

    return required if required else ["EEG"]


def wait_for_marker_inlet(
    marker_stream_name: str,
    stop_event: threading.Event,
    poll_timeout: float = 0.5,
) -> StreamInlet:
    while not stop_event.is_set():
        streams = resolve_byprop("name", marker_stream_name, timeout=poll_timeout)
        if streams:
            return StreamInlet(streams[0], max_buflen=60, max_chunklen=32)
        time.sleep(0.1)

    raise RuntimeError(f"Stopped before marker stream '{marker_stream_name}' became available")

def wait_for_required_streams(
    required_modalities: list[str],
    marker_stream_name: str,
    stop_event: threading.Event,
    timeout_step: float = 1.0,
) -> tuple[list[SignalStreamSpec], list[str]]:
    while not stop_event.is_set():
        signal_specs, missing_modalities = discover_signal_specs(
            required_modalities=required_modalities,
            marker_stream_name=marker_stream_name,
            timeout=timeout_step,
        )
        if not missing_modalities:
            return signal_specs, missing_modalities
        time.sleep(0.2)

    raise RuntimeError("Stopped before required signal streams became available")


@dataclass
class SignalStreamSpec:
    modality: str
    name: str
    stream_type: str
    source_id: str
    channel_count: int
    nominal_srate: float
    hostname: str
    info_obj: object


def classify_stream_modality(name: str, stream_type: str):
    lname = (name or "").lower()
    ltype = (stream_type or "").lower()

    if "marker" in lname or "marker" in ltype:
        return None

    if "quality" in lname:
        return None

    # 1) Trust exact stream type first
    if ltype == "eeg":
        return "EEG"

    if ltype == "imu":
        return "IMU"

    if ltype in {"emg", "exg"}:
        return "EMG"

    # 2) Only then fall back to name heuristics
    if "eeg" in lname:
        return "EEG"

    if any(token in lname for token in ["imu", "accel", "gyro", "daux"]):
        return "IMU"

    if "emg" in lname or "exg" in lname:
        return "EMG"

    return None


def discover_signal_specs(
    required_modalities: list[str],
    marker_stream_name: str,
    timeout: float = 10.0,
) -> tuple[list[SignalStreamSpec], list[str]]:
    deadline = time.time() + timeout
    discovered: dict[tuple[str, str, str], SignalStreamSpec] = {}

    while time.time() < deadline:
        try:
            streams = resolve_streams(wait_time=0.5)
        except Exception:
            streams = []

        for info in streams:
            try:
                name = info.name()
                stream_type = info.type()
                source_id = info.source_id() or ""
                hostname = info.hostname() or ""
                channel_count = int(info.channel_count())
                nominal_srate = float(info.nominal_srate())
            except Exception:
                continue

            if name == marker_stream_name:
                continue

            modality = classify_stream_modality(name, stream_type)
            if modality is None:
                continue

            key = (modality, name, hostname)
            if key not in discovered:
                discovered[key] = SignalStreamSpec(
                    modality=modality,
                    name=name,
                    stream_type=stream_type,
                    source_id=source_id,
                    channel_count=channel_count,
                    nominal_srate=nominal_srate,
                    hostname=hostname,
                    info_obj=info,
                )

        found_modalities = {spec.modality for spec in discovered.values()}
        if all(mod in found_modalities for mod in required_modalities):
            break

    eeg_specs = sorted(
        [s for s in discovered.values() if s.modality == "EEG"],
        key=lambda s: (-s.channel_count, s.name),
    )
    emg_specs = sorted(
        [s for s in discovered.values() if s.modality == "EMG"],
        key=lambda s: (-s.channel_count, s.name),
    )
    imu_specs = sorted(
        [s for s in discovered.values() if s.modality == "IMU"],
        key=lambda s: (s.name,),
    )

    selected_specs: list[SignalStreamSpec] = []
    missing_modalities: list[str] = []

    if "EEG" in required_modalities:
        if eeg_specs:
            selected_specs.append(eeg_specs[0])
        else:
            missing_modalities.append("EEG")

    if "EMG" in required_modalities:
        if emg_specs:
            selected_specs.append(emg_specs[0])
        else:
            missing_modalities.append("EMG")

    if "IMU" in required_modalities:
        if imu_specs:
            selected_specs.extend(imu_specs)
        else:
            missing_modalities.append("IMU")

    return selected_specs, missing_modalities


class RollingStreamBuffer:
    def __init__(self, spec: SignalStreamSpec, buffer_seconds: float):
        self.spec = spec
        self.buffer_seconds = buffer_seconds
        self.inlet = StreamInlet(spec.info_obj, max_buflen=max(10, int(buffer_seconds) + 5), max_chunklen=128)

        self._chunks: deque[tuple[np.ndarray, np.ndarray]] = deque()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.total_samples_received = 0
        self.last_sample_time_lsl: Optional[float] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"collector_{self.spec.name}")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                samples, timestamps = self.inlet.pull_chunk(timeout=0.25, max_samples=256)
            except Exception:
                continue

            if not timestamps:
                continue

            ts = np.asarray(timestamps, dtype=float)
            data = np.asarray(samples, dtype=float)

            if data.ndim == 1:
                data = data.reshape(-1, 1)

            if data.shape[0] != ts.shape[0]:
                if data.ndim == 2 and data.shape[1] == ts.shape[0]:
                    data = data.T
                else:
                    continue

            with self._lock:
                self._chunks.append((ts, data))
                self.total_samples_received += data.shape[0]
                self.last_sample_time_lsl = float(ts[-1])
                self._prune_locked()

    def _prune_locked(self) -> None:
        if not self._chunks:
            return

        newest_time = float(self._chunks[-1][0][-1])
        min_time = newest_time - self.buffer_seconds

        while self._chunks:
            oldest_ts = self._chunks[0][0]
            if float(oldest_ts[-1]) >= min_time:
                break
            self._chunks.popleft()

    def extract_window(self, t_start: float, t_end: float) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            if not self._chunks:
                return np.empty((0,), dtype=float), np.empty((0, self.spec.channel_count), dtype=float)

            ts_all = np.concatenate([chunk_ts for chunk_ts, _ in self._chunks], axis=0)
            data_all = np.concatenate([chunk_data for _, chunk_data in self._chunks], axis=0)

        mask = (ts_all >= t_start) & (ts_all <= t_end)
        if not np.any(mask):
            return np.empty((0,), dtype=float), np.empty((0, data_all.shape[1]), dtype=float)

        return ts_all[mask], data_all[mask]


def save_trial_stream_data(
    trial_dir: Path,
    collector: RollingStreamBuffer,
    t_start: float,
    t_end: float,
) -> dict:
    ts, data = collector.extract_window(t_start, t_end)

    safe_name = sanitize_filename(collector.spec.name)
    samples_path = trial_dir / f"{collector.spec.modality.lower()}__{safe_name}__samples.npy"
    ts_path = trial_dir / f"{collector.spec.modality.lower()}__{safe_name}__timestamps.npy"

    np.save(samples_path, data)
    np.save(ts_path, ts)

    return {
        "modality": collector.spec.modality,
        "name": collector.spec.name,
        "stream_type": collector.spec.stream_type,
        "hostname": collector.spec.hostname,
        "source_id": collector.spec.source_id,
        "channel_count": int(collector.spec.channel_count),
        "nominal_srate": float(collector.spec.nominal_srate),
        "n_samples": int(data.shape[0]),
        "samples_file": samples_path.name,
        "timestamps_file": ts_path.name,
    }


def build_result_payload(
    trial_index: int,
    mode: str,
    target_word: str,
    go_time_lsl: float,
    end_time_lsl: float,
    predicted_word: str,
    confidence: float,
    latency_ms: float,
    selected_model: str | None,
    selected_mode: str | None,
    trial_dir: str,
    stream_summaries: list[dict],
    bundle_path: str | None = None,
    top_predictions: list[dict] | None = None,
    status: str = "ok",
    error_message: str | None = None,
) -> dict:
    payload = {
        "trial_index": trial_index,
        "mode": mode,
        "target_word": target_word,
        "predicted_word": predicted_word,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "status": status,
        "go_time_lsl": go_time_lsl,
        "end_time_lsl": end_time_lsl,
        "selected_model": selected_model,
        "selected_mode": selected_mode,
        "trial_dir": trial_dir,
        "stream_summaries": stream_summaries,
        "bundle_path": bundle_path,
        "top_predictions": top_predictions or [],
        "created_at_unix": time.time(),
    }

    if error_message:
        payload["error_message"] = error_message

    return payload


def update_run_summary(results_dir: Path) -> None:
    trial_jsons = sorted(results_dir.glob("trial_*.json"))
    results = []

    for path in trial_jsons:
        try:
            results.append(read_json(path))
        except Exception:
            continue

    total = len(results)
    correct = sum(1 for r in results if r.get("predicted_word") == r.get("target_word"))
    confidences = [float(r["confidence"]) for r in results if isinstance(r.get("confidence"), (int, float))]
    latencies = [float(r["latency_ms"]) for r in results if isinstance(r.get("latency_ms"), (int, float))]

    summary = {
        "n_trials": total,
        "n_correct": correct,
        "accuracy": (correct / total) if total > 0 else 0.0,
        "mean_confidence": (sum(confidences) / len(confidences)) if confidences else 0.0,
        "mean_latency_ms": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "updated_at_unix": time.time(),
    }

    write_json_atomic(results_dir / "run_summary.json", summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", required=True, help="Path to latest_prepared_run.json")
    parser.add_argument("--results-dir", required=True, help="Directory where per-trial files will be written")
    parser.add_argument("--marker-stream-name", default="PsychoPy_Markers")
    parser.add_argument("--mode", default=None, help="online_silent or online_imagined")
    parser.add_argument("--poll-timeout", type=float, default=0.25)
    parser.add_argument("--stream-timeout", type=float, default=10.0)
    parser.add_argument("--buffer-seconds", type=float, default=20.0)
    parser.add_argument(
        "--pre-seconds",
        type=float,
        default=1.0,
        help="Legacy fallback seconds before GO when bundle metadata is missing.",
    )
    parser.add_argument(
        "--post-seconds",
        type=float,
        default=2.0,
        help="Legacy fallback seconds after GO when bundle metadata is missing.",
    )
    parser.add_argument(
        "--model-window-start-s",
        type=float,
        default=None,
        help="Optional explicit model window start relative to GO. Prefer bundle metadata.",
    )
    parser.add_argument(
        "--model-window-end-s",
        type=float,
        default=None,
        help="Optional explicit model window end relative to GO. Prefer bundle metadata.",
    )
    parser.add_argument("--bundle-path", default=None, help="Path to exported decoder bundle (.pt)")
    parser.add_argument("--brainprint-compare-json", default=None, help="Path to brainprint comparison JSON")
    parser.add_argument("--brainprint-summary-json", default=None, help="Path to brainprint all-vs-all summary.json")
    parser.add_argument("--brainprint-top-k", type=int, default=3)
    parser.add_argument("--eeg-scale-min", type=float, default=0.65)
    parser.add_argument("--eeg-scale-max", type=float, default=1.00)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    bridge_path = resolve_runtime_path(
        args.bridge,
        PROJECT_ROOT,
        label="bridge JSON path passed with --bridge",
        must_exist=True,
    )

    bridge = read_json(bridge_path)
    bridge_project_root = get_bridge_project_root(bridge)

    results_dir = resolve_runtime_path(
        args.results_dir,
        bridge_project_root,
        label="results directory passed with --results-dir",
        must_exist=False,
    )
    trials_root = results_dir / "trials"

    brainprint_compare_json_path = (
        resolve_runtime_path(
            args.brainprint_compare_json,
            bridge_project_root,
            label="brainprint comparison JSON passed with --brainprint-compare-json",
            must_exist=False,
        )
        if args.brainprint_compare_json
        else None
    )
    brainprint_summary_json_path = (
        resolve_runtime_path(
            args.brainprint_summary_json,
            bridge_project_root,
            label="brainprint summary JSON passed with --brainprint-summary-json",
            must_exist=False,
        )
        if args.brainprint_summary_json
        else None
    )

    mode = args.mode or bridge.get("experiment_mode", "online_silent")
    selected_model = bridge.get("selected_model")
    selected_mode = bridge.get("selected_mode")
    ensemble_config = resolve_ensemble_config(bridge)

    ensemble_components = []
    if ensemble_config is not None:
        ensemble_components = prepare_ensemble_components(
            ensemble_config,
            device_arg=args.device,
            project_root=bridge_project_root,
        )
        bundle_path = None
        decoder_runtime = None
        required_modalities = ensemble_required_stream_modalities(ensemble_components)
    else:
        bundle_path = resolve_bundle_path(
            args.bundle_path,
            bridge,
            project_root=bridge_project_root,
        )
        decoder_runtime = BundleDecoderRuntime.load(bundle_path, device_arg=args.device)
        required_modalities = decoder_runtime.required_stream_modalities()

    bundle_path_for_status = str(bundle_path) if bundle_path is not None else None


    # Ensemble-safe metadata object.
    # In single-bundle mode this is the selected BundleDecoderRuntime metadata.
    # In ensemble mode decoder_runtime is intentionally None, so code that reports
    # model/window/preprocessing metadata must use this synthetic ensemble metadata.
    if ensemble_config is not None:
        _component_metas = []
        for _comp in ensemble_components:
            _runtime = _comp.get("runtime")
            if _runtime is not None and hasattr(_runtime, "bundle_meta"):
                _component_metas.append(_runtime.bundle_meta)

        _preprocessing_meta = {}
        if _component_metas:
            # Use the component with the richest modality set as the representative
            # preprocessing/window source. For J+U this is the full EEG+EMG+IMU bundle.
            _component_metas_sorted = sorted(
                _component_metas,
                key=lambda m: len(m.get("required_modalities", [])),
                reverse=True,
            )
            _preprocessing_meta = dict(_component_metas_sorted[0].get("preprocessing_meta", {}))

        runtime_meta_for_status = {
            "model_key": ensemble_config.get("name", "ensemble"),
            "target_task": mode,
            "required_modalities": [m.lower() for m in required_modalities],
            "fusion_mode": ensemble_config.get("method", "weighted_probability_average"),
            "preprocessing_meta": _preprocessing_meta,
            "ensemble_components": [
                {
                    "name": str(c.get("name")),
                    "bundle_path": str(c.get("bundle_path")),
                    "weight": float(c.get("weight", 1.0)),
                    "normalized_weight": float(c.get("normalized_weight", c.get("weight", 1.0))),
                }
                for c in ensemble_components
            ],
        }
    else:
        runtime_meta_for_status = decoder_runtime.bundle_meta
    online_collection_config = build_online_collection_config(
        runtime_meta_for_status,
        ensemble_components=ensemble_components,
    )

    model_window_start_s, model_window_end_s, model_window_source = resolve_model_window_offsets(
        runtime_meta_for_status,
        args,
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    trials_root.mkdir(parents=True, exist_ok=True)

    service_status_path = results_dir / "service_status.json"
    latest_result_path = results_dir / "latest.json"

    stop_event = threading.Event()

    def _handle_stop(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_stop)

    write_json_atomic(
        service_status_path,
        {
            "status": "starting",
            "mode": mode,
            "bridge": str(bridge_path),
            "results_dir": str(results_dir),
            "marker_stream_name": args.marker_stream_name,
            "required_modalities": required_modalities,
            "model_window_start_s": model_window_start_s,
            "model_window_end_s": model_window_end_s,
            "model_window_source": model_window_source,
            "online_collection_config": online_collection_config,
            "started_at_unix": time.time(),
            "bundle_path": bundle_path_for_status,
        },
    )

    write_json_atomic(
        service_status_path,
        {
            "status": "waiting_for_markers",
            "mode": mode,
            "bridge": str(bridge_path),
            "results_dir": str(results_dir),
            "bundle_path": bundle_path_for_status,
            "marker_stream_name": args.marker_stream_name,
            "required_modalities": required_modalities,
            "model_window_start_s": model_window_start_s,
            "model_window_end_s": model_window_end_s,
            "model_window_source": model_window_source,
            "online_collection_config": online_collection_config,
            "started_at_unix": time.time(),
        },
    )
    
    marker_inlet = wait_for_marker_inlet(
        marker_stream_name=args.marker_stream_name,
        stop_event=stop_event,
        poll_timeout=args.poll_timeout,
    )
    
    write_json_atomic(
        service_status_path,
        {
            "status": "waiting_for_streams",
            "mode": mode,
            "bridge": str(bridge_path),
            "results_dir": str(results_dir),
            "bundle_path": bundle_path_for_status,
            "marker_stream_name": args.marker_stream_name,
            "required_modalities": required_modalities,
            "model_window_start_s": model_window_start_s,
            "model_window_end_s": model_window_end_s,
            "model_window_source": model_window_source,
            "online_collection_config": online_collection_config,
            "started_at_unix": time.time(),
        },
    )
    
    signal_specs, missing_modalities = wait_for_required_streams(
        required_modalities=required_modalities,
        marker_stream_name=args.marker_stream_name,
        stop_event=stop_event,
        timeout_step=1.0,
    )

    collectors: list[RollingStreamBuffer] = []
    for spec in signal_specs:
        collector = RollingStreamBuffer(spec=spec, buffer_seconds=args.buffer_seconds)
        collector.start()
        collectors.append(collector)

    write_json_atomic(
        service_status_path,
        {
            "status": "running",
            "mode": mode,
            "bridge": str(bridge_path),
            "results_dir": str(results_dir),
            "bundle_path": bundle_path_for_status,
            "marker_stream_name": args.marker_stream_name,
            "required_modalities": required_modalities,
            "missing_modalities": missing_modalities,
            "resolved_streams": [
                {
                    "modality": spec.modality,
                    "name": spec.name,
                    "stream_type": spec.stream_type,
                    "hostname": spec.hostname,
                    "source_id": spec.source_id,
                    "channel_count": spec.channel_count,
                    "nominal_srate": spec.nominal_srate,
                }
                for spec in signal_specs
            ],
            "started_at_unix": time.time(),
        },
    )

    print(f"[decoder] Running in mode: {mode}")
    print(f"[decoder] Results dir: {results_dir}")
    if ensemble_config is not None:
        print(f"[decoder] Ensemble: {ensemble_config.get('display_name') or ensemble_config.get('name') or 'ensemble'}")
        for comp in ensemble_components:
            print(f"[decoder]   component: {comp['name']} | weight={comp['normalized_weight']:.3f} | bundle={Path(comp['bundle_path']).name}")
    else:
        print(f"[decoder] Bundle: {bundle_path.name}")
    print(f"[decoder] Listening to marker stream: {args.marker_stream_name}")
    print(f"[decoder] Required modalities: {required_modalities}")
    print(
        f"[decoder] Model window relative to GO: "
        f"{model_window_start_s:+.3f}s to {model_window_end_s:+.3f}s "
        f"({model_window_source})"
    )
    if missing_modalities:
        print(f"[decoder] Warning: missing modalities: {missing_modalities}")

    trial_index = 0
    active_trial: dict | None = None

    try:
        while not stop_event.is_set():
            samples, timestamps = marker_inlet.pull_chunk(timeout=args.poll_timeout, max_samples=64)

            if not timestamps:
                continue

            for sample, ts in zip(samples, timestamps):
                if stop_event.is_set():
                    break

                if isinstance(sample, (list, tuple)) and len(sample) > 0:
                    marker = str(sample[0])
                else:
                    marker = str(sample)

                marker = marker.strip()
                go_match = None
                end_match = None

                if mode == "online_silent":
                    go_match = SILENT_GO_RE.match(marker)
                    end_match = SILENT_END_RE.match(marker)
                elif mode == "online_imagined":
                    go_match = IMAGINED_GO_RE.match(marker)
                    end_match = IMAGINED_END_RE.match(marker)

                if go_match:
                    trial_index += 1
                    target_word = go_match.group(1).strip()
                    active_trial = {
                        "trial_index": trial_index,
                        "target_word": target_word,
                        "go_time_lsl": float(ts),
                    }
                    print(f"[decoder] GO trial={trial_index:03d} target={target_word}")
                    continue

                if end_match and active_trial is not None:
                    compute_start = time.time()

                    target_word = active_trial["target_word"]
                    end_word = end_match.group(1).strip()
                    final_target = target_word if target_word else end_word

                    trial_id = active_trial["trial_index"]
                    trial_dir = trials_root / f"trial_{trial_id:03d}"
                    trial_dir.mkdir(parents=True, exist_ok=True)

                    go_time_lsl = float(active_trial["go_time_lsl"])
                    t_start = go_time_lsl + model_window_start_s
                    t_end = go_time_lsl + model_window_end_s

                    stream_summaries = []
                    for collector in collectors:
                        summary = save_trial_stream_data(
                            trial_dir=trial_dir,
                            collector=collector,
                            t_start=t_start,
                            t_end=t_end,
                        )
                        stream_summaries.append(summary)

                    trial_meta = {
                        "trial_index": trial_id,
                        "mode": mode,
                        "target_word": final_target,
                        "go_time_lsl": float(active_trial["go_time_lsl"]),
                        "end_time_lsl": float(ts),
                        "window_start_lsl": t_start,
                        "window_end_lsl": t_end,
                        "model_window_start_s": model_window_start_s,
                        "model_window_end_s": model_window_end_s,
                        "model_window_source": model_window_source,
                        "online_collection_config": online_collection_config,
                        "selected_model": selected_model,
                        "selected_mode": selected_mode,
                        "required_modalities": required_modalities,
                        "missing_modalities": missing_modalities,
                        "stream_summaries": stream_summaries,
                        "created_at_unix": time.time(),
                    }
                    write_json_atomic(trial_dir / "meta.json", trial_meta)

                    modality_scales: dict[str, float] = {}
                    brainprint_info = None
                    model_debug = {"backend": "unknown"}
                    inference_status = "ok"
                    inference_error = None
                    top_predictions: list[dict] = []
                    predicted_word = "INFERENCE_ERROR"
                    confidence = 0.0

                    if brainprint_compare_json_path is not None and "EEG" in required_modalities:
                        eeg_scale, brainprint_info = compute_eeg_scale_from_comparison_json(
                            compare_json_path=brainprint_compare_json_path,
                            summary_json_path=brainprint_summary_json_path,
                            top_k=args.brainprint_top_k,
                            min_scale=args.eeg_scale_min,
                            max_scale=args.eeg_scale_max,
                        )
                        modality_scales["eeg"] = eeg_scale

                    try:
                        modality_windows, modality_sample_rates = collect_modality_windows(
                            collectors=collectors,
                            t_start=t_start,
                            t_end=t_end,
                            required_modalities_upper=required_modalities,
                            online_collection_config=online_collection_config,
                        )

                        if ensemble_config is not None:
                            weighted_probs: dict[str, float] = {}
                            component_debugs: list[dict] = []

                            for comp in ensemble_components:
                                runtime = comp.get("runtime")
                                if runtime is None:
                                    raise RuntimeError(
                                        f"Ensemble component {comp.get('name')} has no loaded runtime"
                                    )

                                comp_required_upper = [
                                    str(m).upper()
                                    for m in comp.get("required_stream_modalities", [])
                                ]
                                comp_windows: dict[str, np.ndarray] = {}
                                comp_sample_rates: dict[str, float] = {}

                                for mod_upper in comp_required_upper:
                                    mod_lower = STREAM_TO_MODALITY[mod_upper]
                                    if mod_lower not in modality_windows:
                                        raise RuntimeError(
                                            f"Missing modality {mod_lower} for ensemble component "
                                            f"{comp.get('name')}"
                                        )
                                    comp_windows[mod_lower] = modality_windows[mod_lower]
                                    if mod_lower in modality_sample_rates:
                                        comp_sample_rates[mod_lower] = modality_sample_rates[mod_lower]

                                comp_scales = None
                                if modality_scales and "eeg" in comp_windows:
                                    comp_scales = modality_scales

                                comp_pred = runtime.predict(
                                    comp_windows,
                                    top_k=3,
                                    modality_scales=comp_scales,
                                    modality_sample_rates=comp_sample_rates,
                                )

                                probs = prediction_probabilities_by_label(
                                    comp_pred.probabilities,
                                    runtime,
                                    str(comp.get("name") or "component"),
                                )

                                weight = float(comp.get("normalized_weight", comp.get("weight", 1.0)))
                                for label, prob in probs.items():
                                    weighted_probs[str(label)] = (
                                        weighted_probs.get(str(label), 0.0) + weight * float(prob)
                                    )

                                component_debugs.append(
                                    {
                                        "name": comp.get("name"),
                                        "bundle_path": str(comp.get("bundle_path")),
                                        "weight": float(comp.get("weight", 1.0)),
                                        "normalized_weight": weight,
                                        "required_modalities": comp_required_upper,
                                        "predicted_word": comp_pred.predicted_label,
                                        "confidence": float(comp_pred.confidence),
                                        "probabilities": probs,
                                        "top_predictions": comp_pred.top_predictions,
                                        "modality_sample_rates": comp_sample_rates,
                                        "modality_scales": comp_scales or {},
                                    }
                                )

                            prob_sum = sum(float(v) for v in weighted_probs.values())
                            if prob_sum <= 0.0:
                                raise RuntimeError("Ensemble probabilities summed to zero")

                            ensemble_probs = {
                                label: float(prob / prob_sum)
                                for label, prob in weighted_probs.items()
                            }
                            sorted_probs = sorted(
                                ensemble_probs.items(),
                                key=lambda item: item[1],
                                reverse=True,
                            )

                            predicted_word = sorted_probs[0][0]
                            confidence = round(float(sorted_probs[0][1]), 4)
                            top_predictions = [
                                {
                                    "rank": rank + 1,
                                    "label": label,
                                    "probability": float(prob),
                                }
                                for rank, (label, prob) in enumerate(sorted_probs[:3])
                            ]
                            model_debug = {
                                "backend": "weighted_probability_ensemble",
                                "probabilities": ensemble_probs,
                                "top_predictions": top_predictions,
                                "ensemble_components": component_debugs,
                                "modality_sample_rates": modality_sample_rates,
                            }

                        else:
                            pred = decoder_runtime.predict(
                                modality_windows,
                                top_k=3,
                                modality_scales=modality_scales if modality_scales else None,
                                modality_sample_rates=modality_sample_rates,
                            )
                            predicted_word = pred.predicted_label
                            confidence = round(float(pred.confidence), 4)
                            top_predictions = pred.top_predictions
                            model_debug = {
                                "backend": "BundleDecoderRuntime.predict",
                                "probabilities": pred.probabilities,
                                "top_predictions": top_predictions,
                                "modality_sample_rates": modality_sample_rates,
                            }

                    except Exception as exc:
                        inference_status = "error"
                        inference_error = str(exc)

                    latency_ms = round((time.time() - compute_start) * 1000.0, 1)

                    payload = build_result_payload(
                        trial_index=trial_id,
                        mode=mode,
                        target_word=final_target,
                        go_time_lsl=float(active_trial["go_time_lsl"]),
                        end_time_lsl=float(ts),
                        predicted_word=predicted_word,
                        confidence=confidence,
                        latency_ms=latency_ms,
                        selected_model=selected_model,
                        selected_mode=selected_mode,
                        trial_dir=str(trial_dir),
                        stream_summaries=stream_summaries,
                        bundle_path=bundle_path_for_status or 'ensemble',
                        top_predictions=top_predictions,
                        status=inference_status,
                        error_message=inference_error,
                    )

                    payload["model_backend"] = model_debug.get("backend", "bundle")
                    payload["bundle_path"] = bundle_path_for_status or "ensemble"
                    payload["model_window_start_s"] = model_window_start_s
                    payload["model_window_end_s"] = model_window_end_s
                    payload["model_window_source"] = model_window_source
                    payload["online_collection_config"] = online_collection_config
                    payload["modality_scales"] = modality_scales
                    payload["brainprint"] = brainprint_info
                    payload["model_debug"] = model_debug
                    payload["decoder_ensemble"] = ensemble_config

                    trial_json_path = results_dir / f"trial_{trial_id:03d}.json"
                    write_json_atomic(trial_json_path, payload)
                    write_json_atomic(latest_result_path, payload)
                    update_run_summary(results_dir)

                    print(
                        f"[decoder] Wrote trial {trial_id:03d} -> {trial_json_path.name} | "
                        f"pred={predicted_word} | conf={confidence}"
                    )

                    active_trial = None

    finally:
        for collector in collectors:
            collector.stop()

        write_json_atomic(
            service_status_path,
            {
                "status": "stopped",
                "mode": mode,
                "bridge": str(bridge_path),
                "results_dir": str(results_dir),
                "marker_stream_name": args.marker_stream_name,
                "required_modalities": required_modalities,
                "missing_modalities": missing_modalities,
                "stopped_at_unix": time.time(),
                "bundle_path": bundle_path_for_status,
            },
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[decoder] Stopped by user.")
        raise SystemExit(0)