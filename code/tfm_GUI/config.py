from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
DECODER_ROOT = CODE_ROOT / "decoder"

# Allow GUI modules to import shared utilities from code/common even when the
# GUI is launched directly as code/tfm_GUI/GUI.py.
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

try:
    from common.config_loader import get_config_path, get_config_value, load_project_config
except Exception:
    # Safe fallback: if the shared config layer is not present yet, keep the old
    # hardcoded defaults instead of breaking the working GUI.
    get_config_path = None
    get_config_value = None
    load_project_config = None


_RAW_PROJECT_CONFIG = (
    load_project_config(PROJECT_ROOT) if load_project_config is not None else {}
)


def _value(dotted_key: str, default):
    if get_config_value is None:
        return default
    return get_config_value(_RAW_PROJECT_CONFIG, dotted_key, default)


def _path(dotted_key: str, default: Path) -> Path:
    if get_config_path is None:
        return default
    resolved = get_config_path(_RAW_PROJECT_CONFIG, dotted_key, PROJECT_ROOT, default)
    return resolved if resolved is not None else default


def _optional_path(dotted_key: str, default: Optional[Path] = None) -> Optional[Path]:
    if get_config_path is None:
        return default
    return get_config_path(_RAW_PROJECT_CONFIG, dotted_key, PROJECT_ROOT, default)

def _default_labrecorder_exe() -> Path:
    """Return the platform-appropriate default LabRecorder executable."""
    if sys.platform == "darwin":
        # Stable Homebrew opt path: survives Homebrew version upgrades.
        return Path(
            "/opt/homebrew/opt/labrecorder/"
            "LabRecorder.app/Contents/MacOS/LabRecorder"
        )

    if sys.platform.startswith("win"):
        return PROJECT_ROOT / "LabRecorder" / "LabRecorder.exe"

    # Linux fallback. This may be overridden in project_config.json.
    return Path("/usr/local/bin/LabRecorder")


@dataclass(frozen=True)
class AppConfig:
    default_mode: str = _value("defaults.mode", "EEG")
    default_experiment_mode: str = _value("defaults.experiment_mode", "recording")

    default_record_xdf: bool = bool(_value("defaults.record_xdf", True))
    default_brainprint_enabled: bool = bool(_value("defaults.brainprint_enabled", True))

    stream_discovery_timeout: float = float(_value("runtime.stream_discovery_timeout", 2.0))

    # GUI-prepared metadata and sidecars
    output_root: Path = _path("paths.data_root", PROJECT_ROOT / "data")
    bridge_file: Path = _path("paths.bridge_file", PROJECT_ROOT / "data" / "latest_prepared_run.json")
    online_results_root: Path = _path("paths.online_results_root", PROJECT_ROOT / "data" / "online_results")
    decoder_models_root: Path = _path("paths.final_decoder_bundles_dir", PROJECT_ROOT / "data" / "final_decoder_bundles" / "silent_v1")
    decoder_manifest_path: Path = _path("paths.decoder_manifest_path", PROJECT_ROOT / "data" / "final_decoder_bundles" / "silent_v1" / "decoder_manifest.json")
    imagined_decoder_manifest_path: Path = _path("paths.imagined_decoder_manifest_path", PROJECT_ROOT / "data" / "final_decoder_bundles" / "imagined_v1" / "decoder_manifest.json")
    default_decoder_window_tag: str = str(_value("runtime.default_decoder_window_tag", "w050_200"))

    # Actual LabRecorder outputs
    recordings_root: Path = _path("paths.recordings_root", PROJECT_ROOT / "recordings")

    # Brainprint pipeline folders
    project_root: Path = PROJECT_ROOT
    brainprint_root: Path = _path("paths.brainprint_root", PROJECT_ROOT / "brainprints")
    brainprint_bank_dir: Path = _path("paths.brainprint_bank_dir", PROJECT_ROOT / "brainprints" / "bank")
    brainprint_candidate_dir: Path = _path("paths.brainprint_candidate_dir", PROJECT_ROOT / "brainprints" / "candidates")
    brainprint_comparison_dir: Path = _path("paths.brainprint_comparison_dir", PROJECT_ROOT / "brainprints" / "comparisons")
    brainprint_rejected_dir: Path = _path("paths.brainprint_rejected_dir", PROJECT_ROOT / "brainprints" / "rejected")
    brainprint_summary_json: Path = _path("paths.brainprint_summary_json", PROJECT_ROOT / "brainprints" / "all_vs_all" / "summary.json")

    # External tools / scripts
    labrecorder_exe: Path = _path("external_tools.labrecorder_exe",_default_labrecorder_exe(),)
    offline_psychopy_script: Path = _path("paths.offline_psychopy_script", PROJECT_ROOT / "psychopy" / "Experiment_Offline.py")
    online_silent_psychopy_script: Path = _path("paths.online_silent_psychopy_script", PROJECT_ROOT / "psychopy" / "Experiment_Online_Silent.py")
    online_imagined_psychopy_script: Path = _path("paths.online_imagined_psychopy_script", PROJECT_ROOT / "psychopy" / "Experiment_Online_Imagined.py")

    compute_brainprint_script: Path = _path("paths.compute_brainprint_script", DECODER_ROOT / "compute_brainprint.py")
    compare_brainprint_script: Path = _path("paths.compare_brainprint_script", DECODER_ROOT / "compare_brainprint.py")
    prepare_online_calibration_script: Path = _path("paths.prepare_online_calibration_script", DECODER_ROOT / "prepare_online_calibration.py")
    online_decoder_service_script: Path = _path("paths.online_decoder_service_script", DECODER_ROOT / "online_decoder_service.py")
    preprocessing_portable_script: Path = _path("paths.preprocessing_portable_script", DECODER_ROOT / "preprocessing_portable.py")
    train_export_decoder_script: Path = _path("paths.train_export_decoder_script", DECODER_ROOT / "train_export_decoder.py")

    brainprint_signal_stream_type: str = str(_value("brainprint.signal_stream_type", "EEG"))
    brainprint_marker_stream_name: str = str(_value("brainprint.marker_stream_name", "PsychoPy_Markers"))
    brainprint_compare_top_k: int = int(_value("brainprint.compare_top_k", 5))

    auto_process_brainprint_after_recording: bool = bool(_value("brainprint.auto_process_after_recording", True))

    psychopy_python_exe: Optional[Path] = _optional_path("external_tools.psychopy_python_exe", None)
    decoder_python_exe: Optional[Path] = _optional_path("external_tools.decoder_python_exe", None)
    decoder_device: str = str(_value("runtime.decoder_device", "cpu"))
    decoder_startup_timeout_s: float = float(_value("runtime.decoder_startup_timeout_s", 15.0))


CONFIG = AppConfig()