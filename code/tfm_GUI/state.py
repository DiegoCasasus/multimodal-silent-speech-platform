from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet, Optional


class Modality(str, Enum):
    EEG = "EEG"
    EMG = "EMG"
    IMU = "IMU"
    MARKERS = "MARKERS"


class ExperimentMode(str, Enum):
    RECORDING = "recording"
    ONLINE_SILENT = "online_silent"
    ONLINE_IMAGINED = "online_imagined"


class AppStatus(str, Enum):
    IDLE = "IDLE"
    DISCOVERING = "DISCOVERING"
    READY = "READY"
    RECORDING = "RECORDING"
    DECODING = "DECODING"
    ERROR = "ERROR"


@dataclass(frozen=True)
class AcquisitionMode:
    name: str
    required_modalities: FrozenSet[Modality]


ACQUISITION_MODES = {
    "EEG": AcquisitionMode(
        name="EEG",
        required_modalities=frozenset({Modality.EEG}),
    ),
    "EMG": AcquisitionMode(
        name="EMG",
        required_modalities=frozenset({Modality.EMG}),
    ),
    "IMU": AcquisitionMode(
        name="IMU",
        required_modalities=frozenset({Modality.IMU}),
    ),
    "EEG+EMG": AcquisitionMode(
        name="EEG+EMG",
        required_modalities=frozenset({Modality.EEG, Modality.EMG}),
    ),
    "EEG+IMU": AcquisitionMode(
        name="EEG+IMU",
        required_modalities=frozenset({Modality.EEG, Modality.IMU}),
    ),
    "EMG+IMU": AcquisitionMode(
        name="EMG+IMU",
        required_modalities=frozenset({Modality.EMG, Modality.IMU}),
    ),
    "EEG+EMG+IMU": AcquisitionMode(
        name="EEG+EMG+IMU",
        required_modalities=frozenset({Modality.EEG, Modality.EMG, Modality.IMU}),
    ),
}


@dataclass(frozen=True)
class StreamDescriptor:
    name: str
    stream_type: str
    source_id: str
    channel_count: int
    nominal_srate: float
    modality: Optional[Modality] = None
    uid: Optional[str] = None
    hostname: Optional[str] = None


@dataclass
class SessionMetadata:
    participant_id: str
    session_id: str
    run_id: str
    experiment_mode: str
    task_mode: str = ""
    notes: str = ""
    operator: str = ""
    site: str = ""


@dataclass
class AppState:
    status: AppStatus = AppStatus.IDLE

    detected_streams: list[StreamDescriptor] = field(default_factory=list)
    detected_modalities: set[Modality] = field(default_factory=set)

    available_experiment_modes: list[str] = field(
        default_factory=lambda: [e.value for e in ExperimentMode]
    )
    selected_experiment_mode: str = ExperimentMode.RECORDING.value

    available_modes: list[str] = field(default_factory=list)
    selected_mode: Optional[str] = None

    available_models: list[str] = field(default_factory=list)
    selected_model: Optional[str] = None

    session: Optional[SessionMetadata] = None

    brainprint_enabled: bool = True
    brainprint_loaded: bool = False
    brainprint_subject_id: Optional[str] = None

    record_xdf: bool = True
    recording_active: bool = False

    psychopy_script_path: Optional[str] = None
    prepared_run_dir: Optional[str] = None
    prepared_run_basename: Optional[str] = None
    prepared_xdf_path: Optional[str] = None
    prepared_manifest_path: Optional[str] = None
    prepared_metadata_path: Optional[str] = None
    prepared_bridge_path: Optional[str] = None
    prepared_decoder_bundle_path: Optional[str] = None
    prepared_online_results_dir: Optional[str] = None

    online_decoder_running: bool = False
    online_decoder_service_status_path: Optional[str] = None
    online_decoder_latest_path: Optional[str] = None
    online_decoder_summary_path: Optional[str] = None

    last_pipeline_status: Optional[str] = None
    last_pipeline_message: Optional[str] = None
    last_brainprint_npz_path: Optional[str] = None
    last_compare_report_path: Optional[str] = None

    error_message: Optional[str] = None
