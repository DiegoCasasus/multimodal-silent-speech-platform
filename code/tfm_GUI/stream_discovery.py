from __future__ import annotations

from typing import Optional

from state import (
    ACQUISITION_MODES,
    Modality,
    StreamDescriptor,
)


def classify_lsl_stream(
    stream_name: str,
    stream_type: str,
    source_id: str = "",
    channel_count: int = 0,
    nominal_srate: float = 0.0,
):
    """
    Classify a raw LSL stream into one of the internal modalities.

    Returns:
        Modality value if recognized, otherwise None.
    """
    name = stream_name.lower().strip()
    stype = stream_type.lower().strip()
    sid = source_id.lower().strip()

    # ------------------------------------------------------------------
    # 1) True marker streams
    # ------------------------------------------------------------------
    # Real event marker streams are usually 1 channel and irregular/0 Hz
    if stype in {"marker", "markers"} and channel_count == 1:
        return Modality.MARKERS

    # Explicit marker-like stream names
    if name in {"sync", "start_stop", "psychopy_markers"} and channel_count == 1:
        return Modality.MARKERS

    # ------------------------------------------------------------------
    # 2) Quality/status streams -> do not treat as real markers
    # ------------------------------------------------------------------
    if "quality" in name or "quality" in sid:
        return None

    # ------------------------------------------------------------------
    # 3) Trust explicit stream TYPE first
    # ------------------------------------------------------------------
    if stype == "eeg":
        return Modality.EEG

    if stype in {"imu", "digital_aux"}:
        return Modality.IMU

    if stype in {"emg", "exg"}:
        return Modality.EMG

    # ------------------------------------------------------------------
    # 4) EEG fallback heuristics
    # ------------------------------------------------------------------
    if "eeg" in sid and channel_count >= 8 and nominal_srate > 0:
        return Modality.EEG

    if "eeg" in name and channel_count >= 8 and nominal_srate > 0:
        return Modality.EEG

    # ------------------------------------------------------------------
    # 5) IMU fallback heuristics
    # ------------------------------------------------------------------
    imu_tokens = ["imu", "acc", "gyro", "accelerometer", "gyroscope", "motion", "daux", "digital_aux"]

    if any(token in name for token in imu_tokens) or any(token in sid for token in imu_tokens):
        return Modality.IMU

    # ------------------------------------------------------------------
    # 6) EMG / EXG fallback heuristics
    # ------------------------------------------------------------------
    emg_tokens = ["emg", "exg", "muscle"]

    if (
        any(token in name for token in emg_tokens)
        or any(token in sid for token in emg_tokens)
    ) and nominal_srate >= 128:
        return Modality.EMG

    return None


def discover_streams(timeout: float = 2.0) -> list[StreamDescriptor]:
    """
    Discover all currently visible LSL streams and convert them into StreamDescriptor objects.
    """
    try:
        from pylsl import resolve_streams
    except ImportError as exc:
        raise RuntimeError(
            "pylsl is not installed in the current environment. "
            "Install it before using LSL stream discovery."
        ) from exc

    resolved_streams = resolve_streams(wait_time=timeout)
    discovered: list[StreamDescriptor] = []

    for info in resolved_streams:
        name = info.name()
        stream_type = info.type()
        source_id = info.source_id()
        channel_count = info.channel_count()
        nominal_srate = info.nominal_srate()
        uid = info.uid()
        hostname = info.hostname()

        modality = classify_lsl_stream(
            stream_name=name,
            stream_type=stream_type,
            source_id=source_id,
            channel_count=channel_count,
            nominal_srate=nominal_srate,
        )

        descriptor = StreamDescriptor(
            name=name,
            stream_type=stream_type,
            source_id=source_id,
            channel_count=channel_count,
            nominal_srate=nominal_srate,
            modality=modality,
            uid=uid,
            hostname=hostname,
        )
        discovered.append(descriptor)

    return discovered


def extract_modalities(streams: list[StreamDescriptor]) -> set[Modality]:
    """
    Extract the set of detected modalities from a list of StreamDescriptor objects.
    """
    return {
        stream.modality
        for stream in streams
        if stream.modality is not None
    }


def resolve_compatible_modes(detected_modalities: set[Modality]) -> list[str]:
    """
    Return acquisition modes whose required modalities are a subset
    of the currently detected modalities.
    """
    compatible_modes: list[str] = []

    for mode_name, mode in ACQUISITION_MODES.items():
        if mode.required_modalities.issubset(detected_modalities):
            compatible_modes.append(mode_name)

    return compatible_modes


def find_streams_by_modality(
    streams: list[StreamDescriptor],
    modality: Modality,
) -> list[StreamDescriptor]:
    """
    Return all streams matching a given modality.
    """
    return [stream for stream in streams if stream.modality == modality]