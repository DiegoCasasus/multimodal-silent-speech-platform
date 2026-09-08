from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, FrozenSet, Optional

from state import Modality

try:
    from config import CONFIG
    PROJECT_ROOT = CONFIG.project_root
    DEFAULT_DECODER_MANIFEST = CONFIG.decoder_manifest_path
    DEFAULT_IMAGINED_DECODER_MANIFEST = CONFIG.imagined_decoder_manifest_path
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DEFAULT_DECODER_MANIFEST = PROJECT_ROOT / "data" / "final_decoder_bundles" / "silent_v1" / "decoder_manifest.json"
    DEFAULT_IMAGINED_DECODER_MANIFEST = PROJECT_ROOT / "data" / "final_decoder_bundles" / "imagined_v1" / "decoder_manifest.json"


@dataclass(frozen=True)
class DecoderSpec:
    """Description of a decoder option shown by the GUI.

    For final deployment decoders, most fields are loaded from
    data/final_decoder_bundles/silent_v1/decoder_manifest.json.
    Legacy fields are kept so older GUI/controller code remains compatible.
    """

    key: str
    display_name: str
    required_modalities: FrozenSet[Modality]
    supports_brainprint: bool
    factory_name: str = ""
    description: str = ""
    bundle_path: Optional[Path] = None
    model_key: Optional[str] = None
    target_task: Optional[str] = None
    fusion_mode: Optional[str] = None
    input_channels: Optional[dict[str, int]] = None
    target_size: Optional[dict[str, int]] = None
    preprocessing_tag: Optional[str] = None
    window_start_s: Optional[float] = None
    window_end_s: Optional[float] = None
    best_val_balanced_accuracy: Optional[float] = None
    best_val_macro_f1: Optional[float] = None
    is_manifest_decoder: bool = False
    ensemble_config: Optional[dict[str, Any]] = None


def _modality_from_manifest_value(value: str) -> Modality:
    normalized = str(value).strip().upper()
    if normalized == "EEG":
        return Modality.EEG
    if normalized == "EMG":
        return Modality.EMG
    if normalized == "IMU":
        return Modality.IMU
    if normalized in {"MARKER", "MARKERS"}:
        return Modality.MARKERS
    raise ValueError(f"Unknown decoder modality in manifest: {value!r}")


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int_dict(value: Any) -> Optional[dict[str, int]]:
    if not isinstance(value, dict):
        return None
    out: dict[str, int] = {}
    for key, val in value.items():
        try:
            out[str(key)] = int(val)
        except (TypeError, ValueError):
            continue
    return out


def _resolve_manifest_path(manifest_path: Path, path_value: str | None) -> Optional[Path]:
    if not path_value:
        return None

    raw_path = Path(path_value)
    candidates = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append(PROJECT_ROOT / raw_path)
        candidates.append(manifest_path.parent / raw_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    # Keep a resolved project-relative path even if the file is not present yet,
    # so the GUI can show a useful error.
    return (PROJECT_ROOT / raw_path).resolve() if not raw_path.is_absolute() else raw_path.resolve()


def _resolve_bundle_path(manifest_path: Path, bundle_path_value: str | None) -> Optional[Path]:
    return _resolve_manifest_path(manifest_path, bundle_path_value)


def _resolve_ensemble_config(manifest_path: Path, item: dict[str, Any]) -> Optional[dict[str, Any]]:
    raw = item.get("ensemble")
    if not isinstance(raw, dict):
        return None

    components = []
    for comp in raw.get("components", []):
        if not isinstance(comp, dict):
            continue
        resolved = _resolve_manifest_path(manifest_path, comp.get("bundle_path"))
        if resolved is None:
            continue
        new_comp = dict(comp)
        new_comp["bundle_path"] = str(resolved)
        try:
            new_comp["weight"] = float(new_comp.get("weight", 1.0))
        except (TypeError, ValueError):
            new_comp["weight"] = 1.0
        components.append(new_comp)

    if not components:
        return None

    cfg = dict(raw)
    cfg["components"] = components
    cfg.setdefault("method", "weighted_probability_average")
    cfg.setdefault("name", item.get("name") or item.get("id") or item.get("model_key"))
    cfg.setdefault("display_name", item.get("display_name") or cfg.get("name"))
    return cfg


def _ensemble_components_exist(ensemble_config: Optional[dict[str, Any]]) -> bool:
    if not isinstance(ensemble_config, dict):
        return False
    components = ensemble_config.get("components", [])
    if not components:
        return False
    for comp in components:
        path = Path(str(comp.get("bundle_path", "")))
        if not path.exists():
            return False
    return True


def load_decoder_manifest(manifest_path: Path = DEFAULT_DECODER_MANIFEST) -> dict[str, DecoderSpec]:
    """Load final deployed decoder specs from a decoder_manifest.json file."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return {}

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    specs: dict[str, DecoderSpec] = {}
    for item in manifest.get("decoders", []):
        try:
            decoder_id = str(item.get("id") or item.get("name") or item.get("model_key")).strip()
            if not decoder_id:
                continue

            required = frozenset(
                _modality_from_manifest_value(m)
                for m in item.get("required_modalities", [])
            )
            if not required:
                continue

            bundle_path = _resolve_bundle_path(manifest_path, item.get("bundle_path"))
            ensemble_config = _resolve_ensemble_config(manifest_path, item)
            display_name = str(item.get("display_name") or decoder_id)
            model_key = item.get("model_key")

            specs[decoder_id] = DecoderSpec(
                key=decoder_id,
                display_name=display_name,
                required_modalities=required,
                supports_brainprint=Modality.EEG in required,
                factory_name=str(model_key or decoder_id),
                description=str(item.get("description") or ""),
                bundle_path=bundle_path,
                model_key=str(model_key) if model_key is not None else decoder_id,
                target_task=str(item.get("target_task")) if item.get("target_task") else None,
                fusion_mode=str(item.get("fusion_mode")) if item.get("fusion_mode") else None,
                input_channels=_safe_int_dict(item.get("input_channels")),
                target_size=_safe_int_dict(item.get("target_size")),
                preprocessing_tag=item.get("preprocessing_tag"),
                window_start_s=_safe_float(item.get("window_start_s")),
                window_end_s=_safe_float(item.get("window_end_s")),
                best_val_balanced_accuracy=_safe_float(item.get("best_val_balanced_accuracy")),
                best_val_macro_f1=_safe_float(item.get("best_val_macro_f1")),
                is_manifest_decoder=True,
                ensemble_config=ensemble_config,
            )
        except Exception:
            # Do not break the GUI because of one malformed manifest entry.
            continue

    return specs


LEGACY_MODEL_REGISTRY = {
    "eeg_only_v1": DecoderSpec(
        key="eeg_only_v1",
        display_name="EEG only v1",
        required_modalities=frozenset({Modality.EEG}),
        supports_brainprint=True,
        factory_name="build_eeg_only_decoder",
        description="Basic EEG-only decoder.",
        model_key="eeg_only_v1",
    ),
    "emg_only_v1": DecoderSpec(
        key="emg_only_v1",
        display_name="EMG only v1",
        required_modalities=frozenset({Modality.EMG}),
        supports_brainprint=False,
        factory_name="build_emg_only_decoder",
        description="EMG-only decoder.",
        model_key="emg_only_v1",
    ),
    "imu_only_v1": DecoderSpec(
        key="imu_only_v1",
        display_name="IMU only v1",
        required_modalities=frozenset({Modality.IMU}),
        supports_brainprint=False,
        factory_name="build_imu_only_decoder",
        description="IMU-only decoder.",
        model_key="imu_only_v1",
    ),
    "eeg_emg_v1": DecoderSpec(
        key="eeg_emg_v1",
        display_name="EEG + EMG v1",
        required_modalities=frozenset({Modality.EEG, Modality.EMG}),
        supports_brainprint=True,
        factory_name="build_eeg_emg_decoder",
        description="Multimodal decoder combining EEG and EMG.",
        model_key="eeg_emg_v1",
    ),
    "eeg_imu_v1": DecoderSpec(
        key="eeg_imu_v1",
        display_name="EEG + IMU v1",
        required_modalities=frozenset({Modality.EEG, Modality.IMU}),
        supports_brainprint=True,
        factory_name="build_eeg_imu_decoder",
        description="Multimodal decoder combining EEG and IMU.",
        model_key="eeg_imu_v1",
    ),
    "emg_imu_v1": DecoderSpec(
        key="emg_imu_v1",
        display_name="EMG + IMU v1",
        required_modalities=frozenset({Modality.EMG, Modality.IMU}),
        supports_brainprint=False,
        factory_name="build_emg_imu_decoder",
        description="Multimodal decoder combining EMG and IMU.",
        model_key="emg_imu_v1",
    ),
    "eeg_emg_imu_v1": DecoderSpec(
        key="eeg_emg_imu_v1",
        display_name="EEG + EMG + IMU v1",
        required_modalities=frozenset({Modality.EEG, Modality.EMG, Modality.IMU}),
        supports_brainprint=True,
        factory_name="build_eeg_emg_imu_decoder",
        description="Multimodal decoder combining EEG, EMG, and IMU.",
        model_key="eeg_emg_imu_v1",
    ),
    "eeg_emg_imu_gated_v1": DecoderSpec(
        key="eeg_emg_imu_gated_v1",
        display_name="EEG + EMG + IMU gated v1",
        required_modalities=frozenset({Modality.EEG, Modality.EMG, Modality.IMU}),
        supports_brainprint=True,
        factory_name="build_eeg_emg_imu_gated_decoder",
        description="Multimodal decoder combining EEG, EMG, and IMU through gated fusion.",
        model_key="eeg_emg_imu_gated_v1",
        fusion_mode="gated",
    ),
}


_silent_manifest_registry = load_decoder_manifest(DEFAULT_DECODER_MANIFEST)
_imagined_manifest_registry = load_decoder_manifest(DEFAULT_IMAGINED_DECODER_MANIFEST)
_manifest_registry = {**_silent_manifest_registry, **_imagined_manifest_registry}
MODEL_REGISTRY = _manifest_registry if _manifest_registry else LEGACY_MODEL_REGISTRY


# Display names are not guaranteed to be unique, but in the deployed GUI they should be.
_DISPLAY_TO_KEY = {spec.display_name: key for key, spec in MODEL_REGISTRY.items()}


def get_model_spec(model_key: str) -> DecoderSpec:
    return MODEL_REGISTRY[model_key]


def list_model_keys() -> list[str]:
    return list(MODEL_REGISTRY.keys())


def get_model_display_name(model_key: str | None) -> str:
    if not model_key:
        return ""
    spec = MODEL_REGISTRY.get(model_key)
    return spec.display_name if spec else str(model_key)


def get_model_key_from_display_name(value: str) -> str:
    value = str(value).strip()
    if value in MODEL_REGISTRY:
        return value
    return _DISPLAY_TO_KEY.get(value, value)


def get_compatible_models(
    available_modalities: set[Modality],
    target_task: str | None = None,
    require_existing_bundle: bool = False,
) -> list[str]:
    compatible_models = []

    for model_key, spec in MODEL_REGISTRY.items():
        if not spec.required_modalities.issubset(available_modalities):
            continue

        if target_task is not None and spec.target_task is not None and spec.target_task != target_task:
            continue

        if require_existing_bundle:
            has_bundle = spec.bundle_path is not None and Path(spec.bundle_path).exists()
            has_ensemble = _ensemble_components_exist(spec.ensemble_config)
            if not has_bundle and not has_ensemble:
                continue

        compatible_models.append(model_key)

    return compatible_models
