from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from config import CONFIG, AppConfig
from state import SessionMetadata, StreamDescriptor


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    basename: str
    xdf_path: Path
    manifest_path: Path
    metadata_path: Path
    bridge_path: Path
    calibration_xdf_path: Path
    silent_xdf_path: Path
    imagined_xdf_path: Path
    online_results_dir: Path


ALL_BLOCK_NAMES = ("calibration", "silent", "imagined")


def path_to_project_relative_string(
    path_value: str | Path | None,
    project_root: Path,
) -> Optional[str]:
    """Return a portable path string for JSON sidecars.

    If the path is inside the project root, it is written relative to the project
    root using POSIX-style separators. If it is outside the project root, it is
    kept absolute because it is probably an intentional external override.

    This function is only for JSON serialization. The application should keep
    using real Path objects internally.
    """
    if path_value is None:
        return None

    text = str(path_value).strip()
    if not text:
        return None

    root = Path(project_root).expanduser().resolve()
    candidate = Path(text).expanduser()

    if candidate.is_absolute():
        candidate = candidate.resolve()
    else:
        candidate = (root / candidate).resolve()

    try:
        return candidate.relative_to(root).as_posix()
    except ValueError:
        return str(candidate)


def make_decoder_ensemble_portable(
    decoder_ensemble: Optional[dict[str, Any]],
    project_root: Path,
) -> Optional[dict[str, Any]]:
    """Convert decoder ensemble component paths to project-relative strings."""
    if decoder_ensemble is None:
        return None

    if not isinstance(decoder_ensemble, dict):
        return decoder_ensemble

    portable = deepcopy(decoder_ensemble)

    if portable.get("bundle_path"):
        portable["bundle_path"] = path_to_project_relative_string(
            portable["bundle_path"],
            project_root,
        )

    components = portable.get("components")
    if isinstance(components, list):
        portable_components = []
        for component in components:
            if isinstance(component, dict):
                component = dict(component)
                if component.get("bundle_path"):
                    component["bundle_path"] = path_to_project_relative_string(
                        component["bundle_path"],
                        project_root,
                    )
            portable_components.append(component)
        portable["components"] = portable_components

    return portable


def get_blocks_for_experiment_mode(experiment_mode: str) -> tuple[str, ...]:
    if experiment_mode == "recording":
        return ("calibration", "silent", "imagined")
    if experiment_mode == "online_silent":
        return ("calibration", "silent")
    if experiment_mode == "online_imagined":
        return ("calibration", "imagined")
    raise ValueError(f"Unknown experiment mode: {experiment_mode}")


def sanitize_label(value: str) -> str:
    cleaned = value.strip().replace(" ", "_")
    allowed = []
    for ch in cleaned:
        if ch.isalnum() or ch in {"-", "_"}:
            allowed.append(ch)
    return "".join(allowed)


def build_run_basename(
    session: SessionMetadata,
    selected_mode: str,
    selected_model: Optional[str] = None,
) -> str:
    participant = sanitize_label(session.participant_id)
    session_id = sanitize_label(session.session_id)
    run_id = sanitize_label(session.run_id)
    task_mode = sanitize_label(session.task_mode)
    mode = sanitize_label(selected_mode)

    parts = [
        f"sub-{participant}",
        f"ses-{session_id}",
        f"run-{run_id}",
        f"task-{task_mode}",
        f"mode-{mode}",
    ]

    if selected_model:
        parts.append(f"model-{sanitize_label(selected_model)}")

    return "_".join(parts)


def build_block_recording_filename(session: SessionMetadata, block: str) -> str:
    participant = sanitize_label(session.participant_id)
    session_id = sanitize_label(session.session_id)
    run_id = sanitize_label(session.run_id)
    experiment_mode = sanitize_label(session.experiment_mode)
    return (
        f"exp_sub_{participant}_ses_{session_id}_run_{run_id}"
        f"_mode_{experiment_mode}_bl_{block}.xdf"
    )


def build_block_recording_path(
    config: AppConfig,
    session: SessionMetadata,
    block: str,
) -> Path:
    experiment_mode = sanitize_label(session.experiment_mode)
    return (
        config.recordings_root
        / experiment_mode
        / block
        / build_block_recording_filename(session, block)
    )


class RecordingService:
    def __init__(self, config: AppConfig = CONFIG):
        self.config = config

    def ensure_output_root(self) -> None:
        self.config.output_root.mkdir(parents=True, exist_ok=True)
        self.config.online_results_root.mkdir(parents=True, exist_ok=True)

    def ensure_recordings_root(self, experiment_mode: str) -> None:
        self.config.recordings_root.mkdir(parents=True, exist_ok=True)

        for block in get_blocks_for_experiment_mode(experiment_mode):
            (self.config.recordings_root / experiment_mode / block).mkdir(parents=True, exist_ok=True)

    def get_bridge_path(self) -> Path:
        self.ensure_output_root()
        self.config.bridge_file.parent.mkdir(parents=True, exist_ok=True)
        return self.config.bridge_file

    def prepare_run_paths(
        self,
        session: SessionMetadata,
        selected_mode: str,
        selected_model: Optional[str] = None,
    ) -> RunPaths:
        self.ensure_output_root()
        self.ensure_recordings_root(session.experiment_mode)

        participant = sanitize_label(session.participant_id)
        session_id = sanitize_label(session.session_id)

        run_dir = self.config.output_root / f"sub-{participant}" / f"ses-{session_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        basename = build_run_basename(
            session=session,
            selected_mode=selected_mode,
            selected_model=selected_model,
        )

        xdf_path = run_dir / f"{basename}.xdf"
        manifest_path = run_dir / f"{basename}_manifest.json"
        metadata_path = run_dir / f"{basename}_metadata.json"
        bridge_path = self.get_bridge_path()

        calibration_xdf_path = build_block_recording_path(self.config, session, "calibration")
        silent_xdf_path = build_block_recording_path(self.config, session, "silent")
        imagined_xdf_path = build_block_recording_path(self.config, session, "imagined")

        online_results_dir = self.config.online_results_root / basename
        online_results_dir.mkdir(parents=True, exist_ok=True)

        return RunPaths(
            run_dir=run_dir,
            basename=basename,
            xdf_path=xdf_path,
            manifest_path=manifest_path,
            metadata_path=metadata_path,
            bridge_path=bridge_path,
            calibration_xdf_path=calibration_xdf_path,
            silent_xdf_path=silent_xdf_path,
            imagined_xdf_path=imagined_xdf_path,
            online_results_dir=online_results_dir,
        )

    def write_run_manifest(
        self,
        run_paths: RunPaths,
        session: SessionMetadata,
        selected_mode: str,
        selected_model: Optional[str],
        record_xdf: bool,
        brainprint_enabled: bool,
        brainprint_loaded: bool,
        brainprint_subject_id: Optional[str],
        streams: list[StreamDescriptor],
        decoder_bundle_path: Optional[str] = None,
        decoder_ensemble: Optional[dict] = None,
        brainprint_compare_json: Optional[str] = None,
        brainprint_summary_json: Optional[str] = None,
    ) -> None:
        project_root = self.config.project_root

        manifest = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "path_format_version": "project_relative_v1",
            "paths_are_relative_to": "project_root",
            "session": asdict(session),
            "selected_mode": selected_mode,
            "selected_model": selected_model,
            "record_xdf": record_xdf,
            "brainprint_enabled": brainprint_enabled,
            "brainprint_loaded": brainprint_loaded,
            "brainprint_subject_id": brainprint_subject_id,
            "prepared_xdf_sidecar_path": path_to_project_relative_string(
                run_paths.xdf_path,
                project_root,
            ),
            "online_results_dir": path_to_project_relative_string(
                run_paths.online_results_dir,
                project_root,
            ),
            "decoder_bundle_path": path_to_project_relative_string(
                decoder_bundle_path,
                project_root,
            ),
            "decoder_ensemble": make_decoder_ensemble_portable(
                decoder_ensemble,
                project_root,
            ),
            "brainprint_compare_json": path_to_project_relative_string(
                brainprint_compare_json,
                project_root,
            ),
            "brainprint_summary_json": path_to_project_relative_string(
                brainprint_summary_json,
                project_root,
            ),
            "recording_block_paths": {
                "calibration": path_to_project_relative_string(
                    run_paths.calibration_xdf_path,
                    project_root,
                ),
                "silent": path_to_project_relative_string(
                    run_paths.silent_xdf_path,
                    project_root,
                ),
                "imagined": path_to_project_relative_string(
                    run_paths.imagined_xdf_path,
                    project_root,
                ),
            },
            "streams": [
                {
                    "name": s.name,
                    "stream_type": s.stream_type,
                    "source_id": s.source_id,
                    "channel_count": s.channel_count,
                    "nominal_srate": s.nominal_srate,
                    "modality": s.modality.value if s.modality is not None else None,
                    "uid": s.uid,
                    "hostname": s.hostname,
                }
                for s in streams
            ],
        }

        with open(run_paths.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def write_psychopy_metadata(
        self,
        run_paths: RunPaths,
        session: SessionMetadata,
        selected_mode: str,
        selected_model: Optional[str],
        decoder_bundle_path: Optional[str] = None,
        decoder_ensemble: Optional[dict] = None,
        brainprint_compare_json: Optional[str] = None,
        brainprint_summary_json: Optional[str] = None,
    ) -> None:
        project_root = self.config.project_root

        payload = {
            "path_format_version": "project_relative_v1",
            "paths_are_relative_to": "project_root",
            "participant_id": session.participant_id,
            "session_id": session.session_id,
            "run_id": session.run_id,
            "experiment_mode": session.experiment_mode,
            "task_mode": session.task_mode,
            "selected_mode": selected_mode,
            "selected_model": selected_model,
            "expected_xdf_path": path_to_project_relative_string(
                run_paths.calibration_xdf_path,
                project_root,
            ),
            "basename": run_paths.basename,
            "online_results_dir": path_to_project_relative_string(
                run_paths.online_results_dir,
                project_root,
            ),
            "decoder_bundle_path": path_to_project_relative_string(
                decoder_bundle_path,
                project_root,
            ),
            "decoder_ensemble": make_decoder_ensemble_portable(
                decoder_ensemble,
                project_root,
            ),
            "brainprint_compare_json": path_to_project_relative_string(
                brainprint_compare_json,
                project_root,
            ),
            "brainprint_summary_json": path_to_project_relative_string(
                brainprint_summary_json,
                project_root,
            ),
            "recording_block_paths": {
                "calibration": path_to_project_relative_string(
                    run_paths.calibration_xdf_path,
                    project_root,
                ),
                "silent": path_to_project_relative_string(
                    run_paths.silent_xdf_path,
                    project_root,
                ),
                "imagined": path_to_project_relative_string(
                    run_paths.imagined_xdf_path,
                    project_root,
                ),
            },
        }

        with open(run_paths.metadata_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def write_latest_prepared_run(
        self,
        run_paths: RunPaths,
        session: SessionMetadata,
        selected_mode: str,
        selected_model: Optional[str],
        decoder_bundle_path: Optional[str] = None,
        decoder_ensemble: Optional[dict] = None,
        brainprint_compare_json: Optional[str] = None,
        brainprint_summary_json: Optional[str] = None,
    ) -> None:
        project_root = self.config.project_root

        payload = {
            "prepared_at": datetime.now().isoformat(timespec="seconds"),
            "path_format_version": "project_relative_v1",
            "paths_are_relative_to": "project_root",
            "participant_id": session.participant_id,
            "session_id": session.session_id,
            "run_id": session.run_id,
            "experiment_mode": session.experiment_mode,
            "task_mode": session.task_mode,
            "selected_mode": selected_mode,
            "selected_model": selected_model,
            "basename": run_paths.basename,
            "expected_xdf_path": path_to_project_relative_string(
                run_paths.calibration_xdf_path,
                project_root,
            ),
            "metadata_path": path_to_project_relative_string(
                run_paths.metadata_path,
                project_root,
            ),
            "manifest_path": path_to_project_relative_string(
                run_paths.manifest_path,
                project_root,
            ),
            "online_results_dir": path_to_project_relative_string(
                run_paths.online_results_dir,
                project_root,
            ),
            "decoder_bundle_path": path_to_project_relative_string(
                decoder_bundle_path,
                project_root,
            ),
            "decoder_ensemble": make_decoder_ensemble_portable(
                decoder_ensemble,
                project_root,
            ),
            "brainprint_compare_json": path_to_project_relative_string(
                brainprint_compare_json,
                project_root,
            ),
            "brainprint_summary_json": path_to_project_relative_string(
                brainprint_summary_json,
                project_root,
            ),
            "recording_block_paths": {
                "calibration": path_to_project_relative_string(
                    run_paths.calibration_xdf_path,
                    project_root,
                ),
                "silent": path_to_project_relative_string(
                    run_paths.silent_xdf_path,
                    project_root,
                ),
                "imagined": path_to_project_relative_string(
                    run_paths.imagined_xdf_path,
                    project_root,
                ),
            },
        }

        with open(run_paths.bridge_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def prepare_run(
        self,
        session: SessionMetadata,
        selected_mode: str,
        selected_model: Optional[str],
        record_xdf: bool,
        brainprint_enabled: bool,
        brainprint_loaded: bool,
        brainprint_subject_id: Optional[str],
        streams: list[StreamDescriptor],
        decoder_bundle_path: Optional[str] = None,
        decoder_ensemble: Optional[dict] = None,
        brainprint_compare_json: Optional[str] = None,
        brainprint_summary_json: Optional[str] = None,
    ) -> RunPaths:
        run_paths = self.prepare_run_paths(
            session=session,
            selected_mode=selected_mode,
            selected_model=selected_model,
        )

        self.write_run_manifest(
            run_paths=run_paths,
            session=session,
            selected_mode=selected_mode,
            selected_model=selected_model,
            record_xdf=record_xdf,
            brainprint_enabled=brainprint_enabled,
            brainprint_loaded=brainprint_loaded,
            brainprint_subject_id=brainprint_subject_id,
            streams=streams,
            decoder_bundle_path=decoder_bundle_path,
            decoder_ensemble=decoder_ensemble,
            brainprint_compare_json=brainprint_compare_json,
            brainprint_summary_json=brainprint_summary_json,
        )

        self.write_psychopy_metadata(
            run_paths=run_paths,
            session=session,
            selected_mode=selected_mode,
            selected_model=selected_model,
            decoder_bundle_path=decoder_bundle_path,
            decoder_ensemble=decoder_ensemble,
            brainprint_compare_json=brainprint_compare_json,
            brainprint_summary_json=brainprint_summary_json,
        )

        self.write_latest_prepared_run(
            run_paths=run_paths,
            session=session,
            selected_mode=selected_mode,
            selected_model=selected_model,
            decoder_bundle_path=decoder_bundle_path,
            decoder_ensemble=decoder_ensemble,
            brainprint_compare_json=brainprint_compare_json,
            brainprint_summary_json=brainprint_summary_json,
        )

        return run_paths
