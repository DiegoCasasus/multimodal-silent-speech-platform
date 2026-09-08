from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from brainprint_service import BrainprintProcessResult, BrainprintService
from config import CONFIG, AppConfig
from models import MODEL_REGISTRY, get_compatible_models, get_model_spec
from recording import RecordingService
from state import (
    ACQUISITION_MODES,
    AppState,
    AppStatus,
    ExperimentMode,
    SessionMetadata,
)
from stream_discovery import (
    discover_streams,
    extract_modalities,
    resolve_compatible_modes,
)


class ExperimentController:
    def __init__(self, config: AppConfig = CONFIG):
        self.config = config
        self.recording_service = RecordingService(config=config)
        self.brainprint_service = BrainprintService(config=config)

        self.psychopy_proc: subprocess.Popen | None = None
        self.last_brainprint_result: BrainprintProcessResult | None = None

        self.state = AppState(
            selected_experiment_mode=config.default_experiment_mode,
            selected_mode=config.default_mode,
            brainprint_enabled=True,
            record_xdf=config.default_record_xdf,
        )
        self.state.online_decoder_running = False

    def _apply_experiment_constraints(self) -> None:
        mode = self.state.selected_experiment_mode

        # Rebuild acquisition choices on every experiment-mode transition.
        # Imagined mode intentionally constrains acquisition to EEG, but those
        # constraints must not leak into recording or online-silent mode.
        if mode == ExperimentMode.ONLINE_IMAGINED.value:
            self.state.available_modes = ["EEG"]
            self.state.selected_mode = "EEG"
        else:
            detected_modes = resolve_compatible_modes(self.state.detected_modalities)
            self.state.available_modes = detected_modes or list(ACQUISITION_MODES.keys())

            if self.state.selected_mode not in self.state.available_modes:
                preferred = self.config.default_mode
                self.state.selected_mode = (
                    preferred
                    if preferred in self.state.available_modes
                    else self.state.available_modes[0]
                )

        if mode == ExperimentMode.RECORDING.value:
            self.state.available_models = []
            self.state.selected_model = None
            return

        available_modalities = set()
        if self.state.selected_mode in ACQUISITION_MODES:
            acq_mode = ACQUISITION_MODES[self.state.selected_mode]
            available_modalities = set(acq_mode.required_modalities)

        target_task = self._target_task_from_mode()
        compatible_models = get_compatible_models(
            available_modalities=available_modalities,
            target_task=target_task,
            require_existing_bundle=self._is_online_mode(),
        )

        self.state.available_models = compatible_models

        if self.state.selected_model not in compatible_models:
            self.state.selected_model = compatible_models[0] if compatible_models else None

    def get_state(self) -> AppState:
        return self.state

    def clear_error(self) -> None:
        self.state.error_message = None
        if self.state.status == AppStatus.ERROR:
            self.state.status = AppStatus.IDLE

    def _clear_prepared_run_info(self) -> None:
        self.state.prepared_run_dir = None
        self.state.prepared_run_basename = None
        self.state.prepared_xdf_path = None
        self.state.prepared_manifest_path = None
        self.state.prepared_metadata_path = None
        self.state.prepared_bridge_path = None
        self.state.psychopy_script_path = None
        self.state.prepared_decoder_bundle_path = None
        self.state.prepared_online_results_dir = None
        self.state.online_decoder_service_status_path = None
        self.state.online_decoder_latest_path = None
        self.state.online_decoder_summary_path = None
        self.state.online_decoder_running = False

    def _legacy_task_mode_from_experiment_mode(self, experiment_mode: str) -> str:
        mapping = {
            ExperimentMode.RECORDING.value: "recording",
            ExperimentMode.ONLINE_SILENT.value: "online_silent",
            ExperimentMode.ONLINE_IMAGINED.value: "online_imagined",
        }
        return mapping.get(experiment_mode, experiment_mode)

    def _resolve_psychopy_script(self) -> Path:
        mode = self.state.selected_experiment_mode

        if mode == ExperimentMode.RECORDING.value:
            return self.config.offline_psychopy_script
        if mode == ExperimentMode.ONLINE_SILENT.value:
            return self.config.online_silent_psychopy_script
        if mode == ExperimentMode.ONLINE_IMAGINED.value:
            return self.config.online_imagined_psychopy_script

        raise ValueError(f"Unknown experiment mode: {mode}")

    def _is_online_mode(self) -> bool:
        return self.state.selected_experiment_mode in {
            ExperimentMode.ONLINE_SILENT.value,
            ExperimentMode.ONLINE_IMAGINED.value,
        }

    def _is_recording_mode(self) -> bool:
        return self.state.selected_experiment_mode == ExperimentMode.RECORDING.value

    def _target_task_from_mode(self) -> Optional[str]:
        if self.state.selected_experiment_mode == ExperimentMode.ONLINE_SILENT.value:
            return "silent"
        if self.state.selected_experiment_mode == ExperimentMode.ONLINE_IMAGINED.value:
            return "imagined"
        return None

    def _resolve_decoder_ensemble_config(self) -> Optional[dict]:
        if not self._is_online_mode() or not self.state.selected_model:
            return None

        task = self._target_task_from_mode()
        if task is None:
            return None

        model_key = self.state.selected_model
        if model_key not in MODEL_REGISTRY:
            return None

        spec = get_model_spec(model_key)
        if spec.target_task is not None and spec.target_task != task:
            return None
        return spec.ensemble_config

    def _resolve_decoder_bundle_path(self) -> Optional[Path]:
        if not self._is_online_mode() or not self.state.selected_model:
            return None

        task = self._target_task_from_mode()
        if task is None:
            return None

        model_key = self.state.selected_model
        if model_key in MODEL_REGISTRY:
            spec = get_model_spec(model_key)
            if spec.target_task is not None and spec.target_task != task:
                return None
            if spec.bundle_path is not None and Path(spec.bundle_path).exists():
                return Path(spec.bundle_path).resolve()

        # Legacy fallback for older folders when no manifest entry is available.
        root = self.config.decoder_models_root
        window_tag = self.config.default_decoder_window_tag

        candidate_dirs = [
            root / f"{model_key}_{task}_{window_tag}",
            root / f"{model_key}_{task}",
            root / model_key,
        ]
        candidate_files = [
            f"{model_key}_{task}_decoder_bundle.pt",
            f"{model_key}_decoder_bundle.pt",
        ]

        for directory in candidate_dirs:
            for filename in candidate_files:
                path = directory / filename
                if path.exists():
                    return path.resolve()

        return None

    def _python_has_psychopy(self, python_exe: Path) -> bool:
        """Return True if this interpreter can import PsychoPy.

        This check intentionally probes the candidate interpreter instead of
        assuming that a path convention is valid. That makes the launcher safer
        across Miniforge/conda environments, PsychoPy standalone installs, and
        future lab computers.
        """
        try:
            python_exe = Path(python_exe).expanduser()
        except Exception:
            return False

        if not python_exe.exists() or python_exe.is_dir():
            return False

        # App bundles are not Python interpreters. They may be useful for manual
        # launching, but this controller needs an executable that accepts
        # "-c import psychopy".
        if python_exe.suffix.lower() == ".app":
            return False

        try:
            probe = subprocess.run(
                [str(python_exe), "-c", "import psychopy"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return probe.returncode == 0
        except Exception:
            return False

    def _deduplicate_paths(self, candidates: list[Path]) -> list[Path]:
        seen: set[str] = set()
        unique: list[Path] = []

        for candidate in candidates:
            try:
                path = Path(candidate).expanduser()
                key = str(path.resolve()) if path.exists() else str(path)
            except Exception:
                continue

            if key in seen:
                continue

            seen.add(key)
            unique.append(path)

        return unique

    def _glob_psychopy_python_candidates(self) -> list[Path]:
        """Return platform-aware PsychoPy Python candidates.

        The explicit config value remains the preferred route. These candidates
        are only fallbacks so the GUI can work on common Windows/macOS lab
        installations without editing source code.
        """
        candidates: list[Path] = []
        home = Path.home()

        system = platform.system().lower()

        if system == "windows":
            psychopy_root = home / "AppData" / "Roaming" / "psychopy4" / ".python"
            if psychopy_root.exists():
                candidates.extend(sorted(psychopy_root.glob("*/Scripts/python.exe"), reverse=True))

            candidates.extend(
                [
                    home / "AppData" / "Roaming" / "psychopy4" / ".python" / "2026.1.1" / "Scripts" / "python.exe",
                    home / "AppData" / "Roaming" / "psychopy4" / ".python" / "2025.2.4" / "Scripts" / "python.exe",
                    home / "AppData" / "Roaming" / "psychopy4" / ".python" / "2025.1.1" / "Scripts" / "python.exe",
                ]
            )

        elif system == "darwin":
            # The most reliable macOS route is still to set
            # external_tools.psychopy_python_exe in config/project_config.json.
            # These are non-destructive fallbacks for common app locations.
            candidates.extend(
                [
                    Path("/Applications/PsychoPy.app/Contents/MacOS/python"),
                    Path("/Applications/PsychoPy.app/Contents/Resources/python"),
                    Path("/Applications/PsychoPy.app/Contents/MacOS/PsychoPy"),
                    home / "Applications" / "PsychoPy.app" / "Contents" / "MacOS" / "python",
                    home / "Applications" / "PsychoPy.app" / "Contents" / "Resources" / "python",
                    home / "Applications" / "PsychoPy.app" / "Contents" / "MacOS" / "PsychoPy",
                ]
            )

        else:
            # Linux or other Unix-like systems: rely mostly on the active
            # environment / PATH, but allow common local installs.
            candidates.extend(
                [
                    Path("/usr/local/bin/python3"),
                    Path("/usr/bin/python3"),
                    home / "miniforge3" / "bin" / "python",
                    home / "miniconda3" / "bin" / "python",
                    home / "anaconda3" / "bin" / "python",
                ]
            )

        return self._deduplicate_paths(candidates)

    def _resolve_psychopy_python(self) -> Path:
        candidates: list[Path] = []

        # 1) Explicit project config override.
        if self.config.psychopy_python_exe is not None:
            candidates.append(Path(self.config.psychopy_python_exe))

        # 2) Environment variable override for portable lab launchers.
        for env_name in ("TFM_PSYCHOPY_PYTHON", "PSYCHOPY_PYTHON", "PSYCHOPY_PYTHON_EXE"):
            env_value = os.environ.get(env_name)
            if env_value:
                candidates.append(Path(env_value))

        # 3) The interpreter running the GUI. This is ideal when the Miniforge/
        # conda environment already contains PsychoPy.
        candidates.append(Path(sys.executable))

        # 4) Common PATH interpreters.
        for exe_name in ("python", "python3", "pythonw"):
            found = shutil.which(exe_name)
            if found:
                candidates.append(Path(found))

        # 5) Platform-specific PsychoPy standalone candidates.
        candidates.extend(self._glob_psychopy_python_candidates())

        candidates = self._deduplicate_paths(candidates)

        checked = []
        for candidate in candidates:
            candidate = Path(candidate)
            checked.append(str(candidate))
            if self._python_has_psychopy(candidate):
                return candidate.resolve() if candidate.exists() else candidate

        raise FileNotFoundError(
            "Could not resolve a PsychoPy Python interpreter with psychopy installed.\n"
            "Checked:\n- " + "\n- ".join(checked) + "\n"
            "Fix options:\n"
            "1) Run the GUI from the same Miniforge/conda environment that contains PsychoPy.\n"
            "2) Set external_tools.psychopy_python_exe in config/project_config.json.\n"
            "3) Set the TFM_PSYCHOPY_PYTHON environment variable in your launcher script."
        )

    def refresh_streams(self) -> None:
        self.state.status = AppStatus.DISCOVERING
        self.state.error_message = None
        self._clear_prepared_run_info()

        try:
            streams = discover_streams(timeout=self.config.stream_discovery_timeout)
            detected_modalities = extract_modalities(streams)
            available_modes = resolve_compatible_modes(detected_modalities)

            self.state.detected_streams = streams
            self.state.detected_modalities = detected_modalities
            self.state.available_modes = available_modes

            if self.state.selected_mode not in available_modes:
                self.state.selected_mode = available_modes[0] if available_modes else self.config.default_mode

            self._update_available_models()
            self.state.status = AppStatus.READY

        except Exception as exc:
            self.state.status = AppStatus.ERROR
            self.state.error_message = f"Stream discovery failed: {exc}"

    def _update_available_models(self) -> None:
        self._apply_experiment_constraints()

    def set_experiment_mode(self, experiment_mode: str) -> None:
        valid_modes = {e.value for e in ExperimentMode}
        if experiment_mode not in valid_modes:
            self.state.error_message = f"Unknown experiment mode '{experiment_mode}'."
            return

        self.state.selected_experiment_mode = experiment_mode
        self.state.error_message = None
        self._clear_prepared_run_info()
        self._update_available_models()

        if self.state.session is not None:
            self.state.session.experiment_mode = experiment_mode
            self.state.session.task_mode = self._legacy_task_mode_from_experiment_mode(experiment_mode)

    def set_selected_mode(self, mode_name: str) -> None:
        if self.state.selected_experiment_mode == ExperimentMode.ONLINE_IMAGINED.value:
            self.state.selected_mode = "EEG"
            self.state.error_message = None
            self._clear_prepared_run_info()
            self._update_available_models()
            return

        if mode_name not in ACQUISITION_MODES:
            self.state.error_message = f"Unknown mode '{mode_name}'."
            return

        self.state.selected_mode = mode_name
        self.state.error_message = None
        self._clear_prepared_run_info()
        self._update_available_models()

    def set_selected_model(self, model_key: str) -> None:
        if model_key not in MODEL_REGISTRY:
            self.state.error_message = f"Unknown model '{model_key}'."
            return

        self.state.selected_model = model_key
        self.state.error_message = None
        self._clear_prepared_run_info()

    def set_session_metadata(
        self,
        participant_id: str,
        session_id: str,
        run_id: str,
        experiment_mode: str,
        notes: str = "",
        operator: str = "",
        site: str = "",
    ) -> None:
        experiment_mode = experiment_mode.strip()
        self.state.selected_experiment_mode = experiment_mode

        self.state.session = SessionMetadata(
            participant_id=participant_id.strip(),
            session_id=session_id.strip(),
            run_id=run_id.strip(),
            experiment_mode=experiment_mode,
            task_mode=self._legacy_task_mode_from_experiment_mode(experiment_mode),
            notes=notes.strip(),
            operator=operator.strip(),
            site=site.strip(),
        )

        self.state.error_message = None
        self._clear_prepared_run_info()
        self._update_available_models()

    def set_brainprint_loaded(
        self,
        loaded: bool,
        subject_id: Optional[str] = None,
    ) -> None:
        self.state.brainprint_loaded = loaded
        self.state.brainprint_subject_id = subject_id if loaded else None
        self._clear_prepared_run_info()

    def validate_prepare_run(self) -> list[str]:
        errors: list[str] = []

        if not self.state.selected_experiment_mode:
            errors.append("No experiment mode selected.")

        if self.state.selected_mode is None:
            errors.append("No acquisition mode selected.")

        if self._is_online_mode() and self.state.selected_model is None:
            errors.append("No decoder model selected.")

        if self.state.session is None:
            errors.append("Session metadata has not been set.")
        else:
            if not self.state.session.participant_id:
                errors.append("Participant ID is missing.")
            if not self.state.session.session_id:
                errors.append("Session ID is missing.")
            if not self.state.session.run_id:
                errors.append("Run ID is missing.")
            if not self.state.session.experiment_mode:
                errors.append("Experiment mode is missing.")

        if self._is_online_mode():
            bundle_path = self._resolve_decoder_bundle_path()
            ensemble_config = self._resolve_decoder_ensemble_config()
            if bundle_path is None and ensemble_config is None:
                errors.append(
                    "Could not resolve a decoder bundle or ensemble for the selected online model. "
                    "Check the task-specific decoder manifest and bundle paths."
                )

        return errors

    def validate_configuration(self) -> list[str]:
        errors = self.validate_prepare_run()

        if not self.state.detected_streams:
            errors.append("No LSL streams detected.")
            return errors

        if self.state.selected_mode in ACQUISITION_MODES:
            required_modalities = ACQUISITION_MODES[self.state.selected_mode].required_modalities
            missing_modalities = [
                modality.value
                for modality in required_modalities
                if modality not in self.state.detected_modalities
            ]
            if missing_modalities:
                errors.append(
                    "Missing required detected modalities for selected acquisition mode: "
                    + ", ".join(missing_modalities)
                )

        if self._is_online_mode() and self.state.selected_model in MODEL_REGISTRY:
            decoder_required = get_model_spec(self.state.selected_model).required_modalities
            missing_decoder_modalities = [
                modality.value
                for modality in decoder_required
                if modality not in self.state.detected_modalities
            ]
            if missing_decoder_modalities:
                errors.append(
                    "Missing required detected modalities for selected decoder: "
                    + ", ".join(missing_decoder_modalities)
                )

        return errors

    def _clear_online_result_placeholders(self, online_results_dir: Path) -> None:
        try:
            online_results_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return

        stale_files = [
            online_results_dir / "service_status.json",
            online_results_dir / "latest.json",
            online_results_dir / "run_summary.json",
        ]

        for path in stale_files:
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass

        try:
            for path in online_results_dir.glob("trial_*.json"):
                path.unlink()
        except Exception:
            pass

    def prepare_run(self) -> list[str]:
        errors = self.validate_prepare_run()
        if errors:
            return errors

        try:
            decoder_bundle_path = self._resolve_decoder_bundle_path()
            decoder_ensemble_config = self._resolve_decoder_ensemble_config()

            # For online modes, PsychoPy will compute a fresh calibration brainprint.
            brainprint_compare_json = None

            brainprint_summary_json = None
            if self.config.brainprint_summary_json.exists():
                brainprint_summary_json = str(self.config.brainprint_summary_json.resolve())

            run_paths = self.recording_service.prepare_run(
                session=self.state.session,
                selected_mode=self.state.selected_mode,
                selected_model=self.state.selected_model,
                record_xdf=self.state.record_xdf,
                brainprint_enabled=self.state.brainprint_enabled,
                brainprint_loaded=self.state.brainprint_loaded,
                brainprint_subject_id=self.state.brainprint_subject_id,
                streams=self.state.detected_streams,
                decoder_bundle_path=str(decoder_bundle_path) if decoder_bundle_path is not None else None,
                decoder_ensemble=decoder_ensemble_config,
                brainprint_compare_json=brainprint_compare_json,
                brainprint_summary_json=brainprint_summary_json,
            )

            calibration_xdf_path = getattr(run_paths, "calibration_xdf_path", None)
            if calibration_xdf_path is None:
                calibration_xdf_path = getattr(run_paths, "xdf_path", None)

            self.state.prepared_run_dir = str(run_paths.run_dir)
            self.state.prepared_run_basename = run_paths.basename
            self.state.prepared_xdf_path = str(calibration_xdf_path) if calibration_xdf_path is not None else None
            self.state.prepared_manifest_path = str(run_paths.manifest_path)
            self.state.prepared_metadata_path = str(run_paths.metadata_path)
            self.state.prepared_bridge_path = str(run_paths.bridge_path)
            self.state.psychopy_script_path = str(self._resolve_psychopy_script())
            self.state.prepared_decoder_bundle_path = str(decoder_bundle_path) if decoder_bundle_path else None
            self.state.prepared_online_results_dir = str(run_paths.online_results_dir)
            self.state.online_decoder_service_status_path = str(run_paths.online_results_dir / "service_status.json")
            self.state.online_decoder_latest_path = str(run_paths.online_results_dir / "latest.json")
            self.state.online_decoder_summary_path = str(run_paths.online_results_dir / "run_summary.json")
            self.state.online_decoder_running = False
            self.state.error_message = None

            if self._is_online_mode():
                self._clear_online_result_placeholders(run_paths.online_results_dir)

            if self.state.status == AppStatus.IDLE:
                self.state.status = AppStatus.READY

            return []

        except Exception as exc:
            self.state.status = AppStatus.ERROR
            self.state.error_message = f"Run preparation failed: {exc}"
            return [self.state.error_message]

    def launch_psychopy(self) -> list[str]:
        if self.psychopy_proc is not None and self.psychopy_proc.poll() is None:
            msg = "PsychoPy is already running."
            self.state.error_message = msg
            return [msg]

        if not self.state.prepared_bridge_path:
            errors = self.prepare_run()
            if errors:
                return errors

        try:
            script_path = Path(self.state.psychopy_script_path or self._resolve_psychopy_script())
            python_exe = self._resolve_psychopy_python()

            if not script_path.exists():
                msg = f"PsychoPy script not found: {script_path}"
                self.state.status = AppStatus.ERROR
                self.state.error_message = msg
                return [msg]

            self.psychopy_proc = subprocess.Popen(
                [str(python_exe), str(script_path)],
                cwd=str(script_path.parent),
            )

            self.state.status = (
                AppStatus.RECORDING
                if self.state.selected_experiment_mode == ExperimentMode.RECORDING.value
                else AppStatus.DECODING
            )
            self.state.online_decoder_running = False
            self.state.error_message = None

            self.state.last_pipeline_status = None
            self.state.last_pipeline_message = None
            self.state.last_brainprint_npz_path = None
            self.state.last_compare_report_path = None

            return []

        except Exception as exc:
            self.state.status = AppStatus.ERROR
            self.state.error_message = f"Failed to launch PsychoPy: {exc}"
            return [self.state.error_message]

    def poll_psychopy_process(self) -> dict | None:
        if self.psychopy_proc is None:
            return None
    
        return_code = self.psychopy_proc.poll()
        if return_code is None:
            return None
    
        self.psychopy_proc = None
        self.state.status = AppStatus.READY
        self.state.online_decoder_running = False
        self.state.error_message = None
    
        # Online experiments now manage calibration brainprint internally in PsychoPy.
        # Do not run or report the old controller-side post-processing path.
        if self._is_online_mode():
            self.last_brainprint_result = None
            self.state.last_pipeline_status = None
            self.state.last_pipeline_message = None
            self.state.last_brainprint_npz_path = None
            self.state.last_compare_report_path = None
    
            return {
                "return_code": return_code,
                "brainprint_result": None,
                "brainprint_skipped_reason": None,
            }
    
        should_process, reason = self.brainprint_service.should_process_after_recording(
            bridge_path=self.state.prepared_bridge_path,
            experiment_mode=self.state.selected_experiment_mode,
            selected_mode=self.state.selected_mode,
        )
    
        brainprint_result = None
    
        if should_process:
            try:
                brainprint_result = self.brainprint_service.process_recording_brainprint(
                    self.state.prepared_bridge_path
                )
                self.last_brainprint_result = brainprint_result
    
                self.state.last_pipeline_status = brainprint_result.status
                self.state.last_pipeline_message = brainprint_result.message
                self.state.last_brainprint_npz_path = brainprint_result.candidate_npz
                self.state.last_compare_report_path = brainprint_result.comparison_json
    
                if brainprint_result.status == "failed":
                    self.state.status = AppStatus.ERROR
                    self.state.error_message = brainprint_result.message
                else:
                    self.state.error_message = None
    
            except Exception as exc:
                error_msg = f"Brainprint pipeline crashed: {exc}"
                self.last_brainprint_result = None
                self.state.last_pipeline_status = "failed"
                self.state.last_pipeline_message = error_msg
                self.state.last_brainprint_npz_path = None
                self.state.last_compare_report_path = None
                self.state.status = AppStatus.ERROR
                self.state.error_message = error_msg
    
                return {
                    "return_code": return_code,
                    "brainprint_result": None,
                    "brainprint_skipped_reason": error_msg,
                }
    
        else:
            self.last_brainprint_result = None
            self.state.last_pipeline_status = "skipped"
            self.state.last_pipeline_message = reason
            self.state.last_brainprint_npz_path = None
            self.state.last_compare_report_path = None
            self.state.error_message = None
    
        return {
            "return_code": return_code,
            "brainprint_result": brainprint_result,
            "brainprint_skipped_reason": None if should_process else reason,
        }