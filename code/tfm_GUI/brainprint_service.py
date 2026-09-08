from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config import CONFIG, AppConfig
try:
    import pyxdf
except Exception:
    pyxdf = None


@dataclass
class BrainprintProcessResult:
    status: str
    message: str
    calibration_xdf: Optional[str] = None
    candidate_npz: Optional[str] = None
    comparison_json: Optional[str] = None
    best_match_file: Optional[str] = None
    best_match_cosine: Optional[float] = None
    stdout: str = ""
    stderr: str = ""


class BrainprintService:
    """
    Post-run brainprint automation.

    Current policy:
    - only process after a full RECORDING run has finished
    - only process if EEG is part of the selected acquisition mode
    - compute from the calibration XDF only
    - save NPZ to candidates/
    - compare against bank/
    - save JSON report to comparisons/
    - do NOT auto-promote to bank yet
    """
    def _precheck_calibration_xdf_for_brainprint(self, xdf_path: Path) -> tuple[bool, str]:
        """
        Return (True, 'ok') only if the calibration XDF appears to contain
        the minimum information needed for brainprint computation.

        We skip, rather than fail, when the run is effectively a dry run:
        marker-only file, missing EEG stream, missing CAL_STIM_ON markers, etc.
        """
        if pyxdf is None:
            return False, "Skipped brainprint: pyxdf is not available in this environment."

        if not xdf_path.exists():
            return False, f"Skipped brainprint: calibration XDF not found: {xdf_path}"

        if xdf_path.stat().st_size == 0:
            return False, f"Skipped brainprint: calibration XDF is empty: {xdf_path}"

        try:
            streams, _ = pyxdf.load_xdf(str(xdf_path))
        except Exception as exc:
            return False, f"Skipped brainprint: could not read calibration XDF ({exc})"

        if not streams:
            return False, "Skipped brainprint: calibration XDF contains no streams."

        eeg_streams = []
        marker_streams = []

        for s in streams:
            info = s.get("info", {})
            name = str(info.get("name", [""])[0]).strip()
            stream_type = str(info.get("type", [""])[0]).strip()

            if stream_type.lower() == "eeg" or "eeg" in name.lower():
                eeg_streams.append(s)

            if (
                stream_type.lower() == "markers"
                or "marker" in stream_type.lower()
                or "marker" in name.lower()
            ):
                marker_streams.append(s)

        if not eeg_streams:
            return False, "Skipped brainprint: no EEG stream found in calibration XDF."

        eeg_has_samples = False
        for s in eeg_streams:
            ts = s.get("time_series", [])
            if ts is not None and len(ts) > 0:
                eeg_has_samples = True
                break

        if not eeg_has_samples:
            return False, "Skipped brainprint: EEG stream found, but it contains no samples."

        if not marker_streams:
            return False, "Skipped brainprint: no marker stream found in calibration XDF."

        found_cal_stim_on = False
        for s in marker_streams:
            ts = s.get("time_series", [])
            for row in ts:
                if isinstance(row, (list, tuple)) and len(row) > 0:
                    value = str(row[0])
                else:
                    value = str(row)

                if "CAL_STIM_ON" in value:
                    found_cal_stim_on = True
                    break

            if found_cal_stim_on:
                break

        if not found_cal_stim_on:
            return False, "Skipped brainprint: no CAL_STIM_ON markers found in calibration XDF."

        return True, "ok"
        
    def __init__(self, config: AppConfig = CONFIG):
        self.config = config

    def _ensure_dirs(self) -> None:
        self.config.brainprint_root.mkdir(parents=True, exist_ok=True)
        self.config.brainprint_bank_dir.mkdir(parents=True, exist_ok=True)
        self.config.brainprint_candidate_dir.mkdir(parents=True, exist_ok=True)
        self.config.brainprint_comparison_dir.mkdir(parents=True, exist_ok=True)
        self.config.brainprint_rejected_dir.mkdir(parents=True, exist_ok=True)

    def _load_bridge_payload(self, bridge_path: Path) -> dict:
        with open(bridge_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _get_calibration_xdf_path(self, payload: dict) -> Optional[Path]:
        """
        Prefer explicit calibration block path.
        Fallback to expected_xdf_path if needed.
        """
        recording_block_paths = payload.get("recording_block_paths", {})
        calibration_path = recording_block_paths.get("calibration")
        if calibration_path:
            return Path(calibration_path)

        expected_xdf_path = payload.get("expected_xdf_path")
        if expected_xdf_path:
            return Path(expected_xdf_path)

        return None

    def should_process_after_recording(
        self,
        bridge_path: str | None,
        experiment_mode: str | None,
        selected_mode: str | None,
    ) -> tuple[bool, str]:
        if not self.config.auto_process_brainprint_after_recording:
            return False, "Automatic brainprint processing is disabled in config."

        if experiment_mode != "recording":
            return False, "Brainprint post-processing is currently only enabled for recording mode."

        if not selected_mode or "EEG" not in selected_mode:
            return False, "Selected acquisition mode does not include EEG."

        if not bridge_path:
            return False, "No prepared bridge path is available."

        return True, "Ready for post-run brainprint processing."

    def _run_subprocess(self, cmd: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(self.config.project_root),
        )

    def process_recording_brainprint(self, bridge_path: str) -> BrainprintProcessResult:
        self._ensure_dirs()

        bridge = Path(bridge_path)
        if not bridge.exists():
            return BrainprintProcessResult(
                status="failed",
                message=f"Bridge file not found: {bridge}",
            )

        try:
            payload = self._load_bridge_payload(bridge)
        except Exception as exc:
            return BrainprintProcessResult(
                status="failed",
                message=f"Could not read bridge JSON: {exc}",
            )

        calibration_xdf = self._get_calibration_xdf_path(payload)
        if calibration_xdf is None:
            return BrainprintProcessResult(
                status="failed",
                message="Could not determine calibration XDF path from bridge JSON.",
            )

        calibration_xdf = calibration_xdf.resolve()

        ok_to_process, precheck_message = self._precheck_calibration_xdf_for_brainprint(calibration_xdf)
        if not ok_to_process:
            return BrainprintProcessResult(
                status="skipped",
                message=precheck_message,
                calibration_xdf=str(calibration_xdf),
                candidate_npz=None,
                comparison_json=None,
                best_match_file=None,
                best_match_cosine=None,
            )

        if not calibration_xdf.exists():
            return BrainprintProcessResult(
                status="failed",
                message=f"Calibration XDF does not exist: {calibration_xdf}",
                calibration_xdf=str(calibration_xdf),
            )

        if calibration_xdf.stat().st_size == 0:
            return BrainprintProcessResult(
                status="skipped",
                message=f"Calibration XDF is empty: {calibration_xdf}",
                calibration_xdf=str(calibration_xdf),
            )

        candidate_npz = self.config.brainprint_candidate_dir / f"{calibration_xdf.stem}.npz"
        comparison_json = self.config.brainprint_comparison_dir / f"{calibration_xdf.stem}_compare.json"

        if candidate_npz.exists():
            candidate_npz.unlink()

        if comparison_json.exists():
            comparison_json.unlink()

        compute_cmd = [
            sys.executable,
            str(self.config.compute_brainprint_script),
            "--xdf",
            str(calibration_xdf),
            "--out",
            str(candidate_npz),
            "--signal-stream-type",
            self.config.brainprint_signal_stream_type,
            "--marker-stream-name",
            self.config.brainprint_marker_stream_name,
        ]

        compute_proc = self._run_subprocess(compute_cmd)

        if compute_proc.returncode != 0:
            return BrainprintProcessResult(
                status="failed",
                message=(
                    "Brainprint computation failed.\n\n"
                    f"STDERR:\n{compute_proc.stderr}\n\n"
                    f"STDOUT:\n{compute_proc.stdout}"
                ),
                calibration_xdf=str(calibration_xdf),
                candidate_npz=str(candidate_npz),
                stdout=compute_proc.stdout,
                stderr=compute_proc.stderr,
            )

        bank_files = sorted(self.config.brainprint_bank_dir.glob("*.npz"))

        if len(bank_files) == 0:
            return BrainprintProcessResult(
                status="ok",
                message=(
                    "Brainprint computed successfully, but comparison was skipped "
                    "because the bank is empty."
                ),
                calibration_xdf=str(calibration_xdf),
                candidate_npz=str(candidate_npz),
                stdout=compute_proc.stdout,
                stderr=compute_proc.stderr,
            )

        compare_cmd = [
            sys.executable,
            str(self.config.compare_brainprint_script),
            "--query",
            str(candidate_npz),
            "--bank-dir",
            str(self.config.brainprint_bank_dir),
            "--top-k",
            str(self.config.brainprint_compare_top_k),
            "--out",
            str(comparison_json),
        ]

        compare_proc = self._run_subprocess(compare_cmd)

        if compare_proc.returncode != 0:
            return BrainprintProcessResult(
                status="failed",
                message=(
                    "Brainprint comparison failed.\n\n"
                    f"STDERR:\n{compare_proc.stderr}\n\n"
                    f"STDOUT:\n{compare_proc.stdout}"
                ),
                calibration_xdf=str(calibration_xdf),
                candidate_npz=str(candidate_npz),
                comparison_json=str(comparison_json),
                stdout=(compute_proc.stdout or "") + "\n" + (compare_proc.stdout or ""),
                stderr=(compute_proc.stderr or "") + "\n" + (compare_proc.stderr or ""),
            )

        best_match_file = None
        best_match_cosine = None

        try:
            with open(comparison_json, "r", encoding="utf-8") as f:
                comparison_payload = json.load(f)

            ranked = comparison_payload.get("ranked_results", [])
            if ranked:
                best_match_file = ranked[0].get("file")
                best_match_cosine = ranked[0].get("cosine_similarity")
        except Exception:
            pass

        msg = "Brainprint computed and compared successfully."
        if best_match_file is not None and best_match_cosine is not None:
            msg += f" Best match: {Path(best_match_file).name} (cosine={best_match_cosine:.4f})."

        return BrainprintProcessResult(
            status="ok",
            message=msg,
            calibration_xdf=str(calibration_xdf),
            candidate_npz=str(candidate_npz),
            comparison_json=str(comparison_json),
            best_match_file=best_match_file,
            best_match_cosine=best_match_cosine,
            stdout=(compute_proc.stdout or "") + "\n" + (compare_proc.stdout or ""),
            stderr=(compute_proc.stderr or "") + "\n" + (compare_proc.stderr or ""),
        )