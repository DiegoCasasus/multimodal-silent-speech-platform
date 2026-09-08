from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_subprocess(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd is not None else None,
    )


def resolve_calibration_xdf(payload: dict) -> Path | None:
    recording_block_paths = payload.get("recording_block_paths", {})
    calibration = recording_block_paths.get("calibration")
    if calibration:
        return Path(calibration)

    expected_xdf_path = payload.get("expected_xdf_path")
    if expected_xdf_path:
        return Path(expected_xdf_path)

    return None


def load_candidate_meta(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    meta = {}

    for key in [
        "signal_name",
        "signal_type",
        "fs",
        "n_target_epochs",
        "n_nontarget_epochs",
        "tmin",
        "tmax",
    ]:
        if key in data:
            value = data[key]
            try:
                meta[key] = value.item()
            except Exception:
                meta[key] = value.tolist()

    if "template_vector" in data:
        meta["template_length"] = int(np.asarray(data["template_vector"]).size)

    return meta


def softmax_with_temperature(scores: list[float], temperature: float) -> list[float]:
    if len(scores) == 0:
        return []

    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    scaled = np.asarray(scores, dtype=float) / temperature
    scaled = scaled - np.max(scaled)
    exp_scores = np.exp(scaled)
    weights = exp_scores / np.sum(exp_scores)
    return [float(w) for w in weights]


def evaluate_bank_promotion(meta: dict, min_target_epochs: int, min_nontarget_epochs: int) -> tuple[bool, str]:
    n_target = int(meta.get("n_target_epochs", 0))
    n_nontarget = int(meta.get("n_nontarget_epochs", 0))

    if n_target < min_target_epochs:
        return False, f"Not promoted: only {n_target} target epochs (< {min_target_epochs})."

    if n_nontarget < min_nontarget_epochs:
        return False, f"Not promoted: only {n_nontarget} non-target epochs (< {min_nontarget_epochs})."

    return True, "Candidate passed minimum epoch-count quality gate."


def copy_with_sidecar(candidate_npz: Path, bank_dir: Path) -> tuple[Path, Path | None]:
    bank_dir.mkdir(parents=True, exist_ok=True)

    dst_npz = bank_dir / candidate_npz.name
    shutil.copy2(candidate_npz, dst_npz)

    src_json = candidate_npz.with_suffix(".json")
    dst_json = None
    if src_json.exists():
        dst_json = bank_dir / src_json.name
        shutil.copy2(src_json, dst_json)

    return dst_npz, dst_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", required=True, help="Path to GUI/PsychoPy bridge JSON")
    parser.add_argument("--out-json", required=True, help="Output JSON that PsychoPy will poll")

    parser.add_argument("--compute-script", required=True, help="Path to compute_brainprint.py")
    parser.add_argument("--compare-script", required=True, help="Path to compare_brainprint.py")

    parser.add_argument("--candidate-dir", required=True, help="Directory for candidate brainprints")
    parser.add_argument("--comparison-dir", required=True, help="Directory for comparison JSON outputs")
    parser.add_argument("--bank-dir", required=True, help="Directory containing accepted bank brainprints")

    parser.add_argument("--signal-stream-type", default="EEG", help="Preferred signal stream type")
    parser.add_argument("--marker-stream-name", default="PsychoPy_Markers", help="Marker stream name")

    parser.add_argument("--top-k", type=int, default=3, help="Top-k neighbors for weighting")
    parser.add_argument("--temperature", type=float, default=0.10, help="Softmax temperature for weights")
    parser.add_argument("--accept-threshold", type=float, default=0.45, help="Minimum best cosine for adaptation")

    parser.add_argument("--promote-to-bank", action="store_true", help="Actually copy accepted candidates to bank")
    parser.add_argument("--min-target-epochs", type=int, default=5, help="Minimum target epochs for bank promotion")
    parser.add_argument("--min-nontarget-epochs", type=int, default=20, help="Minimum non-target epochs for bank promotion")

    args = parser.parse_args()

    bridge_path = Path(args.bridge)
    out_json = Path(args.out_json)
    compute_script = Path(args.compute_script)
    compare_script = Path(args.compare_script)
    candidate_dir = Path(args.candidate_dir)
    comparison_dir = Path(args.comparison_dir)
    bank_dir = Path(args.bank_dir)

    candidate_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    bank_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "status": "failed",
        "mode": None,
        "bridge_path": str(bridge_path),
        "calibration_xdf": None,
        "brainprint_npz": None,
        "comparison_json": None,
        "accept_adaptation": False,
        "reason": "",
        "top_matches": [],
        "best_match_file": None,
        "best_match_cosine": None,
        "promoted_to_bank": False,
        "bank_path": None,
        "candidate_quality_ok": False,
        "candidate_quality_reason": "",
        "stdout": "",
        "stderr": "",
    }

    try:
        if not bridge_path.exists():
            result["reason"] = f"Bridge file not found: {bridge_path}"
            save_json(out_json, result)
            print(result["reason"])
            sys.exit(1)

        payload = load_json(bridge_path)
        result["mode"] = payload.get("experiment_mode")

        calibration_xdf = resolve_calibration_xdf(payload)
        if calibration_xdf is None:
            result["reason"] = "Could not resolve calibration XDF from bridge JSON."
            save_json(out_json, result)
            print(result["reason"])
            sys.exit(1)

        calibration_xdf = calibration_xdf.resolve()
        result["calibration_xdf"] = str(calibration_xdf)

        if not calibration_xdf.exists():
            result["reason"] = f"Calibration XDF does not exist: {calibration_xdf}"
            save_json(out_json, result)
            print(result["reason"])
            sys.exit(1)

        candidate_npz = candidate_dir / f"{calibration_xdf.stem}.npz"
        comparison_json = comparison_dir / f"{calibration_xdf.stem}_compare.json"

        if candidate_npz.exists():
            candidate_npz.unlink()

        if comparison_json.exists():
            comparison_json.unlink()

        compute_cmd = [
            sys.executable,
            str(compute_script),
            "--xdf",
            str(calibration_xdf),
            "--out",
            str(candidate_npz),
            "--signal-stream-type",
            args.signal_stream_type,
            "--marker-stream-name",
            args.marker_stream_name,
        ]

        compute_proc = run_subprocess(compute_cmd, cwd=compute_script.parent)
        result["stdout"] = compute_proc.stdout or ""
        result["stderr"] = compute_proc.stderr or ""

        if compute_proc.returncode != 0:
            result["status"] = "failed"
            result["reason"] = "Brainprint computation failed."
            result["brainprint_npz"] = str(candidate_npz)
            save_json(out_json, result)
            print(result["reason"])
            sys.exit(1)

        result["brainprint_npz"] = str(candidate_npz)

        candidate_meta = load_candidate_meta(candidate_npz)
        quality_ok, quality_reason = evaluate_bank_promotion(
            candidate_meta,
            min_target_epochs=args.min_target_epochs,
            min_nontarget_epochs=args.min_nontarget_epochs,
        )
        result["candidate_quality_ok"] = quality_ok
        result["candidate_quality_reason"] = quality_reason

        bank_files = sorted(bank_dir.glob("*.npz"))

        if len(bank_files) == 0:
            result["status"] = "ready"
            result["accept_adaptation"] = False
            result["reason"] = "Brainprint computed, but bank is empty so adaptation was skipped."

            if quality_ok and args.promote_to_bank:
                dst_npz, _ = copy_with_sidecar(candidate_npz, bank_dir)
                result["promoted_to_bank"] = True
                result["bank_path"] = str(dst_npz)

            save_json(out_json, result)
            print(result["reason"])
            sys.exit(0)

        compare_cmd = [
            sys.executable,
            str(compare_script),
            "--query",
            str(candidate_npz),
            "--bank-dir",
            str(bank_dir),
            "--top-k",
            str(max(args.top_k, 1)),
            "--out",
            str(comparison_json),
        ]

        compare_proc = run_subprocess(compare_cmd, cwd=compare_script.parent)
        result["stdout"] += ("\n" + (compare_proc.stdout or ""))
        result["stderr"] += ("\n" + (compare_proc.stderr or ""))

        if compare_proc.returncode != 0:
            result["status"] = "failed"
            result["reason"] = "Brainprint comparison failed."
            result["comparison_json"] = str(comparison_json)
            save_json(out_json, result)
            print(result["reason"])
            sys.exit(1)

        result["comparison_json"] = str(comparison_json)

        comparison_payload = load_json(comparison_json)
        ranked = comparison_payload.get("ranked_results", [])
        top_matches_raw = ranked[: max(args.top_k, 1)]

        top_cosines = [float(r["cosine_similarity"]) for r in top_matches_raw]
        top_weights = softmax_with_temperature(top_cosines, args.temperature)

        top_matches = []
        for rec, w in zip(top_matches_raw, top_weights):
            top_matches.append(
                {
                    "file": rec.get("file"),
                    "cosine": float(rec.get("cosine_similarity")),
                    "euclidean": float(rec.get("euclidean_distance")),
                    "weight": float(w),
                }
            )

        result["top_matches"] = top_matches

        if len(top_matches) > 0:
            result["best_match_file"] = top_matches[0]["file"]
            result["best_match_cosine"] = top_matches[0]["cosine"]

        if result["best_match_cosine"] is not None and result["best_match_cosine"] >= args.accept_threshold:
            result["accept_adaptation"] = True
            result["reason"] = (
                f"Adaptation accepted: best cosine {result['best_match_cosine']:.4f} "
                f">= threshold {args.accept_threshold:.4f}."
            )
        else:
            result["accept_adaptation"] = False
            if result["best_match_cosine"] is None:
                result["reason"] = "Adaptation rejected: no valid matches found."
            else:
                result["reason"] = (
                    f"Adaptation rejected: best cosine {result['best_match_cosine']:.4f} "
                    f"< threshold {args.accept_threshold:.4f}."
                )

        if quality_ok and args.promote_to_bank:
            dst_npz, _ = copy_with_sidecar(candidate_npz, bank_dir)
            result["promoted_to_bank"] = True
            result["bank_path"] = str(dst_npz)

        result["status"] = "ready"
        save_json(out_json, result)
        print(result["reason"])
        sys.exit(0)

    except Exception as exc:
        result["status"] = "failed"
        result["reason"] = f"Calibration preparation crashed: {exc}"
        save_json(out_json, result)
        print(result["reason"])
        sys.exit(1)


if __name__ == "__main__":
    main()