from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a local brainprint bank from approved calibration XDF files or "
            "from already-computed candidate NPZ files."
        )
    )
    parser.add_argument(
        "--source-mode",
        choices=["xdf", "candidate", "auto"],
        default="auto",
        help="Whether to build from calibration XDF files or existing candidate NPZ files.",
    )
    parser.add_argument("--xdf-root", default=None, help="Root folder containing calibration XDF files.")
    parser.add_argument("--candidate-dir", default=None, help="Folder containing candidate brainprint NPZ files.")
    parser.add_argument("--bank-dir", required=True, help="Destination folder for the local brainprint bank.")
    parser.add_argument(
        "--compute-script",
        default=None,
        help="Path to compute_brainprint.py. Required when source-mode uses XDF files.",
    )
    parser.add_argument(
        "--xdf-glob",
        default="**/*calibration*.xdf",
        help="Glob used to discover calibration XDF files under --xdf-root.",
    )
    parser.add_argument(
        "--candidate-glob",
        default="*.npz",
        help="Glob used to discover candidate NPZ files under --candidate-dir.",
    )
    parser.add_argument(
        "--rename-mode",
        choices=["keep", "hashed"],
        default="keep",
        help="Keep source-derived filenames or replace them with hashed aliases in the bank.",
    )
    parser.add_argument(
        "--hash-salt",
        default="",
        help="Optional site-local salt used when --rename-mode hashed is selected.",
    )
    parser.add_argument(
        "--approved-list",
        default=None,
        help=(
            "Optional newline-delimited text file or JSON list used to whitelist source stems. "
            "Only matching entries will be added to the bank."
        ),
    )
    parser.add_argument(
        "--signal-stream-name",
        default=None,
        help="Forwarded to compute_brainprint.py when building from XDF.",
    )
    parser.add_argument(
        "--signal-stream-type",
        default="EEG",
        help="Forwarded to compute_brainprint.py when building from XDF.",
    )
    parser.add_argument(
        "--marker-stream-name",
        default="PsychoPy_Markers",
        help="Forwarded to compute_brainprint.py when building from XDF.",
    )
    parser.add_argument("--tmin", type=float, default=-0.1)
    parser.add_argument("--tmax", type=float, default=0.6)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing bank entries.")
    parser.add_argument(
        "--manifest-out",
        default=None,
        help="Output JSON manifest path. Defaults to <bank-dir>/bank_manifest.json",
    )
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Output JSON summary path. Defaults to <bank-dir>/bank_build_summary.json",
    )
    return parser.parse_args()


def read_approved_list(path_str: str | None) -> set[str] | None:
    if not path_str:
        return None

    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Approved-list file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return set()

    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError("Approved-list JSON must contain a list of stems.")
        return {str(x).strip() for x in payload if str(x).strip()}

    return {line.strip() for line in text.splitlines() if line.strip()}


def hash_name(stem: str, salt: str) -> str:
    token = hashlib.sha256(f"{salt}::{stem}".encode("utf-8")).hexdigest()[:16]
    return f"bank_{token}"


def choose_output_stem(source_stem: str, rename_mode: str, hash_salt: str) -> str:
    if rename_mode == "keep":
        return source_stem
    return hash_name(source_stem, hash_salt)


def discover_xdf_sources(xdf_root: Path, xdf_glob: str, approved: set[str] | None) -> list[Path]:
    if not xdf_root.exists():
        raise FileNotFoundError(f"XDF root not found: {xdf_root}")

    paths = sorted(xdf_root.glob(xdf_glob))
    if approved is not None:
        paths = [p for p in paths if p.stem in approved]
    return paths


def discover_candidate_sources(candidate_dir: Path, candidate_glob: str, approved: set[str] | None) -> list[Path]:
    if not candidate_dir.exists():
        raise FileNotFoundError(f"Candidate directory not found: {candidate_dir}")

    paths = sorted(candidate_dir.glob(candidate_glob))
    if approved is not None:
        paths = [p for p in paths if p.stem in approved]
    return paths


def compute_from_xdf(
    compute_script: Path,
    xdf_path: Path,
    out_npz: Path,
    signal_stream_name: str | None,
    signal_stream_type: str | None,
    marker_stream_name: str | None,
    tmin: float,
    tmax: float,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(compute_script),
        "--xdf",
        str(xdf_path),
        "--out",
        str(out_npz),
        "--tmin",
        str(tmin),
        "--tmax",
        str(tmax),
    ]

    if signal_stream_name:
        cmd.extend(["--signal-stream-name", signal_stream_name])
    if signal_stream_type:
        cmd.extend(["--signal-stream-type", signal_stream_type])
    if marker_stream_name:
        cmd.extend(["--marker-stream-name", marker_stream_name])

    return subprocess.run(cmd, capture_output=True, text=True)


def copy_candidate(candidate_npz: Path, out_npz: Path) -> None:
    shutil.copy2(candidate_npz, out_npz)
    meta_src = candidate_npz.with_suffix(".json")
    meta_dst = out_npz.with_suffix(".json")
    if meta_src.exists():
        shutil.copy2(meta_src, meta_dst)


def build_manifest_row(
    *,
    source_type: str,
    source_path: Path,
    bank_npz: Path,
    rename_mode: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source_type": source_type,
        "source_path": str(source_path.resolve()),
        "source_name": source_path.name,
        "source_stem": source_path.stem,
        "bank_file": str(bank_npz.resolve()),
        "bank_name": bank_npz.name,
        "bank_stem": bank_npz.stem,
        "rename_mode": rename_mode,
        "local_only": True,
    }

    meta_json = bank_npz.with_suffix(".json")
    if meta_json.exists():
        try:
            row["brainprint_meta"] = json.loads(meta_json.read_text(encoding="utf-8"))
        except Exception:
            row["brainprint_meta"] = None
    else:
        row["brainprint_meta"] = None

    return row


def main() -> int:
    args = parse_args()

    bank_dir = Path(args.bank_dir).resolve()
    bank_dir.mkdir(parents=True, exist_ok=True)

    manifest_out = Path(args.manifest_out).resolve() if args.manifest_out else bank_dir / "bank_manifest.json"
    summary_out = Path(args.summary_out).resolve() if args.summary_out else bank_dir / "bank_build_summary.json"
    approved = read_approved_list(args.approved_list)

    source_mode = args.source_mode
    if source_mode == "auto":
        if args.xdf_root:
            source_mode = "xdf"
        elif args.candidate_dir:
            source_mode = "candidate"
        else:
            raise ValueError("In auto mode you must provide either --xdf-root or --candidate-dir.")

    manifest_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    if source_mode == "xdf":
        if not args.xdf_root:
            raise ValueError("--xdf-root is required in xdf mode.")
        if not args.compute_script:
            raise ValueError("--compute-script is required in xdf mode.")

        compute_script = Path(args.compute_script).resolve()
        if not compute_script.exists():
            raise FileNotFoundError(f"Compute script not found: {compute_script}")

        sources = discover_xdf_sources(Path(args.xdf_root).resolve(), args.xdf_glob, approved)
        if not sources:
            raise RuntimeError("No approved calibration XDF files were found.")

        for xdf_path in sources:
            out_stem = choose_output_stem(xdf_path.stem, args.rename_mode, args.hash_salt)
            out_npz = bank_dir / f"{out_stem}.npz"
            out_json = bank_dir / f"{out_stem}.json"

            if out_npz.exists() and not args.overwrite:
                manifest_rows.append(
                    build_manifest_row(
                        source_type="xdf",
                        source_path=xdf_path,
                        bank_npz=out_npz,
                        rename_mode=args.rename_mode,
                    )
                )
                continue

            if out_npz.exists():
                out_npz.unlink()
            if out_json.exists():
                out_json.unlink()

            proc = compute_from_xdf(
                compute_script=compute_script,
                xdf_path=xdf_path,
                out_npz=out_npz,
                signal_stream_name=args.signal_stream_name,
                signal_stream_type=args.signal_stream_type,
                marker_stream_name=args.marker_stream_name,
                tmin=args.tmin,
                tmax=args.tmax,
            )

            if proc.returncode != 0 or not out_npz.exists():
                failures.append(
                    {
                        "source_type": "xdf",
                        "source_path": str(xdf_path.resolve()),
                        "stdout": proc.stdout,
                        "stderr": proc.stderr,
                    }
                )
                continue

            manifest_rows.append(
                build_manifest_row(
                    source_type="xdf",
                    source_path=xdf_path,
                    bank_npz=out_npz,
                    rename_mode=args.rename_mode,
                )
            )

    elif source_mode == "candidate":
        if not args.candidate_dir:
            raise ValueError("--candidate-dir is required in candidate mode.")

        sources = discover_candidate_sources(Path(args.candidate_dir).resolve(), args.candidate_glob, approved)
        if not sources:
            raise RuntimeError("No approved candidate NPZ files were found.")

        for candidate_npz in sources:
            out_stem = choose_output_stem(candidate_npz.stem, args.rename_mode, args.hash_salt)
            out_npz = bank_dir / f"{out_stem}.npz"
            out_json = bank_dir / f"{out_stem}.json"

            if out_npz.exists() and not args.overwrite:
                manifest_rows.append(
                    build_manifest_row(
                        source_type="candidate",
                        source_path=candidate_npz,
                        bank_npz=out_npz,
                        rename_mode=args.rename_mode,
                    )
                )
                continue

            if out_npz.exists():
                out_npz.unlink()
            if out_json.exists():
                out_json.unlink()

            try:
                copy_candidate(candidate_npz, out_npz)
            except Exception as exc:
                failures.append(
                    {
                        "source_type": "candidate",
                        "source_path": str(candidate_npz.resolve()),
                        "error": str(exc),
                    }
                )
                continue

            manifest_rows.append(
                build_manifest_row(
                    source_type="candidate",
                    source_path=candidate_npz,
                    bank_npz=out_npz,
                    rename_mode=args.rename_mode,
                )
            )
    else:
        raise ValueError(f"Unsupported source mode: {source_mode}")

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest_rows, indent=2), encoding="utf-8")

    summary = {
        "bank_dir": str(bank_dir),
        "source_mode": source_mode,
        "rename_mode": args.rename_mode,
        "n_bank_entries": len(manifest_rows),
        "n_failures": len(failures),
        "approved_filter_used": approved is not None,
        "manifest_path": str(manifest_out),
        "failures": failures,
        "local_only": True,
        "note": (
            "This bank is intended to remain site-local. Distribute the code, not the participant-derived bank files."
        ),
    }

    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[done] bank entries: {len(manifest_rows)}")
    print(f"[done] failures: {len(failures)}")
    print(f"[done] manifest: {manifest_out}")
    print(f"[done] summary: {summary_out}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
