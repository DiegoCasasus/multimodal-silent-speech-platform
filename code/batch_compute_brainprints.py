#python batch_compute_brainprints.py ^
#  --xdf-dir "<PROJECT_ROOT>\recordings\calibration" ^
#  --out-dir "<PROJECT_ROOT>\brainprints" ^
#  --compute-script "<PROJECT_ROOT>\code\compute_brainprint.py" ^
#  --signal-stream-type EEG ^
#  --marker-stream-name PsychoPy_Markers


#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xdf-dir", required=True, help="Folder containing calibration .xdf files")
    parser.add_argument("--out-dir", required=True, help="Folder where .npz brainprints will be saved")
    parser.add_argument("--compute-script", default="compute_brainprint.py", help="Path to compute_brainprint.py")
    parser.add_argument("--signal-stream-name", default=None, help="Exact EEG/EXG stream name")
    parser.add_argument("--signal-stream-type", default="EEG", help="Preferred stream type, e.g. EEG or EXG")
    parser.add_argument("--marker-stream-name", default=None, help="Exact marker stream name")
    parser.add_argument("--tmin", type=float, default=-0.1)
    parser.add_argument("--tmax", type=float, default=0.6)
    parser.add_argument("--recursive", action="store_true", help="Search recursively for .xdf files")
    parser.add_argument("--skip-existing", action="store_true", help="Skip outputs that already exist")
    args = parser.parse_args()

    xdf_dir = Path(args.xdf_dir)
    out_dir = Path(args.out_dir)
    compute_script = Path(args.compute_script)

    if not xdf_dir.exists():
        raise FileNotFoundError(f"XDF directory not found: {xdf_dir}")
    if not compute_script.exists():
        raise FileNotFoundError(f"compute_brainprint.py not found: {compute_script}")

    out_dir.mkdir(parents=True, exist_ok=True)

    xdf_files = sorted(xdf_dir.rglob("*.xdf") if args.recursive else xdf_dir.glob("*.xdf"))
    if not xdf_files:
        raise RuntimeError(f"No .xdf files found in: {xdf_dir}")

    ok = 0
    failed = 0
    skipped = 0

    for xdf_path in xdf_files:
        out_path = out_dir / f"{xdf_path.stem}.npz"

        if args.skip_existing and out_path.exists():
            print(f"[SKIP] {xdf_path.name}")
            skipped += 1
            continue

        cmd = [
            sys.executable,
            str(compute_script),
            "--xdf", str(xdf_path),
            "--out", str(out_path),
            "--tmin", str(args.tmin),
            "--tmax", str(args.tmax),
        ]

        if args.signal_stream_name:
            cmd += ["--signal-stream-name", args.signal_stream_name]
        if args.signal_stream_type:
            cmd += ["--signal-stream-type", args.signal_stream_type]
        if args.marker_stream_name:
            cmd += ["--marker-stream-name", args.marker_stream_name]

        print(f"[RUN ] {xdf_path.name}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"[ OK ] {out_path.name}")
            ok += 1
        else:
            print(f"[FAIL] {xdf_path.name}")
            print(result.stdout)
            print(result.stderr)
            failed += 1

    print()
    print(f"Done. ok={ok} failed={failed} skipped={skipped}")


if __name__ == "__main__":
    main()