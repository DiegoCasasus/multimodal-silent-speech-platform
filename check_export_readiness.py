from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_RELATIVE_VERSION = "project_relative_v1"

REQUIRED_PATHS = [
    "code/tfm_GUI/GUI.py",
    "code/tfm_GUI/controller.py",
    "code/tfm_GUI/recording.py",
    "code/decoder/online_decoder_service.py",
    "code/decoder/model_runtime.py",
    "code/decoder/preprocessing_portable.py",
    "code/common/paths.py",
    "code/common/config_loader.py",
    "config/project_config.json",
    "psychopy/Experiment_Online_Silent.psyexp",
    "psychopy/Experiment_Online_Silent.py",
    "psychopy/Conditions.xlsx",
    "data/final_decoder_bundles/silent_v1/decoder_manifest.json",
]

OPTIONAL_BUT_RECOMMENDED_PATHS = [
    "launch_gui.bat",
    "launch_gui.sh",
    "launch_gui.command",
    "LabRecorder/LabRecorder.exe",
]

JSON_FILES_TO_CHECK = [
    "data/latest_prepared_run.json",
]

TEXT_EXTENSIONS_TO_SCAN = {
    ".py",
    ".json",
    ".psyexp",
    ".bat",
    ".sh",
    ".command",
    ".md",
    ".txt",
    ".cfg",
}

IGNORE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".ipynb_checkpoints",
    "recordings",
    "data/online_results",
    "brainprints/candidates",
    "brainprints/comparisons",
}

WINDOWS_ABSOLUTE_RE_MARKERS = [
    ":\\",  # C:\, D:\ etc.
]

UNIX_USER_PATH_MARKERS = [
    "/Users/",
    "/home/",
]


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def is_ignored(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root).as_posix()
    except Exception:
        return True
    for ignored in IGNORE_DIRS:
        if rel == ignored or rel.startswith(ignored.rstrip("/") + "/"):
            return True
    return False


def check_python_import(module: str) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            [sys.executable, "-c", f"import {module}; print(getattr({module}, '__version__', 'ok'))"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        ok = proc.returncode == 0
        detail = (proc.stdout or proc.stderr).strip()
        return ok, detail
    except Exception as exc:
        return False, str(exc)


def scan_for_machine_paths(root: Path) -> list[tuple[str, int, str]]:
    hits: list[tuple[str, int, str]] = []
    for path in root.rglob("*"):
        if not path.is_file() or is_ignored(path, root):
            continue
        if path.suffix.lower() not in TEXT_EXTENSIONS_TO_SCAN:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for idx, line in enumerate(text.splitlines(), start=1):
            line_lower = line.lower()
            if any(marker.lower() in line_lower for marker in WINDOWS_ABSOLUTE_RE_MARKERS):
                # Allow examples/comments that explicitly mention common Windows locations.
                if "example" in line_lower or "candidate" in line_lower or "program files" in line_lower:
                    continue
                hits.append((path.relative_to(root).as_posix(), idx, line.strip()[:220]))
            elif any(marker in line for marker in UNIX_USER_PATH_MARKERS):
                if "example" in line_lower or "candidate" in line_lower or "/applications/" in line_lower:
                    continue
                hits.append((path.relative_to(root).as_posix(), idx, line.strip()[:220]))
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether the TFM project looks export-ready.")
    parser.add_argument("--root", default=".", help="Project root. Default: current directory.")
    parser.add_argument("--scan-paths", action="store_true", help="Scan text files for obvious machine-specific absolute paths.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    print(f"Project root: {root}")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.executable}")
    print("")

    errors: list[str] = []
    warnings: list[str] = []

    print("Required files:")
    for rel in REQUIRED_PATHS:
        path = root / rel
        marker = "OK" if path.exists() else "MISSING"
        print(f"  [{marker}] {rel}")
        if not path.exists():
            errors.append(f"Missing required file: {rel}")

    print("\nOptional/recommended files:")
    for rel in OPTIONAL_BUT_RECOMMENDED_PATHS:
        path = root / rel
        marker = "OK" if path.exists() else "missing"
        print(f"  [{marker}] {rel}")
        if not path.exists():
            warnings.append(f"Optional/recommended file missing: {rel}")

    print("\nConfig checks:")
    config_path = root / "config" / "project_config.json"
    config = load_json(config_path)
    if config is None:
        errors.append("Could not parse config/project_config.json")
        print("  [MISSING/INVALID] config/project_config.json")
    else:
        print("  [OK] config/project_config.json parses as JSON")
        external_tools = config.get("external_tools", {}) if isinstance(config.get("external_tools"), dict) else {}
        for key in ["labrecorder_exe", "psychopy_python_exe", "decoder_python_exe"]:
            value = external_tools.get(key)
            print(f"  [info] external_tools.{key} = {value!r}")

    print("\nRuntime JSON path-format checks:")
    for rel in JSON_FILES_TO_CHECK:
        path = root / rel
        payload = load_json(path)
        if payload is None:
            warnings.append(f"Could not parse or missing runtime JSON: {rel}")
            print(f"  [missing/invalid] {rel}")
            continue
        version = payload.get("path_format_version")
        base = payload.get("paths_are_relative_to")
        ok = version == PROJECT_RELATIVE_VERSION and base == "project_root"
        print(f"  [{'OK' if ok else 'WARN'}] {rel}: path_format_version={version!r}, paths_are_relative_to={base!r}")
        if not ok:
            warnings.append(f"Runtime JSON does not declare project-relative paths: {rel}")

    print("\nPython package checks:")
    for module in ["numpy", "pylsl", "torch", "psychopy"]:
        ok, detail = check_python_import(module)
        print(f"  [{'OK' if ok else 'MISSING'}] import {module}: {detail}")
        if not ok and module in {"numpy", "pylsl", "torch"}:
            errors.append(f"Required Python package missing from current environment: {module}")
        elif not ok:
            warnings.append(f"PsychoPy is not importable from current Python: {module}")

    labrecorder_from_path = shutil.which("LabRecorder") or shutil.which("LabRecorder.exe")
    if labrecorder_from_path:
        print(f"\nLabRecorder on PATH: {labrecorder_from_path}")
    else:
        print("\nLabRecorder on PATH: not found")
        warnings.append("LabRecorder not found on PATH; project config or bundled LabRecorder must resolve it.")

    if args.scan_paths:
        print("\nScanning text files for obvious machine-specific absolute paths...")
        hits = scan_for_machine_paths(root)
        if hits:
            print(f"  Found {len(hits)} possible machine-specific path(s):")
            for rel, line_no, snippet in hits[:50]:
                print(f"  - {rel}:{line_no}: {snippet}")
            if len(hits) > 50:
                print(f"  ... plus {len(hits) - 50} more")
            warnings.append("Possible machine-specific absolute paths found in text files.")
        else:
            print("  [OK] no obvious machine-specific absolute paths found in scanned text files")

    print("\nSummary:")
    if errors:
        print("  ERRORS:")
        for item in errors:
            print(f"  - {item}")
    else:
        print("  No blocking errors found.")

    if warnings:
        print("  WARNINGS:")
        for item in warnings:
            print(f"  - {item}")
    else:
        print("  No warnings found.")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
