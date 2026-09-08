from __future__ import annotations

import argparse
import fnmatch
import shutil
from pathlib import Path

INCLUDE_DIRS = [
    "code",
    "config",
    "psychopy",
]

INCLUDE_ROOT_FILES = [
    "launch_gui.bat",
    "launch_gui.sh",
    "launch_gui.command",

    "check_export_readiness.py",
    "create_safe_export_package.py",
    "verify_environment.py",

    "environment_windows.yml",
    "environment_macos.yml",
    "environment-lock.yml",

    "README.md",
    "INSTALL_FRESH_WINDOWS.md",
    "LICENSE",
    "VERSION",
    "EXPORT_VALIDATION_REPORT.txt",
]

# Keep only final decoder bundles from data by default. Avoid subject/run outputs.
INCLUDE_DATA_SUBDIRS = [
    "data/final_decoder_bundles",
]

EXCLUDE_PATTERNS = [
    "__pycache__",
    "*.pyc",
    ".ipynb_checkpoints",
    ".git",
    "*.zip",
    "psychopy/data",
    "psychopy/*.log",
    "psychopy/*.csv",
    "psychopy/*.psydat",
    "recordings",
    "data/online_results",
    "data/sub-*",
    "brainprints/bank/*",
    "brainprints/all_vs_all/*",
    "brainprints/candidates/*",
    "brainprints/comparisons/*",
    "brainprints/queries/*",
    "brainprints/holdout/*",
    "brainprints/rejected/*",
    "code/decoder/sweep_silent_small.json",
    "data/final_decoder_bundles/**/*_metrics.json",
]

BRAINPRINT_SUBDIRS = ["bank", "all_vs_all", "candidates", "comparisons", "rejected"]

README_BRAINPRINTS = """# Brainprints scaffold\n\nThis export package includes the brainprints folder structure but intentionally\nexcludes private biometric/template data.\n\nInstall an authorized local bank into `brainprints/bank/` only if permitted by\nethics/consent rules. The app can launch without a bank; comparison will simply\nproduce no reference matches until a local bank is installed.\n"""


def rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def should_exclude(path: Path, root: Path) -> bool:
    r = rel(path, root)
    # Training/evaluation metadata can contain historical machine-specific
    # dataset paths and is not required for runtime decoder deployment.
    if path.name.endswith("_metrics.json"):
        return True
    parts = set(path.parts)
    if "__pycache__" in parts or ".ipynb_checkpoints" in parts or ".git" in parts:
        return True
    for pattern in EXCLUDE_PATTERNS:
        if fnmatch.fnmatch(r, pattern) or any(fnmatch.fnmatch(parent.as_posix(), pattern) for parent in path.relative_to(root).parents):
            return True
    return False


def copy_tree_filtered(src: Path, dst: Path, root: Path) -> None:
    for path in src.rglob("*"):
        if should_exclude(path, root):
            continue
        target = dst / path.relative_to(src)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def create_brainprints_scaffold(target_root: Path) -> None:
    brainprints = target_root / "brainprints"
    brainprints.mkdir(parents=True, exist_ok=True)
    (brainprints / "README_BRAINPRINTS.md").write_text(README_BRAINPRINTS, encoding="utf-8")
    for subdir in BRAINPRINT_SUBDIRS:
        d = brainprints / subdir
        d.mkdir(parents=True, exist_ok=True)
        (d / ".gitkeep").touch()


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a clean TFM source export package without private data.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--target", required=True, help="Target directory for the clean export copy.")
    parser.add_argument("--include-labrecorder", action="store_true", help="Copy LabRecorder folder if present.")
    parser.add_argument("--overwrite", action="store_true", help="Delete target directory first if it exists.")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    target_root = Path(args.target).expanduser().resolve()

    if target_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Target already exists: {target_root}\nUse --overwrite to replace it.")
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    for dirname in INCLUDE_DIRS:
        src = project_root / dirname
        if src.exists():
            dst = target_root / dirname
            copy_tree_filtered(src, dst, project_root)

    for dirname in INCLUDE_DATA_SUBDIRS:
        src = project_root / dirname
        if src.exists():
            dst = target_root / dirname
            copy_tree_filtered(src, dst, project_root)

    if args.include_labrecorder:
        src = project_root / "LabRecorder"
        if src.exists():
            copy_tree_filtered(src, target_root / "LabRecorder", project_root)

    for filename in INCLUDE_ROOT_FILES:
        src = project_root / filename
        if src.exists() and src.is_file():
            shutil.copy2(src, target_root / filename)

    create_brainprints_scaffold(target_root)

    print(f"Clean export package created at: {target_root}")
    print("Excluded by default: recordings, data/sub-*, data/online_results, private brainprints content, zips, caches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
