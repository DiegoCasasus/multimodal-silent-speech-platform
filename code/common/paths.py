from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def expand_path_string(value: str | os.PathLike[str]) -> Path:
    """Return a Path after expanding ~ and environment variables."""
    text = os.path.expandvars(os.path.expanduser(str(value)))
    return Path(text)


def resolve_project_path(
    value: str | os.PathLike[str] | Path | None,
    project_root: str | os.PathLike[str] | Path,
    *,
    default: str | os.PathLike[str] | Path | None = None,
) -> Path | None:
    """Resolve a configured path using project_root as the base for relatives.

    Rules:
    - None or empty string -> default, if provided; otherwise None.
    - Absolute path -> returned as an absolute Path.
    - Relative path -> resolved relative to project_root.

    This function deliberately avoids resolving relative paths against the current
    working directory, because that makes the app fragile when launched from .bat
    files, PsychoPy, IDEs, or terminals with different cwd values.
    """
    if value is None or str(value).strip() == "":
        value = default

    if value is None or str(value).strip() == "":
        return None

    root = Path(project_root).expanduser().resolve()
    path = expand_path_string(value)

    if path.is_absolute():
        return path.resolve()

    return (root / path).resolve()


def to_project_relative(
    path: str | os.PathLike[str] | Path | None,
    project_root: str | os.PathLike[str] | Path,
) -> str | None:
    """Return a portable project-relative string when possible.

    If the path is outside the project root, the absolute path is kept. This is
    useful for legitimate user-configured external tools such as a LabRecorder
    executable installed outside the repository.
    """
    if path is None or str(path).strip() == "":
        return None

    root = Path(project_root).expanduser().resolve()
    candidate = expand_path_string(path)
    if not candidate.is_absolute():
        candidate = root / candidate

    try:
        return candidate.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(candidate.resolve())


def normalise_path_dict(value: Any, project_root: str | os.PathLike[str] | Path) -> Any:
    """Recursively convert Path objects to portable strings inside dict/list data.

    This is intentionally conservative. It does not guess which arbitrary strings
    are paths; it only converts actual Path objects.
    """
    if isinstance(value, Path):
        return to_project_relative(value, project_root)
    if isinstance(value, dict):
        return {k: normalise_path_dict(v, project_root) for k, v in value.items()}
    if isinstance(value, list):
        return [normalise_path_dict(v, project_root) for v in value]
    if isinstance(value, tuple):
        return [normalise_path_dict(v, project_root) for v in value]
    return value