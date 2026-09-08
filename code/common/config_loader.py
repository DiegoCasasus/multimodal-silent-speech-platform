from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import resolve_project_path


DEFAULT_CONFIG_PATHS = (
    Path("config") / "project_config.json",
    Path("config") / "project_config.yaml",
    Path("config") / "project_config.yml",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            f"YAML config file found but PyYAML is not installed: {path}. "
            "Either install PyYAML, use project_config.json, or remove the YAML file."
        ) from exc

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Project config must contain a dictionary at the top level: {path}")
    return data


def load_project_config(
    project_root: str | Path,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load optional project configuration.

    Missing config files are allowed. That is important for safe incremental
    adoption: the app must keep its current defaults if config/project_config.json
    has not been created yet.
    """
    root = Path(project_root).expanduser().resolve()

    if config_path is not None:
        candidate = resolve_project_path(config_path, root)
        candidates = [candidate] if candidate is not None else []
    else:
        candidates = [root / p for p in DEFAULT_CONFIG_PATHS]

    existing = [p for p in candidates if p.exists()]
    if not existing:
        return {}

    path = existing[0]
    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif suffix in {".yaml", ".yml"}:
        data = _load_yaml(path)
    else:
        raise ValueError(f"Unsupported project config extension: {path.suffix}")

    if not isinstance(data, dict):
        raise ValueError(f"Project config must contain a dictionary at the top level: {path}")
    return data


def get_config_value(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Read nested config values using keys like 'paths.recordings_root'."""
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def get_config_path(
    config: dict[str, Any],
    dotted_key: str,
    project_root: str | Path,
    default: str | Path | None = None,
) -> Path | None:
    """Read and resolve a path config value."""
    value = get_config_value(config, dotted_key, None)
    return resolve_project_path(value, project_root, default=default)