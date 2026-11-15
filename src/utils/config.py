"""Configuration helpers: YAML loading, dotted overrides and cross-platform path resolution."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


def project_root() -> Path:
    """Absolute path of the project root (the folder that contains `src/` and `configs/`)."""
    return Path(__file__).resolve().parents[2]


def resolve_path(p: Any, root: Optional[Path] = None) -> Path:
    """Resolve a (possibly relative) path against the project root. Works on Windows and Linux."""
    path = Path(str(p)).expanduser()
    if path.is_absolute():
        return path
    return (root or project_root()) / path


def deep_update(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: Any = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load a YAML config. `default.yaml` is always loaded first, `path` (if given) is layered on top."""
    default_file = resolve_path("configs/default.yaml")
    with open(default_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if path:
        p = resolve_path(path)
        if p.resolve() != default_file.resolve():
            if not p.exists():
                raise FileNotFoundError(f"Config file not found: {p}")
            with open(p, "r", encoding="utf-8") as f:
                deep_update(cfg, yaml.safe_load(f) or {})
    if overrides:
        deep_update(cfg, copy.deepcopy(overrides))
    return cfg


def apply_set_overrides(cfg: Dict[str, Any], items: Optional[Iterable[str]]) -> Dict[str, Any]:
    """Apply CLI overrides such as ``detector.conf=0.3`` (value parsed as YAML)."""
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got: {item}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg
