"""Minimal YAML config loader with single-level ``base:`` inheritance."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config.

    If the file contains a top-level ``base:`` key naming another YAML file
    (resolved relative to this file's directory), that base is loaded first and
    the current file's keys are deep-merged on top.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    base_ref = cfg.pop("base", None)
    if base_ref is not None:
        base_cfg = load_config(path.parent / base_ref)
        cfg = _deep_merge(base_cfg, cfg)
    return cfg
