"""Hashed YAML configs (Section 13.2, 13.3)."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def merge(base: dict, override: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _resolve(name_or_path: str | os.PathLike) -> Path:
    p = Path(name_or_path)
    if p.exists():
        return p
    q = CONFIG_DIR / p
    if q.exists():
        return q
    q = CONFIG_DIR / f"{p}.yaml"
    if q.exists():
        return q
    raise FileNotFoundError(f"config not found: {name_or_path}")


def load_config(name_or_path: str | os.PathLike = "default", overrides: dict | None = None) -> dict:
    """Load a YAML config. A top level key ``extends`` names a base config to merge under it."""
    path = _resolve(name_or_path)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    base_name = cfg.pop("extends", None)
    if base_name:
        cfg = merge(load_config(base_name), cfg)
    cfg = merge(cfg, overrides)
    cfg["_config_path"] = str(path)
    cfg["_config_hash"] = config_hash(cfg)
    return cfg


def config_hash(cfg: dict) -> str:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    s = json.dumps(clean, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
