"""Provenance records for results (Section 13.3): config hash, commit, library versions, machine."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "nogit"


def library_versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for name in ["numpy", "scipy", "numba", "matplotlib", "osqp", "trimesh", "mujoco", "pybullet", "pydrake", "skimage"]:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "present")
        except Exception:
            out[name] = None
    return out


def machine() -> dict:
    return {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(), "node": "redacted"}


def record(cfg: dict, extra: dict | None = None) -> dict:
    rec = {
        "config_hash": cfg.get("_config_hash"),
        "config_path": cfg.get("_config_path"),
        "commit": git_commit(),
        "libraries": library_versions(),
        "machine": machine(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    rec.update(extra or {})
    return rec


def content_address(obj) -> str:
    """Content hash of a JSON serialisable object, used to name result directories."""
    s = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def results_dir(root: Path, cfg: dict, tag: str) -> Path:
    d = Path(root) / f"{tag}_{cfg.get('_config_hash', 'nohash')}"
    d.mkdir(parents=True, exist_ok=True)
    return d
