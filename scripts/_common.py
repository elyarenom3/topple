"""Shared helpers for the figure and experiment scripts: paths, config, record loading."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topple.core.config import load_config  # noqa: E402

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
STILLS = RESULTS / "stills"
VIDEOS = RESULTS / "videos"


def parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--profile", default="quick", help="config name under configs/ (default quick; use default for full scale)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output directory (default results/figures or results/stills)")
    ap.add_argument("--scale", type=float, default=1.0, help="multiplier on render resolution and samples")
    return ap


def config(args):
    cfg = load_config(args.profile)
    cfg["experiment"]["seed"] = args.seed
    return cfg


def find_experiments(condition_prefix: str) -> list[Path]:
    """Experiment directories whose name starts with the prefix, oldest first (by modification time)."""
    import os

    paths = [Path(p) for p in glob.glob(str(RESULTS / "experiments" / f"{condition_prefix}*")) if (Path(p) / "records.json").exists()]
    return sorted(paths, key=lambda p: os.path.getmtime(p / "records.json"))


def load_experiment(path: Path) -> tuple[list[dict], dict]:
    from topple.pipeline import load_records

    rows = load_records(path / "records.json")
    table = json.load(open(path / "table.json"))[0] if (path / "table.json").exists() else {}
    return rows, table


def out_dir(args, default: Path) -> Path:
    d = Path(args.out) if args.out else default
    d.mkdir(parents=True, exist_ok=True)
    return d
