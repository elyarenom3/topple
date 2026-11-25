#!/usr/bin/env python3
"""E5: depth noise scale in {0.5, 1, 2, 4} and friction prior width in {0.05, 0.1, 0.25, 0.5}."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="quick")
    ap.add_argument("--towers", type=int, default=None)
    ap.add_argument("--noise", default="0.5,1,2,4")
    ap.add_argument("--friction", default="0.05,0.1,0.25,0.5")
    args = ap.parse_args()
    for ns in [float(x) for x in args.noise.split(",")]:
        for fs in [float(x) for x in args.friction.split(",")]:
            argv = ["--condition", "E5", "--profile", args.profile, "--noise-scale", str(ns), "--friction-log-sigma", str(fs)]
            if args.towers is not None:
                argv += ["--towers", str(args.towers)]
            run_experiment.main(argv)
