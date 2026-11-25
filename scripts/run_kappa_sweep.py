#!/usr/bin/env python3
"""E4: risk weight kappa in {0, 0.5, 1, 2} with the full autonomous loop (arm placement)."""

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
    ap.add_argument("--kappas", default="0,0.5,1,2")
    ap.add_argument("--placement", default="arm")
    args = ap.parse_args()
    for k in [float(x) for x in args.kappas.split(",")]:
        argv = ["--condition", "E4", "--profile", args.profile, "--kappa", str(k), "--placement", args.placement]
        if args.towers is not None:
            argv += ["--towers", str(args.towers)]
        run_experiment.main(argv)
