#!/usr/bin/env python3
"""E6 (stone class), E7 (cone sides) and E8 (simulation scoring) sweeps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["E6", "E7", "E8"])
    ap.add_argument("--profile", default="quick")
    ap.add_argument("--towers", type=int, default=None)
    args = ap.parse_args()
    extra = [] if args.towers is None else ["--towers", str(args.towers)]
    if args.which == "E6":
        for cls in ["flat_river", "blocky_quarry", "rounded_cobble"]:
            run_experiment.main(["--condition", "E6", "--profile", args.profile, "--stone-class", cls] + extra)
    elif args.which == "E7":
        for n in [4, 8, 16, 32]:
            run_experiment.main(["--condition", "E7", "--profile", args.profile, "--cone-sides", str(n)] + extra)
    else:
        run_experiment.main(["--condition", "E8", "--profile", args.profile] + extra)
