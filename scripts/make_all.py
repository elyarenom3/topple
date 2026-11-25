#!/usr/bin/env python3
"""Run every figure, still and video script in order (Section 11.5), one command each.

python scripts/make_all.py --profile quick        # smoke scale, minutes
python scripts/make_all.py --profile default      # full scale, long
python scripts/make_all.py --only F2,F3,H1        # subset
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

COMMANDS = {
    "F1": ["fig_pipeline.py"],
    "F2": ["fig_stones.py"],
    "F3": ["fig_solver_validation.py"],
    "F4": ["fig_engine_comparison.py"],
    "F5": ["fig_perception.py", "F5"],
    "F6": ["fig_perception.py", "F6"],
    "E1": ["run_experiment.py", "--condition", "E1"],
    "E2": ["run_experiment.py", "--condition", "E2"],
    "E2_point": ["run_experiment.py", "--condition", "E2_point"],
    "E2_ut": ["run_experiment.py", "--condition", "E2_ut"],
    "E2_fo": ["run_experiment.py", "--condition", "E2_fo"],
    "F7": ["fig_results.py", "F7"],
    "F8": ["fig_results.py", "F8"],
    "F9": ["fig_sobol.py"],
    "E4": ["run_kappa_sweep.py"],
    "F10": ["fig_results.py", "F10"],
    "E5": ["run_sensitivity_grid.py"],
    "F11": ["fig_results.py", "F11"],
    "F12": ["fig_results.py", "F12"],
    "R1": ["fig_mitsuba_comparison.py"],
    "showcase": ["build_showcase.py", "--stone-class", "blocky_quarry"],
    "H": ["render_stills.py", "all"],
    "V": ["render_videos.py", "all"],
    "viewer": ["export_viewer.py"],
    "arm": ["arm_demo.py"],
    "report": ["make_report.py"],
}


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="quick")
    ap.add_argument("--only", default=None)
    ap.add_argument("--towers", type=int, default=None)
    args = ap.parse_args()
    keys = list(COMMANDS) if not args.only else args.only.split(",")
    for k in keys:
        cmd = [sys.executable, str(HERE / COMMANDS[k][0])] + COMMANDS[k][1:]
        if COMMANDS[k][0] not in ("fig_pipeline.py", "fig_mitsuba_comparison.py", "export_viewer.py", "make_report.py", "render_stills.py", "render_videos.py"):
            cmd += ["--profile", args.profile]
        if args.towers is not None and COMMANDS[k][0].startswith("run_"):
            cmd += ["--towers", str(args.towers)]
        t0 = time.time()
        print(f"==> {k}: {' '.join(cmd[1:])}", flush=True)
        r = subprocess.run(cmd, cwd=str(HERE))
        print(f"<== {k}: exit {r.returncode} in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
