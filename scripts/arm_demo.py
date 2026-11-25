#!/usr/bin/env python3
"""Arm placement demonstration (milestones M8 and M9, Section 9).

Builds towers with exact inputs where every stone is placed by the arm: grasp selection on the
stone, damped least squares IK, RRT-Connect around the inflated tower, Cartesian impedance descent
at 1 cm/s, contact detection from the wrist force, release and retract. Reports for every placement
whether the arm path completed or fell back to the oracle lowering, the reason, and the distance
between the released pose and the planned pose.

python scripts/arm_demo.py --profile quick --towers 3 --max-stones 3
"""

from __future__ import annotations

import json

import numpy as np

from _common import RESULTS, config, parser

from topple.core.transforms import SE3, quat_conj, quat_mul, quat_to_rotvec
from topple.pipeline import Condition, Experiment


def main(args):
    cfg = config(args)
    cfg["experiment"]["max_stones"] = args.max_stones
    cfg["experiment"]["probes_per_step"] = 0
    cfg["stones"]["pile_layout"] = "arc"  # within the reach of the arm
    cfg["arm"]["finger_max"] = args.finger_max
    rows = []
    for t in range(args.towers):
        exp = Experiment(cfg, Condition(name="arm", oracle=True, propagation="none", placement="arm", tests=("tilt",), stone_class=args.stone_class), verbose=True)
        recs = exp.run_tower(t)
        ends = {e["k"]: e for e in exp.events if e["kind"] == "place_end"}
        starts = {e["k"]: e for e in exp.events if e["kind"] == "place_start"}
        world = exp.last_world
        for r in recs:
            if r.meta.get("probe"):
                continue
            target = SE3.from_matrix(np.array(starts[r.placement]["target"]))
            released = SE3.from_matrix(np.array(ends[r.placement]["pose"]))
            rows.append({
                "tower": t, "placement": r.placement, "stone": r.stone,
                "arm_completed": r.meta.get("arm_fallback") in (None, False),
                "fallback_reason": r.meta.get("arm_fallback"),
                "position_error_mm": 1000 * float(np.linalg.norm(released.t - target.t)),
                "rotation_error_deg": float(np.degrees(np.linalg.norm(quat_to_rotvec(quat_mul(released.q, quat_conj(target.q)))))),
                "collapsed": r.collapsed, "m_hat": r.m_hat, "m_true": r.m_true,
            })
            print(rows[-1], flush=True)
    done = [r for r in rows if r["arm_completed"]]
    summary = {
        "placements": len(rows),
        "arm_completed": len(done),
        "fallback_reasons": sorted({str(r["fallback_reason"])[:60] for r in rows if not r["arm_completed"]}),
        "median_position_error_mm_arm": float(np.median([r["position_error_mm"] for r in done])) if done else None,
        "median_rotation_error_deg_arm": float(np.median([r["rotation_error_deg"] for r in done])) if done else None,
        "collapses_arm": int(sum(r["collapsed"] for r in done)),
    }
    out = RESULTS / "arm_demo.json"
    with open(out, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1, default=str)
    print(json.dumps(summary, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    ap = parser("arm placement demonstration")
    ap.add_argument("--towers", type=int, default=3)
    ap.add_argument("--max-stones", type=int, default=3)
    ap.add_argument("--stone-class", default="blocky_quarry")
    ap.add_argument("--finger-max", type=float, default=0.12, help="finger opening in m (a wider gripper than the Panda hand)")
    main(ap.parse_args())
