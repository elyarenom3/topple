#!/usr/bin/env python3
"""E3 and F9: first order and total Sobol indices of the tilt margin by uncertainty source
(Section 11.3, hypothesis H4).

Towers are built from perception (no dynamic tests are needed). For every placed configuration,
the reduced posterior vector of all stones (Section 8.8) is split into five groups: shape modes
spread over the surface, shape modes concentrated on the underside at the placed orientation,
centre of mass, friction coefficient and pose. A Saltelli design with independent
standard normal blocks per group gives first order (Saltelli 2002 form) and total (Jansen)
indices, averaged over placements and reported per stone class.

python scripts/fig_sobol.py --profile study --towers 2 --n 64 --classes flat_river,blocky_quarry
"""

from __future__ import annotations

import json

import numpy as np

from _common import FIGURES, RESULTS, config, out_dir, parser

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from topple.core.rng import Stage, make_rng  # noqa: E402
from topple.eval.figures import COLUMN_WIDTH_IN, palette, save  # noqa: E402
from topple.eval.sobol import sobol_indices  # noqa: E402
from topple.pipeline import Condition, Experiment  # noqa: E402
from topple.statics.analysis import MarginModel  # noqa: E402

GROUPS = ["surface shape", "underside shape", "centre of mass", "friction", "pose"]


def group_index(params) -> dict:
    """Indices of each group in the concatenated reduced vector of all stones."""
    out = {g: [] for g in GROUPS}
    off = 0
    for p in params:
        k = p.n_modes
        # underside: modes concentrated on the vertices that face down at the placed orientation
        from topple.core.transforms import quat_to_rot
        from topple.perception.estimate_view import underside_modes

        under = underside_modes(p.mean_mesh, p.modes, quat_to_rot(p.pose.q)) if k else np.zeros(0, dtype=bool)
        modes = off + 6 + np.arange(k)
        out["surface shape"] += modes[~under].tolist()
        out["underside shape"] += modes[under].tolist()
        out["centre of mass"] += [off + 0, off + 1, off + 2]
        out["friction"] += [off + 3]
        out["pose"] += list(range(off + 6 + k, off + 12 + k)) + [off + 4, off + 5]  # pose and contact plane
        off += p.dim
    return {g: np.array(v, dtype=int) for g, v in out.items()}, off


def sobol_for(model, placed_below, top, n, rng):
    params = model._params(placed_below, top["estimate"], top["pose"])
    groups, dim = group_index(params)
    A = rng.standard_normal((n, dim))
    B = rng.standard_normal((n, dim))
    f = lambda X: np.array([model._margin_at(params, x, n_azimuth=12)[0] for x in X])
    fA, fB = f(A), f(B)
    fAB = []
    for g in GROUPS:
        AB = A.copy()
        idx = groups[g]
        if len(idx):
            AB[:, idx] = B[:, idx]
        fAB.append(f(AB))
    return sobol_indices(fA, fB, np.array(fAB), n_boot=100, rng=rng), float(np.var(np.concatenate([fA, fB])))


def main(args):
    cfg = config(args)
    cfg["experiment"]["probes_per_step"] = 0
    cfg["experiment"]["max_stones"] = args.max_stones
    cfg["experiment"]["stop_when_infeasible"] = False
    results = {}
    for cls in args.classes.split(","):
        cls = None if cls == "all" else cls
        S_all, ST_all, cases = [], [], []
        for t in range(args.towers):
            cond = Condition(name="E3", oracle=False, propagation="mc", margin="tilt", tests=(), stone_class=cls)
            exp = Experiment(cfg, cond, verbose=False)
            exp.run_tower(t)
            placed = exp.last_placed
            model = MarginModel(cfg, cond, make_rng(args.seed, t, Stage.EVAL))
            rng = np.random.default_rng(args.seed + t)
            for j in range(len(placed)):
                r, var = sobol_for(model, placed[:j], placed[j], args.n, rng)
                if var <= 1e-9:
                    continue
                S_all.append(r["S"])
                ST_all.append(r["ST"])
                cases.append({"tower": t, "level": j, "var": var, "S": r["S"].tolist(), "ST": r["ST"].tolist()})
                print(f"{cls or 'all'} tower {t} level {j}: var {var:.2f} S {np.round(r['S'], 2)} ST {np.round(r['ST'], 2)}", flush=True)
        if S_all:
            results[cls or "all"] = {"S": np.mean(S_all, axis=0).tolist(), "ST": np.mean(ST_all, axis=0).tolist(), "S_sd": np.std(S_all, axis=0).tolist(), "n": len(S_all), "cases": cases}
    with open(RESULTS / "sobol_indices.json", "w") as f:
        json.dump({"groups": GROUPS, "by_class": results, "n_design": args.n}, f, indent=1)
    if not results:
        print("no placements")
        return
    fig, axes = plt.subplots(1, len(results), figsize=(1.1 * COLUMN_WIDTH_IN * len(results), 0.8 * COLUMN_WIDTH_IN), squeeze=False)
    c = palette(2)
    x = np.arange(len(GROUPS))
    for ax, (cls, r) in zip(axes[0], results.items()):
        ax.bar(x - 0.2, np.clip(r["S"], 0, 1), 0.4, color=c[0], label="first order")
        ax.bar(x + 0.2, np.clip(r["ST"], 0, 1), 0.4, color=c[1], label="total")
        ax.set_xticks(x)
        ax.set_xticklabels(GROUPS, rotation=30, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(f"{cls.replace('_', ' ')} ({r['n']} placements)")
    axes[0][0].set_ylabel("Sobol index of the tilt margin")
    axes[0][0].legend(frameon=False)
    fig.tight_layout()
    print("wrote", save(fig, out_dir(args, FIGURES) / "F9_sobol"))


if __name__ == "__main__":
    ap = parser("F9 Sobol indices")
    ap.add_argument("--towers", type=int, default=2)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--max-stones", type=int, default=3)
    ap.add_argument("--classes", default="flat_river,blocky_quarry,rounded_cobble")
    main(ap.parse_args())
