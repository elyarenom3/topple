#!/usr/bin/env python3
"""F4: cross engine validation (Section 12.2). The same towers run in the developer's engine,
MuJoCo, PyBullet and Drake (those importable); settled pose differences, critical tilt and
binary survival are compared as distributions over towers."""

from __future__ import annotations

import json

import numpy as np

from _common import FIGURES, RESULTS, config, out_dir, parser

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from topple.core.rng import Stage, make_rng  # noqa: E402
from topple.eval.figures import COLUMN_WIDTH_IN, palette, save  # noqa: E402
from topple.physics.body import RigidBody  # noqa: E402
from topple.physics.world import World  # noqa: E402
from topple.stonegen.generator import StoneGenerator, stone_to_body  # noqa: E402
from topple.validation import available_engines, compare_engines  # noqa: E402


def random_tower(cfg, seed, n_stones=3):
    """A hand stacked tower: each stone on its most probable resting facet, centred on the stone
    below with a small offset, settled in the developer's engine. The resting state is the common
    starting point of every engine."""
    from topple.core.transforms import quat_rotate
    from topple.planning.resting import resting_orientations

    rng = make_rng(seed, 0, Stage.STONEGEN)
    gen = StoneGenerator(cfg)
    w = World(cfg["world"])
    w.add_body(RigidBody.ground(friction=0.6))
    z_top = 0.0
    for k in range(n_stones):
        spec = gen.sample(rng, "flat_river" if k == 0 else "blocky_quarry")
        b = stone_to_body(spec, friction=0.6, sdf_spacing=None)
        rest = resting_orientations(b.shape.mesh, n_drops=100, rng=rng)
        b.q = rest[0][0] if rest else b.q
        V = quat_rotate(b.q, b.shape.mesh.vertices)
        b.x = np.array([rng.uniform(-0.005, 0.005), rng.uniform(-0.005, 0.005), z_top - V[:, 2].min() + 0.002])
        w.add_body(b)
        z_top = float(b.x[2] + V[:, 2].max())
    w.damped_settle(seconds=1.0)
    return w


def holds(w, seconds=1.0, fail_disp_m=0.005, fail_rot_deg=2.0) -> bool:
    """The tower stays within the failure thresholds at zero tilt in the developer's engine."""
    from topple.core.transforms import quat_conj, quat_mul, quat_to_rotvec

    c = w.clone()
    s0 = c.state()
    c.step(int(seconds / c.dt))
    s1 = c.state()
    for i, b in enumerate(c.bodies):
        if b.static:
            continue
        rot = np.degrees(np.linalg.norm(quat_to_rotvec(quat_mul(s1.q[i], quat_conj(s0.q[i])))))
        if np.linalg.norm(s1.x[i] - s0.x[i]) > fail_disp_m or rot > fail_rot_deg:
            return False
    return True


def main(args):
    cfg = config(args)
    cfg["harness"].update({"tilt_azimuths": 4, "coarse_rate_deg_s": 20.0, "tilt_rate_deg_s": 2.0, "tilt_max_deg": 60.0, "fixed_hold_s": 1.0})
    n = int(args.towers)
    path = RESULTS / "engine_comparison.json"
    old = json.load(open(path)) if (args.merge and path.exists()) else None
    want = ["topple"] + available_engines() if not args.engines else [e.strip() for e in args.engines.split(",")]
    external = [e for e in want if e != "topple"]
    if old is not None and "seeds" in old:
        seeds = [int(x) for x in old["seeds"]]
    else:
        # towers that hold at zero tilt in the developer's engine (a tower that creeps by itself
        # says nothing about the engines' tilt margins)
        seeds, s = [], args.seed
        while len(seeds) < n and s < args.seed + 10 * n:
            if holds(random_tower(cfg, s)):
                seeds.append(s)
            s += 1
    out = old if old is not None else {"seeds": seeds, "tilt": {}, "fixed": {}, "pose_diff": {}, "settled_x": {}}
    for e in want:
        out["tilt"][e] = {}
        out["fixed"][e] = {}
        if e != "topple":
            out["pose_diff"][e] = {}
    for s in seeds:
        w = random_tower(cfg, s)
        res = compare_engines(w, cfg, engines=external, tests=("settle", "tilt", "fixed"))
        if "topple" in want:
            out["settled_x"][str(s)] = res["topple"]["settled_x"].tolist()
        ref_x = np.array(out["settled_x"].get(str(s), res["topple"]["settled_x"]))
        for e in want:
            if e not in res or "error" in res[e]:
                print(f"tower {s}: {e} {res.get(e, {}).get('error', 'missing')}", flush=True)
                continue
            out["tilt"][e][str(s)] = float(res[e]["tilt"])
            out["fixed"][e][str(s)] = float(res[e]["fixed"])
            if e != "topple":
                d = np.linalg.norm(res[e]["settled_x"] - ref_x, axis=1)
                out["pose_diff"][e][str(s)] = float(d[1:].max())
        print(f"tower {s}: " + ", ".join(f"{e} tilt {out['tilt'][e][str(s)]:.1f}" for e in want if str(s) in out["tilt"][e]), flush=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
    plot(args)


def plot(args):
    with open(RESULTS / "engine_comparison.json") as f:
        d = json.load(f)
    if "seeds" in d:
        # per seed dictionaries: align every engine on the towers that the developer's engine ran
        seeds = [str(s) for s in d["seeds"] if str(s) in d["tilt"].get("topple", {})]
        order = ["topple"] + [e for e in ("mujoco", "drake", "pybullet") if e in d["tilt"]]
        tilt = {e: [d["tilt"][e].get(s, np.nan) for s in seeds] for e in order}
        fixed = {e: [d["fixed"][e][s] for s in seeds if s in d["fixed"][e]] for e in order}
        pose_diff = {e: [d["pose_diff"][e][s] for s in seeds if s in d["pose_diff"].get(e, {})] for e in order if e != "topple"}
    else:
        tilt, fixed, pose_diff = d["tilt"], d["fixed"], d["pose_diff"]
    engines = list(tilt.keys())
    c = palette(len(engines))
    fig, axes = plt.subplots(1, 3, figsize=(2.2 * COLUMN_WIDTH_IN, 0.8 * COLUMN_WIDTH_IN))
    ax = axes[0]
    x = np.array(tilt["topple"])
    lim = max(5.0, np.nanmax([np.nanmax(v) for v in tilt.values() if len(v)]) * 1.1)
    ax.plot([0, lim], [0, lim], "--", color="0.5", lw=0.8)
    for i, e in enumerate(engines[1:]):
        ax.scatter(x, tilt[e], s=14, color=c[i + 1], label=e)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("critical tilt, topple (deg)")
    ax.set_ylabel("critical tilt, other engine (deg)")
    ax.set_title("(a) tilt margin per tower")
    ax.legend(frameon=False)
    ax = axes[1]
    xs = np.arange(len(engines))
    ax.bar(xs, [np.mean(fixed[e]) if fixed[e] else np.nan for e in engines], color=c)
    ax.set_xticks(xs)
    ax.set_xticklabels(engines, rotation=20)
    ax.set_ylabel("survival rate at 3 deg")
    ax.set_title("(b) fixed perturbation")
    ax = axes[2]
    for i, e in enumerate(engines[1:]):
        if pose_diff.get(e):
            ax.hist(np.array(pose_diff[e]) * 1000, bins=10, histtype="step", color=c[i + 1], label=e, lw=1)
    ax.set_xlabel("settled pose difference (mm)")
    ax.set_ylabel("towers")
    ax.set_title("(c) settled poses after 1 s")
    ax.legend(frameon=False)
    fig.tight_layout()
    p = save(fig, out_dir(args, FIGURES) / "F4_engine_comparison")
    print("wrote", p)


if __name__ == "__main__":
    ap = parser("F4 engine comparison")
    ap.add_argument("--towers", type=int, default=10)
    ap.add_argument("--plot-only", action="store_true", help="redraw from results/engine_comparison.json")
    ap.add_argument("--engines", default=None, help="comma separated engines to run (default: topple and every importable engine)")
    ap.add_argument("--merge", action="store_true", help="add the engines of this run to results/engine_comparison.json (same towers)")
    a = ap.parse_args()
    plot(a) if a.plot_only else main(a)
