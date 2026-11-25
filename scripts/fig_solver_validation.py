#!/usr/bin/env python3
"""F3: solver validation. Panels: (a) inclined plane slip onset against tan(alpha) = mu, (b) block
tipping angle against the analytic min(atan mu, atan b/h), (c) drift of a ten cube stack over 10 s,
(d) energy of a torque free asymmetric body."""

from __future__ import annotations

import numpy as np

from _common import FIGURES, config, out_dir, parser

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from topple.core.mesh import box  # noqa: E402
from topple.eval.figures import COLUMN_WIDTH_IN, palette, save  # noqa: E402
from topple.harness.tests import tilt_test  # noqa: E402
from topple.physics.body import RigidBody  # noqa: E402
from topple.physics.world import World  # noqa: E402
from topple.statics.margins import analytic_block_tilt  # noqa: E402


def block_world(cfg, mu, b, h):
    w = World(cfg["world"])
    g = RigidBody.ground(friction=mu)
    w.add_body(g)
    blk = RigidBody.from_mesh("block", box((b, b, h)), 2600.0, friction=mu, sdf_spacing=None)
    blk.x = np.array([0.0, 0.0, h / 2])
    w.add_body(blk)
    w.set_pair_friction(g.id, blk.id, mu)
    w.step(100)
    return w, blk


def main(args):
    cfg = config(args)
    cfg["harness"].update({"tilt_azimuths": 1, "coarse_rate_deg_s": 20.0, "tilt_rate_deg_s": 2.0, "tilt_max_deg": 70.0})
    c = palette(3)
    fig, axes = plt.subplots(2, 2, figsize=(2 * COLUMN_WIDTH_IN, 1.7 * COLUMN_WIDTH_IN))
    # (a) slip onset for a flat wide block (tipping far away) over mu
    mus = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.8])
    meas = []
    for mu in mus:
        w, blk = block_world(cfg, mu, 0.2, 0.02)
        meas.append(tilt_test(w, cfg, azimuths=[0.0]).value)
    ax = axes[0, 0]
    ax.plot(np.degrees(np.arctan(mus)), meas, "o", color=c[0], ms=4, label="measured")
    ax.plot([0, 45], [0, 45], "--", color="0.5", lw=0.8, label="tan alpha = mu")
    ax.set_xlabel("atan(mu) (deg)")
    ax.set_ylabel("slip onset (deg)")
    ax.set_title("(a) inclined plane slip")
    ax.legend(frameon=False)
    # (b) tipping over aspect ratio with high friction
    ratios = np.array([0.3, 0.5, 0.7, 1.0])
    meas2, ana = [], []
    for r in ratios:
        w, blk = block_world(cfg, 1.2, 0.1, 0.1 / r)
        meas2.append(tilt_test(w, cfg, azimuths=[0.0]).value)
        ana.append(analytic_block_tilt(1.2, 0.1, 0.1 / r))
    ax = axes[0, 1]
    ax.plot(ratios, ana, "-", color="0.5", lw=0.8, label="analytic")
    ax.plot(ratios, meas2, "o", color=c[1], ms=4, label="measured")
    ax.set_xlabel("b / h")
    ax.set_ylabel("critical tilt (deg)")
    ax.set_title("(b) block tipping")
    ax.legend(frameon=False)
    # (c) ten cube stack drift
    w = World(cfg["world"])
    g = RigidBody.ground(friction=0.6)
    w.add_body(g)
    cubes = []
    for k in range(10):
        cb = RigidBody.from_mesh(f"cube{k}", box((0.05, 0.05, 0.05)), 2600.0, friction=0.6, sdf_spacing=None)
        cb.x = np.array([0.0, 0.0, 0.025 + 0.05 * k])
        w.add_body(cb)
        cubes.append(cb)
    x0 = np.array([cb.x.copy() for cb in cubes])
    ts, drift = [], []
    n = int(10.0 / w.dt)
    every = int(0.1 / w.dt)
    for k in range(0, n, every):
        w.step(every)
        ts.append(w.time)
        drift.append(max(np.linalg.norm(cb.x - x0[i]) for i, cb in enumerate(cubes)))
    ax = axes[1, 0]
    ax.plot(ts, np.array(drift) * 1000, color=c[2], lw=1)
    ax.axhline(0.1, color="0.5", lw=0.8, ls="--")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("max drift (mm)")
    ax.set_title("(c) ten stacked cubes")
    # (d) torque free precession energy
    w = World(cfg["world"])
    w.gravity = np.zeros(3)
    body = RigidBody.from_mesh("brick", box((0.03, 0.06, 0.12)), 2600.0, sdf_spacing=None)
    body.x = np.array([0.0, 0.0, 1.0])
    body.w = np.array([0.1, 8.0, 0.1])
    w.add_body(body)
    E0 = body.kinetic_energy()
    ts2, es = [], []
    for k in range(200):
        w.step(10)
        ts2.append(w.time)
        es.append(body.kinetic_energy() / E0 - 1)
    ax = axes[1, 1]
    ax.plot(ts2, np.array(es) * 1e6, color=c[0], lw=1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("relative energy change (1e-6)")
    ax.set_title("(d) torque free asymmetric body")
    fig.tight_layout()
    p = save(fig, out_dir(args, FIGURES) / "F3_solver_validation")
    print("wrote", p)


if __name__ == "__main__":
    main(parser("F3 solver validation").parse_args())
