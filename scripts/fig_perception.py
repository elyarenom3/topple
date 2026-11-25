#!/usr/bin/env python3
"""F5 and F6: depth noise and fusion error; centre of mass error against view count (Section 8).

F5: for one stone, the per pixel depth noise sigma of the configured model against distance and
incidence, and the fused mesh surface error (median distance to the true surface) against the
noise scale.
F6: volume and centre of mass error of the estimate against the number of views, with and
without the in hand views, averaged over a few stones.

python scripts/fig_perception.py F5 --profile quick
python scripts/fig_perception.py F6 --profile quick --stones 3
"""

from __future__ import annotations

import numpy as np

from _common import FIGURES, config, out_dir, parser

from topple.core.rng import Stage, make_rng
from topple.core.transforms import SE3
from topple.eval.figures import COLUMN_WIDTH_IN, lines, palette, save
from topple.perception.camera import PinholeCamera
from topple.perception.depth_sensor import DepthSensor
from topple.perception.estimator import Estimator
from topple.perception.noise import khoshelham_sigma, nguyen_sigma
from topple.perception.viewpoints import hemisphere_views, in_hand_views
from topple.stonegen.generator import StoneGenerator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def scan(cfg, spec, n_views, n_hand, noise_scale, seed):
    """Scan a stone resting at the origin on the ground; return (estimate, true mesh world)."""
    scfg = dict(cfg["sensor"])
    scfg["noise_scale"] = noise_scale
    cam = PinholeCamera.from_fov(int(scfg["width"]), int(scfg["height"]), float(scfg["fov_deg"]))
    rng = make_rng(seed, 0, Stage.SENSOR)
    sensor = DepthSensor(cam, scfg, rng)
    from topple.core.mesh import box

    ground = box((1.0, 1.0, 0.02), center=(0, 0, -0.01))
    mesh = spec.mesh.translated([0, 0, -spec.mesh.bounds()[0][2]])
    radius = max(0.35, 4 * mesh.bounding_radius())
    views = []
    for T in hemisphere_views(mesh.centroid(), radius, n_views, rng):
        views.append(sensor.render_depth([ground, mesh], T, ids=[0, 1]))
    for T in in_hand_views(mesh.centroid(), radius, n_hand, rng):
        views.append(sensor.render_depth([mesh], T, ids=[1]))
    est = Estimator(cfg, make_rng(seed, 0, Stage.ESTIMATOR)).estimate(views, int(cfg["tsdf"]["posterior_samples"]), target_id=1)
    return est, mesh


def f5(args):
    cfg = config(args)
    c = palette(3)
    fig, axes = plt.subplots(1, 3, figsize=(2.2 * COLUMN_WIDTH_IN, 0.75 * COLUMN_WIDTH_IN))
    z = np.linspace(0.4, 3.0, 50)
    axes[0].plot(z, 1000 * khoshelham_sigma(z), color=c[0], label="Khoshelham and Elberink")
    for th, ls in ((0.0, "-"), (np.radians(45), "--"), (np.radians(70), ":")):
        axes[0].plot(z, 1000 * nguyen_sigma(z, np.full_like(z, th))[0], color=c[1], ls=ls, label=f"Nguyen et al., {np.degrees(th):.0f} deg")
    axes[0].set_xlabel("distance (m)")
    axes[0].set_ylabel("axial sigma (mm)")
    axes[0].legend(frameon=False, fontsize=5.5)
    axes[0].set_title("(a) depth noise models")
    gen = StoneGenerator(cfg)
    spec = gen.sample(make_rng(args.seed, 0, Stage.STONEGEN), "rounded_cobble")
    scales = [0.0, 0.5, 1.0, 2.0, 4.0]
    errs, vols = [], []
    for s in scales:
        est, mesh = scan(cfg, spec, int(cfg["sensor"]["views_per_stone"]), int(cfg["sensor"].get("in_hand_views", 4)), s, args.seed)
        d = np.abs(mesh.signed_distance(est.mean_mesh.vertices))
        errs.append(1000 * float(np.median(d)))
        vols.append(100 * (est.mean_volume - mesh.volume()) / mesh.volume())
        print(f"noise scale {s}: median surface error {errs[-1]:.2f} mm, volume error {vols[-1]:+.1f}%", flush=True)
    axes[1].plot(scales, errs, "o-", color=c[2], ms=3)
    axes[1].set_xlabel("noise scale")
    axes[1].set_ylabel("median surface error (mm)")
    axes[1].set_title("(b) fused mesh error")
    axes[2].plot(scales, vols, "o-", color=c[0], ms=3)
    axes[2].axhline(0, color="0.5", lw=0.7, ls="--")
    axes[2].set_xlabel("noise scale")
    axes[2].set_ylabel("volume error (%)")
    axes[2].set_title("(c) volume error")
    fig.tight_layout()
    p = save(fig, out_dir(args, FIGURES) / "F5_depth_noise_fusion")
    print("wrote", p)


def f6(args):
    cfg = config(args)
    gen = StoneGenerator(cfg)
    counts = [2, 4, 8, 12]
    com_err = {"with in hand views": [], "hemisphere only": []}
    vol_err = {"with in hand views": [], "hemisphere only": []}
    for n in counts:
        for label, n_hand in (("with in hand views", int(cfg["sensor"].get("in_hand_views", 4))), ("hemisphere only", 0)):
            ce, ve = [], []
            for k in range(args.stones):
                spec = gen.sample(make_rng(args.seed, k, Stage.STONEGEN))
                est, mesh = scan(cfg, spec, n, n_hand, 1.0, args.seed + k)
                ce.append(1000 * np.linalg.norm(est.mean_com - mesh.centroid()))
                ve.append(100 * abs(est.mean_volume - mesh.volume()) / mesh.volume())
            com_err[label].append(float(np.mean(ce)))
            vol_err[label].append(float(np.mean(ve)))
            print(f"views {n} {label}: com error {com_err[label][-1]:.2f} mm, volume error {vol_err[label][-1]:.1f}%", flush=True)
    p1 = lines(counts, com_err, out_dir(args, FIGURES) / "F6_com_error_views", xlabel="hemisphere views", ylabel="centre of mass error (mm)", title="Centre of mass error against view count")
    p2 = lines(counts, vol_err, out_dir(args, FIGURES) / "F6_volume_error_views", xlabel="hemisphere views", ylabel="volume error (%)", title="Volume error against view count")
    print("wrote", p1, p2)


if __name__ == "__main__":
    ap = parser("F5 and F6 perception figures")
    ap.add_argument("figure", choices=["F5", "F6"])
    ap.add_argument("--stones", type=int, default=3)
    args = ap.parse_args()
    {"F5": f5, "F6": f6}[args.figure](args)
