#!/usr/bin/env python3
"""F2: stone gallery and descriptors (Section 5.3). Left: rendered silhouettes of three stones per
class (matplotlib 3D shaded triangles, the path traced version is H4). Right: the four descriptors
of a sampled population per class, with the class boxes."""

from __future__ import annotations

import numpy as np

from _common import FIGURES, RESULTS, config, out_dir, parser

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from topple.core.rng import Stage, make_rng  # noqa: E402
from topple.eval.figures import COLUMN_WIDTH_IN, palette, save  # noqa: E402
from topple.stonegen.generator import StoneGenerator  # noqa: E402


def shade(ax, mesh, colour):
    tri = mesh.triangles()
    n = mesh.face_normals()
    light = np.array([0.4, 0.3, 0.85])
    light /= np.linalg.norm(light)
    lam = np.clip(n @ light, 0, 1) * 0.75 + 0.25
    cols = np.array(colour)[None, :3] * lam[:, None]
    pc = Poly3DCollection(tri, facecolors=np.clip(cols, 0, 1), edgecolors="none")
    ax.add_collection3d(pc)
    r = mesh.bounding_radius()
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_zlim(-r, r)
    ax.set_axis_off()
    ax.view_init(25, 35)


def main(args):
    cfg = config(args)
    gen = StoneGenerator(cfg)
    classes = list(cfg["stones"]["classes"])
    c = palette(len(classes))
    fig = plt.figure(figsize=(2.1 * COLUMN_WIDTH_IN, 1.5 * COLUMN_WIDTH_IN))
    gs = fig.add_gridspec(len(classes), 5, width_ratios=[1, 1, 1, 1.6, 1.6])
    rows = []
    for ci, cls in enumerate(classes):
        rng = make_rng(args.seed, ci, Stage.STONEGEN)
        for k in range(3):
            spec = gen.sample(rng, cls)
            ax = fig.add_subplot(gs[ci, k], projection="3d")
            shade(ax, spec.mesh, c[ci])
            if k == 0:
                ax.set_title(cls.replace("_", " "), fontsize=8, loc="left")
            rows.append((cls, spec.name, spec.descriptors))
        for _ in range(args.n - 3):
            spec = gen.sample(rng, cls)
            rows.append((cls, spec.name, spec.descriptors))
    ax1 = fig.add_subplot(gs[:, 3])
    ax2 = fig.add_subplot(gs[:, 4])
    boxes = cfg["stones"]["class_boxes"]
    for ci, cls in enumerate(classes):
        d = [r[2] for r in rows if r[0] == cls]
        ax1.scatter([x["elongation"] for x in d], [x["flatness"] for x in d], s=8, color=c[ci], label=cls.replace("_", " "))
        ax2.scatter([x["convexity"] for x in d], [x["sphericity"] for x in d], s=8, color=c[ci])
        b = boxes[cls]
        ax1.add_patch(plt.Rectangle((b["elongation"][0], b["flatness"][0]), b["elongation"][1] - b["elongation"][0], b["flatness"][1] - b["flatness"][0], fill=False, ec=c[ci], lw=0.7))
        ax2.add_patch(plt.Rectangle((b["convexity"][0], b["sphericity"][0]), b["convexity"][1] - b["convexity"][0], b["sphericity"][1] - b["sphericity"][0], fill=False, ec=c[ci], lw=0.7))
    ax1.set_xlabel("elongation b/a")
    ax1.set_ylabel("flatness c/b")
    ax1.legend(frameon=False, fontsize=6)
    ax2.set_xlabel("convexity V / V_hull")
    ax2.set_ylabel("sphericity")
    fig.tight_layout()
    p = save(fig, out_dir(args, FIGURES) / "F2_stone_gallery")
    import csv

    with open(RESULTS / "stone_descriptors.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "name", "elongation", "flatness", "convexity", "sphericity"])
        for cls, name, d in rows:
            w.writerow([cls, name, d["elongation"], d["flatness"], d["convexity"], d["sphericity"]])
    print("wrote", p)


if __name__ == "__main__":
    ap = parser("F2 stone gallery")
    ap.add_argument("--n", type=int, default=30, help="stones per class for the descriptor scatter")
    main(ap.parse_args())
