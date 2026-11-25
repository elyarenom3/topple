#!/usr/bin/env python3
"""F1: pipeline diagram (Section 4.1) as a vector figure."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _common import FIGURES, out_dir, parser

STAGES = [
    ("Stone generator", "watertight meshes,\ntrue density, true mu"),
    ("World", "rigid body state,\nphysics engine"),
    ("Depth sensor", "pinhole rays,\nnoise model"),
    ("Estimator", "TSDF fusion,\nposterior samples"),
    ("Stability analyser", "contact force LP,\nmargin samples, p hat"),
    ("Planner", "stone, pose, grasp,\nIK, trajectory"),
    ("Controller", "impedance descent,\nrelease"),
    ("Test harness", "tilt, shake, push,\nM star and y"),
    ("Evaluator", "(M hat, p hat, M star, y),\ncalibration"),
    ("Renderer", "path traced stills,\nvideo, overlays"),
]


def main(args):
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.set_xlim(0, 7.2)
    ax.set_ylim(0, 3.5)
    ax.axis("off")
    w, h = 1.25, 0.72
    pos = {}
    for i, (name, sub) in enumerate(STAGES):
        row, col = divmod(i, 5)
        x = 0.1 + col * 1.42
        y = 2.5 - row * 1.05
        pos[i] = (x, y)
        colour = "#dfe7f0" if i not in (0, 7, 8) else "#f3e3d3"
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=colour, ec="0.3", lw=0.8))
        ax.text(x + w / 2, y + h - 0.16, name, ha="center", va="center", fontsize=6.8, weight="bold")
        ax.text(x + w / 2, y + 0.24, sub, ha="center", va="center", fontsize=5.4)
    for i in range(len(STAGES) - 2):
        x0, y0 = pos[i]
        x1, y1 = pos[i + 1]
        if (i + 1) % 5 == 0:
            ax.add_patch(FancyArrowPatch((x0 + w / 2, y0), (x1 + w / 2, y1 + h), arrowstyle="-|>", mutation_scale=7, lw=0.7, color="0.3", connectionstyle="arc3,rad=0.0"))
        else:
            ax.add_patch(FancyArrowPatch((x0 + w, y0 + h / 2), (x1, y1 + h / 2), arrowstyle="-|>", mutation_scale=7, lw=0.7, color="0.3"))
    # hidden ground truth store
    gx, gy = 0.1, 0.15
    ax.add_patch(FancyBboxPatch((gx, gy), 2.7, 0.7, boxstyle="round,pad=0.02", fc="#f9d9d9", ec="0.3", lw=0.8))
    ax.text(gx + 1.35, gy + 0.54, "Hidden ground truth store", ha="center", fontsize=6.8, weight="bold")
    ax.text(gx + 1.35, gy + 0.22, "true meshes, densities, friction table\nread by the harness and the evaluator only", ha="center", va="center", fontsize=5.0)
    x0, y0 = pos[0]
    ax.add_patch(FancyArrowPatch((x0 + 0.2, y0), (gx + 0.2, gy + 0.7), arrowstyle="-|>", mutation_scale=7, lw=0.7, color="0.3", connectionstyle="arc3,rad=0.3"))
    for j in (7, 8):
        xj, yj = pos[j]
        ax.add_patch(FancyArrowPatch((gx + 2.7, gy + 0.3), (xj + w / 2, yj), arrowstyle="-|>", mutation_scale=7, lw=0.7, color="0.3", connectionstyle="arc3,rad=-0.2"))
    ax.text(5.2, 0.3, "the estimator and the planner never\nimport the ground truth (unit test)", ha="center", va="center", fontsize=5.4, style="italic")
    xr, yr = pos[9]
    x1, y1 = pos[1]
    ax.add_patch(FancyArrowPatch((x1 + w / 2, y1 + h), (xr + w / 2, yr + h), arrowstyle="-|>", mutation_scale=7, lw=0.7, color="0.3", connectionstyle="arc3,rad=-0.12"))
    ax.text(3.6, 3.38, "recorded states at 1 kHz", ha="center", fontsize=5.4)
    d = out_dir(args, FIGURES)
    fig.savefig(d / "F1_pipeline.pdf", bbox_inches="tight")
    fig.savefig(d / "F1_pipeline.png", dpi=250, bbox_inches="tight")
    print("wrote", d / "F1_pipeline.pdf")


if __name__ == "__main__":
    main(parser("F1 pipeline figure").parse_args())
