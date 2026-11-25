#!/usr/bin/env python3
"""Renderer comparison with Mitsuba 3 on a shared scene (Sections 10.9 and 12.5).

The same scene (a diffuse ground, two diffuse spheres and a diffuse box under one rectangular area
light, pinhole camera) is rendered by the developer's path tracer and by Mitsuba 3 (scalar RGB
variant, path tracer with the same maximum depth). Both images are linear RGB; they are exposed
identically, encoded to sRGB and compared with FLIP (target mean error below 0.05, author's choice).

python scripts/fig_mitsuba_comparison.py --width 256 --spp 1024
"""

from __future__ import annotations

import json
import time

import numpy as np

from _common import FIGURES, RESULTS, out_dir, parser

from topple.core.mesh import box, icosphere, plane
from topple.core.transforms import SE3
from topple.render import AreaLight, Camera, Material, Scene, render, write_png
from topple.render.color import srgb_encode
from topple.render.flip import flip_error

EYE = np.array([0.9, -1.1, 0.55])
TARGET = np.array([0.0, 0.0, 0.12])
FOCAL_MM = 40.0
LIGHT = dict(center=np.array([0.1, -0.2, 1.0]), size=(0.6, 0.4), radiance=6.0)
OBJECTS = [
    ("sphere", dict(center=(0.0, 0.0, 0.12), radius=0.12, albedo=(0.7, 0.7, 0.7))),
    ("sphere", dict(center=(0.3, 0.15, 0.07), radius=0.07, albedo=(0.75, 0.3, 0.2))),
    ("box", dict(center=(-0.28, 0.12, 0.08), size=(0.16, 0.16, 0.16), albedo=(0.2, 0.35, 0.7))),
]
GROUND = dict(size=4.0, albedo=(0.5, 0.5, 0.5))


def render_topple(w, h, spp, max_depth):
    sc = Scene()
    lam = lambda a: Material(name="lambert", base_color=a, roughness=1.0, specular=0.0, metallic=0.0, diffuse_model=0)
    sc.add(plane(GROUND["size"], 0.0), lam(GROUND["albedo"]), name="ground")
    for kind, o in OBJECTS:
        if kind == "sphere":
            sc.add(icosphere(5, o["radius"]), lam(o["albedo"]), SE3(t=o["center"]))
        else:
            sc.add(box(o["size"], center=o["center"]), lam(o["albedo"]))
    sc.add_light(AreaLight.rect(LIGHT["center"], [0, 0, -1.0], [0, 1.0, 0], LIGHT["size"][0], LIGHT["size"][1], LIGHT["radiance"]))
    sc.background = np.zeros(3)
    cam = Camera(position=EYE, look_at=TARGET, focal_mm=FOCAL_MM, f_number=1e6, width=w, height=h)
    t0 = time.time()
    out = render(sc, cam, spp=spp, max_depth_diffuse=max_depth, max_depth_specular=max_depth, seed=1, denoise=False)
    return out["beauty"], time.time() - t0


def render_mitsuba(w, h, spp, max_depth):
    import mitsuba as mi

    mi.set_variant("scalar_rgb")
    T = mi.ScalarTransform4f
    fov_x = float(np.degrees(2 * np.arctan(18.0 / FOCAL_MM)))
    diffuse = lambda a: {"type": "diffuse", "reflectance": {"type": "rgb", "value": list(a)}}
    scene = {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": max_depth + 1},
        "sensor": {
            "type": "perspective", "fov": fov_x, "fov_axis": "x",
            "to_world": T().look_at(origin=EYE.tolist(), target=TARGET.tolist(), up=[0, 0, 1]),
            "film": {"type": "hdrfilm", "width": w, "height": h, "rfilter": {"type": "box"}, "pixel_format": "rgb"},
            "sampler": {"type": "independent", "sample_count": spp},
        },
        "ground": {"type": "rectangle", "to_world": T().scale([GROUND["size"] / 2, GROUND["size"] / 2, 1.0]), "bsdf": diffuse(GROUND["albedo"])},
        "light": {
            "type": "rectangle",
            # rectangle normal is +z; flip to face down, scale to the light size
            "to_world": T().translate(LIGHT["center"].tolist()) @ T().scale([LIGHT["size"][0] / 2, LIGHT["size"][1] / 2, 1.0]) @ T().rotate([1, 0, 0], 180),
            "emitter": {"type": "area", "radiance": {"type": "rgb", "value": [LIGHT["radiance"]] * 3}},
        },
    }
    for k, (kind, o) in enumerate(OBJECTS):
        if kind == "sphere":
            scene[f"obj{k}"] = {"type": "sphere", "center": list(o["center"]), "radius": o["radius"], "bsdf": diffuse(o["albedo"])}
        else:
            scene[f"obj{k}"] = {"type": "cube", "to_world": T().translate(list(o["center"])) @ T().scale([s / 2 for s in o["size"]]), "bsdf": diffuse(o["albedo"])}
    t0 = time.time()
    img = np.array(mi.render(mi.load_dict(scene), spp=spp))
    return img[..., :3], time.time() - t0


def main(args):
    w, h = args.width, int(args.width * 2 / 3)
    ours, t_ours = render_topple(w, h, args.spp, args.max_depth)
    ref, t_ref = render_mitsuba(w, h, args.spp_ref, args.max_depth)
    # identical exposure: the mean of the reference maps to 0.18
    k = 0.18 / max(float(ref.mean()), 1e-9)
    a, b = srgb_encode(ref * k), srgb_encode(ours * k)
    err, emap = flip_error(a, b)
    ratio = float(ours.mean() / max(ref.mean(), 1e-12))
    d = out_dir(args, FIGURES)
    write_png(d / "R1_mitsuba_reference.png", a)
    write_png(d / "R1_topple.png", b)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from topple.eval.figures import COLUMN_WIDTH_IN, save, shared_cmap

    fig, axes = plt.subplots(1, 3, figsize=(2.3 * COLUMN_WIDTH_IN, 0.6 * COLUMN_WIDTH_IN))
    for ax, im, t in ((axes[0], a, "Mitsuba 3"), (axes[1], b, "topple path tracer"), (axes[2], emap, f"FLIP, mean {err:.3f}")):
        ax.imshow(im, cmap=shared_cmap() if im.ndim == 2 else None, vmin=0, vmax=1)
        ax.set_title(t)
        ax.axis("off")
    fig.tight_layout()
    save(fig, d / "R1_mitsuba_comparison")
    res = {"flip_mean": float(err), "target": 0.05, "mean_radiance_ratio_ours_over_ref": ratio, "width": w, "height": h, "spp": args.spp, "spp_ref": args.spp_ref, "seconds_ours": t_ours, "seconds_ref": t_ref}
    with open(RESULTS / "mitsuba_comparison.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = parser("renderer comparison with Mitsuba 3")
    ap.add_argument("--width", type=int, default=240)
    ap.add_argument("--spp", type=int, default=512)
    ap.add_argument("--spp-ref", type=int, default=512)
    ap.add_argument("--max-depth", type=int, default=8)
    main(ap.parse_args())
