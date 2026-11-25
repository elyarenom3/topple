#!/usr/bin/env python3
"""Hero stills H1 to H12 (Section 10.7) from the recorded showcase tower.

Each still is produced by one function from stored camera, material and lighting presets with a
fixed seed. Resolution and samples default to a laptop scale; ``--scale`` multiplies both and
``--width 4096 --spp 4096`` reaches the specification's hero settings.

python scripts/render_stills.py all --width 960 --spp 128
python scripts/render_stills.py H1 --width 4096 --spp 4096
"""

from __future__ import annotations

import json
import time

import numpy as np

from _common import RESULTS, STILLS, parser

from topple.core.mesh import TriMesh, box
from topple.core.transforms import SE3
from topple.render import key_light_exposure, overlays as ov, render, tonemap, write_exr, write_png
from topple.render.color import colormap
from topple.render.showcase import Showcase


def finish(name, sc, cam, args, spp=None, focus_point=None, exposure_point=None, denoise=True, exposure_override=None):
    out = STILLS
    out.mkdir(parents=True, exist_ok=True)
    spp = int((spp or args.spp) * args.scale)
    if exposure_point is None:
        exposure_point = SHOW.meter_point()
    exposure = exposure_override if exposure_override is not None else key_light_exposure(sc, exposure_point, [0, 0, 1])
    t0 = time.time()
    ckpt = out / f".{name}_{cam.width}x{cam.height}_{spp}_ckpt.npz"
    # spectral transport and manifold NEE of sun caustics where the scene holds a dielectric (water)
    wet = any(getattr(i.material, "dielectric", False) for i in sc.instances)
    res = render(sc, cam, spp=spp, seed=args.seed, denoise=denoise, checkpoint=str(ckpt), spectral=wet, mnee=wet)
    dt = time.time() - t0
    if ckpt.exists():
        ckpt.unlink()  # finished: the checkpoint only serves to resume an interrupted render
    img = res["composite_denoised"] if denoise and "composite_denoised" in res else res["composite"]
    write_png(out / f"{name}.png", tonemap(img, exposure))
    write_exr(out / f"{name}.exr", {k: res[k] for k in ("beauty", "albedo", "normal", "depth", "overlay") if k in res})
    st = res["stats"]
    meta = {"name": name, "width": cam.width, "height": cam.height, "spp": spp, "seconds": dt, "rays": int(st.rays), "rays_per_s": float(st.rays_per_second), "focal_mm": cam.focal_mm, "f_number": cam.f_number, "dof": [float(x) for x in cam.dof_limits()], "exposure": float(exposure), "spectral": bool(wet), "mnee": bool(wet), "lens": cam.lens or "thin"}
    with open(out / f"{name}.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"{name}: {cam.width}x{cam.height} at {spp} spp in {dt:.0f} s ({st.rays_per_second / 1e6:.2f} M rays/s) -> {out / (name + '.png')}", flush=True)


def size(args, aspect=1.5):
    w = int(args.width * args.scale ** 0.5)
    return w, int(round(w / aspect))


def tower_scene(ground="studio_slab", lighting="studio", tint=None, ids=None, sun=210.0):
    lo, hi = SHOW.tower_bounds()
    c = 0.5 * (lo + hi)
    sc = SHOW.base_scene(ground, lighting, center=c, size=float(np.max(hi - lo)), sun_azimuth_deg=sun)
    ids = SHOW.tower_ids if ids is None else ids
    SHOW.add_stones(sc, ids, {i: SHOW.pose(i) for i in ids}, tint=tint)
    return sc


def H1(args):
    """Tower at dusk (the specification's "seven stones at dusk"; the showcase tower has as many
    stones as the planner could stack): the tower as an object, wet ground, sparkle on granite."""
    sc = tower_scene("riverbank", "dusk", sun=200.0)
    w, h = size(args)
    lo, hi = SHOW.tower_bounds()
    third = SHOW.pose(SHOW.tower_ids[min(2, len(SHOW.tower_ids) - 1)]).t
    cam = SHOW.camera("H1", w, h)
    cam.focus_distance = float(np.linalg.norm(cam.position - third))
    finish("H1_tower_at_dusk", sc, cam, args)


def _within_focus_range(cam, target, margin: float = 1.05) -> None:
    """Move the camera back along its line of sight when ``target`` is closer than the traced
    double Gauss lens (at the camera's focal length) can focus."""
    from topple.render.lens import DOUBLE_GAUSS_50, LensSystem

    lens = LensSystem(DOUBLE_GAUSS_50, cam.focal_mm, cam.f_number)
    f = lens.effective_focal_length()
    s = np.array([lens.object_distance(x) for x in np.linspace(0.2 * f, 3.0 * f, 300)])
    s = s[np.isfinite(s) & (s > 0)]
    d_min = margin * float(s.min())
    d = float(np.linalg.norm(cam.position - target))
    if d < d_min:
        cam.position = target + (cam.position - target) * (d_min / d)


def H2(args):
    """Contact: one interface under grazing light, with the same frame plus O1 forces."""
    j = min(1, len(SHOW.tower_ids) - 1)
    ids = SHOW.tower_ids[: j + 1]
    lo, hi = SHOW.tower_bounds(ids)
    z_int = float(SHOW.world_mesh(ids[0], SHOW.pose(ids[0])).vertices[:, 2].max())
    c = np.array([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), z_int])
    w, h = size(args)
    for with_forces in (False, True):
        sc = SHOW.base_scene("granite_table", "macro", center=c, size=0.15)
        SHOW.add_stones(sc, ids, {i: SHOW.pose(i) for i in ids})
        if with_forces:
            SHOW.add_force_overlays(sc, n=j + 1, scale=0.02)
        cam = SHOW.camera("H2", w, h)
        cam.look_at = c
        cam.position = c + np.array([0.3, -0.22, 0.035])
        _within_focus_range(cam, c)
        cam.focus_distance = float(np.linalg.norm(cam.position - c))
        cam.lens = "dgauss50"  # traced double Gauss (Kolb et al. 1995): real vignetting and bokeh
        cam.__post_init__()
        # meter on the face of the lower stone lit by the grazing light (the light sits along +x)
        lower = SHOW.world_mesh(ids[0], SHOW.pose(ids[0]))
        meter = np.array([lower.vertices[:, 0].max() + 0.002, c[1], c[2] - 0.01])
        exposure = key_light_exposure(sc, meter, [1.0, 0.0, 0.2])
        finish("H2_contact" + ("_forces" if with_forces else ""), sc, cam, args, exposure_override=exposure)


def _arm_meshes_at_release():
    """Arm configuration holding the top stone at its final pose (IK on the planned grasp)."""
    from topple.physics.arm import ArmModel
    from topple.planning.grasp import antipodal_grasps, rank_grasps
    from topple.planning.kinematics import ik_dls

    top = SHOW.tower_ids[-1]
    T_top = SHOW.pose(top)
    model = ArmModel("panda7", base=SE3(t=np.array([-0.45, 0.0, 0.0])))
    mesh = SHOW.meshes[top]
    gs = rank_grasps(antipodal_grasps(mesh, 1.0, 300, np.random.default_rng(0)), mesh.centroid())
    best = None
    for g in gs[:20]:
        if abs(T_top.apply_vec(g.axis)[2]) > 0.5:
            continue
        T_tcp = T_top @ g.pose(T_top.R.T @ np.array([0.0, 0.0, -1.0]))
        q, e = ik_dls(model, T_tcp, model.q_home)
        if np.linalg.norm(e[:3]) < 3e-3:
            best = (q, g)
            break
    if best is None:
        return []
    q, g = best
    out = []
    for m, T in zip(model.link_meshes(), model.link_poses(q)):
        out.append(m.transformed(T))
    T = model.fk(q)
    for f in model.finger_meshes(min(model.finger_max, g.width + 0.012)):
        out.append(f.transformed(T))
    out.append(model.base_mesh().transformed(model.base))
    return out


def H3(args):
    """Arm at release: fingers open around the top stone, with motion blur on the fingers."""
    from topple.render import Material

    sc = tower_scene()
    alu = Material(name="arm", base_color=(0.85, 0.85, 0.86), roughness=0.35, metallic=0.0)
    dark = Material(name="fingers", base_color=(0.05, 0.05, 0.05), roughness=0.5)
    for k, m in enumerate(_arm_meshes_at_release()):
        sc.add(m, dark if k >= 8 else alu, name=f"arm{k}")
    w, h = size(args)
    # frame the tower and the hand: a looser fit than the tower only shots
    cam = SHOW.camera("H3", w, h, fill=0.38)
    top = SHOW.pose(SHOW.tower_ids[-1]).t
    cam.look_at = top - np.array([0.0, 0.0, 0.03])
    cam.focus_distance = float(np.linalg.norm(cam.position - top))
    cam.__post_init__()
    finish("H3_arm_at_release", sc, cam, args)


def H4(args):
    """Field of candidates: the scattered pile from above, each stone tinted by its best margin."""
    ids = [s["id"] for s in SHOW.stones]
    best = {}
    for r in SHOW.data["records"]:
        for s in SHOW.stones:
            if s["name"] == r["stone"]:
                best[s["id"]] = max(best.get(s["id"], -1.0), r["m_hat"])
    vals = np.array([best.get(i, 0.0) for i in ids])
    cols = colormap(vals, "sequential", 0.0, max(10.0, float(vals.max())))
    tint = {i: tuple(c) for i, c in zip(ids, cols)}
    P = np.array([SHOW.pose(i, "pile").t for i in ids])
    c = P.mean(axis=0)
    sc = SHOW.base_scene("studio_slab", "studio", center=c, size=0.8)
    SHOW.add_stones(sc, ids, {i: SHOW.pose(i, "pile") for i in ids}, tint=tint)
    w, h = size(args)
    span = float(np.max(P[:, :2].max(axis=0) - P[:, :2].min(axis=0))) + 0.25
    from topple.render import Camera

    cam = Camera(position=c + np.array([0, 0, 3.0]), look_at=c, up=np.array([0, 1.0, 0]), ortho=True, ortho_width=span * w / h, width=w, height=h)
    finish("H4_field_of_candidates", sc, cam, args, exposure_point=np.array([c[0], c[1] - 0.6, 0.002]))


def H5(args):
    """Posterior ghosts: one stone with 32 shape samples as a haze around the mean mesh."""
    from topple.render import Camera

    top = SHOW.tower_ids[-1]
    T = SE3(t=np.array([0.0, 0.0, 0.0]))
    m = SHOW.meshes[top]
    T = SE3(SHOW.pose(top).q, np.array([0.0, 0.0, -m.transformed(SE3(SHOW.pose(top).q)).vertices[:, 2].min()]))
    sc = SHOW.base_scene("granite_table", "studio", center=T.t, size=0.2)
    sc.background = np.array([0.005, 0.005, 0.006])
    post = SHOW.data["posterior"]
    mean = TriMesh(post["mean_V"], post["mean_F"])
    scan = SE3.from_matrix(post["scan_pose"])
    T_show = T @ SHOW.pose(top, "pile").inverse() @ SE3() if False else T
    # posterior meshes are in the scan time world frame; map them onto the displayed stone pose
    T_map = T @ SHOW.pose(top, "pile").inverse()
    sc.add(mean.transformed(T_map), SHOW.materials[top], name="mean")
    for V, F in post["samples"][:32]:
        sc.add_overlay(TriMesh(V, F).transformed(T_map), (0.55, 0.75, 1.0), alpha=0.018, blend="add", outline=0.0)
    C = T_map.apply(np.asarray(post["coms"]))
    if len(C) > 3:
        S = np.cov(C.T) + 1e-10 * np.eye(3)
        sc.add_overlay(ov.ellipsoid(C.mean(axis=0), S * 25.0), (1.0, 0.8, 0.3), alpha=0.4)
    w, h = size(args)
    c = T_map.apply(mean.centroid())
    cam = Camera(position=c + np.array([0.26, -0.2, 0.12]), look_at=c, focal_mm=85, f_number=5.6, width=w, height=h)
    cam.focus_distance = float(np.linalg.norm(cam.position - c))
    cam.__post_init__()
    finish("H5_posterior_ghosts", sc, cam, args, exposure_point=np.array([c[0] + 0.1, c[1] - 0.1, 0.002]))


def _clip_pose(rec, sub_id, t):
    return rec.pose_at(sub_id, t)


def H6(args):
    """Collapse at 1/30 s: falling stones streak, standing stones stay sharp (180 degree shutter)."""
    rec = SHOW.clips.get("collapse")
    if rec is None:
        print("no collapse clip")
        return
    # the frame of fastest motion in the collapse clip (largest stone speed)
    inv0 = SHOW.subset_map
    ts = np.array(rec.times)
    X = np.array([s.x[[inv0[i] for i in SHOW.tower_ids]] for s in rec.states])
    speed = np.linalg.norm(np.diff(X, axis=0), axis=2).max(axis=1) / np.maximum(np.diff(ts), 1e-9)
    t0 = float(ts[int(np.argmax(speed))])
    t1 = t0 + 1.0 / 30.0 * 0.5
    inv = SHOW.subset_map
    lo, hi = SHOW.tower_bounds()
    c = 0.5 * (lo + hi)
    sc = SHOW.base_scene("riverbank", "dusk", center=c, size=float(np.max(hi - lo)))
    for i in SHOW.tower_ids:
        j = inv[i]
        sc.add(SHOW.meshes[i], SHOW.materials[i], _clip_pose(rec, j, t0), _clip_pose(rec, j, t1), object_id=int(i))
    w, h = size(args)
    cam = SHOW.camera("H6", w, h)
    cam.shutter = (0.0, 1.0)  # the instance transforms already span the exposure
    cam.__post_init__()
    finish("H6_collapse", sc, cam, args)


def H7(args):
    """Section with thrust line: the tower cut by a vertical plane (O2, O3)."""
    lo, hi = SHOW.tower_bounds()
    c = 0.5 * (lo + hi)
    sc = SHOW.base_scene("studio_slab", "studio", center=c, size=float(np.max(hi - lo)))
    from topple.render import Material

    for i in SHOW.tower_ids:
        mw = SHOW.world_mesh(i, SHOW.pose(i))
        dens = SHOW.stones_by_id(i)["mass"] / max(SHOW.meshes[i].volume(), 1e-9)
        tint = 0.35 * colormap(np.array([dens]), "sequential", 2400, 2900)[0] + 0.65 * np.array([0.6, 0.6, 0.6])
        clipped, cap, col = ov.hatched_section_cap(mw, c, [0.0, -1.0, 0.0], 0.004, tint)
        sc.add(clipped, SHOW.materials[i], name=f"clip{i}")
        if cap is not None:
            sc.add(cap, Material(name="cap", base_color=tuple(tint), roughness=0.9, use_vertex_colors=True), vertex_colors=col, name=f"cap{i}")
    plane = (c, np.array([0.0, -1.0, 0.0]), 0.002)
    SHOW.add_thrust_line(sc, plane=plane)
    SHOW.add_force_overlays(sc, scale=0.03, cones=False, plane=plane)
    w, h = size(args)
    cam = SHOW.camera("H7", w, h)
    finish("H7_section_thrust_line", sc, cam, args)


def H8(args):
    """Stability landscape: the objective over the tower top as a heat map plane, chosen pose
    outlined and the next stone drawn as a ghost (O6)."""
    recs = [r for r in SHOW.data["records"] if not r["meta"].get("probe")]
    ids = SHOW.tower_ids[:-1] if len(SHOW.tower_ids) > 1 else SHOW.tower_ids
    nxt = SHOW.tower_ids[-1]
    lo, hi = SHOW.tower_bounds(ids)
    top_z = float(hi[2])
    c = np.array([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), top_z])
    # landscape: support polygon tipping margin of the next stone over a grid of positions
    from topple.statics.analysis import HullGeom, settle_hull_stone

    top_mesh = SHOW.world_mesh(ids[-1], SHOW.pose(ids[-1]))
    support = HullGeom(top_mesh.vertices)
    T_next = SHOW.pose(nxt)
    m_next = SHOW.meshes[nxt]
    n = 24
    half = 0.5 * float(np.max(hi[:2] - lo[:2]))
    xs = np.linspace(c[0] - half, c[0] + half, n)
    ys = np.linspace(c[1] - half, c[1] + half, n)
    vals = np.zeros((n, n))
    from topple.statics.tower import StaticBody, StaticContact, Tower
    from topple.statics import margins

    for a, x in enumerate(xs):
        for b, y in enumerate(ys):
            T = SE3(T_next.q, np.array([x, y, T_next.t[2] + 0.01]))
            h = HullGeom(m_next.transformed(T).vertices)
            h2, com, rot = settle_hull_stone(h, T.t.copy(), [None, support], 5e-4, max_tilt_deg=6.0)
            if rot >= 6.0:
                vals[a, b] = 0.0
                continue
            from topple.statics.analysis import _support_points

            P = _support_points(h2, [support], 5e-4)
            if len(P) < 3:
                vals[a, b] = 0.0
                continue
            d = np.linalg.norm(P[:, :2] - com[:2], axis=1)
            vals[a, b] = float(np.degrees(np.arctan2(np.min(d) * 0.5, max(com[2] - P[:, 2].mean(), 1e-3))))
    mesh, cols = ov.heatmap_plane(np.array([xs[0], ys[0], top_z + 0.03]), np.array([xs[-1] - xs[0], 0, 0]), np.array([0, ys[-1] - ys[0], 0]), vals)
    sc = SHOW.base_scene("studio_slab", "studio", center=c, size=0.4)
    SHOW.add_stones(sc, ids, {i: SHOW.pose(i) for i in ids})
    sc.add_overlay(mesh, (1, 1, 1), alpha=0.75, vertex_colors=cols)
    ghost = m_next.transformed(T_next)
    sc.add_overlay(ghost, (0.8, 0.9, 1.0), alpha=0.18, glass=True)
    outline = ghost.vertices[ghost.vertices[:, 2] < ghost.vertices[:, 2].min() + 0.004]
    if len(outline) >= 3:
        from scipy.spatial import ConvexHull

        hpts = outline[ConvexHull(outline[:, :2]).vertices]
        sc.add_overlay(ov.polygon_outline(hpts + np.array([0, 0, 0.031]), 0.0012), (1.0, 0.95, 0.4), alpha=1.0)
    w, h = size(args)
    cam = SHOW.camera("H8", w, h)
    finish("H8_stability_landscape", sc, cam, args)


def H9(args):
    """Long exposure shake: the shake test integrated over 2 s with centre of mass trails (O9)."""
    rec = SHOW.clips.get("shake")
    if rec is None:
        print("no shake clip")
        return
    end = rec.times[-1]
    t0 = max(rec.times[0], end - 2.0)
    inv = SHOW.subset_map
    lo, hi = SHOW.tower_bounds()
    c = 0.5 * (lo + hi)
    sc = SHOW.base_scene("studio_slab", "studio", center=c, size=float(np.max(hi - lo)))
    sc.background = np.array([0.004, 0.004, 0.005])
    for i in SHOW.tower_ids:
        j = inv[i]
        # sample the exposure with several transform keys through motion blur segments
        sc.add(SHOW.meshes[i], SHOW.materials[i], _clip_pose(rec, j, t0), _clip_pose(rec, j, end), object_id=int(i))
        trail = np.array([_clip_pose(rec, j, t).t for t in np.linspace(rec.times[0], end, 200)])
        sc.add_overlay(ov.polyline(trail, 0.0012), tuple(colormap(np.array([SHOW.tower_ids.index(i) / max(len(SHOW.tower_ids) - 1, 1)]), "sequential", 0, 1)[0]), alpha=0.9, blend="add")
    w, h = size(args)
    cam = SHOW.camera("H9", w, h)
    cam.shutter = (0.0, 1.0)
    cam.__post_init__()
    finish("H9_long_exposure_shake", sc, cam, args)


def H10(args):
    """Triptych: beauty, section with thrust line, and margin tint side by side."""
    import imageio.v2 as iio

    names = []
    w, h = size(args)
    w3 = max(160, w // 3)
    for kind in ("beauty", "section", "tint"):
        lo, hi = SHOW.tower_bounds()
        c = 0.5 * (lo + hi)
        if kind == "beauty":
            sc = tower_scene()
        elif kind == "section":
            sc = SHOW.base_scene("studio_slab", "studio", center=c, size=float(np.max(hi - lo)))
            for i in SHOW.tower_ids:
                clipped, cap, col = ov.hatched_section_cap(SHOW.world_mesh(i, SHOW.pose(i)), c, [0.0, -1.0, 0.0])
                sc.add(clipped, SHOW.materials[i])
                if cap is not None:
                    from topple.render import Material

                    sc.add(cap, Material(name="cap", base_color=(0.7, 0.7, 0.7), roughness=0.9, use_vertex_colors=True), vertex_colors=col)
            SHOW.add_thrust_line(sc, plane=(c, np.array([0.0, -1.0, 0.0]), 0.002))
        else:
            sc = tower_scene(tint=SHOW.margin_tint())
        cam = SHOW.camera("H10", w3, int(w3 * 1.4))
        finish(f"H10_triptych_{kind}", sc, cam, args)
        names.append(STILLS / f"H10_triptych_{kind}.png")
    imgs = [iio.imread(p) for p in names]
    iio.imwrite(STILLS / "H10_triptych.png", np.concatenate(imgs, axis=1))


def H11(args):
    """Hundred towers: a grid of final towers, each tinted by outcome with its predicted
    probability as a bar. Uses the study experiment records (one tile per tower)."""
    import glob
    import pickle

    from topple.pipeline import load_records

    paths = sorted(glob.glob(str(RESULTS / "experiments" / "E2all*" / "records.json")))
    if not paths:
        print("no E2 records for H11")
        return
    rows = [r for r in load_records(paths[-1]) if not r["meta"].get("probe")]
    towers = sorted({r["tower"] for r in rows})[:100]
    k = int(np.ceil(np.sqrt(len(towers))))
    sc = SHOW.base_scene("studio_slab", "studio", center=np.zeros(3), size=1.0)
    pitch = 0.3
    base = SHOW.meshes[SHOW.tower_ids[0]]
    for n, t in enumerate(towers):
        rt = [r for r in rows if r["tower"] == t]
        ok = not any(r["collapsed"] or r["y"] < 0.5 for r in rt)
        p = float(np.mean([r["p_hat"] for r in rt]))
        gx, gy = divmod(n, k)
        off = np.array([(gx - (k - 1) / 2) * pitch, (gy - (k - 1) / 2) * pitch, 0.0])
        col = (0.25, 0.6, 0.35) if ok else (0.75, 0.25, 0.2)
        zc = 0.0
        for j, r in enumerate(rt):
            # tiles show the same stone shapes stacked as many times as the tower had stones
            m = SHOW.meshes[SHOW.tower_ids[j % len(SHOW.tower_ids)]]
            T = SE3(SHOW.pose(SHOW.tower_ids[j % len(SHOW.tower_ids)]).q, off + np.array([0, 0, 0]))
            mw = m.transformed(T)
            mw = mw.translated([0, 0, zc - mw.vertices[:, 2].min()])
            zc = float(mw.vertices[:, 2].max())
            from topple.render import Material

            sc.add(mw, Material(name="tile", base_color=col, roughness=0.6))
        sc.add_overlay(ov.bar(off + np.array([0.11, -0.11, 0.0]), off + np.array([0.11, -0.11, 0.2 * p])), (0.95, 0.85, 0.3), alpha=1.0)
    from topple.render import Camera

    w, h = size(args)
    span = k * pitch + 0.2
    cam = Camera(position=np.array([0, 0, 6.0]), look_at=np.zeros(3), up=np.array([0, 1.0, 0]), ortho=True, ortho_width=span * w / h, width=w, height=h)
    finish("H11_hundred_towers", sc, cam, args, exposure_point=np.zeros(3))


def H12(args):
    """Tilt strip: five frames of the tilt test, camera fixed to the ground, with the predicted
    critical angle marked on every frame."""
    import imageio.v2 as iio

    rec = SHOW.clips.get("tilt")
    if rec is None:
        print("no tilt clip")
        return
    info = SHOW.data["tilt"]
    rate = info["rate_deg_s"]
    fail = info.get("fail_time") or rec.times[-1]
    f_deg = max((fail - rec.times[0]) * rate, 0.5)
    a_max = (rec.times[-1] - rec.times[0]) * rate
    if f_deg > 16.0:
        angles = [0.0, 5.0, 10.0, 15.0, f_deg]
    else:
        # a marginal tower: the onset, then the progression after it up to the end of the clip
        angles = sorted({0.0, round(f_deg, 1), round(0.3 * a_max, 1), round(0.6 * a_max, 1), round(a_max, 1)})[:5]
    m_hat = [r for r in SHOW.data["records"] if not r["meta"].get("probe")][-1]["m_hat"]
    inv = SHOW.subset_map
    frames = []
    w, h = size(args)
    wf = max(160, w // 5)
    for a in angles:
        t = rec.times[0] + a / rate
        if t > rec.times[-1]:
            t = rec.times[-1]
        lo, hi = SHOW.tower_bounds()
        c = 0.5 * (lo + hi)
        sc = SHOW.base_scene("studio_slab", "studio", center=c, size=float(np.max(hi - lo)))
        # the camera is fixed to the ground: gravity rotates, so the scene is shown in the ground frame
        for i in SHOW.tower_ids:
            sc.add(SHOW.meshes[i], SHOW.materials[i], _clip_pose(rec, inv[i], t), object_id=int(i))
        for mesh, col in ov.dial(c + np.array([0.12, -0.05, 0.15]), 0.04, a, (m_hat, m_hat), vmin=0.0, vmax=max(30.0, m_hat * 1.5), normal=(0, -1.0, 0)):
            sc.add_overlay(mesh, col, alpha=1.0)
        cam = SHOW.camera("H12", wf, int(wf * 1.5))
        name = f"H12_tilt_{a:.1f}deg".replace(".", "p")
        finish(name, sc, cam, args)
        frames.append(iio.imread(STILLS / f"{name}.png"))
    iio.imwrite(STILLS / "H12_tilt_strip.png", np.concatenate(frames, axis=1))


SHOTS = {"H1": H1, "H2": H2, "H3": H3, "H4": H4, "H5": H5, "H6": H6, "H7": H7, "H8": H8, "H9": H9, "H10": H10, "H11": H11, "H12": H12}
SHOW = None

if __name__ == "__main__":
    ap = parser("hero stills")
    ap.add_argument("shot", help="H1..H12 or all")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--spp", type=int, default=128)
    args = ap.parse_args()
    SHOW = Showcase(RESULTS / "showcase")
    names = list(SHOTS) if args.shot == "all" else args.shot.split(",")
    for n in names:
        SHOTS[n](args)
