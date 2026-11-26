#!/usr/bin/env python3
"""Video shots V1 to V9 (Section 10.8) from the recorded showcase tower.

Every frame is path traced from the recorded simulation (positions interpolated linearly and
orientations by slerp at the sampled shutter times, 180 degree shutter at 24 frames per second),
denoised with the albedo and normal buffers, tone mapped, written as PNG (and EXR with --exr),
then encoded to H.265 when ffmpeg is available. Defaults are laptop scale (480 x 270, few samples,
shortened durations); --width 3840 --spp 512 --full-length reaches the specification.

python scripts/render_videos.py all
python scripts/render_videos.py V3 --width 960 --spp 32
"""

from __future__ import annotations

import json
import shutil
import time

import numpy as np

from _common import RESULTS, VIDEOS, parser

from topple.core.mesh import TriMesh
from topple.core.transforms import SE3, quat_slerp
from topple.render import Camera, FrameWriter, encode, key_light_exposure, overlays as ov, render
from topple.render.color import colormap
from topple.render.showcase import Showcase
from topple.render.video import camera_path_min_jerk, min_jerk, orbit_keyframes

FPS = 24.0
SHUTTER = 1.0 / 48.0  # 180 degree shutter at 24 frames per second


def _writer(name, exposure, args):
    d = VIDEOS / name
    if d.exists():
        shutil.rmtree(d)
    return FrameWriter(d, exposure=exposure, exr=args.exr)


def _encode(name, args, meta):
    status = encode(VIDEOS / name, VIDEOS / f"{name}.mp4", fps=FPS)
    meta["encode"] = status
    with open(VIDEOS / f"{name}.json", "w") as f:
        json.dump(meta, f, indent=1, default=float)
    print(f"{name}: {status}", flush=True)


def _size(args):
    w = int(args.width)
    return w, int(round(w * 9 / 16))


def _render_frame(sc, cam, args, writer):
    res = render(sc, cam, spp=args.spp, seed=args.seed + writer.count, denoise=True)
    writer.write(res, "composite_denoised" if "composite_denoised" in res else "composite")
    return res


def _cam_from(fr, w, h, focal=50.0, f_number=5.6):
    cam = Camera(position=fr["position"], look_at=fr["look_at"], focal_mm=fr.get("focal_mm", focal), f_number=fr.get("f_number", f_number), width=w, height=h, shutter=(0.0, SHUTTER))
    return cam


def _pose_lerp(T0: SE3, T1: SE3, u: float, lift: float = 0.0) -> SE3:
    u = float(np.clip(u, 0.0, 1.0))
    s = float(min_jerk(np.array([u]))[0])
    t = (1 - s) * T0.t + s * T1.t + np.array([0.0, 0.0, lift * np.sin(np.pi * s)])
    return SE3(quat_slerp(T0.q, T1.q, s), t)


def V1(args):
    """Build: the full construction under a circular dolly of radius 1.2 m with a 50 mm lens.
    Each stone travels from the pile on a lifted arc (the carry is drawn kinematically, its
    timing compressed), is lowered and released; settling comes from the recorded simulation.
    A second pass repeats the last turn with the margin tint (O7)."""
    show = SHOW
    rec = show.build
    events = [e for e in show.data["events"] if e["kind"] in ("place_start", "place_end")]
    starts = [e for e in events if e["kind"] == "place_start"]
    ends = {e["k"]: e for e in events if e["kind"] == "place_end"}
    per = args.seconds_per_stone
    n_frames = int(per * FPS * len(starts)) + int(1.5 * FPS)
    lo, hi = show.tower_bounds()
    c = 0.5 * (lo + hi)
    keys = orbit_keyframes(c, 1.2, c[2] + 0.25, n=8, start_deg=-60, sweep_deg=120)
    path = camera_path_min_jerk(keys, list(np.linspace(0, n_frames / FPS, len(keys))), n_frames, FPS)
    w, h = _size(args)
    all_ids = [s["id"] for s in show.stones]
    sc0 = show.base_scene("studio_slab", "studio", center=c, size=0.6)
    show.add_stones(sc0, all_ids, {i: show.pose(i, "pile") for i in all_ids})
    exposure = key_light_exposure(sc0, SHOW.meter_point(), [0, 0, 1])
    writer = _writer("V1_build", exposure, args)
    for f in range(n_frames):
        t_video = f / FPS
        k = min(int(t_video // per), len(starts) - 1)
        u = (t_video - k * per) / per
        poses, poses_end = {}, {}
        for kk, e in enumerate(starts):
            bid = e["body_id"]
            T_from = SE3.from_matrix(np.array(e["from"]))
            T_rel = SE3.from_matrix(np.array(ends[e["k"]]["pose"])) if e["k"] in ends else SE3.from_matrix(np.array(e["target"]))
            if kk < k or (kk == k and u >= 0.7):
                # released: settled pose from the record (end of the settle after this placement)
                t_rec = ends[e["k"]]["time"] if e["k"] in ends else rec.times[-1]
                if kk == k:
                    t_rec = e["time"] + (ends[e["k"]]["time"] - e["time"]) * min(1.0, (u - 0.7) / 0.3) if e["k"] in ends else t_rec
                poses[bid] = rec.pose_at(bid, t_rec)
                poses_end[bid] = rec.pose_at(bid, t_rec + SHUTTER)
            elif kk == k:
                # carry: lifted arc to 3 cm above the release pose, then the descent
                above = SE3(T_rel.q, T_rel.t + np.array([0, 0, 0.03]))
                if u < 0.55:
                    poses[bid] = _pose_lerp(T_from, above, u / 0.55, lift=0.25)
                    poses_end[bid] = _pose_lerp(T_from, above, min(1, (u + 1 / (per * FPS) / 2) / 0.55), lift=0.25)
                else:
                    s = (u - 0.55) / 0.15
                    poses[bid] = _pose_lerp(above, T_rel, s)
                    poses_end[bid] = poses[bid]
            else:
                poses[bid] = show.pose(bid, "pile")
        for i in all_ids:
            poses.setdefault(i, show.pose(i, "pile"))
        tint = show.margin_tint() if f > n_frames - int(1.5 * FPS) else None
        sc = show.base_scene("studio_slab", "studio", center=c, size=0.6)
        show.add_stones(sc, all_ids, poses, poses_end, tint=tint)
        cam = _cam_from(path[f], w, h, 50.0, 5.6)
        cam.focus_distance = float(np.linalg.norm(cam.position - c))
        _render_frame(sc, cam, args, writer)
    _encode("V1_build", args, {"frames": n_frames, "width": w, "height": h, "spp": args.spp, "seconds_per_stone": per})


def V2(args):
    """Scan: the wrist camera point of view moving through the scan views, with the live depth
    image as an inset."""
    show = SHOW
    post = show.data["posterior"]
    poses = [SE3.from_matrix(np.array(m)) for m in post["view_poses"]]
    depth = post["depth"]
    all_ids = [s["id"] for s in show.stones]
    n_views = len(poses)
    frames_per_view = max(4, int(args.scan_seconds * FPS / n_views))
    n_frames = frames_per_view * (n_views - 1)
    w, h = _size(args)
    c = np.array(post["scan_pose"])[:3, 3]
    sc = show.base_scene("studio_slab", "studio", center=c, size=0.6)
    show.add_stones(sc, all_ids, {i: show.pose(i, "pile") for i in all_ids})
    exposure = key_light_exposure(sc, SHOW.meter_point(), [0, 0, 1])
    writer = _writer("V2_scan", exposure, args)
    import imageio.v2 as iio

    for f in range(n_frames):
        v = f // frames_per_view
        u = (f % frames_per_view) / frames_per_view
        T = poses[v].interpolate(poses[v + 1], float(min_jerk(np.array([u]))[0]))
        fwd = T.apply_vec(np.array([0.0, 0.0, 1.0]))
        up = -T.apply_vec(np.array([0.0, 1.0, 0.0]))
        cam = Camera(position=T.t, look_at=T.t + 0.3 * fwd, up=up, focal_mm=24.0, f_number=8.0, width=w, height=h)
        res = render(sc, cam, spp=args.spp, seed=args.seed + f, denoise=True)
        img = res["composite_denoised"]
        # depth inset of the current view (bottom right quarter)
        d, valid = depth[v]
        dn = np.where(valid, (d - d[valid].min()) / max(np.ptp(d[valid]), 1e-6), 0.0) if valid.any() else np.zeros_like(d)
        dc = colormap(dn.ravel(), "sequential", 0, 1).reshape(d.shape + (3,))
        ih, iw = h // 3, w // 3
        ys = (np.arange(ih) * d.shape[0] / ih).astype(int)
        xs = (np.arange(iw) * d.shape[1] / iw).astype(int)
        inset = dc[ys][:, xs] / max(exposure, 1e-9)
        img = img.copy()
        img[h - ih - 8 : h - 8, w - iw - 8 : w - 8] = inset
        writer.write({"composite": img, "beauty": img})
    _encode("V2_scan", args, {"frames": n_frames, "views": n_views})


def _clip_video(name, rec, t_list, args, cam_fn, overlay_fn=None, trail=False, ground="studio_slab"):
    show = SHOW
    inv = show.subset_map
    lo, hi = show.tower_bounds()
    c = 0.5 * (lo + hi)
    w, h = _size(args)
    sc0 = show.base_scene(ground, "studio", center=c, size=0.6)
    show.add_stones(sc0, show.tower_ids, {i: show.pose(i) for i in show.tower_ids})
    exposure = key_light_exposure(sc0, SHOW.meter_point(), [0, 0, 1])
    writer = _writer(name, exposure, args)
    for f, t in enumerate(t_list):
        sc = show.base_scene(ground, "studio", center=c, size=0.6)
        for i in show.tower_ids:
            j = inv[i]
            sc.add(show.meshes[i], show.materials[i], rec.pose_at(j, t), rec.pose_at(j, t + SHUTTER * args.time_scale), object_id=int(i))
            if trail:
                pts = np.array([rec.pose_at(j, tt).t for tt in np.linspace(rec.times[0], t, 60)])
                if np.linalg.norm(pts[-1] - pts[0]) > 1e-4:
                    sc.add_overlay(ov.polyline(pts, 0.001), tuple(colormap(np.array([show.tower_ids.index(i) / max(1, len(show.tower_ids) - 1)]), "sequential", 0, 1)[0]), alpha=0.9, blend="add")
        if overlay_fn is not None:
            overlay_fn(sc, t, c)
        cam = cam_fn(f, c, w, h)
        _render_frame(sc, cam, args, writer)
    _encode(name, args, {"frames": len(t_list), "time_scale": args.time_scale})


def V3(args):
    """Tilt test: ground tilts at 0.5 degrees per second to failure; the camera is fixed to the ground
    frame (gravity rotates in the simulation, so the ground frame is the world frame) with the O8 gauge.
    Played as a time lapse (``--tilt-speedup``)."""
    rec = SHOW.clips["tilt"]
    info = SHOW.data["tilt"]
    m_hat = [r for r in SHOW.data["records"] if not r["meta"].get("probe")][-1]
    t_end = rec.times[-1]
    n = int((t_end - rec.times[0]) / args.tilt_speedup * FPS)
    ts = list(np.linspace(rec.times[0], t_end, max(n, 24)))

    def gauge(sc, t, c):
        a = info["rate_deg_s"] * (t - rec.times[0])
        lo_, hi_ = m_hat["m_lo"], m_hat["m_hi"]
        for mesh, col in ov.dial(c + np.array([0.14, -0.06, 0.12]), 0.045, a, (lo_, hi_), vmin=0.0, vmax=max(30.0, m_hat["m_hat"] * 1.5)):
            sc.add_overlay(mesh, col, alpha=1.0)

    def cam(f, c, w, h):
        return Camera(position=c + np.array([0.0, -1.1, 0.05]), look_at=c, focal_mm=50.0, f_number=8.0, width=w, height=h, shutter=(0.0, SHUTTER))

    _clip_video("V3_tilt_test", rec, ts, args, cam, gauge)


def V4(args):
    """Collapse: slow motion of the fall (``--collapse-fps`` effective), slow orbit, trajectory
    painting O9."""
    rec = SHOW.clips["collapse"]
    fail = SHOW.data["collapse"].get("fail_time") or rec.times[0]
    t0 = max(rec.times[0], fail - 0.1)
    dur = min(rec.times[-1] - t0, args.collapse_seconds)
    n = int(dur * args.collapse_fps)
    ts = list(np.linspace(t0, t0 + dur, max(n, 24)))
    lo, hi = SHOW.tower_bounds()
    c0 = 0.5 * (lo + hi)
    keys = orbit_keyframes(c0, 1.0, c0[2] + 0.1, n=4, start_deg=-30, sweep_deg=40)
    path = camera_path_min_jerk(keys, list(np.linspace(0, len(ts) / FPS, len(keys))), len(ts), FPS)

    def cam(f, c, w, h):
        return _cam_from(path[f], w, h, 50.0, 4.0)

    _clip_video("V4_collapse", rec, ts, args, cam, trail=True, ground="studio_slab")
    if not args.no_sound:
        _v4_sound(rec, ts)


def _v4_sound(rec, ts):
    """Modal impact sound of the collapse (topple.render.sound), at the slow motion time base with
    the real pitch, muxed into V4_collapse.mp4 and kept as V4_collapse.wav."""
    import os

    from topple.render.sound import impacts_from_clip, mux_audio, surface_modes, synthesize, write_wav

    inv = SHOW.subset_map
    masses = {inv[i]: float(SHOW.stones_by_id(i)["mass"]) for i in SHOW.tower_ids}
    modes = {inv[i]: surface_modes(SHOW.meshes[i], masses[inv[i]]) for i in SHOW.tower_ids}
    t0, t1 = ts[0], ts[-1]
    events = [e for e in impacts_from_clip(rec, masses) if t0 <= e[0] <= t1]
    video_s = len(ts) / FPS

    def tmap(t):
        return (t - t0) / max(t1 - t0, 1e-9) * (len(ts) - 1) / FPS

    y = synthesize(events, modes, {j: (lambda t, j=j: rec.pose_at(j, t)) for j in masses}, video_s, time_map=tmap)
    wav = VIDEOS / "V4_collapse.wav"
    write_wav(wav, y)
    mp4 = VIDEOS / "V4_collapse.mp4"
    tmp = VIDEOS / "V4_collapse_sound.mp4"
    if mp4.exists() and mux_audio(mp4, wav, tmp) == 0:
        os.replace(tmp, mp4)
    print(f"V4 sound: {len(events)} impacts, {wav}", flush=True)


def V5(args):
    """Shake table: rising sinusoidal excitation at 2 Hz, fixed camera, margin readout as a bar and
    O9 trails."""
    rec = SHOW.clips["shake"]
    ts = list(np.arange(rec.times[0], rec.times[-1], 1.0 / FPS))[: int(args.shake_seconds * FPS)]
    info = SHOW.data["shake"]

    def readout(sc, t, c):
        A = info["amax_g"] * min(1.0, (t - rec.times[0]) / info["ramp_s"])
        sc.add_overlay(ov.bar(c + np.array([0.15, 0, -c[2] + 0.01]), c + np.array([0.15, 0, -c[2] + 0.01 + 0.4 * A])), (0.95, 0.6, 0.2), alpha=1.0)

    def cam(f, c, w, h):
        return Camera(position=c + np.array([0.8, -0.9, 0.15]), look_at=c, focal_mm=50.0, f_number=8.0, width=w, height=h, shutter=(0.0, SHUTTER))

    _clip_video("V5_shake_table", rec, ts, args, cam, readout, trail=True)


def V6(args):
    """Hundred towers: the grid of H11 building in parallel, outcome tint resolving per tower."""
    import glob

    from topple.pipeline import load_records
    from topple.render import Material

    paths = sorted(glob.glob(str(RESULTS / "experiments" / "E2all*" / "records.json")))
    if not paths:
        print("no records for V6")
        return
    rows = [r for r in load_records(paths[-1]) if not r["meta"].get("probe")]
    towers = sorted({r["tower"] for r in rows})[:100]
    k = int(np.ceil(np.sqrt(len(towers))))
    pitch = 0.3
    n_frames = int(args.grid_seconds * FPS)
    w, h = _size(args)
    exposure = None
    writer = None
    for f in range(n_frames):
        u = f / max(n_frames - 1, 1)
        sc = SHOW.base_scene("studio_slab", "studio", center=np.zeros(3), size=1.0)
        for n, t in enumerate(towers):
            rt = sorted([r for r in rows if r["tower"] == t], key=lambda r: r["placement"])
            gx, gy = divmod(n, k)
            off = np.array([(gx - (k - 1) / 2) * pitch, (gy - (k - 1) / 2) * pitch, 0.0])
            shown = int(np.ceil(u * len(rt) * 1.2))
            ok = not any(r["collapsed"] or r["y"] < 0.5 for r in rt)
            col = (0.55, 0.55, 0.55) if shown < len(rt) else ((0.25, 0.6, 0.35) if ok else (0.75, 0.25, 0.2))
            zc = 0.0
            for j in range(min(shown, len(rt))):
                i = SHOW.tower_ids[j % len(SHOW.tower_ids)]
                mw = SHOW.meshes[i].transformed(SE3(SHOW.pose(i).q, off))
                mw = mw.translated([0, 0, zc - mw.vertices[:, 2].min()])
                zc = float(mw.vertices[:, 2].max())
                sc.add(mw, Material(name="tile", base_color=col, roughness=0.6))
        span = k * pitch + 0.2
        zoom = 1.0 - 0.25 * float(min_jerk(np.array([u]))[0])
        # orthographic, so the height only sets what lies between: below the lower edge of the key softbox (0.71 m)
        cam = Camera(position=np.array([0, 0, 0.5]), look_at=np.zeros(3), up=np.array([0, 1.0, 0]), ortho=True, ortho_width=span * zoom * w / h, width=w, height=h)
        if writer is None:
            # meter on the ground between tiles (a tile sits at the origin when the grid side is odd)
            exposure = key_light_exposure(sc, np.array([0.5 * pitch, 0.5 * pitch, 0.001]), [0, 0, 1])
            writer = _writer("V6_hundred_towers", exposure, args)
        _render_frame(sc, cam, args, writer)
    _encode("V6_hundred_towers", args, {"frames": n_frames, "towers": len(towers)})


def V7(args):
    """Posterior to decision: shells tighten as views arrive (the shell spread is scaled by
    1 / sqrt(views) around the mean), then the chosen pose locks in."""
    show = SHOW
    post = show.data["posterior"]
    top = show.tower_ids[-1]
    mean = TriMesh(post["mean_V"], post["mean_F"])
    T_map = SE3(show.pose(top).q, np.zeros(3))
    T_map = SE3(T_map.q, -T_map.apply(mean.centroid()) * 0 + np.array([0, 0, 0.0]))
    T_map = show.pose(top) @ show.pose(top, "pile").inverse()
    n_frames = int(args.posterior_seconds * FPS)
    w, h = _size(args)
    c = show.pose(top).t
    keys = orbit_keyframes(c, 0.6, c[2] + 0.15, n=4, start_deg=0, sweep_deg=60)
    path = camera_path_min_jerk(keys, list(np.linspace(0, n_frames / FPS, len(keys))), n_frames, FPS)
    writer = None
    samples = [TriMesh(V, F) for V, F in post["samples"][:16]]
    for f in range(n_frames):
        u = f / max(n_frames - 1, 1)
        views = 1 + int(u * 15)
        spread = 1.0 / np.sqrt(views)
        sc = show.base_scene("granite_table", "studio", center=c, size=0.3)
        below = show.tower_ids[:-1]
        show.add_stones(sc, below, {i: show.pose(i) for i in below})
        for s in samples:
            V = mean.vertices + spread * 4.0 * (s.vertices[: mean.n_vertices] - mean.vertices) if s.n_vertices == mean.n_vertices else s.vertices
            sc.add_overlay(TriMesh(V, s.faces if s.n_vertices == mean.n_vertices else s.faces).transformed(T_map), (0.55, 0.75, 1.0), alpha=0.06, blend="add", outline=0.0)
        if u > 0.75:
            sc.add(show.meshes[top], show.materials[top], show.pose(top), name="locked")
        cam = _cam_from(path[f], w, h, 85.0, 5.6)
        cam.focus_distance = float(np.linalg.norm(cam.position - c))
        if writer is None:
            writer = _writer("V7_posterior_to_decision", key_light_exposure(sc, SHOW.meter_point(), [0, 0, 1]), args)
        _render_frame(sc, cam, args, writer)
    _encode("V7_posterior_to_decision", args, {"frames": n_frames})


def V8(args):
    """Force flow: the top stone is lowered, contacts appear, arrows grow with the load carried and
    the thrust line follows (O1, O2 from the static solution of the final tower)."""
    show = SHOW
    top = show.tower_ids[-1]
    T = show.pose(top)
    n_frames = int(args.force_seconds * FPS)
    lo, hi = show.tower_bounds()
    c = 0.5 * (lo + hi)
    w, h = _size(args)
    writer = None
    st = show.statics_for(None)
    for f in range(n_frames):
        u = f / max(n_frames - 1, 1)
        drop = 0.03 * max(0.0, 1.0 - u / 0.4)
        load = float(np.clip((u - 0.4) / 0.4, 0, 1))
        sc = show.base_scene("studio_slab", "studio", center=c, size=0.5)
        ids = show.tower_ids[:-1]
        show.add_stones(sc, ids, {i: show.pose(i) for i in ids})
        sc.add(show.meshes[top], show.materials[top], SE3(T.q, T.t + np.array([0, 0, drop])), name="top")
        if load > 0:
            show.add_force_overlays(sc, scale=0.03 * load, cones=load > 0.5)
            if load > 0.3:
                show.add_thrust_line(sc)
        cam = Camera(position=c + np.array([0.55, -0.6, 0.12]), look_at=c, focal_mm=85.0, f_number=8.0, width=w, height=h)
        if writer is None:
            writer = _writer("V8_force_flow", key_light_exposure(sc, SHOW.meter_point(), [0, 0, 1]), args)
        _render_frame(sc, cam, args, writer)
    _encode("V8_force_flow", args, {"frames": n_frames})


def V9(args):
    """Sweeping section: the clipping plane sweeps through the tower (O2, O3)."""
    from topple.render import Material

    show = SHOW
    lo, hi = show.tower_bounds()
    c = 0.5 * (lo + hi)
    n_frames = int(args.section_seconds * FPS)
    w, h = _size(args)
    writer = None
    for f in range(n_frames):
        u = f / max(n_frames - 1, 1)
        y = lo[1] - 0.005 + (hi[1] - lo[1] + 0.01) * u
        sc = show.base_scene("studio_slab", "studio", center=c, size=0.5)
        for i in show.tower_ids:
            clipped, cap, col = ov.hatched_section_cap(show.world_mesh(i, show.pose(i)), np.array([c[0], y, c[2]]), [0.0, -1.0, 0.0])
            if clipped.n_faces:
                sc.add(clipped, show.materials[i])
            if cap is not None:
                sc.add(cap, Material(name="cap", base_color=(0.7, 0.7, 0.7), roughness=0.9, use_vertex_colors=True), vertex_colors=col)
        show.add_thrust_line(sc, plane=(np.array([c[0], y, c[2]]), np.array([0.0, -1.0, 0.0]), 0.002))
        cam = Camera(position=c + np.array([0.0, -1.3, 0.05]), look_at=c, focal_mm=85.0, f_number=8.0, width=w, height=h)
        if writer is None:
            writer = _writer("V9_sweeping_section", key_light_exposure(sc, SHOW.meter_point(), [0, 0, 1]), args)
        _render_frame(sc, cam, args, writer)
    _encode("V9_sweeping_section", args, {"frames": n_frames})


SHOTS = {"V1": V1, "V2": V2, "V3": V3, "V4": V4, "V5": V5, "V6": V6, "V7": V7, "V8": V8, "V9": V9}
SHOW = None

if __name__ == "__main__":
    ap = parser("video shots")
    ap.add_argument("shot", help="V1..V9 or all")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--spp", type=int, default=12)
    ap.add_argument("--exr", action="store_true")
    ap.add_argument("--seconds-per-stone", type=float, default=2.5)
    ap.add_argument("--scan-seconds", type=float, default=5.0)
    ap.add_argument("--tilt-speedup", type=float, default=8.0)
    ap.add_argument("--collapse-fps", type=float, default=240.0, help="effective frame rate of the slow motion")
    ap.add_argument("--no-sound", action="store_true", help="skip the modal impact sound of V4")
    ap.add_argument("--collapse-seconds", type=float, default=0.6)
    ap.add_argument("--shake-seconds", type=float, default=6.0)
    ap.add_argument("--grid-seconds", type=float, default=3.0)
    ap.add_argument("--posterior-seconds", type=float, default=3.0)
    ap.add_argument("--force-seconds", type=float, default=3.0)
    ap.add_argument("--section-seconds", type=float, default=3.0)
    ap.add_argument("--time-scale", type=float, default=1.0)
    args = ap.parse_args()
    VIDEOS.mkdir(parents=True, exist_ok=True)
    SHOW = Showcase(RESULTS / "showcase")
    names = list(SHOTS) if args.shot == "all" else args.shot.split(",")
    for n in names:
        t0 = time.time()
        SHOTS[n](args)
        print(f"{n} done in {time.time() - t0:.0f} s", flush=True)
