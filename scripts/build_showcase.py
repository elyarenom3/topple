#!/usr/bin/env python3
"""Build and record the showcase tower used by every still and video (Sections 10.7, 10.8).

One tower is built with exact inputs (so the planner can stack it as high as the stones allow),
the true world is recorded at 1 kHz (every ``--every`` steps), and the dynamic tests used by the
shots are recorded on clones of the final tower: a tilt to failure (V3, H12), a collapse (V4, H6),
a shake (V5, H9). One stone is scanned with the simulated depth camera to supply depth images and
posterior shells (V2, V7, H5). Everything goes to results/showcase/.

python scripts/build_showcase.py --profile study --max-stones 8 --seed 3
"""

from __future__ import annotations

import json
import pickle
import time

import numpy as np

from _common import RESULTS, config, parser

from topple.core.rng import Stage, make_rng
from topple.core.transforms import SE3
from topple.harness.tests import _rotated_gravity, failure_check
from topple.physics.recorder import StateRecorder
from topple.pipeline import Condition, Experiment
from topple.statics.equilibrium import min_norm_forces
from topple.statics.tower import Tower


def record_clip(world, seconds, every, gravity_fn=None, ids=None, stop_on_fail=False, ref=None, cfg=None):
    """Step a clone of ``world`` for ``seconds`` while recording; gravity_fn(t) updates gravity."""
    w = world.clone()
    rec = StateRecorder(w, every=every)
    g0 = w.gravity.copy()
    n = int(seconds / w.dt)
    fail_t = None
    for k in range(n):
        if gravity_fn is not None:
            w.gravity = gravity_fn(k * w.dt, g0)
        w.step(1)
        if stop_on_fail and fail_t is None and k % 10 == 0 and ids:
            failed, _, _ = failure_check(w, ref, ids, cfg["harness"]["fail_disp_m"], cfg["harness"]["fail_rot_deg"])
            if failed:
                fail_t = w.time
    return rec, fail_t


def main(args):
    cfg = config(args)
    cfg["experiment"]["max_stones"] = args.max_stones
    cfg["experiment"]["probes_per_step"] = 0
    cfg["stones"]["n_pile"] = args.pile
    cfg["planner"]["n_candidates"] = args.candidates
    out = RESULTS / (args.showcase_dir or "showcase")
    out.mkdir(parents=True, exist_ok=True)
    cond = Condition(name="showcase", oracle=True, propagation="none", margin="tilt", tests=("tilt",), stone_class=args.stone_class)
    exp = Experiment(cfg, cond, verbose=True)
    exp.record_every = args.every
    t0 = time.time()
    records = exp.run_tower(args.tower)
    print(f"built tower with {len([r for r in records if not r.meta.get('probe')])} placements in {time.time() - t0:.0f} s")
    world = exp.last_world
    truth = exp.last_truth
    placed = exp.last_placed
    ground_id = next(b.id for b in world.bodies if b.static)
    tower_ids = [p["body_id"] for p in placed]
    # stones: body frame meshes, materials and classes
    stones = []
    for b in world.bodies:
        if b.static:
            continue
        spec = truth.spec(b.name)
        stones.append({"name": b.name, "id": b.id, "V": b.shape.mesh.vertices, "F": b.shape.mesh.faces, "material": spec.material, "class": spec.stone_class, "mass": b.mass})
    # static analysis of the final tower and of every partial tower (forces for O1 and O2)
    statics = []
    tw, idmap = world.subset([ground_id] + tower_ids)
    for j in range(1, len(tower_ids) + 1):
        sub, m2 = world.subset([ground_id] + tower_ids[:j])
        T = Tower.from_world(sub, margin=float(cfg["statics"].get("contact_band", 5e-4)))
        try:
            F = min_norm_forces(T, None, int(cfg["statics"]["cone_sides"]))
        except Exception:
            F = None
        inv = {v: k for k, v in m2.items()}
        statics.append({
            "n": j,
            "contacts": [{"a": inv[c.body_a], "b": inv[c.body_b], "p": np.asarray(c.point).tolist(), "n": np.asarray(c.normal).tolist(), "mu": float(c.mu)} for c in T.contacts],
            "forces": None if F is None else np.asarray(F).tolist(),
        })
    # recorded dynamic clips on the final tower (ground plus tower only)
    ref = tw.state()
    ids = [idmap[i] for i in tower_ids]
    build = [r for r in records if not r.meta.get("probe")]
    last = build[-1] if build else None
    m_true = last.m_true if last is not None and np.isfinite(last.m_true) else 10.0
    per = last.meta.get("tilt_per_azimuth") if last is not None else None
    az = float(min(per, key=per.get)) if per else 0.0
    rate = 0.5  # deg per second, the specified tilt rate
    tilt_s = min(60.0, (m_true + 4.0) / rate)
    print(f"recording tilt clip ({tilt_s:.0f} s simulated)")
    tilt_rec, tilt_fail = record_clip(tw, tilt_s, args.every, lambda t, g0: _rotated_gravity(g0, rate * t, az), ids, True, ref, cfg)
    # collapse: tilt quickly past the critical angle and keep recording the fall
    print("recording collapse clip")
    crit = m_true + 3.0
    collapse_rec, coll_fail = record_clip(tw, 3.0, 1, lambda t, g0: _rotated_gravity(g0, min(crit, 20.0 * t), az), ids, True, ref, cfg)
    # shake: rising sinusoidal ground acceleration at 2 Hz
    print("recording shake clip")
    f, amax, ramp = cfg["harness"]["shake_freq_hz"], cfg["harness"]["shake_max_g"], 6.0

    def shake_g(t, g0):
        A = amax * min(1.0, t / ramp)
        return g0 - A * 9.81 * np.sin(2 * np.pi * f * t) * np.array([1.0, 0.0, 0.0])

    shake_rec, shake_fail = record_clip(tw, ramp, args.every, shake_g, ids, True, ref, cfg)
    # perception of the top stone: depth images and posterior shells
    print("scanning the top stone")
    from topple.perception.camera import PinholeCamera
    from topple.perception.depth_sensor import DepthSensor
    from topple.perception.estimator import Estimator
    from topple.perception.viewpoints import hemisphere_views, in_hand_views

    top = world.bodies[tower_ids[-1]]
    pile_state = exp.pile_state
    scfg = dict(cfg["sensor"])
    cam = PinholeCamera.from_fov(int(scfg["width"]), int(scfg["height"]), float(scfg["fov_deg"]))
    sensor = DepthSensor(cam, scfg, make_rng(args.seed, 0, Stage.SENSOR, 99))
    pile_world = world.clone()
    pile_world.set_state(pile_state)
    meshes = [b.world_mesh() for b in pile_world.bodies]
    centre = pile_world.bodies[top.id].x.copy()
    radius = max(0.35, 4 * top.shape.bounding_radius)
    views, view_poses = [], []
    for T in hemisphere_views(centre, radius, int(scfg["views_per_stone"]), make_rng(args.seed, 0, Stage.SENSOR, 98)):
        views.append(sensor.render_depth(meshes, T, ids=list(range(len(meshes)))))
        view_poses.append(T.matrix().tolist())
    for T in in_hand_views(centre, radius, int(scfg.get("in_hand_views", 4)), make_rng(args.seed, 0, Stage.SENSOR, 97)):
        views.append(sensor.render_depth([meshes[top.id]], T, ids=[top.id]))
        view_poses.append(T.matrix().tolist())
    est = Estimator(cfg, make_rng(args.seed, 0, Stage.ESTIMATOR, 5)).estimate(views, 32, target_id=top.id)
    posterior = {
        "mean_V": est.mean_mesh.vertices, "mean_F": est.mean_mesh.faces,
        "samples": [(s.mesh.vertices, s.mesh.faces) for s in est.samples],
        "coms": np.array([s.com for s in est.samples]),
        "scan_pose": pile_world.bodies[top.id].pose.matrix(),
        "depth": [(v.depth, v.valid) for v in views],
        "view_poses": view_poses,
    }
    data = {
        "cfg": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "stones": stones,
        "ground_id": ground_id,
        "tower_ids": tower_ids,
        "pile_state": {"q": exp.pile_state.q, "x": exp.pile_state.x},
        "final_state": {"q": world.state().q, "x": world.state().x},
        "events": exp.events,
        "records": [r.as_row() for r in records],
        "statics": statics,
        "subset_map": idmap,
        "tilt": {"azimuth": az, "rate_deg_s": rate, "fail_time": tilt_fail},
        "collapse": {"azimuth": az, "angle": crit, "fail_time": coll_fail},
        "shake": {"freq": f, "amax_g": amax, "ramp_s": ramp, "fail_time": shake_fail},
        "posterior": posterior,
    }
    exp.last_recorder.save(out / "build_record.npz")
    tilt_rec.save(out / "tilt_record.npz")
    collapse_rec.save(out / "collapse_record.npz")
    shake_rec.save(out / "shake_record.npz")
    with open(out / "showcase.pkl", "wb") as fh:
        pickle.dump(data, fh)
    # overlay forces from the hull contact model (one normal per contact; the engine manifold shares
    # one normal per stone pair, which can make a resting stack look infeasible)
    from topple.render.showcase import Showcase

    data["statics"] = Showcase(out).hull_statics(float(cfg["statics"].get("contact_band", 5e-4)), int(cfg["statics"]["cone_sides"]))
    with open(out / "showcase.pkl", "wb") as fh:
        pickle.dump(data, fh)
    with open(out / "summary.json", "w") as fh:
        json.dump({"n_stones": len(tower_ids), "stones": [world.bodies[i].name for i in tower_ids], "height_m": float(max(world.bodies[i].world_mesh().vertices[:, 2].max() for i in tower_ids)), "m_true_final": m_true, "tilt_fail_time": tilt_fail, "collapse_fail_time": coll_fail, "shake_fail_time": shake_fail}, fh, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    ap = parser("build the showcase tower")
    ap.add_argument("--max-stones", type=int, default=8)
    ap.add_argument("--pile", type=int, default=12)
    ap.add_argument("--candidates", type=int, default=16)
    ap.add_argument("--tower", type=int, default=0)
    ap.add_argument("--every", type=int, default=10, help="record every n physics steps (10 = 100 Hz)")
    ap.add_argument("--stone-class", default=None, help="draw the pile from one class (blocky_quarry stacks best)")
    ap.add_argument("--showcase-dir", default=None, help="output directory under results (default showcase)")
    main(ap.parse_args())
