"""Replay of finished experiment towers for scoring against other physics engines.

Each tower is rebuilt exactly as the experiment ran it: the same stones, true masses and friction,
the same pile and, under perception, the same scans, all drawn from the keyed random streams. The
recorded placements (building and probe) are lowered again in the developer's engine in the same
order, which reproduces the state the harness tested. That state is handed to MuJoCo and Drake;
each settles it for ``settle_s`` under its own contact model and runs the same tilt and fixed
perturbation tests. The tests of the building placements can also be rerun in the developer's
engine, to check that the replay reproduces the recorded outcomes.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np


def load_experiment(exp_dir, profile: str = "study"):
    """(cfg, condition, records) of a finished experiment directory."""
    from ..core.config import load_config
    from ..pipeline import Condition

    exp_dir = Path(exp_dir)
    prov = json.load(open(exp_dir / "provenance.json"))
    cfg = load_config(profile)
    if cfg.get("_config_hash") != prov["config_hash"]:
        raise ValueError(f"config hash {cfg.get('_config_hash')} differs from the experiment's {prov['config_hash']}")
    c = dict(prov["condition"])
    c["tests"] = tuple(c["tests"])
    if c.get("placement") == "arm":
        raise ValueError("replay supports the oracle lowering only")
    return cfg, Condition(**c), json.load(open(exp_dir / "records.json"))


def setup_tower(exp, t: int):
    """The true world of tower ``t`` after the pile settled, as in ``Experiment.run_tower``, with
    the estimate frame to body frame offsets from perception."""
    from ..core.config import get
    from ..core.rng import Stage, make_rng
    from ..harness import GroundTruthStore
    from ..physics.body import RigidBody
    from ..physics.world import World
    from ..stonegen.generator import StoneGenerator

    cfg, cond = exp.cfg, exp.cond
    cfg_w = dict(cfg["world"])
    cfg_w["cone_sides"] = cond.cone_sides
    rng_gen = make_rng(exp.seed, t, Stage.STONEGEN)
    gen = StoneGenerator(cfg)
    specs = [gen.sample(rng_gen, cond.stone_class) for _ in range(int(get(cfg, "stones.n_pile", 12)))]
    cfg_truth = dict(cfg)
    cfg_truth["friction"] = dict(cfg["friction"], prior_log_sigma=cond.friction_log_sigma)
    truth = GroundTruthStore.build(specs, cfg_truth, exp.seed, t)
    world = World(cfg_w)
    world.add_body(RigidBody.ground(friction=truth.mu_ground))
    sdf_cfg = get(cfg, "stones.sdf_spacing", 0.001)
    sdf_spacing = None if sdf_cfg is None else float(sdf_cfg)
    bodies = []
    for s in specs:
        b = truth.make_body(s.name, sdf_spacing=sdf_spacing)
        bodies.append(b)
        world.add_body(b)
    truth.apply_friction(world)
    exp._scatter_pile(world, bodies, make_rng(exp.seed, t, Stage.PHYSICS))
    _, offsets = exp._perceive(world, bodies, truth, t)
    return world, bodies, truth, offsets


def engine_tests(tw, cfg: dict, engines, settle_s: float, tmpdir: str) -> dict:
    """Settle the test world ``tw`` in each engine, then run the tilt and fixed tests."""
    from ..harness.tests import fixed_perturbation, tilt_test
    from .common import build_engine_world

    out = {}
    x0 = tw.state().x
    dyn = [i for i, b in enumerate(tw.bodies) if not b.static]
    for e in engines:
        t0 = time.perf_counter()
        try:
            w = build_engine_world(tw, e, tmpdir)
            w.step(int(round(settle_s / w.dt)))
            shift = float(np.max(np.linalg.norm(w.state().x[dyn] - x0[dyn], axis=1)))
            out[e] = {"tilt": float(tilt_test(w, cfg).value), "fixed": float(fixed_perturbation(w, cfg).value), "settle_shift_m": shift, "seconds": time.perf_counter() - t0}
        except Exception as exc:  # an engine failure is recorded, the replay continues
            out[e] = {"error": repr(exc), "seconds": time.perf_counter() - t0}
    return out


def replay_tower(exp_dir, t: int, engines=("mujoco", "drake"), settle_s: float = 1.0, check_topple: bool = True, max_records: int | None = None, profile: str = "study", log=None) -> list[dict]:
    """Outcomes of every recorded placement of tower ``t`` in each engine."""
    from ..core.transforms import SE3
    from ..harness.tests import failure_check, fixed_perturbation, tilt_test
    from ..pipeline import Experiment
    from ..planning.placement import lower_and_release

    cfg, cond, recs = load_experiment(exp_dir, profile)
    exp = Experiment(cfg, cond, verbose=False)
    world, bodies, truth, offsets = setup_tower(exp, t)
    by_name = {b.name: i for i, b in enumerate(bodies)}
    mine = [r for r in recs if r["tower"] == t]
    ground_id = next(b.id for b in world.bodies if b.static)
    h = cfg["harness"]
    tower_ids: list[int] = []
    out = []
    n_done = 0
    with tempfile.TemporaryDirectory() as td:
        for k in sorted({r["placement"] for r in mine}):
            step = [r for r in mine if r["placement"] == k]
            ordered = [(r, True) for r in step if r["meta"].get("probe")] + [(r, False) for r in step if not r["meta"].get("probe")]
            for r, is_probe in ordered:
                if max_records is not None and n_done >= max_records:
                    return out
                j = by_name[r["stone"]]
                body = bodies[j]
                T = SE3.from_matrix(np.asarray(r["meta"]["pose"], dtype=float)) @ offsets[j]
                t0 = time.perf_counter()
                if is_probe:
                    w2, idmap = world.subset([ground_id] + tower_ids + [body.id])
                    b2 = w2.bodies[idmap[body.id]]
                    ids2 = [idmap[i] for i in tower_ids]
                    ref = w2.state()
                    ok = lower_and_release(w2, b2, T, cfg)
                    collapsed = (failure_check(w2, ref, ids2, h["fail_disp_m"], h["fail_rot_deg"])[0] if ids2 else False) or not ok
                    tw, _ = w2.subset([idmap[ground_id]] + ids2 + [b2.id])
                else:
                    ref = world.state()
                    ok = lower_and_release(world, body, T, cfg)
                    collapsed = (failure_check(world, ref, tower_ids, h["fail_disp_m"], h["fail_rot_deg"])[0] if tower_ids else False) or not ok
                    tw, _ = world.subset([ground_id] + tower_ids + [body.id])
                row = {"tower": t, "placement": k, "stone": r["stone"], "probe": is_probe, "tag": r["meta"].get("tag"),
                       "recorded": {"y": r["y"], "m_true": r["m_true"], "collapsed": r["collapsed"], "p_hat": r["p_hat"], "m_hat": r["m_hat"]},
                       "replay_collapsed": bool(collapsed)}
                if collapsed:
                    row["engines"] = {e: {"tilt": 0.0, "fixed": 0.0, "collapsed": True} for e in engines}
                else:
                    if check_topple and not is_probe:
                        row["topple_replay"] = {"tilt": float(tilt_test(tw, cfg).value), "fixed": float(fixed_perturbation(tw, cfg).value)}
                    row["engines"] = engine_tests(tw, cfg, engines, settle_s, td)
                row["seconds"] = time.perf_counter() - t0
                out.append(row)
                n_done += 1
                if log:
                    log(f"tower {t} placement {k} {'probe' if is_probe else 'build'} {r['stone']}: recorded y {r['y']:.0f} tilt {r['m_true']:.1f} | " + " | ".join(f"{e} " + (f"y {v['fixed']:.0f} tilt {v['tilt']:.1f}" if "error" not in v else "error") for e, v in row["engines"].items()) + f" ({row['seconds']:.0f} s)")
                if not is_probe:
                    if collapsed:
                        return out
                    tower_ids.append(body.id)
    return out
