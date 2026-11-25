"""End to end pipeline: one placement at a time (Section 4.1).

For each tower: generate stones, build the hidden ground truth world, scan every stone
with the simulated depth camera, then loop: the planner chooses a stone and a pose from
posterior samples (stating a margin distribution and a survival probability), the stone is
placed in the true world, the harness measures the true margin and the binary outcome on
clones of the post release state, and the evaluator stores (M_hat, p_hat, M_star, y).

Two placement modes exist: ``teleport`` lowers the stone kinematically to contact and
releases it (oracle placement, milestone M7), ``arm`` runs grasp planning, inverse
kinematics, motion planning and the impedance controller (milestone M9).

The estimator and the planner only ever see depth images and the observed poses of the
placed stones (true pose plus camera pose noise). The transform between a stone's
estimated frame and its true body frame is known to the harness only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

import numpy as np

from .core.config import get
from .core.rng import Stage, make_rng
from .core.transforms import SE3, quat_from_rotvec, quat_mul


@dataclass
class Condition:
    name: str = "E2"
    oracle: bool = False  # exact geometry, com and friction
    propagation: str = "mc"  # mc | unscented | first_order | none
    margin: str = "tilt"  # tilt | push | fc | support | simulation
    kappa: float = 1.0
    placement: str = "teleport"  # teleport | arm
    tests: tuple = ("tilt", "fixed")
    cone_sides: int = 8
    stone_class: str | None = None
    noise_scale: float = 1.0
    friction_log_sigma: float = 0.25
    extra: dict = field(default_factory=dict)


@dataclass
class PlacementRecord:
    tower: int
    placement: int
    stone: str
    condition: str
    margin_kind: str
    propagation: str
    m_hat: float
    m_lo: float
    m_hi: float
    samples: np.ndarray
    p_hat: float
    m_true: float
    y: float
    height: float
    collapsed: bool
    ms_decision: float
    shake_true: float = float("nan")
    push_true: float = float("nan")
    support_true: float = float("nan")
    meta: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        d = asdict(self)
        d["samples"] = np.asarray(self.samples).tolist()
        return d


class Experiment:
    def __init__(self, cfg: dict, cond: Condition, verbose: bool = True):
        self.cfg = cfg
        self.cond = cond
        self.verbose = verbose
        self.seed = int(get(cfg, "experiment.seed", 0))

    # ------------------------------------------------------------------ logging
    def log(self, *a):
        if self.verbose:
            print(f"[{self.cond.name}]", *a, flush=True)

    # ------------------------------------------------------------------ run
    def run(self, towers: list[int] | None = None) -> list[PlacementRecord]:
        n = int(get(self.cfg, "experiment.n_towers", 50))
        towers = list(range(n)) if towers is None else towers
        out = []
        for t in towers:
            out += self.run_tower(t)
        return out

    def run_tower(self, t: int) -> list[PlacementRecord]:
        from .harness import GroundTruthStore, run_protocol
        from .harness.tests import failure_check
        from .physics.world import World
        from .stonegen.generator import StoneGenerator

        cfg = self.cfg
        cond = self.cond
        cfg_w = dict(cfg["world"])
        cfg_w["cone_sides"] = cond.cone_sides
        rng_gen = make_rng(self.seed, t, Stage.STONEGEN)
        n_pile = int(get(cfg, "stones.n_pile", 12))
        gen = StoneGenerator(cfg)
        specs = [gen.sample(rng_gen, cond.stone_class) for _ in range(n_pile)]
        cfg_truth = dict(cfg)
        cfg_truth["friction"] = dict(cfg["friction"], prior_log_sigma=cond.friction_log_sigma)
        truth = GroundTruthStore.build(specs, cfg_truth, self.seed, t)

        # ---------------------------------------------------------- true world
        world = World(cfg_w)
        from .physics.body import RigidBody

        ground = RigidBody.ground(friction=truth.mu_ground)
        world.add_body(ground)
        sdf_cfg = get(cfg, "stones.sdf_spacing", 0.001)
        sdf_spacing = None if sdf_cfg is None else float(sdf_cfg)
        bodies = []
        for s in specs:
            b = truth.make_body(s.name, sdf_spacing=sdf_spacing)
            bodies.append(b)
            world.add_body(b)
        truth.apply_friction(world)
        self._scatter_pile(world, bodies, make_rng(self.seed, t, Stage.PHYSICS))
        self.log(f"tower {t}: pile of {len(bodies)} stones settled")
        # optional recording of the true world for the renderer (showcase builds)
        self.events = []
        self.last_recorder = None
        if int(getattr(self, "record_every", 0)) > 0:
            from .physics.recorder import StateRecorder

            self.last_recorder = StateRecorder(world, every=int(self.record_every))
        self.pile_state = world.state()

        # ---------------------------------------------------------- perception
        estimates, est_offsets = self._perceive(world, bodies, truth, t)

        # ---------------------------------------------------------- build loop
        from .planning.planner import PlacedStone, Planner, TowerState
        from .statics.analysis import MarginModel

        cfg_p = dict(cfg)
        cfg_p["planner"] = dict(cfg["planner"], kappa=cond.kappa)
        if cond.placement == "arm":
            cfg_p["planner"]["keep_pile_orientation"] = True
        planner = Planner(cfg_p, make_rng(self.seed, t, Stage.PLANNER))
        if get(cfg, "planner.dynamic_settle", True):
            from .planning.settle import SupportWorld

            planner.settler = SupportWorld(cfg, ground_mu=float(get(cfg, "friction.prior_median", 0.6)))
        model = MarginModel(cfg, cond, make_rng(self.seed, t, Stage.ESTIMATOR))
        screen_samples = int(get(cfg, "planner.screen_samples", 8))
        screen_azimuth = int(get(cfg, "planner.screen_azimuths", 12))
        placed: list[dict] = []  # dicts: estimate, pose (SE3, observed), body_id, height, stone_index
        state = TowerState(stones=[], ground_mu=float(get(cfg, "friction.prior_median", 0.6)), gravity=tuple(get(cfg, "world.gravity", (0, 0, -9.81))))
        available = list(range(len(bodies)))
        records = []
        h = cfg["harness"]
        max_stones = int(get(cfg, "experiment.max_stones", 12))
        stop_infeasible = bool(get(cfg, "experiment.stop_when_infeasible", True))
        n_probes = int(get(cfg, "experiment.probes_per_step", 0))
        rng_probe = make_rng(self.seed, t, Stage.EVAL, 77)
        for k in range(max_stones):
            if not available:
                break
            t0 = time.perf_counter()
            pile = [estimates[i] for i in available]

            def margin_fn(tower, cand, _pile=pile):
                # coarse evaluation for screening: few samples and azimuths (a point estimate when the
                # condition propagates nothing); the chosen pose is re-evaluated at full resolution
                est_c = _pile[cand.stone_index]
                if model.method == "none":
                    return model.evaluate(placed, est_c, cand.pose, n_azimuth=screen_azimuth).samples
                return model.evaluate(placed, est_c, cand.pose, n_samples=screen_samples, n_azimuth=screen_azimuth, method="mc").samples

            decision = planner.choose(state, pile, margin_fn, with_grasp=(cond.placement == "arm"))
            t_choose = time.perf_counter() - t0
            if decision is None:
                self.log(f"  placement {k}: no candidate, stopping")
                break
            # probe placements: extra candidates tested on clones of the true world (calibration data
            # across the range of predicted survival probability; the tower itself is unchanged)
            if n_probes > 0:
                for cand in self._pick_probes(decision, n_probes, rng_probe):
                    j = available[cand.stone_index]
                    rec = self._probe(world, bodies, j, estimates[j], est_offsets[j], cand.pose, placed, model, t, k, cfg)
                    if rec is not None:
                        records.append(rec)
            if stop_infeasible and not decision.feasible:
                # no candidate meets the chance constraint: the tower is complete. The best effort
                # choice is still predicted and tested as a probe for the calibration record.
                j = available[decision.stone_index]
                rec = self._probe(world, bodies, j, estimates[j], est_offsets[j], decision.pose, placed, model, t, k, cfg, tag="final_infeasible")
                if rec is not None:
                    records.append(rec)
                self.log(f"  placement {k}: no candidate meets p_hat >= 1 - delta, tower complete")
                break
            i = available[decision.stone_index]
            body = bodies[i]
            est = estimates[i]
            t1 = time.perf_counter()
            final = model.evaluate(placed, est, decision.pose)
            # decision time: the choice plus the final evaluation (probe placements and their tests,
            # which run in between, are excluded)
            ms = 1000 * (t_choose + time.perf_counter() - t1)
            preds = self._extra_predictions(placed, est, decision.pose, model)
            # map the estimated frame pose to the true body frame pose
            T_true = decision.pose @ est_offsets[i]
            tower_ids = [p["body_id"] for p in placed]
            ref = world.state()
            self.events.append({"kind": "place_start", "k": k, "stone": body.name, "body_id": body.id, "time": world.time, "from": body.pose.matrix().tolist(), "target": T_true.matrix().tolist()})
            ok = self._place(world, body, T_true, mode=cond.placement, rng=make_rng(self.seed, t, Stage.CONTROLLER, k), decision=decision, T_est_body=est_offsets[i])
            self.events.append({"kind": "place_end", "k": k, "stone": body.name, "body_id": body.id, "time": world.time, "pose": body.pose.matrix().tolist(), "m_hat": float(final.mean), "p_hat": float(final.p_hat)})
            collapsed, dmax, rmax = failure_check(world, ref, tower_ids, h["fail_disp_m"], h["fail_rot_deg"]) if tower_ids else (False, 0.0, 0.0)
            collapsed = collapsed or not ok
            top_z = float(body.world_mesh().vertices[:, 2].max())
            height = top_z if not collapsed else max([0.0] + [p["height"] for p in placed])
            if collapsed:
                m_true, y = 0.0, 0.0
                outcomes = {}
                self.log(f"  placement {k}: stone {body.name} collapsed the tower (dmax {dmax*1000:.1f} mm, placed ok {ok})")
            else:
                outcomes, m_true, y = self._test(world, tower_ids, body.id, cfg)
            rec = PlacementRecord(
                tower=t,
                placement=k,
                stone=body.name,
                condition=cond.name,
                margin_kind=cond.margin,
                propagation=cond.propagation,
                m_hat=final.mean,
                m_lo=final.lo,
                m_hi=final.hi,
                samples=np.asarray(final.samples),
                p_hat=final.p_hat,
                m_true=m_true,
                y=y,
                height=height,
                collapsed=collapsed,
                ms_decision=ms,
                shake_true=outcomes["shake"].value if "shake" in outcomes else float("nan"),
                push_true=outcomes["push"].value if "push" in outcomes else float("nan"),
                meta={"J": decision.J, "kappa": cond.kappa, "pose": decision.pose.matrix().tolist(), "n_candidates": len(decision.candidates), "n_contacts": final.n_contacts, "screen_p_hat": decision.p_hat, "arm_fallback": body.meta.get("arm_fallback"), "probe": False, "feasible": decision.feasible, "release": body.meta.get("release_check"), "tilt_per_azimuth": outcomes["tilt"].per_direction if "tilt" in outcomes else None, "preds": preds, "stone_class": body.meta.get("stone_class")},
            )
            records.append(rec)
            self.log(f"  placement {k}: {body.name} m_hat {rec.m_hat:.1f} [{rec.m_lo:.1f}, {rec.m_hi:.1f}] p_hat {rec.p_hat:.2f} | m_true {m_true:.1f} y {y:.0f} | {ms:.0f} ms")
            if collapsed:
                break
            # observed pose of the placed stone (true pose plus pose noise) in the estimated frame
            T_obs_true = body.pose @ est_offsets[i].inverse()
            obs = T_obs_true if cond.oracle else self._observe_pose(T_obs_true, make_rng(self.seed, t, Stage.SENSOR, 1000 + k))
            placed.append({"estimate": est, "pose": obs, "body_id": body.id, "height": height, "stone_index": i})
            state.stones.append(PlacedStone(body.name, est.mean_mesh.transformed(obs), float(est.mass), obs.apply(np.asarray(est.com, dtype=float)), float(est.mu), body.id))
            available.remove(i)
        self.last_placed = placed
        self.last_world = world
        self.last_truth = truth
        self.last_estimates = estimates
        return records

    # ------------------------------------------------------------------ testing helpers
    def _test(self, world, tower_ids, top_id, cfg):
        """Run the dynamic tests on a clone holding the ground and the tower only."""
        from .harness import run_protocol

        ground_id = next(b.id for b in world.bodies if b.static)
        tw, idmap = world.subset([ground_id] + list(tower_ids) + [top_id])
        outcomes = run_protocol(tw, cfg, idmap[top_id], [idmap[j] for j in list(tower_ids) + [top_id]], which=self.cond.tests)
        m_true = outcomes["tilt"].value if "tilt" in outcomes else float("nan")
        y = outcomes["fixed"].value if "fixed" in outcomes else float(m_true > get(cfg, "propagation.survival_tilt_deg", 3.0))
        return outcomes, m_true, y

    def _extra_predictions(self, placed, est, pose, model) -> dict:
        """Predictions of other propagation methods, margin definitions, cone discretisations and
        the simulation score on the same placement (conditions E1, E2, E7, E8 share the tests)."""
        import copy

        ex = self.cond.extra or {}
        out = {}

        def pack(res):
            s = np.asarray(res.samples, dtype=float)
            return {"mean": float(res.mean), "std": float(res.std), "lo": float(res.lo), "hi": float(res.hi), "p_hat": float(res.p_hat), "samples": s[:512].tolist()}

        for method in ex.get("eval_methods", []):
            if method == model.method:
                continue
            try:
                out[f"method:{method}"] = pack(model.evaluate(placed, est, pose, method=method))
            except Exception as exc:  # pragma: no cover
                out[f"method:{method}"] = {"error": repr(exc)}
        for kind in ex.get("eval_margins", []):
            if kind == model.kind:
                continue
            m2 = copy.copy(model)
            m2.kind = kind
            try:
                out[f"margin:{kind}"] = pack(m2.evaluate(placed, est, pose, method="none" if self.cond.oracle else model.method, n_samples=min(model.n_samples, 32)))
            except Exception as exc:  # pragma: no cover
                out[f"margin:{kind}"] = {"error": repr(exc)}
        for n_sides in ex.get("eval_cones", []):
            if int(n_sides) == model.cone_sides:
                continue
            m2 = copy.copy(model)
            m2.cone_sides = int(n_sides)
            out[f"cone:{int(n_sides)}"] = pack(m2.evaluate(placed, est, pose, method="none"))
        if ex.get("eval_simulation", False):
            m2 = copy.copy(model)
            m2.kind = "simulation"
            try:
                out["margin:simulation"] = pack(m2.evaluate(placed, est, pose, method="none"))
            except Exception as exc:  # pragma: no cover
                out["margin:simulation"] = {"error": repr(exc)}
        return out

    def _pick_probes(self, decision, n: int, rng) -> list:
        """Stable candidates other than the chosen pose, spread over predicted survival probability:
        candidates are binned by their screening p_hat into five bins and drawn from distinct bins."""
        cands = [c for c in decision.candidates if getattr(c.settled, "stable", False) and c.pose is not decision.pose and c.margin_samples is not None]
        if not cands:
            return []
        bins = {}
        for c in cands:
            bins.setdefault(min(int(c.p_hat * 5), 4), []).append(c)
        keys = list(bins.keys())
        rng.shuffle(keys)
        out = []
        for kb in keys[:n]:
            group = bins[kb]
            out.append(group[int(rng.integers(len(group)))])
        return out

    def _probe(self, world, bodies, j, est, offset, pose, placed, model, t, k, cfg, tag="probe"):
        """Predict and test one candidate placement on a clone of the true world."""
        from .harness.tests import failure_check
        from .planning.placement import lower_and_release

        body = bodies[j]
        tower_ids = [p["body_id"] for p in placed]
        ground_id = next(b.id for b in world.bodies if b.static)
        t0 = time.perf_counter()
        final = model.evaluate(placed, est, pose)
        ms = 1000 * (time.perf_counter() - t0)
        preds = self._extra_predictions(placed, est, pose, model)
        w2, idmap = world.subset([ground_id] + tower_ids + [body.id])
        b2 = w2.bodies[idmap[body.id]]
        ref = w2.state()
        ok = lower_and_release(w2, b2, pose @ offset, cfg)
        h = cfg["harness"]
        ids2 = [idmap[i] for i in tower_ids]
        collapsed = (failure_check(w2, ref, ids2, h["fail_disp_m"], h["fail_rot_deg"])[0] if ids2 else False) or not ok
        if collapsed:
            outcomes, m_true, y = {}, 0.0, 0.0
        else:
            outcomes, m_true, y = self._test(w2, ids2, b2.id, cfg)
        rec = PlacementRecord(
            tower=t, placement=k, stone=body.name, condition=self.cond.name, margin_kind=self.cond.margin, propagation=self.cond.propagation,
            m_hat=final.mean, m_lo=final.lo, m_hi=final.hi, samples=np.asarray(final.samples), p_hat=final.p_hat, m_true=m_true, y=y,
            height=float("nan"), collapsed=collapsed, ms_decision=ms,
            shake_true=outcomes["shake"].value if "shake" in outcomes else float("nan"),
            push_true=outcomes["push"].value if "push" in outcomes else float("nan"),
            meta={"probe": True, "tag": tag, "n_contacts": final.n_contacts, "pose": pose.matrix().tolist(), "tilt_per_azimuth": outcomes["tilt"].per_direction if "tilt" in outcomes else None, "preds": preds, "stone_class": body.meta.get("stone_class")},
        )
        self.log(f"    {tag} {body.name}: m_hat {rec.m_hat:.1f} [{rec.m_lo:.1f}, {rec.m_hi:.1f}] p_hat {rec.p_hat:.2f} | m_true {m_true:.1f} y {y:.0f}")
        return rec

    # ------------------------------------------------------------------ pieces
    def _scatter_pile(self, world, bodies, rng):
        """Drop the stones on a grid in random orientations and let them settle."""
        n = len(bodies)
        cols = int(np.ceil(np.sqrt(n)))
        pitch = 0.22
        layout = str(get(self.cfg, "stones.pile_layout", "grid"))
        base = np.asarray(get(self.cfg, "arm.base", [-0.45, 0.0, 0.0]), dtype=float)
        spots = []
        if layout == "arc":
            # within reach of the arm: two arcs around the arm base on both sides of the tower
            for radius in (0.5, 0.68):
                m = max(1, int(np.ceil(n / 4)))
                for side in (1.0, -1.0):
                    for a in np.linspace(np.radians(55.0), np.radians(125.0), m):
                        spots.append(base[:2] + radius * np.array([np.cos(side * a), np.sin(side * a)]))
        for k, b in enumerate(bodies):
            if layout == "arc":
                x, y = spots[k] + rng.uniform(-0.01, 0.01, size=2)
            else:
                r, c = divmod(k, cols)
                x = 0.25 + c * pitch + rng.uniform(-0.02, 0.02)
                y = -0.5 * (cols - 1) * pitch + r * pitch + rng.uniform(-0.02, 0.02)
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            b.q = q
            b.x = np.array([x, y, b.shape.bounding_radius + 0.01])
            b.v[:] = 0
            b.w[:] = 0
        world.step(int(0.5 / world.dt))
        world.damped_settle(seconds=0.7, damping=float(get(self.cfg, "planner.settle_damping", 60.0)))

    def _perceive(self, world, bodies, truth, t):
        """Scan every stone; return estimates and the estimate frame to true body frame offsets."""
        cfg = self.cfg
        cond = self.cond
        estimates, offsets = [], []
        from .perception.estimate_view import EstimateView, OracleEstimate

        if cond.oracle:
            for b in bodies:
                spec = truth.spec(b.name)
                mu = {bb.name: truth.mu(b.name, bb.name) for bb in world.bodies}
                est = OracleEstimate(spec, mu, b.pose, body=b)
                estimates.append(est)
                offsets.append(SE3())  # the estimate frame is the true body frame
            return estimates, offsets
        from .perception.camera import PinholeCamera
        from .perception.depth_sensor import DepthSensor
        from .perception.estimator import Estimator
        from .perception.viewpoints import hemisphere_views, in_hand_views

        scfg = dict(cfg["sensor"])
        scfg["noise_scale"] = cond.noise_scale
        cam = PinholeCamera.from_fov(int(scfg["width"]), int(scfg["height"]), float(scfg["fov_deg"]))
        meshes = [b.world_mesh() for b in world.bodies]
        ids = list(range(len(world.bodies)))
        n_modes = int(get(cfg, "propagation.shape_modes", 10))
        for i, b in enumerate(bodies):
            rng = make_rng(self.seed, t, Stage.SENSOR, i)
            sensor = DepthSensor(cam, scfg, rng)
            target = world.bodies.index(b)
            views = []
            center = b.x.copy()
            radius = max(0.35, 4 * b.shape.bounding_radius)
            for T in hemisphere_views(center, radius, int(scfg["views_per_stone"]), rng):
                views.append(sensor.render_depth(meshes, T, ids=ids))
            # in hand views: the stone alone (lifted from the pile), seen from below, fingers masked
            # by rendering the stone only; expressed in the stone's resting world frame
            for T in in_hand_views(center, radius, int(scfg.get("in_hand_views", 4)), rng):
                views.append(sensor.render_depth([meshes[target]], T, ids=[target]))
            cfg_e = dict(cfg)
            cfg_e["friction"] = dict(cfg["friction"], prior_log_sigma=cond.friction_log_sigma)
            raw = Estimator(cfg_e, make_rng(self.seed, t, Stage.ESTIMATOR, i)).estimate(views, int(get(cfg, "tsdf.posterior_samples", 32)), target_id=target)
            est = EstimateView(raw, n_modes=n_modes, rng=make_rng(self.seed, t, Stage.ESTIMATOR, 500 + i), smooth_iterations=int(get(cfg, "tsdf.smooth_iterations", 10)))
            offsets.append(est.frame.inverse() @ b.pose)  # est frame -> true body frame
            estimates.append(est)
            self.log(f"  scanned {b.name}: {len(views)} views, volume error {100 * (est.volume - b.meta['volume']) / b.meta['volume']:+.1f}%, {len(est.mode_std)} modes")
        return estimates, offsets

    def _observe_pose(self, T: SE3, rng) -> SE3:
        s = self.cfg["sensor"]
        rot = np.radians(float(s.get("pose_noise_rot_deg", 0.3))) * rng.standard_normal(3)
        tr = float(s.get("pose_noise_trans_m", 0.002)) * rng.standard_normal(3)
        return SE3(quat_mul(quat_from_rotvec(rot), T.q), T.t + tr)

    def _place(self, world, body, T_true: SE3, mode: str, rng, decision=None, T_est_body=None) -> bool:
        """Place ``body`` at ``T_true``: lower kinematically from 3 cm above until contact, then release.

        Returns False if the placement failed (no contact reached or the stone fell off).
        """
        from .planning.placement import lower_and_release

        if mode == "arm":
            from .planning.placement import arm_place

            return arm_place(world, body, T_true, self.cfg, rng, grasp=getattr(decision, "grasp", None), T_est_body=T_est_body)
        return lower_and_release(world, body, T_true, self.cfg)


def records_to_rows(records: list[PlacementRecord]) -> list[dict]:
    return [
        {
            "tower": r.tower,
            "placement": r.placement,
            "p_hat": r.p_hat,
            "y": r.y,
            "m_hat": r.m_hat,
            "m_true": r.m_true,
            "samples": np.asarray(r.samples),
            "height": r.height,
            "collapsed": r.collapsed,
            "ms_decision": r.ms_decision,
            "probe": bool(r.meta.get("probe", False)),
        }
        for r in records
    ]


def save_records(records: list[PlacementRecord], path):
    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([r.as_row() for r in records], f, default=float)


def load_records(path) -> list[dict]:
    import json

    with open(path) as f:
        rows = json.load(f)
    for r in rows:
        r["samples"] = np.asarray(r["samples"], dtype=float)
    return rows
