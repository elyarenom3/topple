"""Next best object and pose planner (Section 9).

For each pile stone and candidate pose the planner settles the stone quasi statically on the
tower, builds a static ``Tower`` and evaluates margin samples through a caller supplied
``margin_fn(tower, candidate)``; the objective J = E[M] - kappa Std[M] + lambda_h dh is maximised
under the chance constraint p_hat >= 1 - delta, and the best pose is refined over (x, y, yaw).
Stone selection is greedy by best J, matching the next best object structure of Furrer et al.
(2017) and Liu, Choi and Napp (2021).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.mesh import TriMesh, box
from ..core.transforms import SE3, quat_from_axis_angle, quat_mul
from ..statics.tower import StaticBody, StaticContact, Tower, contacts_from_meshes
from ..statics.survival import survival_probability
from .candidates import sample_candidates, settle_pose, upper_surface_points
from .objective import chance_ok, objective
from .optimise import cma_es, gradient_ascent, nelder_mead
from .resting import resting_orientations


@dataclass
class PlacedStone:
    name: str
    mesh: TriMesh  # world frame
    mass: float
    com: np.ndarray
    mu: float = 0.6
    body_id: int = -1


@dataclass
class TowerState:
    stones: list = field(default_factory=list)
    ground_mu: float = 0.6
    gravity: tuple = (0.0, 0.0, -9.81)
    ground_size: float = 4.0

    def ground_mesh(self) -> TriMesh:
        return box((self.ground_size, self.ground_size, 0.2), center=(0.0, 0.0, -0.1))

    def support_meshes(self) -> list[TriMesh]:
        return [self.ground_mesh()] + [s.mesh for s in self.stones]

    def top_height(self) -> float:
        if not self.stones:
            return 0.0
        return float(max(s.mesh.vertices[:, 2].max() for s in self.stones))

    def top_mesh(self) -> TriMesh:
        return self.ground_mesh() if not self.stones else self.stones[-1].mesh


@dataclass
class CandidatePlacement:
    stone_index: int
    pose: SE3
    settled: object  # SettleResult
    tower: Tower | None = None
    margin_samples: np.ndarray | None = None
    p_hat: float = 0.0
    J: float = -np.inf
    dh: float = 0.0
    feasible: bool = False
    p_raw: float = float("nan")  # Monte Carlo probability before recalibration (when used)


@dataclass
class Decision:
    stone_index: int
    pose: SE3
    grasp: object
    margin_samples: np.ndarray
    p_hat: float
    J: float
    candidates: list
    tower: Tower | None = None
    feasible: bool = True  # False when no candidate met the chance constraint (best effort choice)
    p_raw: float = float("nan")  # screening probability before recalibration (when used)


def _estimate_fields(est):
    """Duck typed access to a stone estimate: mesh (body frame), mass, com, mu."""
    mesh = getattr(est, "mean_mesh", None) or getattr(est, "mesh", None)
    if mesh is None and isinstance(est, dict):
        mesh = est.get("mean_mesh", est.get("mesh"))
    mass = getattr(est, "mass", None)
    if mass is None and isinstance(est, dict):
        mass = est.get("mass")
    if mass is None:
        density = getattr(est, "density", 2650.0) if not isinstance(est, dict) else est.get("density", 2650.0)
        mass = mesh.volume() * density
    com = getattr(est, "com", None) if not isinstance(est, dict) else est.get("com")
    com = mesh.centroid() if com is None else np.asarray(com)
    mu = getattr(est, "mu", None) if not isinstance(est, dict) else est.get("mu")
    if mu is None:
        mus = getattr(est, "mu_samples", None) if not isinstance(est, dict) else est.get("mu_samples")
        mu = float(np.median(mus)) if mus is not None and len(mus) else 0.6
    return mesh, float(mass), com, float(mu)


def build_static_tower(state: TowerState, new_mesh_world: TriMesh | None = None, new_mass: float = 0.0, new_com=None, new_mu: float = 0.6, margin: float = 1.5e-3, max_points: int = 8, rng=None) -> Tower:
    """Static tower from the placed stones plus an optional new stone, with proximity contacts."""
    meshes = [s.mesh for s in state.stones]
    masses = [s.mass for s in state.stones]
    coms = [s.com for s in state.stones]
    mus = [s.mu for s in state.stones]
    names = [s.name for s in state.stones]
    if new_mesh_world is not None:
        meshes.append(new_mesh_world)
        masses.append(new_mass)
        coms.append(new_mesh_world.centroid() if new_com is None else new_com)
        mus.append(new_mu)
        names.append("candidate")
    return Tower.from_meshes(meshes, masses, coms, mus, ground_mu=state.ground_mu, ground_size=state.ground_size, margin=margin, max_points=max_points, rng=rng, names=names, gravity=state.gravity)


class Planner:
    def __init__(self, cfg: dict | None = None, rng=None):
        cfg = cfg or {}
        pc = cfg.get("planner", cfg)
        self.kappa = float(pc.get("kappa", 1.0))
        self.lambda_h = float(pc.get("lambda_h", 2.0))
        self.delta = float(pc.get("delta", 0.05))
        self.n_candidates = int(pc.get("n_candidates", 24))
        self.optimiser = str(pc.get("optimiser", "nelder_mead"))
        self.refine_iters = int(pc.get("refine_iters", 12))
        self.tau_deg = float(cfg.get("propagation", {}).get("survival_tilt_deg", pc.get("survival_tilt_deg", 3.0)))
        self.contact_margin = float(cfg.get("world", {}).get("contact_margin", pc.get("contact_margin", 1.5e-3)))
        self.max_points = int(cfg.get("world", {}).get("max_manifold_points", pc.get("max_manifold_points", 8)))
        self.rng = np.random.default_rng(0) if rng is None else rng
        self._resting_cache: dict = {}
        self.top_min_nz = float(pc.get("top_min_nz", 0.9))
        self.top_height_fraction = float(pc.get("top_height_fraction", 0.25))
        self.top_radius_fraction = float(pc.get("top_radius_fraction", 0.5))
        self.require_feasible = bool(pc.get("require_feasible", False))
        self.keep_pile_orientation = bool(pc.get("keep_pile_orientation", False))
        self.settler = None  # optional SupportWorld for dynamic settling (Section 9.1)
        # optional learned recalibration of the survival probability used in the chance constraint:
        # a fitted eval.recalibration.Recalibrator, and the context it needs (stone classes of the
        # pile in candidate order, experiment settings)
        self.recalibrator = None
        self.recal_context: dict = {}
        self._support_world = None

    # ------------------------------------------------------------------ pieces
    def resting(self, key, mesh: TriMesh):
        if key not in self._resting_cache:
            self._resting_cache[key] = resting_orientations(mesh, n_drops=200, rng=self.rng)
        return self._resting_cache[key]

    def evaluate(self, state: TowerState, est, pose: SE3, margin_fn, stone_index: int = -1) -> CandidatePlacement:
        """Settle one candidate pose and score it."""
        mesh_b, mass, com_b, mu = _estimate_fields(est)
        if self.settler is not None:
            # dynamic settling on the estimated geometry (Section 9.1), dropped from the sampled pose
            settled = self.settler.settle(state, id(est), mesh_b, mass, com_b, mu, pose, world=self._support_world)
        else:
            settled = settle_pose(mesh_b, pose, state.support_meshes(), rng=self.rng, com=com_b)
            if state.stones and len(settled.support_index):
                # a tower is one column: the new stone must rest on the current top stone only
                top_id = len(state.stones)
                settled.stable = bool(settled.stable and np.all(settled.support_index == top_id))
        cand = CandidatePlacement(stone_index, settled.pose, settled)
        mesh_w = mesh_b.transformed(settled.pose)
        cand.dh = float(mesh_w.vertices[:, 2].max() - state.top_height())
        tower = build_static_tower(state, mesh_w, mass, settled.pose.apply(com_b), mu, self.contact_margin, self.max_points, self.rng)
        cand.tower = tower
        samples = np.asarray(margin_fn(tower, cand), dtype=np.float64).reshape(-1)
        cand.margin_samples = samples
        if 1 < samples.size < 32:
            # screening uses few samples: a normal approximation from their mean and spread gives a
            # smoother survival estimate than the empirical fraction (which jumps in steps of 1 / n)
            from scipy.stats import norm

            sd = float(np.std(samples, ddof=1))
            cand.p_hat = float(norm.sf((self.tau_deg - float(np.mean(samples))) / max(sd, 1e-6)))
        else:
            cand.p_hat = survival_probability(samples, self.tau_deg)
        if self.recalibrator is not None:
            cand.p_raw = cand.p_hat
            cand.p_hat = self._recalibrated(state, cand, samples)
        cand.feasible = settled.stable and chance_ok(cand.p_hat, self.delta)
        cand.J = objective(samples, cand.dh, self.kappa, self.lambda_h)
        return cand

    def _recalibrated(self, state: TowerState, cand: CandidatePlacement, samples) -> float:
        from ..eval.recalibration import candidate_record

        classes = self.recal_context.get("classes") or []
        cls = classes[cand.stone_index] if 0 <= cand.stone_index < len(classes) else None
        nc = cand.tower.n_contacts if cand.tower is not None else 0
        rec = candidate_record(cand.p_hat, samples, len(state.stones), state.top_height() + cand.dh, nc, cls, self.recal_context.get("settings", {}))
        return float(self.recalibrator.predict([rec])[0])

    def refine(self, state: TowerState, est, cand: CandidatePlacement, margin_fn) -> CandidatePlacement:
        """Refine (x, y, yaw) of the best candidate with the configured optimiser."""
        base = cand.pose
        best = cand

        def pose_of(x):
            q = quat_mul(quat_from_axis_angle([0, 0, 1], x[2]), base.q)
            t = base.t + np.array([x[0], x[1], 0.02])
            return SE3(q, t)

        cache = {}

        def f(x):
            key = tuple(np.round(x, 5))
            if key in cache:
                return cache[key]
            c = self.evaluate(state, est, pose_of(x), margin_fn, cand.stone_index)
            val = c.J if c.feasible else c.J - 1e3
            cache[key] = (val, c)
            return val

        def fval(x):
            v = f(x)
            return v[0] if isinstance(v, tuple) else v

        x0 = np.zeros(3)
        bounds = (np.array([-0.02, -0.02, -0.3]), np.array([0.02, 0.02, 0.3]))
        if self.optimiser == "cma_es":
            x, _, _ = cma_es(fval, x0, sigma0=0.008, iters=self.refine_iters, rng=self.rng, bounds=bounds)
        elif self.optimiser == "gradient":
            def grad(x, h=2e-3):
                g = np.zeros(3)
                for k in range(3):
                    e = np.zeros(3)
                    e[k] = h
                    g[k] = (fval(x + e) - fval(x - e)) / (2 * h)
                return g

            x, _, _ = gradient_ascent(fval, grad, x0, step=0.002, iters=self.refine_iters, bounds=bounds)
        else:
            x, _, _ = nelder_mead(fval, x0, step=np.array([0.005, 0.005, 0.1]), iters=self.refine_iters, bounds=bounds)
        for val, c in cache.values():
            if c.feasible and c.J > (best.J if best.feasible else -np.inf):
                best = c
        if not best.feasible:
            # keep the candidate with the highest J even if infeasible
            allc = [cand] + [c for _, c in cache.values()]
            best = max(allc, key=lambda c: c.J)
        return best

    def clearance_fn(self, state: TowerState, cand: CandidatePlacement, mesh_b: TriMesh):
        """Finger clearance at the target pose: distance of the finger tips from the supports."""
        supports = state.support_meshes()

        def fn(grasp, target_pose):
            T = cand.pose
            p1, p2 = T.apply(grasp.p1), T.apply(grasp.p2)
            tips = np.array([p1 - 0.01 * T.apply_vec(grasp.axis), p2 + 0.01 * T.apply_vec(grasp.axis)])
            best = np.inf
            for m in supports:
                best = min(best, float(m.signed_distance(tips).min()))
            return float(np.clip(best, -0.02, 0.05)) / 0.05

        return fn

    def plan_grasp(self, est, state: TowerState, cand: CandidatePlacement, shape_samples=None):
        from .grasp import antipodal_grasps, rank_grasps

        mesh_b, mass, com_b, mu = _estimate_fields(est)
        grasps = antipodal_grasps(mesh_b, mu_g=1.0, n_pairs=300, rng=self.rng)
        if not grasps:
            return None
        ranked = rank_grasps(grasps, com_b, cand.pose, self.clearance_fn(state, cand, mesh_b), shape_samples=shape_samples, rng=self.rng)
        return ranked[0]

    # ------------------------------------------------------------------ main entry
    def choose(self, tower_estimate: TowerState, pile_estimates: list, margin_fn, placed: set | None = None, refine: bool = True, with_grasp: bool = True) -> Decision | None:
        """Choose the next stone and pose.

        ``margin_fn(tower, candidate)`` returns margin samples (degrees) for a static ``Tower``
        containing the candidate as its top body. Returns None when no stone can be placed.
        """
        placed = placed or set()
        if self.settler is not None:
            self._support_world = self.settler.world_for(tower_estimate)
        if tower_estimate.stones:
            # candidate centres over the inner part of the top stone: upward facing points whose
            # horizontal distance from the top stone's centroid is within a fraction of its radius
            top = tower_estimate.top_mesh()
            pts = upper_surface_points(top, 400, self.rng, min_nz=self.top_min_nz)
            c = top.centroid()
            r_h = float(np.max(np.linalg.norm(top.vertices[:, :2] - c[:2], axis=1)))
            d = np.linalg.norm(pts[:, :2] - c[:2], axis=1)
            keep = d < self.top_radius_fraction * r_h
            # prefer the highest, most horizontal region: keep points in the upper part of the top
            # stone's height range (a domed top is near horizontal only around its apex)
            zt = float(top.vertices[:, 2].max())
            hz = zt - float(top.vertices[:, 2].min())
            keep &= pts[:, 2] > zt - self.top_height_fraction * hz
            top_pts = pts[keep] if keep.sum() >= 4 else np.array([[c[0], c[1], zt]])
        else:
            top_pts = np.array([[0.0, 0.0, 0.0]]) + self.rng.uniform(-0.02, 0.02, size=(32, 3)) * np.array([1, 1, 0])
        all_candidates = []
        best = None
        for si, est in enumerate(pile_estimates):
            if si in placed:
                continue
            mesh_b, mass, com_b, mu = _estimate_fields(est)
            if self.keep_pile_orientation and getattr(est, "frame", None) is not None:
                # arm placement without regrasping: the stone keeps its pile orientation up to a yaw,
                # so a grasp axis that is horizontal at the pick stays horizontal at the place
                resting = [(np.asarray(est.frame.q, dtype=float), 1.0)]
            else:
                resting = self.resting(id(est), mesh_b)
            poses = sample_candidates(top_pts, mesh_b, self.n_candidates, self.rng, resting=resting)
            for pose in poses:
                cand = self.evaluate(tower_estimate, est, pose, margin_fn, si)
                all_candidates.append(cand)
                if cand.feasible and (best is None or cand.J > best.J):
                    best = cand
        feasible = best is not None
        if best is None:
            if not all_candidates or self.require_feasible:
                return None
            # no candidate meets the chance constraint: take the best stable one, else the best J
            stable = [c for c in all_candidates if getattr(c.settled, "stable", False)]
            best = max(stable or all_candidates, key=lambda c: c.J)
        if refine:
            best = self.refine(tower_estimate, pile_estimates[best.stone_index], best, margin_fn)
        grasp = self.plan_grasp(pile_estimates[best.stone_index], tower_estimate, best) if with_grasp else None
        if refine and feasible:
            feasible = bool(best.feasible)
        return Decision(best.stone_index, best.pose, grasp, best.margin_samples, best.p_hat, best.J, all_candidates, best.tower, feasible, best.p_raw)
