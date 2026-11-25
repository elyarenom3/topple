"""Dynamic candidate settling on the estimated geometry (Section 9.1).

"Each candidate is settled in a short simulation on the estimated geometry." A support world
holds the ground and the placed stones (posterior mean meshes at their observed poses) as static
bodies. For each candidate the stone (posterior mean mesh in its estimate frame, estimated mass,
principal inertia and friction) is placed at the candidate orientation, lowered by conservative
advancement until it touches, released at rest and stepped until its linear and angular speeds
fall below thresholds or ``max_steps`` is reached. The quasi static tipping of
``candidates.settle_pose`` serves as the initial guess, so most candidates come to rest within a
few hundred steps.

The support stones are static here: the settling answers "where does the new stone come to rest",
and whether the tower below survives is the job of the margin analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.mesh import TriMesh
from ..core.transforms import SE3
from ..physics.body import CollisionShape, RigidBody
from ..physics.world import World
from .candidates import SettleResult


def _static_body_from_world_mesh(name: str, mesh_world: TriMesh, mu: float) -> RigidBody:
    shape = CollisionShape.from_mesh(mesh_world, sdf_spacing=None)
    b = RigidBody(name, np.inf, np.diag([np.inf] * 3), shape, friction=mu, static=True)
    return b


class SupportWorld:
    """Ground plus the placed stones as static bodies, cached across candidates and decisions."""

    def __init__(self, cfg: dict | None = None, ground_mu: float = 0.6):
        cfg = cfg or {}
        wc = dict(cfg.get("world", {}))
        wc["merging"] = False
        self.cfg = {"world": wc}
        self.ground_mu = float(ground_mu)
        self._static_cache: dict = {}  # name -> RigidBody (static, world frame)
        self._stone_cache: dict = {}  # key -> RigidBody template (dynamic, estimate frame)
        ps = cfg.get("planner", {})
        self.max_steps = int(ps.get("settle_steps", 800))
        self.v_tol = float(ps.get("settle_v_tol", 2e-3))
        self.w_tol = float(ps.get("settle_w_tol", 2e-2))
        self.min_steps = int(ps.get("settle_min_steps", 100))
        self.rest_dx = float(ps.get("settle_rest_dx", 1e-4))  # m over 50 steps
        self.damping = float(ps.get("settle_damping", 60.0))  # 1/s during settling only (author's choice)
        self.rest_dth = float(np.radians(ps.get("settle_rest_dth_deg", 0.1)))

    def world_for(self, state) -> World:
        w = World(self.cfg)
        w.add_body(RigidBody.ground(friction=self.ground_mu))
        for s in state.stones:
            key = (s.name, round(float(s.mesh.vertices[0, 0]), 9), round(float(s.mesh.vertices[0, 2]), 9))
            if key not in self._static_cache:
                self._static_cache[key] = _static_body_from_world_mesh(s.name, s.mesh, s.mu)
            w.add_body(self._static_cache[key].copy())
        return w

    def stone_body(self, key, mesh_body: TriMesh, mass: float, com_body: np.ndarray, mu: float) -> RigidBody:
        """Dynamic template body for a stone estimate. The estimate frame mesh is re expressed about
        the estimated centre of mass so the body frame origin is the centre of mass."""
        if key not in self._stone_cache:
            m = mesh_body.translated(-np.asarray(com_body, dtype=float))
            density = mass / max(m.volume(), 1e-12)
            mp = m.mass_properties(density)
            shape = CollisionShape.from_mesh(m, sdf_spacing=None)
            b = RigidBody("candidate", mp["mass"], mp["inertia_com"], shape, friction=mu)
            b.meta["com_body"] = np.asarray(com_body, dtype=float).copy()
            self._stone_cache[key] = b
        return self._stone_cache[key]

    def settle(self, state, key, mesh_body: TriMesh, mass: float, com_body, mu: float, pose: SE3, world: World | None = None) -> SettleResult:
        """Settle the stone dropped at ``pose`` (estimate frame) and return the rest pose (estimate frame)."""
        from .placement import _min_separation

        base = self.world_for(state) if world is None else world
        w = base.clone()
        tmpl = self.stone_body(key, mesh_body, mass, com_body, mu)
        b = tmpl.copy()
        com_b = b.meta["com_body"]
        # body frame = estimate frame translated to the centre of mass
        T_body = SE3(pose.q, pose.apply(com_b))
        b.pose = SE3(T_body.q, T_body.t + np.array([0.0, 0.0, 0.01]))
        bid = w.add_body(b)
        for k in range(len(w.bodies) - 1):
            w.set_pair_friction(k, bid, np.sqrt(w.bodies[k].friction * mu) if k > 0 else np.sqrt(self.ground_mu * mu))
        # conservative advancement to first touch
        z = float(b.x[2])
        for _ in range(100):
            sep = _min_separation(w, b, 0.01)
            if sep <= 2e-4:
                break
            step = 0.009 if not np.isfinite(sep) else max(sep - 1e-4, 5e-5)
            z -= step
            b.x = np.array([b.x[0], b.x[1], z])
            if z < -0.5:
                break
        b.v[:] = 0
        b.w[:] = 0
        w.damping = self.damping
        steps = 0
        at_rest = False
        window = 50
        x_prev, q_prev = b.x.copy(), b.q.copy()
        while steps < self.max_steps:
            w.step(window)
            steps += window
            # rest by displacement over the window: solver chatter can keep instantaneous speeds
            # around 0.1 rad/s while the stone moves less than a few hundredths of a millimetre
            dx = float(np.linalg.norm(b.x - x_prev))
            dth = 2.0 * float(np.arccos(min(1.0, abs(float(np.dot(b.q, q_prev))))))
            x_prev, q_prev = b.x.copy(), b.q.copy()
            slow = np.linalg.norm(b.v) < self.v_tol and np.linalg.norm(b.w) < self.w_tol
            still = dx < self.rest_dx and dth < self.rest_dth
            if steps >= self.min_steps and (slow or still):
                at_rest = True
                break
        T_est = SE3(b.q.copy(), b.x - _rot(b.q) @ com_b)
        contacts = [c for c in w.contacts if bid in (c.body_a, c.body_b)]
        pts = np.array([c.point for c in contacts]) if contacts else np.zeros((0, 3))
        nrm = np.array([c.normal if c.body_b == bid else -c.normal for c in contacts]) if contacts else np.zeros((0, 3))
        sup = np.array([c.body_a if c.body_b == bid else c.body_b for c in contacts], dtype=int) if contacts else np.zeros(0, dtype=int)
        fell = float(b.x[2]) < 0.5 * float(T_body.t[2]) - 0.02
        return SettleResult(T_est, pts, nrm, sup, bool(at_rest and len(contacts) >= 1 and not fell), steps, 0.0)


def _rot(q):
    from ..core.transforms import quat_to_rot

    return quat_to_rot(q)
