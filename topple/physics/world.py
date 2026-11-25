"""The rigid body world (Section 6): bodies, contacts and the time step.

Each step: unconstrained velocities (gravity, external forces, implicit midpoint
gyroscopic update), contact detection with the proximity margin, warm started impulse
solve (PGS by default, Lemke reference, or the XPBD secondary solver), and position
update with exact quaternion integration.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from numba import njit

from ..core.transforms import quat_integrate, quat_rotate, quat_to_rot
from .body import Contact, RigidBody, WorldState
from .collision import CollisionCache, find_contacts
from .solver import apply_impulses, cone_residuals, contact_arrays, pgs_solve, solve_lemke, stabilisation_bias


@njit(cache=True)
def _gyro_implicit_midpoint(I_body, w_body, h, iters=3):
    """Solve I (w+ - w) + h wm x (I wm) = 0 with wm = (w + w+)/2 by Newton's method.

    The implicit midpoint rule conserves the quadratic invariants (kinetic energy and
    the angular momentum magnitude) of torque free rotation.
    """
    w0 = w_body
    w1 = w0.copy()
    Jf = np.empty((3, 3))
    for _ in range(iters):
        wm = 0.5 * (w0 + w1)
        Iwm = I_body @ wm
        cr = np.empty(3)
        cr[0] = wm[1] * Iwm[2] - wm[2] * Iwm[1]
        cr[1] = wm[2] * Iwm[0] - wm[0] * Iwm[2]
        cr[2] = wm[0] * Iwm[1] - wm[1] * Iwm[0]
        f = I_body @ (w1 - w0) + h * cr
        # d/dw1 of h * wm x (I wm) = 0.5 h [skew(wm) I - skew(I wm)]
        for i in range(3):
            for j in range(3):
                Jf[i, j] = I_body[i, j]
        Sw = np.array([[0.0, -wm[2], wm[1]], [wm[2], 0.0, -wm[0]], [-wm[1], wm[0], 0.0]])
        SI = np.array([[0.0, -Iwm[2], Iwm[1]], [Iwm[2], 0.0, -Iwm[0]], [-Iwm[1], Iwm[0], 0.0]])
        Jf += 0.5 * h * (Sw @ I_body - SI)
        w1 = w1 - np.linalg.solve(Jf, f)
    return w1


@njit(cache=True)
def _integrate_free(q, v, w, dyn, I_body, g, f_ext, tau_ext, inv_m, h):
    """Unconstrained velocity update for all dynamic bodies (gravity, external wrench,
    implicit midpoint gyroscopic term)."""
    n = q.shape[0]
    for i in range(n):
        if not dyn[i]:
            continue
        for k in range(3):
            v[i, k] += h * g[k] + h * inv_m[i] * f_ext[i, k]
        R = _quat_to_rot_nb(q[i])
        wb = R.T @ w[i]
        wb = _gyro_implicit_midpoint(I_body[i], wb, h)
        if tau_ext[i, 0] != 0.0 or tau_ext[i, 1] != 0.0 or tau_ext[i, 2] != 0.0:
            wb = wb + h * np.linalg.solve(I_body[i], R.T @ tau_ext[i])
        w[i] = R @ wb


@njit(cache=True)
def _quat_to_rot_nb(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    R = np.empty((3, 3))
    R[0, 0] = 1 - 2 * (y * y + z * z)
    R[0, 1] = 2 * (x * y - z * w)
    R[0, 2] = 2 * (x * z + y * w)
    R[1, 0] = 2 * (x * y + z * w)
    R[1, 1] = 1 - 2 * (x * x + z * z)
    R[1, 2] = 2 * (y * z - x * w)
    R[2, 0] = 2 * (x * z - y * w)
    R[2, 1] = 2 * (y * z + x * w)
    R[2, 2] = 1 - 2 * (x * x + y * y)
    return R


@njit(cache=True)
def _integrate_positions(q, x, v, w, movable, dyn, I_body, L_target, h):
    """Position update with exact quaternion integration. Dynamic bodies then take the
    angular velocity that realises the target angular momentum in the new orientation,
    so that angular momentum is conserved exactly through the rotation."""
    n = q.shape[0]
    for i in range(n):
        if not movable[i]:
            continue
        for k in range(3):
            x[i, k] += h * v[i, k]
        ang = math.sqrt(w[i, 0] ** 2 + w[i, 1] ** 2 + w[i, 2] ** 2)
        if ang > 0.0:
            half = 0.5 * ang * h
            s = math.sin(half) / ang
            aw, ax, ay, az = math.cos(half), w[i, 0] * s, w[i, 1] * s, w[i, 2] * s
            bw, bx, by, bz = q[i, 0], q[i, 1], q[i, 2], q[i, 3]
            nw = aw * bw - ax * bx - ay * by - az * bz
            nx = aw * bx + ax * bw + ay * bz - az * by
            ny = aw * by - ax * bz + ay * bw + az * bx
            nz = aw * bz + ax * by - ay * bx + az * bw
            nn = math.sqrt(nw * nw + nx * nx + ny * ny + nz * nz)
            q[i, 0] = nw / nn
            q[i, 1] = nx / nn
            q[i, 2] = ny / nn
            q[i, 3] = nz / nn
        if dyn[i]:
            R = _quat_to_rot_nb(q[i])
            Lb = R.T @ L_target[i]
            wb = np.linalg.solve(I_body[i], Lb)
            w[i] = R @ wb


class World:
    def __init__(
        self,
        cfg: Optional[dict] = None,
        gravity=(0.0, 0.0, -9.81),
        dt: float = 1e-3,
        solver: str = "pgs",
        cone_sides: int = 8,
        pgs_iterations: int = 100,
        contact_margin: float = 1.5e-3,
        max_manifold_points: int = 8,
        merging: bool = False,
        baumgarte: float = 0.2,
        slop: float = 5e-4,
        restitution: float = 0.0,
        impact_mode: str = "simultaneous",
        pgs_tol: float = 1e-9,
        torsion_radius: float = 0.002,
    ):
        w = (cfg or {}).get("world", {}) if cfg else {}
        self.gravity = np.asarray(w.get("gravity", gravity), dtype=np.float64).copy()
        self.dt = float(w.get("dt", dt))
        self.solver = str(w.get("solver", solver))
        self.cone_sides = int(w.get("cone_sides", cone_sides))
        self.pgs_iterations = int(w.get("pgs_iterations", pgs_iterations))
        self.contact_margin = float(w.get("contact_margin", contact_margin))
        self.max_manifold_points = int(w.get("max_manifold_points", max_manifold_points))
        self.merging = bool(w.get("merging", merging))
        self.baumgarte = float(w.get("baumgarte", baumgarte))
        self.slop = float(w.get("slop", slop))
        self.restitution = float(w.get("restitution", restitution))
        self.impact_mode = str(w.get("impact_mode", impact_mode))
        self.pgs_tol = float(pgs_tol)
        # radius (m) of the contact patch that resists spin about the normal (soft contact model,
        # author's choice); 0 recovers point contacts with no torsional friction
        self.torsion_radius = float(w.get("torsion_radius", torsion_radius))
        self.restitution_threshold = 0.05  # m/s, approach speed below which impacts are inelastic
        self.bodies: List[RigidBody] = []
        self.time = 0.0
        self.contacts: List[Contact] = []
        self.last_residuals: dict = {"max_cone_violation": 0.0, "max_complementarity": 0.0, "pgs_iterations": 0}
        self.recorder = None
        self._pair_mu: Dict[Tuple[int, int], float] = {}
        self._collision_cache = CollisionCache(fat_margin=0.01)
        self._warm: Dict[int, np.ndarray] = {}
        self._warm_t: Dict[int, float] = {}
        self._ext_force: Dict[int, np.ndarray] = {}
        self._ext_torque: Dict[int, np.ndarray] = {}
        self._skip_internal = False
        self.merge_manager = None
        if self.merging:
            from .merging import MergeManager

            self.merge_manager = MergeManager(self)
        self.step_count = 0
        # velocity damping (1/s); zero for all dynamic tests. The settling procedures switch it on
        # to remove the persistent rocking that rigid rolling contacts cannot dissipate.
        self.damping = 0.0

    # ------------------------------------------------------------------ bodies
    def add_body(self, body: RigidBody) -> int:
        body.id = len(self.bodies)
        self.bodies.append(body)
        return body.id

    def remove_body(self, body_id: int) -> None:
        """Remove one body; later ids shift down by one and the pair friction table is remapped."""
        self.bodies.pop(body_id)
        for i, b in enumerate(self.bodies):
            b.id = i

        def remap(i):
            return i - 1 if i > body_id else i

        self._pair_mu = {(remap(a), remap(b)): mu for (a, b), mu in self._pair_mu.items() if a != body_id and b != body_id}
        self._warm = {}
        self._collision_cache = CollisionCache(fat_margin=0.01)

    def subset(self, keep_ids) -> tuple["World", dict]:
        """Clone holding only the listed bodies (the ground is kept when listed). Returns the new
        world and a map from old ids to new ids. Used by the harness so that the dynamic tests run
        on the tower alone, without the remaining pile stones."""
        keep = sorted(set(int(i) for i in keep_ids))
        w = self.clone_empty()
        old_to_new = {}
        for old in keep:
            old_to_new[old] = w.add_body(self.bodies[old].copy())
        for (a, b), mu in self._pair_mu.items():
            if a in old_to_new and b in old_to_new:
                na, nb = old_to_new[a], old_to_new[b]
                w._pair_mu[(min(na, nb), max(na, nb))] = mu
        w.time = self.time
        return w, old_to_new

    def clone_empty(self) -> "World":
        """A world with the same parameters and no bodies."""
        w = World(
            gravity=self.gravity,
            dt=self.dt,
            solver=self.solver,
            cone_sides=self.cone_sides,
            pgs_iterations=self.pgs_iterations,
            contact_margin=self.contact_margin,
            max_manifold_points=self.max_manifold_points,
            merging=False,
            baumgarte=self.baumgarte,
            slop=self.slop,
            restitution=self.restitution,
            impact_mode=self.impact_mode,
            pgs_tol=self.pgs_tol,
            torsion_radius=self.torsion_radius,
        )
        w.restitution_threshold = self.restitution_threshold
        return w

    def body_by_name(self, name: str) -> RigidBody:
        for b in self.bodies:
            if b.name == name:
                return b
        raise KeyError(name)

    def set_pair_friction(self, id_a: int, id_b: int, mu: float) -> None:
        self._pair_mu[(min(id_a, id_b), max(id_a, id_b))] = float(mu)

    def pair_friction(self, id_a: int, id_b: int) -> float:
        key = (min(id_a, id_b), max(id_a, id_b))
        if key in self._pair_mu:
            return self._pair_mu[key]
        return math.sqrt(self.bodies[id_a].friction * self.bodies[id_b].friction)

    def set_kinematic_velocity(self, body_id: int, v, w) -> None:
        b = self.bodies[body_id]
        b.kinematic = True
        b.static = False
        b.v = np.asarray(v, dtype=np.float64).copy()
        b.w = np.asarray(w, dtype=np.float64).copy()

    def external_force(self, body_id: int, f, p_world=None) -> None:
        """Apply force f (N) at world point p for the next step only."""
        f = np.asarray(f, dtype=np.float64)
        self._ext_force[body_id] = self._ext_force.get(body_id, np.zeros(3)) + f
        if p_world is not None:
            r = np.asarray(p_world, dtype=np.float64) - self.bodies[body_id].x
            self._ext_torque[body_id] = self._ext_torque.get(body_id, np.zeros(3)) + np.cross(r, f)

    def external_torque(self, body_id: int, tau) -> None:
        self._ext_torque[body_id] = self._ext_torque.get(body_id, np.zeros(3)) + np.asarray(tau, dtype=np.float64)

    # ------------------------------------------------------------------- state
    def state(self) -> WorldState:
        n = len(self.bodies)
        q = np.empty((n, 4))
        x = np.empty((n, 3))
        v = np.empty((n, 3))
        w = np.empty((n, 3))
        for i, b in enumerate(self.bodies):
            q[i] = b.q
            x[i] = b.x
            v[i] = b.v
            w[i] = b.w
        return WorldState(self.time, q, x, v, w)

    def set_state(self, s: WorldState) -> None:
        for i, b in enumerate(self.bodies):
            b.q = s.q[i].copy()
            b.x = s.x[i].copy()
            b.v = s.v[i].copy()
            b.w = s.w[i].copy()
        self.time = float(s.time)

    def clone(self) -> "World":
        w = World(
            gravity=self.gravity,
            dt=self.dt,
            solver=self.solver,
            cone_sides=self.cone_sides,
            pgs_iterations=self.pgs_iterations,
            contact_margin=self.contact_margin,
            max_manifold_points=self.max_manifold_points,
            merging=False,
            baumgarte=self.baumgarte,
            slop=self.slop,
            restitution=self.restitution,
            impact_mode=self.impact_mode,
            pgs_tol=self.pgs_tol,
            torsion_radius=self.torsion_radius,
        )
        for b in self.bodies:
            w.add_body(b.copy())
        w.time = self.time
        w.damping = self.damping
        w._pair_mu = dict(self._pair_mu)
        w._warm = {k: v.copy() for k, v in self._warm.items()}
        w._warm_t = dict(self._warm_t)
        w.restitution_threshold = self.restitution_threshold
        if self.merging:
            w.merging = True
            from .merging import MergeManager

            w.merge_manager = MergeManager(w)
        return w

    def kinetic_energy(self) -> float:
        return float(sum(b.kinetic_energy() for b in self.bodies))

    def linear_momentum(self) -> np.ndarray:
        return sum((b.mass * b.v for b in self.bodies if np.isfinite(b.mass) and not b.kinematic), np.zeros(3))

    def angular_momentum(self, about=None) -> np.ndarray:
        o = np.zeros(3) if about is None else np.asarray(about)
        L = np.zeros(3)
        for b in self.bodies:
            if not np.isfinite(b.mass) or b.kinematic:
                continue
            L += b.inertia_world() @ b.w + np.cross(b.x - o, b.mass * b.v)
        return L

    # -------------------------------------------------------------------- step
    def step(self, n: int = 1) -> None:
        for _ in range(int(n)):
            self._step_once()

    def _gather(self):
        bodies = self.bodies
        n = len(bodies)
        q = np.empty((n, 4))
        x = np.empty((n, 3))
        v = np.empty((n, 3))
        w = np.empty((n, 3))
        dyn = np.zeros(n, dtype=np.bool_)
        movable = np.zeros(n, dtype=np.bool_)
        inv_m = np.zeros(n)
        I_body = np.zeros((n, 3, 3))
        for i, b in enumerate(bodies):
            q[i] = b.q
            x[i] = b.x
            v[i] = b.v
            w[i] = b.w
            if not b.static:
                movable[i] = True
            if b.inv_mass > 0.0:
                dyn[i] = True
                inv_m[i] = b.inv_mass
                I_body[i] = b.inertia_body
            else:
                I_body[i] = np.eye(3)
        return q, x, v, w, dyn, movable, inv_m, I_body

    def _unconstrained_velocities(self, h: float):
        q, x, v, w, dyn, movable, inv_m, I_body = self._gather()
        n = len(self.bodies)
        f_ext = np.zeros((n, 3))
        tau_ext = np.zeros((n, 3))
        for bid, f in self._ext_force.items():
            f_ext[bid] = f
        for bid, t in self._ext_torque.items():
            tau_ext[bid] = t
        _integrate_free(q, v, w, dyn, I_body, self.gravity, f_ext, tau_ext, inv_m, h)
        f = math.exp(-self.damping * h) if self.damping > 0.0 else 1.0
        for i, b in enumerate(self.bodies):
            if dyn[i]:
                b.v = v[i] * f
                b.w = w[i] * f

    def _solver_arrays(self):
        nb = len(self.bodies)
        x = np.empty((nb, 3))
        v = np.empty((nb, 3))
        w = np.empty((nb, 3))
        inv_m = np.zeros(nb)
        inv_I = np.zeros((nb, 3, 3))
        for i, b in enumerate(self.bodies):
            x[i] = b.x
            v[i] = b.v
            w[i] = b.w
            inv_m[i] = b.inv_mass
            if inv_m[i] > 0.0:
                R = quat_to_rot(b.q)
                inv_I[i] = R @ b.inv_inertia_body @ R.T
        return x, v, w, inv_m, inv_I

    def _step_once(self) -> None:
        h = self.dt
        if self.solver == "xpbd":
            from .xpbd import step_xpbd

            step_xpbd(self)
            self._finish_step()
            return
        # angular momentum before the step (world frame) for exact conservation
        q, x, v, w, dyn, movable, inv_m, I_body = self._gather()
        nb = len(self.bodies)
        L0 = np.zeros((nb, 3))
        Iw_old = np.zeros((nb, 3, 3))
        for i in range(nb):
            if dyn[i]:
                R = quat_to_rot(q[i])
                Iw_old[i] = R @ I_body[i] @ R.T
                L0[i] = Iw_old[i] @ w[i]
        w_before = w.copy()
        self._unconstrained_velocities(h)
        w_unc = np.array([b.w for b in self.bodies])
        contacts = find_contacts(self, self.contact_margin)
        if self.merge_manager is not None:
            contacts = self.merge_manager.before_solve(contacts)
        self.contacts = contacts
        if contacts:
            self._solve_contacts(contacts, h)
        else:
            self.last_residuals = {"max_cone_violation": 0.0, "max_complementarity": 0.0, "pgs_iterations": 0, "max_penetration_velocity": 0.0}
        if self.merge_manager is not None:
            self.merge_manager.after_solve()
        # integrate positions
        q, x, v, w, dyn, movable, inv_m, I_body = self._gather()
        L_target = np.zeros((nb, 3))
        for i in range(nb):
            if dyn[i]:
                L_target[i] = L0[i] + Iw_old[i] @ (w[i] - w_unc[i])
                tau = self._ext_torque.get(i)
                if tau is not None:
                    L_target[i] += h * tau
        # rotate with the midpoint of the gyroscopic update (second order), keeping the
        # contact impulse part in full so that constraints are satisfied at the new positions
        w_rot = w.copy()
        for i in range(nb):
            if dyn[i]:
                w_rot[i] = w[i] - 0.5 * (w_unc[i] - w_before[i])
        _integrate_positions(q, x, v, w_rot, movable, dyn, I_body, L_target, h)
        w = w_rot
        for i, b in enumerate(self.bodies):
            if movable[i]:
                b.x = x[i]
                b.q = q[i]
                if dyn[i]:
                    b.w = w[i]
        self._finish_step()

    def _finish_step(self):
        self.time += self.dt
        self.step_count += 1
        self._ext_force = {}
        self._ext_torque = {}
        if self.recorder is not None:
            self.recorder.append(self.state())

    def _solve_contacts(self, contacts: List[Contact], h: float) -> None:
        body_index = {b.id: i for i, b in enumerate(self.bodies)}
        if self.merge_manager is not None and self.merge_manager.active():
            x, v, w, inv_m, inv_I, body_index = self.merge_manager.aggregate_arrays()
        else:
            x, v, w, inv_m, inv_I = self._solver_arrays()
        ia, ib, p, n, t1, t2, mu, depth = contact_arrays(contacts, body_index)
        bias = stabilisation_bias(depth, h, self.baumgarte, self.slop)
        nc = len(contacts)
        if self.restitution > 0.0:
            if self.impact_mode == "sequential":
                # resolve impacts one at a time, then solve the rest as inelastic contacts
                self._sequential_impacts(ia, ib, p, n, x, v, w, inv_m, inv_I)
            else:
                # simultaneous treatment: restitution targets on approaching contacts
                for c in range(nc):
                    a, b = ia[c], ib[c]
                    dv = (v[b] + np.cross(w[b], p[c] - x[b])) - (v[a] + np.cross(w[a], p[c] - x[a]))
                    vn = float(dv @ n[c])
                    if vn < -self.restitution_threshold:
                        bias[c] = max(bias[c], -self.restitution * vn)
        lam = np.zeros((nc, 3))
        # warm start
        for c, ct in enumerate(contacts):
            prev = self._warm.get(ct.feature)
            if prev is not None:
                lam[c] = prev
        if np.any(lam != 0.0):
            apply_impulses(ia, ib, p, n, t1, t2, x, v, w, inv_m, inv_I, lam)
        lam_t = np.zeros(nc)
        if self.solver == "lemke":
            lam, status = solve_lemke(ia, ib, p, n, t1, t2, bias, mu, x, v, w, inv_m, inv_I, self.cone_sides)
            iters = status
        else:
            if self.torsion_radius > 0.0:
                # translational rows plus the spin friction of the contact patches in one sweep
                from .solver import apply_torsion, pgs_solve_t, torsion_rows_mass

                kt = torsion_rows_mass(ia, ib, n, inv_I)
                for c, ct in enumerate(contacts):
                    lam_t[c] = self._warm_t.get(ct.feature, 0.0)
                apply_torsion(ia, ib, n, w, inv_I, lam_t)
                iters = pgs_solve_t(ia, ib, p, n, t1, t2, bias, mu, x, v, w, inv_m, inv_I, lam, self.pgs_iterations, self.cone_sides, self.pgs_tol, lam_t, kt, self.torsion_radius)
            else:
                iters = pgs_solve(ia, ib, p, n, t1, t2, bias, mu, x, v, w, inv_m, inv_I, lam, self.pgs_iterations, self.cone_sides, self.pgs_tol)
        max_cone, max_comp, max_pen = cone_residuals(ia, ib, p, n, t1, t2, bias, mu, x, v, w, lam, self.cone_sides)
        self.last_residuals = {"max_cone_violation": float(max_cone), "max_complementarity": float(max_comp), "max_penetration_velocity": float(max_pen), "pgs_iterations": int(iters)}
        # scatter velocities
        if self.merge_manager is not None and self.merge_manager.active():
            self.merge_manager.scatter_velocities(v, w)
        else:
            for i, b in enumerate(self.bodies):
                if b.inv_mass > 0.0:
                    b.v = v[i].copy()
                    b.w = w[i].copy()
        self._warm = {}
        self._warm_t = {}
        for c, ct in enumerate(contacts):
            ct.fn, ct.ft1, ct.ft2 = float(lam[c, 0]), float(lam[c, 1]), float(lam[c, 2])
            self._warm[ct.feature] = lam[c].copy()
            if lam_t[c] != 0.0:
                self._warm_t[ct.feature] = float(lam_t[c])

    def _sequential_impacts(self, ia, ib, p, n, x, v, w, inv_m, inv_I, max_rounds: int = 200):
        """Sequential impact treatment: resolve the most approaching body pair one at a time
        with the restitution law until no pair approaches (Newton's cradle test).

        The contact points of one pair form a manifold on a shared face, so each pair is
        treated as one impact at the manifold centroid along the mean normal. Resolving the
        corner points one by one would inject spurious spin into face contacts."""
        from .solver import _apply_impulse, _row_mass

        e = self.restitution
        pairs: dict = {}
        for c in range(ia.shape[0]):
            pairs.setdefault((int(ia[c]), int(ib[c])), []).append(c)
        groups = []
        for (a, b), cs in pairs.items():
            pc = p[cs].mean(axis=0)
            nc = n[cs].mean(axis=0)
            nn = np.linalg.norm(nc)
            if nn < 1e-12:
                continue
            groups.append((a, b, pc, nc / nn))
        for _ in range(max_rounds):
            worst, wg = -self.restitution_threshold, -1
            for g, (a, b, pc, nc) in enumerate(groups):
                dv = (v[b] + np.cross(w[b], pc - x[b])) - (v[a] + np.cross(w[a], pc - x[a]))
                vn = float(dv @ nc)
                if vn < worst:
                    worst, wg = vn, g
            if wg < 0:
                break
            a, b, pc, nc = groups[wg]
            ra, rb = pc - x[a], pc - x[b]
            k = _row_mass(inv_m, inv_I, a, b, ra, rb, nc)
            j = -(1.0 + e) * worst / k
            imp = j * nc
            _apply_impulse(v, w, inv_m, inv_I, a, ra, imp, -1.0)
            _apply_impulse(v, w, inv_m, inv_I, b, rb, imp, 1.0)

    # ---------------------------------------------------------------- helpers
    def max_body_speed(self) -> float:
        return max((float(np.linalg.norm(b.v)) for b in self.bodies if not b.static and not b.kinematic), default=0.0)

    def damped_settle(self, ids=None, seconds: float = 0.6, damping: float = 60.0, max_undamped_s: float = 1.5, window: int = 50, still_dx: float = 1e-4, still_dth_deg: float = 0.05, still_windows: int = 4, cycles: int = 4, wobble_deg: float = 0.5, wobble_hz: float = 2.0) -> dict:
        """Settling procedure used after releases and drops.

        Each cycle steps with velocity damping (rigid rolling contacts cannot dissipate rocking
        energy), then without damping until every listed body moves less than ``still_dx`` and
        ``still_dth_deg`` in each of ``still_windows`` consecutive windows of ``window`` steps. The
        thresholds sit above the contact solver's chatter (a few hundredths of a millimetre per
        window). A body balanced in an unstable pose tips during the undamped phase and the next
        cycle settles the new pose. Velocities are zeroed at the end. Returns whether rest was
        reached, the cycles used and the undamped time.

        During the first part of each damped phase the gravity direction wobbles on a cone of
        half angle ``wobble_deg`` (``wobble_hz`` turns per second), so that a stone resting in
        an unstable balance on a mesh edge or vertex tips onto a stable support before the tests
        start (author's choice; a tower in a room is never free of such small disturbances). The
        last 0.15 s of each damped phase uses the true gravity."""
        ids = [b.id for b in self.bodies if not b.static and not b.kinematic] if ids is None else list(ids)
        old = self.damping
        undamped = 0.0
        still = False
        dx = dth = 0.0
        n_cycles = 0
        for cycle in range(int(cycles)):
            n_cycles = cycle + 1
            self.damping = float(damping)
            n_damped = int(round((seconds if cycle == 0 else 0.3) / self.dt))
            n_final = min(n_damped, int(round(0.15 / self.dt)))
            if wobble_deg > 0.0:
                g0 = np.array(self.gravity, dtype=float).copy()
                gm = float(np.linalg.norm(g0))
                down = g0 / gm
                e1 = np.cross(down, [1.0, 0.0, 0.0])
                if np.linalg.norm(e1) < 1e-6:
                    e1 = np.cross(down, [0.0, 1.0, 0.0])
                e1 /= np.linalg.norm(e1)
                e2 = np.cross(down, e1)
                ta = np.tan(np.radians(wobble_deg))
                try:
                    for k in range(n_damped - n_final):
                        ph = 2.0 * np.pi * wobble_hz * k * self.dt
                        gd = down + ta * (np.cos(ph) * e1 + np.sin(ph) * e2)
                        self.gravity = gm * gd / np.linalg.norm(gd)
                        self.step(1)
                finally:
                    self.gravity = g0
                self.step(n_final)
            else:
                self.step(n_damped)
            self.damping = old
            run = 0
            steps = 0
            while steps * self.dt < max_undamped_s:
                x0 = {i: self.bodies[i].x.copy() for i in ids}
                q0 = {i: self.bodies[i].q.copy() for i in ids}
                self.step(window)
                steps += window
                dx = max((float(np.linalg.norm(self.bodies[i].x - x0[i])) for i in ids), default=0.0)
                dth = max((2.0 * float(np.arccos(min(1.0, abs(float(np.dot(self.bodies[i].q, q0[i])))))) for i in ids), default=0.0)
                run = run + 1 if (dx < still_dx and np.degrees(dth) < still_dth_deg) else 0
                if run >= still_windows:
                    still = True
                    break
            undamped += steps * self.dt
            if still:
                break
        self.damping = old
        for b in self.bodies:
            if not b.static and not b.kinematic:
                b.v[:] = 0.0
                b.w[:] = 0.0
        return {"undamped_s": undamped, "still": still, "cycles": n_cycles, "last_dx": dx, "last_dth_deg": float(np.degrees(dth))}

    def settle(self, max_steps: int = 2000, v_tol: float = 1e-3, w_tol: float = 1e-2, min_steps: int = 10) -> int:
        """Step until all dynamic bodies are slower than the tolerances. Returns steps taken."""
        for i in range(max_steps):
            self.step()
            if i >= min_steps:
                vmax = max((float(np.linalg.norm(b.v)) for b in self.bodies if b.inv_mass > 0), default=0.0)
                wmax = max((float(np.linalg.norm(b.w)) for b in self.bodies if b.inv_mass > 0), default=0.0)
                if vmax < v_tol and wmax < w_tol:
                    return i + 1
        return max_steps
