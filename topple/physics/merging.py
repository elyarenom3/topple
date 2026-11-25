"""Adaptive merging for tall static towers (Section 6.5), after Coevoet, Benchekroun and
Kry (2020). Bodies that share a common (near zero) spatial velocity over a window and
touch each other are merged into one rigid aggregate that the solver treats as a single
body; internal contacts are not solved. The aggregate unmerges when an external impulse
arrives, when gravity changes, or when a friction cone check on its internal contacts
fails (a full solve of the group's contacts shows an internal contact at its cone limit
or an internal relative velocity)."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .body import Contact


class MergeManager:
    def __init__(self, world, v_tol: float = 2e-3, w_tol: float = 2e-2, window: int = 50, check_every: int = 25, impulse_tol: float = 1e-3):
        self.world = world
        self.v_tol = v_tol
        self.w_tol = w_tol
        self.window = window
        self.check_every = check_every
        self.impulse_tol = impulse_tol
        self.groups: List[List[int]] = []
        self._rest_count: Dict[int, int] = {}
        self._internal: List[Contact] = []
        self._external: List[Contact] = []
        self._prev_ext_impulse: Dict[int, float] = {}
        self._last_gravity = world.gravity.copy()
        self._agg: Dict[int, dict] = {}
        self._body_slot: Dict[int, int] = {}
        self._active = False

    def active(self) -> bool:
        return self._active

    # ----------------------------------------------------------- merge logic
    def _update_rest(self):
        for b in self.world.bodies:
            if b.inv_mass == 0.0:
                continue
            if np.linalg.norm(b.v) < self.v_tol and np.linalg.norm(b.w) < self.w_tol:
                self._rest_count[b.id] = self._rest_count.get(b.id, 0) + 1
            else:
                self._rest_count[b.id] = 0

    def _try_merge(self, contacts: List[Contact]):
        resting = {bid for bid, n in self._rest_count.items() if n >= self.window and self.world.bodies[bid].group < 0}
        if len(resting) < 2:
            return
        # connected components over contacts among resting bodies
        parent = {b: b for b in resting}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for c in contacts:
            if c.body_a in resting and c.body_b in resting:
                parent[find(c.body_a)] = find(c.body_b)
        comps: Dict[int, List[int]] = {}
        for b in resting:
            comps.setdefault(find(b), []).append(b)
        for members in comps.values():
            if len(members) >= 2:
                gid = len(self.groups)
                self.groups.append(sorted(members))
                # members share the mass weighted mean velocity, which still carries this step's
                # gravity increment, so the contact impulses keep supporting the weight
                bs = [self.world.bodies[m] for m in members]
                m_tot = sum(b.mass for b in bs)
                v_mean = sum(b.mass * b.v for b in bs) / m_tot
                for b in bs:
                    b.group = gid
                    b.v = v_mean.copy()
                    b.w[:] = 0.0

    def _unmerge(self, gid: int):
        for m in self.groups[gid]:
            self.world.bodies[m].group = -1
            self._rest_count[m] = 0
        self.groups[gid] = []

    def before_solve(self, contacts: List[Contact]) -> List[Contact]:
        world = self.world
        if np.any(world.gravity != self._last_gravity) or world._ext_force or world._ext_torque:
            for gid, g in enumerate(self.groups):
                if g:
                    self._unmerge(gid)
            self._last_gravity = world.gravity.copy()
        self._try_merge(contacts)
        groups = [g for g in self.groups if g]
        self._internal = []
        self._external = []
        if not groups:
            self._active = False
            return contacts
        for c in contacts:
            ga = world.bodies[c.body_a].group
            gb = world.bodies[c.body_b].group
            if ga >= 0 and ga == gb:
                self._internal.append(c)
            else:
                self._external.append(c)
        # periodic friction cone check with a full solve on a clone
        if world.step_count % self.check_every == 0 and self._internal:
            self._cone_check(contacts)
            groups = [g for g in self.groups if g]
            if not groups:
                self._active = False
                return contacts
            self._external = [c for c in contacts if not (world.bodies[c.body_a].group >= 0 and world.bodies[c.body_a].group == world.bodies[c.body_b].group)]
        self._active = True
        self._build_aggregates()
        return self._external

    def _cone_check(self, contacts: List[Contact]):
        """Full unmerged solve of the current contacts on a scratch copy; unmerge groups whose
        internal contacts reach the cone limit or move."""
        world = self.world
        scratch = world.clone()
        scratch.merging = False
        scratch.merge_manager = None
        for b in scratch.bodies:
            b.group = -1
        scratch._warm = {}
        from .solver import contact_arrays, pgs_solve, stabilisation_bias

        x, v, w, inv_m, inv_I = scratch._solver_arrays()
        idx = {b.id: i for i, b in enumerate(scratch.bodies)}
        ia, ib, p, n, t1, t2, mu, depth = contact_arrays(contacts, idx)
        bias = stabilisation_bias(depth, world.dt, world.baumgarte, world.slop)
        lam = np.zeros((len(contacts), 3))
        for c_i, c in enumerate(contacts):
            prev = world._warm.get(c.feature)
            if prev is not None:
                lam[c_i] = prev
        from .solver import apply_impulses

        if np.any(lam != 0.0):
            apply_impulses(ia, ib, p, n, t1, t2, x, v, w, inv_m, inv_I, lam)
        pgs_solve(ia, ib, p, n, t1, t2, bias, mu, x, v, w, inv_m, inv_I, lam, 4 * world.pgs_iterations, world.cone_sides, world.pgs_tol)
        bad = set()
        for c_i, c in enumerate(contacts):
            ga = world.bodies[c.body_a].group
            if ga < 0 or ga != world.bodies[c.body_b].group:
                continue
            ft = np.hypot(lam[c_i, 1], lam[c_i, 2])
            if lam[c_i, 0] > 0 and ft >= 0.999 * c.mu * lam[c_i, 0] * np.cos(np.pi / world.cone_sides):
                bad.add(ga)
            a, b = ia[c_i], ib[c_i]
            dv = (v[b] + np.cross(w[b], p[c_i] - x[b])) - (v[a] + np.cross(w[a], p[c_i] - x[a]))
            if np.linalg.norm(dv) > 5.0 * self.v_tol:
                bad.add(ga)
        for gid in bad:
            self._unmerge(gid)

    # -------------------------------------------------------- aggregate arrays
    def _build_aggregates(self):
        world = self.world
        nb = len(world.bodies)
        slot = {}
        rows = []
        for b in world.bodies:
            if b.group < 0 or not self.groups[b.group]:
                slot[b.id] = len(rows)
                rows.append(("body", b.id))
        self._agg = {}
        for gid, g in enumerate(self.groups):
            if not g:
                continue
            bodies = [world.bodies[m] for m in g]
            m_tot = sum(b.mass for b in bodies)
            com = sum(b.mass * b.x for b in bodies) / m_tot
            I = np.zeros((3, 3))
            for b in bodies:
                r = b.x - com
                I += b.inertia_world() + b.mass * (float(r @ r) * np.eye(3) - np.outer(r, r))
            self._agg[gid] = {"mass": m_tot, "com": com, "I": I, "row": len(rows)}
            for m in g:
                slot[m] = len(rows)
            rows.append(("group", gid))
        self._rows = rows
        self._body_slot = slot

    def aggregate_arrays(self):
        world = self.world
        n = len(self._rows)
        x = np.zeros((n, 3))
        v = np.zeros((n, 3))
        w = np.zeros((n, 3))
        inv_m = np.zeros(n)
        inv_I = np.zeros((n, 3, 3))
        for i, (kind, ident) in enumerate(self._rows):
            if kind == "body":
                b = world.bodies[ident]
                x[i], v[i], w[i] = b.x, b.v, b.w
                inv_m[i] = b.inv_mass
                if inv_m[i] > 0:
                    inv_I[i] = b.inv_inertia_world()
            else:
                a = self._agg[ident]
                x[i] = a["com"]
                # unconstrained aggregate velocity: members fell by h g during this step
                v[i] = world.bodies[self.groups[ident][0]].v
                w[i] = 0.0
                inv_m[i] = 1.0 / a["mass"]
                inv_I[i] = np.linalg.inv(a["I"])
        return x, v, w, inv_m, inv_I, dict(self._body_slot)

    def scatter_velocities(self, v, w):
        world = self.world
        for i, (kind, ident) in enumerate(self._rows):
            if kind == "body":
                b = world.bodies[ident]
                if b.inv_mass > 0:
                    b.v, b.w = v[i].copy(), w[i].copy()
            else:
                a = self._agg[ident]
                for m in self.groups[ident]:
                    b = world.bodies[m]
                    b.v = v[i] + np.cross(w[i], b.x - a["com"])
                    b.w = w[i].copy()

    def after_solve(self):
        # rest counters use the constrained (post solve) velocities
        self._update_rest()
        # unmerge on a change of external impulse (a perturbation arrived)
        world = self.world
        cur: Dict[int, float] = {}
        for c in self._external:
            for bid in (c.body_a, c.body_b):
                if world.bodies[bid].group >= 0:
                    cur[world.bodies[bid].group] = cur.get(world.bodies[bid].group, 0.0) + abs(c.fn)
        for gid, val in cur.items():
            prev = self._prev_ext_impulse.get(gid)
            if prev is not None and abs(val - prev) > self.impulse_tol * max(prev, 1e-9) + 1e-9 * 0:
                if abs(val - prev) > self.impulse_tol:
                    self._unmerge(gid)
        self._prev_ext_impulse = cur
