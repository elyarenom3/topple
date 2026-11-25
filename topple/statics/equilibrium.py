"""Static equilibrium as a feasibility program (Section 7.1).

Contact force f_i = (ft1, ft2, fn) in the local basis (t1, t2, n). B_i = [I; [p_i]x] R_i maps the
local force to a world wrench about the origin. Per body equilibrium: the contact force acts on
body_b and its negative on body_a. The inscribed N sided polyhedral cone of Section 6.2 is

    [cos(2 pi k / N), sin(2 pi k / N), -mu cos(pi / N)] f_i <= 0,  k = 0 .. N - 1,  fn >= 0.

Feasibility is decided by a linear program (HiGHS through scipy). The displayed contact forces
come from the quadratic program that minimises the sum of squared contact forces (OSQP when
importable, scipy SLSQP otherwise).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog, minimize

from ..core.transforms import skew, tangent_basis

try:  # optional
    import osqp  # type: ignore
    import scipy.sparse as sp

    _HAS_OSQP = True
except Exception:  # pragma: no cover
    _HAS_OSQP = False

try:  # optional fast path: a persistent HiGHS model whose right hand side changes per query
    import highspy  # type: ignore

    _HAS_HIGHSPY = True
except Exception:  # pragma: no cover
    _HAS_HIGHSPY = False

_HIGHS_INSTANCE = None
_HIGHS_OWNER = [None]


def _get_highs():
    """One HiGHS instance per process (its banner is written once, to a silenced descriptor)."""
    global _HIGHS_INSTANCE
    if _HIGHS_INSTANCE is None:
        import os
        import sys

        saved = devnull = None
        try:
            sys.stdout.flush()
            saved = os.dup(1)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 1)
        except Exception:
            saved = None
        try:
            h = highspy.Highs()
        finally:
            if saved is not None:
                os.dup2(saved, 1)
                os.close(saved)
                os.close(devnull)
        h.setOptionValue("output_flag", False)
        h.setOptionValue("log_to_console", False)
        _HIGHS_INSTANCE = h
    return _HIGHS_INSTANCE


class _HighsFeasibility:
    """Persistent LP  min sum fn  s.t.  A f = -w,  C f <= 0,  fn >= 0  with a changeable w."""

    def __init__(self, A: np.ndarray, C: np.ndarray, nc: int):
        self.A, self.C, self.nc = A, C, nc
        self.n_eq = A.shape[0]
        self.token = object()
        inf = highspy.kHighsInf
        M = np.vstack([A, C])
        nz = np.nonzero(M.T)  # column wise
        cols, rows = nz
        start = np.searchsorted(cols, np.arange(M.shape[1] + 1))
        lp = highspy.HighsLp()
        lp.num_col_ = M.shape[1]
        lp.num_row_ = M.shape[0]
        cost = np.tile(np.array([0.0, 0.0, 1.0]), nc)
        lp.col_cost_ = cost
        lp.col_lower_ = np.tile(np.array([-inf, -inf, 0.0]), nc)
        lp.col_upper_ = np.full(M.shape[1], inf)
        lp.row_lower_ = np.concatenate([np.zeros(self.n_eq), np.full(C.shape[0], -inf)])
        lp.row_upper_ = np.zeros(M.shape[0])
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = start.astype(np.int32)
        lp.a_matrix_.index_ = rows.astype(np.int32)
        lp.a_matrix_.value_ = M.T[nz].astype(np.float64)
        self.lp = lp

    def solve(self, w: np.ndarray):
        h = _get_highs()
        if _HIGHS_OWNER[0] is not self.token:
            h.clearModel()
            h.passModel(self.lp)
            _HIGHS_OWNER[0] = self.token
        b = -np.asarray(w, dtype=np.float64)
        h.changeRowsBounds(self.n_eq, np.arange(self.n_eq, dtype=np.int32), b, b)
        h.run()
        if h.getModelStatus() == highspy.HighsModelStatus.kOptimal:
            return True, np.array(h.getSolution().col_value).reshape(self.nc, 3)
        return False, None


def cone_matrix(mu: float, cone_sides: int) -> np.ndarray:
    k = np.arange(cone_sides)
    th = 2.0 * np.pi * k / cone_sides
    return np.column_stack([np.cos(th), np.sin(th), -mu * np.cos(np.pi / cone_sides) * np.ones(cone_sides)])


def contact_basis(contact, cone_sides: int = 8):
    """Return (B (6, 3), C (N, 3)) for one contact."""
    n = contact.normal / max(np.linalg.norm(contact.normal), 1e-300)
    t1, t2 = tangent_basis(n)
    R = np.column_stack([t1, t2, n])
    B = np.vstack([R, skew(contact.point) @ R])
    return B, cone_matrix(contact.mu, cone_sides)


def gravity_wrench(tower, gravity=None) -> np.ndarray:
    """Stacked gravity wrenches w_g = (m g, c x m g) of the free bodies (6 n_free,)."""
    g = tower.gravity if gravity is None else np.asarray(gravity, dtype=np.float64)
    free = tower.free_bodies()
    w = np.zeros(6 * len(free))
    for i, b in enumerate(free):
        mg = b.mass * g
        w[6 * i : 6 * i + 3] = mg
        w[6 * i + 3 : 6 * i + 6] = np.cross(b.com, mg)
    return w


def assemble(tower, gravity=None, cone_sides: int = 8):
    """Assemble A (6 n_free, 3 n_c), w_g, cone blocks C_i and the body index map.

    Equilibrium reads A f = -w_g with C_i f_i <= 0 and fn_i >= 0.
    """
    free = tower.free_bodies()
    index = {b.id: i for i, b in enumerate(free)}
    nb, nc = len(free), tower.n_contacts
    A = np.zeros((6 * nb, 3 * nc))
    C_blocks = []
    for j, c in enumerate(tower.contacts):
        B, C = contact_basis(c, cone_sides)
        if c.body_b in index:
            r = 6 * index[c.body_b]
            A[r : r + 6, 3 * j : 3 * j + 3] += B
        if c.body_a in index:
            r = 6 * index[c.body_a]
            A[r : r + 6, 3 * j : 3 * j + 3] -= B
        C_blocks.append(C)
    return A, gravity_wrench(tower, gravity), C_blocks, index


def block_diag(blocks: list[np.ndarray]) -> np.ndarray:
    if not blocks:
        return np.zeros((0, 0))
    r = sum(b.shape[0] for b in blocks)
    c = sum(b.shape[1] for b in blocks)
    out = np.zeros((r, c))
    i = j = 0
    for b in blocks:
        out[i : i + b.shape[0], j : j + b.shape[1]] = b
        i += b.shape[0]
        j += b.shape[1]
    return out


class ContactLP:
    """Cached linear program for one tower: the matrices do not depend on gravity, so tilt and
    push margins reuse them and only change the right hand side."""

    def __init__(self, tower, cone_sides: int = 8, use_highspy: bool = True):
        self.tower = tower
        self.cone_sides = cone_sides
        self.A, self.w_g, self.C_blocks, self.index = assemble(tower, None, cone_sides)
        self.C = block_diag(self.C_blocks)
        self.nc = tower.n_contacts
        self.nb = len(self.index)
        self.bounds = [(None, None), (None, None), (0.0, None)] * self.nc
        self.c_obj = np.tile(np.array([0.0, 0.0, 1.0]), self.nc)  # minimise total normal force for determinacy
        self.b_ub = np.zeros(self.C.shape[0])
        self._fast = None
        if _HAS_HIGHSPY and use_highspy and self.nc > 0 and self.nb > 0:
            try:
                self._fast = _HighsFeasibility(self.A, self.C, self.nc)
            except Exception:
                self._fast = None

    def feasible(self, gravity=None):
        """Return (feasible, forces (nc, 3) in local bases or None)."""
        if self.nb == 0:
            return True, np.zeros((self.nc, 3))
        if self.nc == 0:
            return False, None
        w = self.w_g if gravity is None else gravity_wrench(self.tower, gravity)
        if self._fast is not None:
            return self._fast.solve(w)
        res = linprog(self.c_obj, A_ub=self.C, b_ub=self.b_ub, A_eq=self.A, b_eq=-w, bounds=self.bounds, method="highs")
        if res.status == 0:
            return True, res.x.reshape(self.nc, 3)
        return False, None

    def push(self, direction: np.ndarray, top_id: int, gravity=None, s_max: float = 1e6):
        """Maximise the horizontal force magnitude s applied at the top body's centre of mass.

        Returns (s, forces, res) with the full linprog result (duals available) or (0, None, res).
        """
        if self.nc == 0 or top_id not in self.index:
            return 0.0, None, None
        e = np.asarray(direction, dtype=np.float64)
        e = e / max(np.linalg.norm(e), 1e-300)
        top = self.tower.body(top_id)
        col = np.zeros(6 * self.nb)
        r = 6 * self.index[top_id]
        col[r : r + 3] = e
        col[r + 3 : r + 6] = np.cross(top.com, e)
        A_eq = np.column_stack([self.A, col])
        A_ub = np.column_stack([self.C, np.zeros(self.C.shape[0])])
        w = self.w_g if gravity is None else gravity_wrench(self.tower, gravity)
        c = np.zeros(3 * self.nc + 1)
        c[-1] = -1.0
        bounds = self.bounds + [(0.0, s_max)]
        res = linprog(c, A_ub=A_ub, b_ub=self.b_ub, A_eq=A_eq, b_eq=-w, bounds=bounds, method="highs")
        if res.status == 0:
            return float(res.x[-1]), res.x[:-1].reshape(self.nc, 3), res
        if res.status == 3:  # unbounded (cannot happen with the cap but kept for safety)
            return s_max, None, res
        return 0.0, None, res


def feasible(tower, gravity=None, cone_sides: int = 8):
    """Equilibrium feasibility of the tower under ``gravity`` (defaults to tower.gravity)."""
    return ContactLP(tower, cone_sides).feasible(gravity)


def world_forces(tower, local_forces: np.ndarray) -> np.ndarray:
    """Convert local (ft1, ft2, fn) forces to world force vectors acting on body_b."""
    out = np.zeros_like(local_forces)
    for j, c in enumerate(tower.contacts):
        n = c.normal
        t1, t2 = tangent_basis(n)
        out[j] = local_forces[j, 0] * t1 + local_forces[j, 1] * t2 + local_forces[j, 2] * n
    return out


def min_norm_forces(tower, gravity=None, cone_sides: int = 8, solver: str = "auto"):
    """Contact forces (nc, 3, local basis) minimising the sum of squared forces (Section 7.1).

    Raises ValueError if the tower is not equilibrium feasible.
    """
    lp = ContactLP(tower, cone_sides)
    ok, f0 = lp.feasible(gravity)
    if not ok:
        raise ValueError("tower is not equilibrium feasible")
    if lp.nb == 0:
        return np.zeros((lp.nc, 3))
    w = lp.w_g if gravity is None else gravity_wrench(tower, gravity)
    n = 3 * lp.nc
    use_osqp = _HAS_OSQP and solver in ("auto", "osqp")
    if use_osqp:
        try:
            fn_rows = np.zeros((lp.nc, n))
            for j in range(lp.nc):
                fn_rows[j, 3 * j + 2] = 1.0
            Acon = sp.csc_matrix(np.vstack([lp.A, lp.C, fn_rows]))
            l = np.concatenate([-w, -np.inf * np.ones(lp.C.shape[0]), np.zeros(lp.nc)])
            u = np.concatenate([-w, np.zeros(lp.C.shape[0]), np.inf * np.ones(lp.nc)])
            prob = osqp.OSQP()
            prob.setup(sp.identity(n, format="csc"), np.zeros(n), Acon, l, u, verbose=False, eps_abs=1e-9, eps_rel=1e-9, max_iter=20000, polish=True)
            try:
                res = prob.solve(raise_error=False)
            except TypeError:
                res = prob.solve()
            status = str(getattr(res.info, "status", "")).lower()
            if "solved" in status and res.x is not None and np.all(np.isfinite(res.x)):
                x = np.asarray(res.x)
                if np.max(np.abs(lp.A @ x + w)) < 1e-5 * max(1.0, np.max(np.abs(w))):
                    return x.reshape(lp.nc, 3)
        except Exception:
            pass
    # scipy fallback: SLSQP from the LP solution
    x0 = f0.reshape(-1)
    cons = [
        {"type": "eq", "fun": lambda x: lp.A @ x + w, "jac": lambda x: lp.A},
        {"type": "ineq", "fun": lambda x: -(lp.C @ x), "jac": lambda x: -lp.C},
    ]
    bounds = lp.bounds
    res = minimize(lambda x: 0.5 * x @ x, x0, jac=lambda x: x, constraints=cons, bounds=bounds, method="SLSQP", options={"maxiter": 500, "ftol": 1e-12})
    return res.x.reshape(lp.nc, 3)


def equilibrium_residual(tower, local_forces: np.ndarray, gravity=None, cone_sides: int = 8) -> float:
    """Max absolute residual of A f + w_g (for tests and display)."""
    A, w, _, _ = assemble(tower, gravity, cone_sides)
    return float(np.max(np.abs(A @ local_forces.reshape(-1) + w))) if A.size else 0.0
