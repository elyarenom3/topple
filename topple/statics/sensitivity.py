"""Differentiable margins (Section 7.7).

The push margin s* is the optimal value of a parametric linear program. With the HiGHS dual
variables y_s = d(-s*)/d b_eq (``res.eqlin.marginals``) and z_s = d(-s*)/d b_ub
(``res.ineqlin.marginals``), the sensitivity of the optimal value V = -s* to a parameter xi is

    dV/dxi = y_s^T (d b_eq/dxi - dA_eq/dxi x*) + z_s^T (d b_ub/dxi - dA_ub/dxi x*)

so that ds*/dxi = -dV/dxi. Derivatives are undefined where the active set changes, which is
expected near a margin of zero. The tilt margin gradient is provided by finite differences.
"""

from __future__ import annotations

import numpy as np

from ..core.transforms import skew, tangent_basis
from .equilibrium import ContactLP
from .margins import horizontal_basis, tilt_margin


def push_margin_gradient(tower, direction, wrt: str = "com", cone_sides: int = 8, top_body: int | None = None):
    """Gradient of s* (force in newtons, before dividing by the weight) for one push direction.

    ``wrt``: "com" returns (n_free, 3) in the order of ``tower.free_bodies()``; "mu" returns
    (n_contacts,); "points" returns (n_contacts, 3). Returns None when the LP is infeasible.
    """
    top_id = tower.top_body().id if top_body is None else top_body
    lp = ContactLP(tower, cone_sides)
    s, forces, res = lp.push(direction, top_id)
    if res is None or res.status != 0:
        return None
    x = res.x
    y = np.asarray(res.eqlin.marginals)  # dV/db_eq
    z = np.asarray(res.ineqlin.marginals)  # dV/db_ub
    free = tower.free_bodies()
    index = {b.id: i for i, b in enumerate(free)}
    g = tower.gravity
    N = cone_sides
    if wrt == "com":
        grad = np.zeros((len(free), 3))
        for b in free:
            i = index[b.id]
            rows = slice(6 * i + 3, 6 * i + 6)
            # b_eq = -w_g, torque rows of w_g are c x (m g): d/dc = -[m g]x, so d b_eq / dc = +[m g]x
            dV = y[rows] @ skew(b.mass * g)
            if b.id == top_id:
                # A_eq column of s has (e, c x e): d(A_eq x*)/dc = s* d(c x e)/dc = -s* [e]x
                e = np.asarray(direction, dtype=np.float64)
                e = e / max(np.linalg.norm(e), 1e-300)
                dV = dV - y[rows] @ (-s * skew(e))
            grad[i] = -dV
        return grad
    if wrt == "mu":
        grad = np.zeros(tower.n_contacts)
        for j in range(tower.n_contacts):
            fn = x[3 * j + 2]
            rows = slice(N * j, N * (j + 1))
            # A_ub rows: [cos, sin, -mu cos(pi/N)]: d(A_ub x*)/dmu = -cos(pi/N) fn for each row
            dV = z[rows] @ (-(-np.cos(np.pi / N) * fn) * np.ones(N))
            grad[j] = -dV
        return grad
    if wrt == "points":
        grad = np.zeros((tower.n_contacts, 3))
        for j, c in enumerate(tower.contacts):
            n = c.normal
            t1, t2 = tangent_basis(n)
            Fw = x[3 * j] * t1 + x[3 * j + 1] * t2 + x[3 * j + 2] * n  # world force on body_b
            # torque rows: p x F, d(p x F)/dp = -[F]x ; sign + for body_b rows, - for body_a rows
            dV = np.zeros(3)
            if c.body_b in index:
                rows = slice(6 * index[c.body_b] + 3, 6 * index[c.body_b] + 6)
                dV += -(y[rows] @ (-skew(Fw)))
            if c.body_a in index:
                rows = slice(6 * index[c.body_a] + 3, 6 * index[c.body_a] + 6)
                dV += -(y[rows] @ (skew(Fw)))
            grad[j] = -dV
        return grad
    raise ValueError("wrt must be com, mu or points")


def margin_gradient(tower, wrt: str = "com", kind: str = "push", direction=None, cone_sides: int = 8, **kw):
    """Gradient of a margin. ``kind="push"`` uses the LP duals for one direction (default: the
    direction of the minimum push margin); ``kind="tilt"`` uses finite differences."""
    if kind == "push":
        if direction is None:
            from .margins import push_margin

            _, direction = push_margin(tower, cone_sides=cone_sides, **kw)
        return push_margin_gradient(tower, direction, wrt, cone_sides)
    if kind == "tilt":
        return margin_jacobian_fd(lambda t: tilt_margin(t, cone_sides=cone_sides, **kw)[0], tower, wrt)
    raise ValueError("kind must be push or tilt")


def margin_jacobian_fd(fn, tower, wrt: str = "com", eps: float = 1e-4):
    """Central finite difference gradient of ``fn(tower)`` with respect to the chosen parameters."""
    base = tower.copy()
    if wrt == "com":
        free = base.free_bodies()
        grad = np.zeros((len(free), 3))
        for i, b in enumerate(free):
            for k in range(3):
                tp, tm = base.copy(), base.copy()
                tp.body(b.id).com[k] += eps
                tm.body(b.id).com[k] -= eps
                grad[i, k] = (fn(tp) - fn(tm)) / (2 * eps)
        return grad
    if wrt == "mu":
        grad = np.zeros(base.n_contacts)
        for j in range(base.n_contacts):
            tp, tm = base.copy(), base.copy()
            tp.contacts[j].mu += eps
            tm.contacts[j].mu -= eps
            grad[j] = (fn(tp) - fn(tm)) / (2 * eps)
        return grad
    if wrt == "points":
        grad = np.zeros((base.n_contacts, 3))
        for j in range(base.n_contacts):
            for k in range(3):
                tp, tm = base.copy(), base.copy()
                tp.contacts[j].point[k] += eps
                tm.contacts[j].point[k] -= eps
                grad[j, k] = (fn(tp) - fn(tm)) / (2 * eps)
        return grad
    raise ValueError("wrt must be com, mu or points")
