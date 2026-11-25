"""Velocity level contact solvers (Section 6.3).

Time stepping follows Stewart and Trinkle (1996) and Anitescu and Potra (1997). With
generalised velocity u, mass matrix M, contact Jacobian J, step h, gap phi and impulses
lambda:

    M (u+ - u) = h f_ext + J^T lambda,
    0 <= lambda_n  perp  J_n u+ + phi / h >= 0,

plus the friction conditions on an inscribed N sided polygon (Section 6.2). The default
solver is projected Gauss Seidel (PGS) over contacts with warm starting; the normal
impulse is projected onto [0, inf) and the tangential impulse pair onto the regular
N-gon of circumradius mu * lambda_n (vertices in the Stewart Trinkle directions
2 pi k / N). The tangential projection uses the metric of the 2 x 2 effective mass
block, which makes each contact update the exact Gauss Seidel sub problem (maximal
dissipation) even when the two tangent directions are strongly coupled. ``solve_lemke`` builds the Stewart Trinkle LCP explicitly and solves it with
Lemke's complementary pivoting, as the small problem reference (Erleben 2013).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit


# ----------------------------------------------------------------------------------
# Projection onto the inscribed N-gon
# ----------------------------------------------------------------------------------


@njit(cache=True)
def project_polygon(f1, f2, radius, N):
    """Project (f1, f2) onto the regular N-gon with vertices at angles 2 pi k / N and
    circumradius ``radius``. Returns the projected pair."""
    if radius <= 0.0:
        return 0.0, 0.0
    sector = 2.0 * math.pi / N
    phi = math.atan2(f2, f1)
    if phi < 0.0:
        phi += 2.0 * math.pi
    k = math.floor(phi / sector)
    psi = (k + 0.5) * sector  # edge normal angle
    ex = math.cos(psi)
    ey = math.sin(psi)
    apothem = radius * math.cos(math.pi / N)
    s = f1 * ex + f2 * ey
    if s <= apothem:
        return f1, f2
    # project onto the edge line and clamp along the edge
    px = -ey
    py = ex
    tau = f1 * px + f2 * py
    half = radius * math.sin(math.pi / N)
    if tau > half:
        tau = half
    elif tau < -half:
        tau = -half
    return apothem * ex + tau * px, apothem * ey + tau * py


@njit(cache=True)
def project_polygon_metric(g1, g2, radius, N, k11, k12, k22):
    """Projection of (g1, g2) onto the regular N-gon of circumradius ``radius`` in the metric
    of the 2 x 2 effective mass matrix K = [[k11, k12], [k12, k22]].

    This is the exact Gauss Seidel sub problem for one contact: minimise
    0.5 (f - g)^T K (f - g) over the polygon, where g is the unconstrained optimum.
    """
    if radius <= 0.0:
        return 0.0, 0.0
    sector = 2.0 * math.pi / N
    apothem = radius * math.cos(math.pi / N)
    inside = True
    for k in range(N):
        psi = (k + 0.5) * sector
        if g1 * math.cos(psi) + g2 * math.sin(psi) > apothem:
            inside = False
            break
    if inside:
        return g1, g2
    best = 1e300
    bx = 0.0
    by = 0.0
    for k in range(N):
        ax = radius * math.cos(k * sector)
        ay = radius * math.sin(k * sector)
        cx = radius * math.cos((k + 1) * sector)
        cy = radius * math.sin((k + 1) * sector)
        ex = cx - ax
        ey = cy - ay
        # minimise over s in [0, 1] of (a + s e - g)^T K (a + s e - g)
        dx = ax - g1
        dy = ay - g2
        eKe = ex * (k11 * ex + k12 * ey) + ey * (k12 * ex + k22 * ey)
        dKe = dx * (k11 * ex + k12 * ey) + dy * (k12 * ex + k22 * ey)
        if eKe > 1e-300:
            s = -dKe / eKe
        else:
            s = 0.0
        if s < 0.0:
            s = 0.0
        elif s > 1.0:
            s = 1.0
        px = ax + s * ex
        py = ay + s * ey
        rx = px - g1
        ry = py - g2
        val = rx * (k11 * rx + k12 * ry) + ry * (k12 * rx + k22 * ry)
        if val < best:
            best = val
            bx = px
            by = py
    return bx, by


# ----------------------------------------------------------------------------------
# PGS kernel
# ----------------------------------------------------------------------------------


@njit(cache=True)
def _apply_impulse(v, w, inv_m, inv_I, b, r, imp, sign):
    if inv_m[b] > 0.0:
        v[b, 0] += sign * inv_m[b] * imp[0]
        v[b, 1] += sign * inv_m[b] * imp[1]
        v[b, 2] += sign * inv_m[b] * imp[2]
        tx = r[1] * imp[2] - r[2] * imp[1]
        ty = r[2] * imp[0] - r[0] * imp[2]
        tz = r[0] * imp[1] - r[1] * imp[0]
        w[b, 0] += sign * (inv_I[b, 0, 0] * tx + inv_I[b, 0, 1] * ty + inv_I[b, 0, 2] * tz)
        w[b, 1] += sign * (inv_I[b, 1, 0] * tx + inv_I[b, 1, 1] * ty + inv_I[b, 1, 2] * tz)
        w[b, 2] += sign * (inv_I[b, 2, 0] * tx + inv_I[b, 2, 1] * ty + inv_I[b, 2, 2] * tz)


@njit(cache=True)
def _row_mass(inv_m, inv_I, a, b, ra, rb, d):
    """Effective inverse mass of a contact row with direction d: d^T (M^-1) d summed over both bodies."""
    k = inv_m[a] + inv_m[b]
    ca = np.empty(3)
    ca[0] = ra[1] * d[2] - ra[2] * d[1]
    ca[1] = ra[2] * d[0] - ra[0] * d[2]
    ca[2] = ra[0] * d[1] - ra[1] * d[0]
    cb = np.empty(3)
    cb[0] = rb[1] * d[2] - rb[2] * d[1]
    cb[1] = rb[2] * d[0] - rb[0] * d[2]
    cb[2] = rb[0] * d[1] - rb[1] * d[0]
    for i in range(3):
        for j in range(3):
            k += ca[i] * inv_I[a, i, j] * ca[j] + cb[i] * inv_I[b, i, j] * cb[j]
    return k


@njit(cache=True)
def _row_cross_mass(inv_m, inv_I, a, b, ra, rb, d1, d2):
    k = inv_m[a] * (d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]) + inv_m[b] * (d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2])
    ca1 = np.empty(3)
    ca1[0] = ra[1] * d1[2] - ra[2] * d1[1]
    ca1[1] = ra[2] * d1[0] - ra[0] * d1[2]
    ca1[2] = ra[0] * d1[1] - ra[1] * d1[0]
    ca2 = np.empty(3)
    ca2[0] = ra[1] * d2[2] - ra[2] * d2[1]
    ca2[1] = ra[2] * d2[0] - ra[0] * d2[2]
    ca2[2] = ra[0] * d2[1] - ra[1] * d2[0]
    cb1 = np.empty(3)
    cb1[0] = rb[1] * d1[2] - rb[2] * d1[1]
    cb1[1] = rb[2] * d1[0] - rb[0] * d1[2]
    cb1[2] = rb[0] * d1[1] - rb[1] * d1[0]
    cb2 = np.empty(3)
    cb2[0] = rb[1] * d2[2] - rb[2] * d2[1]
    cb2[1] = rb[2] * d2[0] - rb[0] * d2[2]
    cb2[2] = rb[0] * d2[1] - rb[1] * d2[0]
    for i in range(3):
        for j in range(3):
            k += ca1[i] * inv_I[a, i, j] * ca2[j] + cb1[i] * inv_I[b, i, j] * cb2[j]
    return k


@njit(cache=True)
def pgs_solve(ia, ib, p, n, t1, t2, bias, mu, x, v, w, inv_m, inv_I, lam, iters, N, tol):
    """Projected Gauss Seidel over contacts. ``lam`` (nc, 3) holds warm start impulses that
    have already been applied to v, w. Returns the number of iterations used."""
    nc = ia.shape[0]
    ra = np.empty((nc, 3))
    rb = np.empty((nc, 3))
    kn = np.empty(nc)
    k11 = np.empty(nc)
    k12 = np.empty(nc)
    k22 = np.empty(nc)
    for c in range(nc):
        a = ia[c]
        b = ib[c]
        for i in range(3):
            ra[c, i] = p[c, i] - x[a, i]
            rb[c, i] = p[c, i] - x[b, i]
        kn[c] = _row_mass(inv_m, inv_I, a, b, ra[c], rb[c], n[c])
        k11[c] = _row_mass(inv_m, inv_I, a, b, ra[c], rb[c], t1[c])
        k22[c] = _row_mass(inv_m, inv_I, a, b, ra[c], rb[c], t2[c])
        k12[c] = _row_cross_mass(inv_m, inv_I, a, b, ra[c], rb[c], t1[c], t2[c])
    imp = np.empty(3)
    used = 0
    for it in range(iters):
        max_change = 0.0
        for c in range(nc):
            a = ia[c]
            b = ib[c]
            # relative velocity of b with respect to a at the contact
            dvx = (v[b, 0] + w[b, 1] * rb[c, 2] - w[b, 2] * rb[c, 1]) - (v[a, 0] + w[a, 1] * ra[c, 2] - w[a, 2] * ra[c, 1])
            dvy = (v[b, 1] + w[b, 2] * rb[c, 0] - w[b, 0] * rb[c, 2]) - (v[a, 1] + w[a, 2] * ra[c, 0] - w[a, 0] * ra[c, 2])
            dvz = (v[b, 2] + w[b, 0] * rb[c, 1] - w[b, 1] * rb[c, 0]) - (v[a, 2] + w[a, 0] * ra[c, 1] - w[a, 1] * ra[c, 0])
            vn = dvx * n[c, 0] + dvy * n[c, 1] + dvz * n[c, 2]
            if kn[c] > 0.0:
                dl = (bias[c] - vn) / kn[c]
                new = lam[c, 0] + dl
                if new < 0.0:
                    new = 0.0
                dl = new - lam[c, 0]
                if dl != 0.0:
                    lam[c, 0] = new
                    for i in range(3):
                        imp[i] = dl * n[c, i]
                    _apply_impulse(v, w, inv_m, inv_I, a, ra[c], imp, -1.0)
                    _apply_impulse(v, w, inv_m, inv_I, b, rb[c], imp, 1.0)
                    if abs(dl) > max_change:
                        max_change = abs(dl)
            # friction block
            if lam[c, 0] > 0.0 and mu[c] > 0.0:
                dvx = (v[b, 0] + w[b, 1] * rb[c, 2] - w[b, 2] * rb[c, 1]) - (v[a, 0] + w[a, 1] * ra[c, 2] - w[a, 2] * ra[c, 1])
                dvy = (v[b, 1] + w[b, 2] * rb[c, 0] - w[b, 0] * rb[c, 2]) - (v[a, 1] + w[a, 2] * ra[c, 0] - w[a, 0] * ra[c, 2])
                dvz = (v[b, 2] + w[b, 0] * rb[c, 1] - w[b, 1] * rb[c, 0]) - (v[a, 2] + w[a, 0] * ra[c, 1] - w[a, 1] * ra[c, 0])
                vt1 = dvx * t1[c, 0] + dvy * t1[c, 1] + dvz * t1[c, 2]
                vt2 = dvx * t2[c, 0] + dvy * t2[c, 1] + dvz * t2[c, 2]
                det = k11[c] * k22[c] - k12[c] * k12[c]
                if det > 1e-300:
                    d1 = -(k22[c] * vt1 - k12[c] * vt2) / det
                    d2 = -(-k12[c] * vt1 + k11[c] * vt2) / det
                else:
                    d1 = -vt1 / max(k11[c], 1e-300)
                    d2 = -vt2 / max(k22[c], 1e-300)
                f1 = lam[c, 1] + d1
                f2 = lam[c, 2] + d2
                f1, f2 = project_polygon_metric(f1, f2, mu[c] * lam[c, 0], N, k11[c], k12[c], k22[c])
                d1 = f1 - lam[c, 1]
                d2 = f2 - lam[c, 2]
                if d1 != 0.0 or d2 != 0.0:
                    lam[c, 1] = f1
                    lam[c, 2] = f2
                    for i in range(3):
                        imp[i] = d1 * t1[c, i] + d2 * t2[c, i]
                    _apply_impulse(v, w, inv_m, inv_I, a, ra[c], imp, -1.0)
                    _apply_impulse(v, w, inv_m, inv_I, b, rb[c], imp, 1.0)
                    ch = math.sqrt(d1 * d1 + d2 * d2)
                    if ch > max_change:
                        max_change = ch
            elif lam[c, 1] != 0.0 or lam[c, 2] != 0.0:
                # normal impulse vanished: remove any friction impulse
                for i in range(3):
                    imp[i] = -(lam[c, 1] * t1[c, i] + lam[c, 2] * t2[c, i])
                _apply_impulse(v, w, inv_m, inv_I, a, ra[c], imp, -1.0)
                _apply_impulse(v, w, inv_m, inv_I, b, rb[c], imp, 1.0)
                lam[c, 1] = 0.0
                lam[c, 2] = 0.0
        used = it + 1
        if max_change < tol:
            break
    return used


@njit(cache=True)
def apply_impulses(ia, ib, p, n, t1, t2, x, v, w, inv_m, inv_I, lam):
    """Apply impulses lam (nc, 3) to the velocity arrays (used for warm starting)."""
    imp = np.empty(3)
    for c in range(ia.shape[0]):
        a = ia[c]
        b = ib[c]
        ra = p[c] - x[a]
        rb = p[c] - x[b]
        for i in range(3):
            imp[i] = lam[c, 0] * n[c, i] + lam[c, 1] * t1[c, i] + lam[c, 2] * t2[c, i]
        _apply_impulse(v, w, inv_m, inv_I, a, ra, imp, -1.0)
        _apply_impulse(v, w, inv_m, inv_I, b, rb, imp, 1.0)


@njit(cache=True)
def cone_residuals(ia, ib, p, n, t1, t2, bias, mu, x, v, w, lam, N):
    """Max cone violation (relative) and max complementarity residual after a solve."""
    max_cone = 0.0
    max_comp = 0.0
    max_pen_vel = 0.0
    apo = math.cos(math.pi / N)
    for c in range(ia.shape[0]):
        a = ia[c]
        b = ib[c]
        ft = math.sqrt(lam[c, 1] ** 2 + lam[c, 2] ** 2)
        lim = mu[c] * lam[c, 0]
        if lim > 0.0:
            viol = (ft - lim) / lim
        else:
            viol = ft
        if viol > max_cone:
            max_cone = viol
        ra = p[c] - x[a]
        rb = p[c] - x[b]
        dvx = (v[b, 0] + w[b, 1] * rb[2] - w[b, 2] * rb[1]) - (v[a, 0] + w[a, 1] * ra[2] - w[a, 2] * ra[1])
        dvy = (v[b, 1] + w[b, 2] * rb[0] - w[b, 0] * rb[2]) - (v[a, 1] + w[a, 2] * ra[0] - w[a, 0] * ra[2])
        dvz = (v[b, 2] + w[b, 0] * rb[1] - w[b, 1] * rb[0]) - (v[a, 2] + w[a, 0] * ra[1] - w[a, 1] * ra[0])
        vn = dvx * n[c, 0] + dvy * n[c, 1] + dvz * n[c, 2]
        slack = vn - bias[c]
        comp = abs(lam[c, 0] * slack)
        if comp > max_comp:
            max_comp = comp
        if -slack > max_pen_vel:
            max_pen_vel = -slack
    return max_cone, max_comp, max_pen_vel


# ----------------------------------------------------------------------------------
# Problem assembly helpers (Python level)
# ----------------------------------------------------------------------------------


def contact_arrays(contacts, body_index):
    """Pack a list of Contact into solver arrays. ``body_index`` maps body id to solver row."""
    nc = len(contacts)
    ia = np.empty(nc, dtype=np.int64)
    ib = np.empty(nc, dtype=np.int64)
    p = np.empty((nc, 3))
    n = np.empty((nc, 3))
    t1 = np.empty((nc, 3))
    t2 = np.empty((nc, 3))
    mu = np.empty(nc)
    depth = np.empty(nc)
    for c, ct in enumerate(contacts):
        ia[c] = body_index[ct.body_a]
        ib[c] = body_index[ct.body_b]
        p[c] = ct.point
        n[c] = ct.normal
        t1[c] = ct.t1
        t2[c] = ct.t2
        mu[c] = ct.mu
        depth[c] = ct.depth
    return ia, ib, p, n, t1, t2, mu, depth


def stabilisation_bias(depth, h, baumgarte, slop):
    """Target normal velocity: approach allowed up to the gap for separated contacts,
    Baumgarte push out beyond the slop for penetrating ones."""
    phi = -depth
    bias = np.where(phi >= 0.0, -phi / h, baumgarte * np.maximum(-phi - slop, 0.0) / h)
    return bias


# ----------------------------------------------------------------------------------
# Lemke reference
# ----------------------------------------------------------------------------------


def lemke(M: np.ndarray, q: np.ndarray, max_iter: int = 2000, tol: float = 1e-12):
    """Lemke's complementary pivoting for the LCP w = M z + q, w, z >= 0, w^T z = 0.

    Returns (z, status) with status 0 on success, 1 on ray termination, 2 on iteration limit.
    """
    n = len(q)
    if np.all(q >= -tol):
        return np.zeros(n), 0
    # tableau columns: w (n) | z (n) | z0 | rhs
    T = np.hstack([np.eye(n), -M, -np.ones((n, 1)), q.reshape(-1, 1)]).astype(np.float64)
    basis = list(range(n))
    z0 = 2 * n
    r = int(np.argmin(q))
    _pivot(T, r, z0)
    basis[r] = z0
    entering = n + r
    for _ in range(max_iter):
        col = T[:, entering]
        rhs = T[:, -1]
        cand = np.where(col > tol)[0]
        if len(cand) == 0:
            return _extract(T, basis, n), 1
        ratios = rhs[cand] / col[cand]
        rmin = ratios.min()
        ties = cand[ratios <= rmin + 1e-12]
        # prefer leaving z0 when tied (termination)
        r = int(ties[0])
        for t in ties:
            if basis[t] == z0:
                r = int(t)
        leaving = basis[r]
        _pivot(T, r, entering)
        basis[r] = entering
        if leaving == z0:
            return _extract(T, basis, n), 0
        entering = leaving + n if leaving < n else leaving - n
    return _extract(T, basis, n), 2


def _pivot(T, r, c):
    T[r] /= T[r, c]
    for i in range(T.shape[0]):
        if i != r and T[i, c] != 0.0:
            T[i] -= T[i, c] * T[r]


def _extract(T, basis, n):
    z = np.zeros(n)
    for i, b in enumerate(basis):
        if n <= b < 2 * n:
            z[b - n] = T[i, -1]
    return z


def build_stewart_trinkle_lcp(ia, ib, p, n, t1, t2, bias, mu, x, v_free, w_free, inv_m, inv_I, N):
    """Assemble the Stewart Trinkle LCP with N friction directions per contact.

    Unknowns per contact: lambda_n, beta_1..beta_N (>= 0), and the cone slack lambda_s.
    Returns (Mlcp, qlcp, Jn, Jd, dirs, Minv_blocks) for use by ``solve_lemke``.
    """
    nc = ia.shape[0]
    nb = inv_m.shape[0]
    ndof = 6 * nb
    Minv = np.zeros((ndof, ndof))
    for b in range(nb):
        Minv[6 * b : 6 * b + 3, 6 * b : 6 * b + 3] = np.eye(3) * inv_m[b]
        Minv[6 * b + 3 : 6 * b + 6, 6 * b + 3 : 6 * b + 6] = inv_I[b]
    u = np.zeros(ndof)
    for b in range(nb):
        u[6 * b : 6 * b + 3] = v_free[b]
        u[6 * b + 3 : 6 * b + 6] = w_free[b]

    def row(c, d):
        J = np.zeros(ndof)
        a, b = ia[c], ib[c]
        ra = p[c] - x[a]
        rb = p[c] - x[b]
        J[6 * b : 6 * b + 3] += d
        J[6 * b + 3 : 6 * b + 6] += np.cross(rb, d)
        J[6 * a : 6 * a + 3] -= d
        J[6 * a + 3 : 6 * a + 6] -= np.cross(ra, d)
        return J

    Jn = np.array([row(c, n[c]) for c in range(nc)])
    dirs = np.zeros((nc, N, 3))
    Jd = np.zeros((nc * N, ndof))
    for c in range(nc):
        for k in range(N):
            th = 2 * np.pi * k / N
            d = np.cos(th) * t1[c] + np.sin(th) * t2[c]
            dirs[c, k] = d
            Jd[c * N + k] = row(c, d)
    E = np.zeros((nc * N, nc))
    for c in range(nc):
        E[c * N : (c + 1) * N, c] = 1.0
    Mu = np.diag(mu)
    A = Jn @ Minv @ Jn.T
    B = Jn @ Minv @ Jd.T
    C = Jd @ Minv @ Jn.T
    D = Jd @ Minv @ Jd.T
    Z = np.zeros((nc, nc))
    Mlcp = np.block([[A, B, Z], [C, D, E], [Mu, -E.T, Z]])
    qlcp = np.concatenate([Jn @ u - bias, Jd @ u, np.zeros(nc)])
    return Mlcp, qlcp, Jn, Jd, dirs, Minv


def solve_lemke(ia, ib, p, n, t1, t2, bias, mu, x, v, w, inv_m, inv_I, N):
    """Reference solve: assemble the LCP, run Lemke, apply the impulses to v, w in place.

    Returns (lam (nc, 3) in the (n, t1, t2) basis, status).
    """
    nc = ia.shape[0]
    Mlcp, qlcp, Jn, Jd, dirs, Minv = build_stewart_trinkle_lcp(ia, ib, p, n, t1, t2, bias, mu, x, v, w, inv_m, inv_I, N)
    z, status = lemke(Mlcp, qlcp)
    lam_n = z[:nc]
    beta = z[nc : nc + nc * N].reshape(nc, N)
    lam = np.zeros((nc, 3))
    for c in range(nc):
        ft = np.zeros(3)
        for k in range(N):
            ft += beta[c, k] * dirs[c, k]
        lam[c, 0] = lam_n[c]
        lam[c, 1] = ft @ t1[c]
        lam[c, 2] = ft @ t2[c]
    apply_impulses(ia, ib, p, n, t1, t2, x, v, w, inv_m, inv_I, lam)
    return lam, status


# ----------------------------------------------------------------------------------
# Torsional (spin) friction of a contact patch
# ----------------------------------------------------------------------------------


@njit(cache=True)
def _apply_angular(w, inv_I, b, n, dl, sign):
    for i in range(3):
        w[b, i] += sign * dl * (inv_I[b, i, 0] * n[0] + inv_I[b, i, 1] * n[1] + inv_I[b, i, 2] * n[2])


@njit(cache=True)
def torsion_rows_mass(ia, ib, n, inv_I):
    nc = ia.shape[0]
    kt = np.zeros(nc)
    for c in range(nc):
        a = ia[c]
        b = ib[c]
        s = 0.0
        for i in range(3):
            for j in range(3):
                s += n[c, i] * (inv_I[a, i, j] + inv_I[b, i, j]) * n[c, j]
        kt[c] = s
    return kt


@njit(cache=True)
def apply_torsion(ia, ib, n, w, inv_I, lam_t):
    for c in range(ia.shape[0]):
        if lam_t[c] != 0.0:
            _apply_angular(w, inv_I, ia[c], n[c], lam_t[c], -1.0)
            _apply_angular(w, inv_I, ib[c], n[c], lam_t[c], 1.0)


@njit(cache=True)
def pgs_torsion_pass(ia, ib, n, w, inv_I, lam, lam_t, kt, mu, r_t):
    """One Gauss Seidel pass over the torsional rows: the spin impulse about the contact normal
    is bounded by mu * r_t * lambda_n (a contact patch of radius r_t, soft contact model).
    Returns the largest change."""
    max_change = 0.0
    for c in range(ia.shape[0]):
        a = ia[c]
        b = ib[c]
        lim = mu[c] * r_t * lam[c, 0]
        if kt[c] <= 0.0:
            continue
        if lim <= 0.0:
            if lam_t[c] != 0.0:
                _apply_angular(w, inv_I, a, n[c], lam_t[c], 1.0)
                _apply_angular(w, inv_I, b, n[c], lam_t[c], -1.0)
                lam_t[c] = 0.0
            continue
        wn = 0.0
        for i in range(3):
            wn += (w[b, i] - w[a, i]) * n[c, i]
        new = lam_t[c] - wn / kt[c]
        if new > lim:
            new = lim
        elif new < -lim:
            new = -lim
        dl = new - lam_t[c]
        if dl != 0.0:
            lam_t[c] = new
            _apply_angular(w, inv_I, a, n[c], dl, -1.0)
            _apply_angular(w, inv_I, b, n[c], dl, 1.0)
            if abs(dl) > max_change:
                max_change = abs(dl)
    return max_change


@njit(cache=True)
def pgs_solve_t(ia, ib, p, n, t1, t2, bias, mu, x, v, w, inv_m, inv_I, lam, iters, N, tol, lam_t, kt, r_t):
    """Projected Gauss Seidel with a torsional row per contact: the spin impulse about the normal
    is bounded by mu * r_t * lambda_n (contact patch of radius r_t). ``lam`` and ``lam_t`` hold
    warm start impulses already applied to v, w. Returns the number of iterations used."""
    nc = ia.shape[0]
    ra = np.empty((nc, 3))
    rb = np.empty((nc, 3))
    kn = np.empty(nc)
    k11 = np.empty(nc)
    k12 = np.empty(nc)
    k22 = np.empty(nc)
    for c in range(nc):
        a = ia[c]
        b = ib[c]
        for i in range(3):
            ra[c, i] = p[c, i] - x[a, i]
            rb[c, i] = p[c, i] - x[b, i]
        kn[c] = _row_mass(inv_m, inv_I, a, b, ra[c], rb[c], n[c])
        k11[c] = _row_mass(inv_m, inv_I, a, b, ra[c], rb[c], t1[c])
        k22[c] = _row_mass(inv_m, inv_I, a, b, ra[c], rb[c], t2[c])
        k12[c] = _row_cross_mass(inv_m, inv_I, a, b, ra[c], rb[c], t1[c], t2[c])
    imp = np.empty(3)
    used = 0
    for it in range(iters):
        max_change = 0.0
        for c in range(nc):
            a = ia[c]
            b = ib[c]
            # relative velocity of b with respect to a at the contact
            dvx = (v[b, 0] + w[b, 1] * rb[c, 2] - w[b, 2] * rb[c, 1]) - (v[a, 0] + w[a, 1] * ra[c, 2] - w[a, 2] * ra[c, 1])
            dvy = (v[b, 1] + w[b, 2] * rb[c, 0] - w[b, 0] * rb[c, 2]) - (v[a, 1] + w[a, 2] * ra[c, 0] - w[a, 0] * ra[c, 2])
            dvz = (v[b, 2] + w[b, 0] * rb[c, 1] - w[b, 1] * rb[c, 0]) - (v[a, 2] + w[a, 0] * ra[c, 1] - w[a, 1] * ra[c, 0])
            vn = dvx * n[c, 0] + dvy * n[c, 1] + dvz * n[c, 2]
            if kn[c] > 0.0:
                dl = (bias[c] - vn) / kn[c]
                new = lam[c, 0] + dl
                if new < 0.0:
                    new = 0.0
                dl = new - lam[c, 0]
                if dl != 0.0:
                    lam[c, 0] = new
                    for i in range(3):
                        imp[i] = dl * n[c, i]
                    _apply_impulse(v, w, inv_m, inv_I, a, ra[c], imp, -1.0)
                    _apply_impulse(v, w, inv_m, inv_I, b, rb[c], imp, 1.0)
                    if abs(dl) > max_change:
                        max_change = abs(dl)
            # friction block
            if lam[c, 0] > 0.0 and mu[c] > 0.0:
                dvx = (v[b, 0] + w[b, 1] * rb[c, 2] - w[b, 2] * rb[c, 1]) - (v[a, 0] + w[a, 1] * ra[c, 2] - w[a, 2] * ra[c, 1])
                dvy = (v[b, 1] + w[b, 2] * rb[c, 0] - w[b, 0] * rb[c, 2]) - (v[a, 1] + w[a, 2] * ra[c, 0] - w[a, 0] * ra[c, 2])
                dvz = (v[b, 2] + w[b, 0] * rb[c, 1] - w[b, 1] * rb[c, 0]) - (v[a, 2] + w[a, 0] * ra[c, 1] - w[a, 1] * ra[c, 0])
                vt1 = dvx * t1[c, 0] + dvy * t1[c, 1] + dvz * t1[c, 2]
                vt2 = dvx * t2[c, 0] + dvy * t2[c, 1] + dvz * t2[c, 2]
                det = k11[c] * k22[c] - k12[c] * k12[c]
                if det > 1e-300:
                    d1 = -(k22[c] * vt1 - k12[c] * vt2) / det
                    d2 = -(-k12[c] * vt1 + k11[c] * vt2) / det
                else:
                    d1 = -vt1 / max(k11[c], 1e-300)
                    d2 = -vt2 / max(k22[c], 1e-300)
                f1 = lam[c, 1] + d1
                f2 = lam[c, 2] + d2
                f1, f2 = project_polygon_metric(f1, f2, mu[c] * lam[c, 0], N, k11[c], k12[c], k22[c])
                d1 = f1 - lam[c, 1]
                d2 = f2 - lam[c, 2]
                if d1 != 0.0 or d2 != 0.0:
                    lam[c, 1] = f1
                    lam[c, 2] = f2
                    for i in range(3):
                        imp[i] = d1 * t1[c, i] + d2 * t2[c, i]
                    _apply_impulse(v, w, inv_m, inv_I, a, ra[c], imp, -1.0)
                    _apply_impulse(v, w, inv_m, inv_I, b, rb[c], imp, 1.0)
                    ch = math.sqrt(d1 * d1 + d2 * d2)
                    if ch > max_change:
                        max_change = ch
            elif lam[c, 1] != 0.0 or lam[c, 2] != 0.0:
                # normal impulse vanished: remove any friction impulse
                for i in range(3):
                    imp[i] = -(lam[c, 1] * t1[c, i] + lam[c, 2] * t2[c, i])
                _apply_impulse(v, w, inv_m, inv_I, a, ra[c], imp, -1.0)
                _apply_impulse(v, w, inv_m, inv_I, b, rb[c], imp, 1.0)
                lam[c, 1] = 0.0
                lam[c, 2] = 0.0
            # torsional row
            if kt[c] > 0.0:
                lim = mu[c] * r_t * lam[c, 0]
                wn = 0.0
                for i in range(3):
                    wn += (w[b, i] - w[a, i]) * n[c, i]
                newt = lam_t[c] - wn / kt[c]
                if newt > lim:
                    newt = lim
                elif newt < -lim:
                    newt = -lim
                dlt = newt - lam_t[c]
                if dlt != 0.0:
                    lam_t[c] = newt
                    _apply_angular(w, inv_I, a, n[c], dlt, -1.0)
                    _apply_angular(w, inv_I, b, n[c], dlt, 1.0)
                    # the torsional row converges with the translational rows it is coupled to;
                    # its change enters the stopping test as an equivalent force at the patch rim
                    if abs(dlt) / max(r_t, 1e-9) * 0.01 > max_change:
                        max_change = abs(dlt) / max(r_t, 1e-9) * 0.01
        used = it + 1
        if max_change < tol:
            break
    return used


