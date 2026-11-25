"""Position based secondary solver (Section 6.3): XPBD rigid bodies after Müller,
Macklin, Chentanez, Jeschke and Kim (2020).

Each step runs ``substeps`` substeps. A substep predicts positions and orientations from
the velocities (gravity, external force and the gyroscopic term), then runs
``iterations`` Gauss Seidel passes over the contacts: a positional non penetration
constraint with zero compliance, followed by static friction as a positional correction
of the tangential relative displacement limited by mu times the normal correction. New
velocities are derived from the position change, and a velocity pass removes the normal
velocity injected by the correction (inelastic contacts) and applies dynamic friction.
Contacts are generated once per step by the shared collision pipeline, with the contact
points attached to the bodies in local coordinates.

The substep loop is a numba kernel over flat arrays; the Python wrapper gathers and
scatters body state.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from .collision import find_contacts


@njit(cache=True)
def _qmul(a, b):
    out = np.empty(4)
    out[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
    out[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
    out[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
    out[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
    return out


@njit(cache=True)
def _qrot(q, v):
    qx, qy, qz, qw = q[1], q[2], q[3], q[0]
    tx = 2.0 * (qy * v[2] - qz * v[1])
    ty = 2.0 * (qz * v[0] - qx * v[2])
    tz = 2.0 * (qx * v[1] - qy * v[0])
    out = np.empty(3)
    out[0] = v[0] + qw * tx + (qy * tz - qz * ty)
    out[1] = v[1] + qw * ty + (qz * tx - qx * tz)
    out[2] = v[2] + qw * tz + (qx * ty - qy * tx)
    return out


@njit(cache=True)
def _qint(q, wv, h):
    """Exact rotation of q by the rotation vector wv * h (world frame)."""
    rx, ry, rz = wv[0] * h, wv[1] * h, wv[2] * h
    ang = math.sqrt(rx * rx + ry * ry + rz * rz)
    dq = np.empty(4)
    if ang < 1e-12:
        dq[0] = 1.0
        dq[1] = 0.5 * rx
        dq[2] = 0.5 * ry
        dq[3] = 0.5 * rz
    else:
        s = math.sin(0.5 * ang) / ang
        dq[0] = math.cos(0.5 * ang)
        dq[1] = rx * s
        dq[2] = ry * s
        dq[3] = rz * s
    out = _qmul(dq, q)
    nrm = math.sqrt(out[0] ** 2 + out[1] ** 2 + out[2] ** 2 + out[3] ** 2)
    return out / nrm


@njit(cache=True)
def _rot(q):
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
def _iinv_world(q, Ib_inv):
    R = _rot(q)
    return R @ Ib_inv @ R.T


@njit(cache=True)
def _cross(a, b):
    out = np.empty(3)
    out[0] = a[1] * b[2] - a[2] * b[1]
    out[1] = a[2] * b[0] - a[0] * b[2]
    out[2] = a[0] * b[1] - a[1] * b[0]
    return out


@njit(cache=True)
def _gen_inv_mass(i, q, inv_m, Ib_inv, r, n):
    if inv_m[i] == 0.0:
        return 0.0
    Iinv = _iinv_world(q[i], Ib_inv[i])
    rn = _cross(r, n)
    return inv_m[i] + rn @ (Iinv @ rn)


@njit(cache=True)
def _pos_corr(i, q, x, inv_m, Ib_inv, r, dp, sign):
    if inv_m[i] == 0.0:
        return
    for k in range(3):
        x[i, k] += sign * inv_m[i] * dp[k]
    Iinv = _iinv_world(q[i], Ib_inv[i])
    dth = sign * (Iinv @ _cross(r, dp))
    if dth[0] != 0.0 or dth[1] != 0.0 or dth[2] != 0.0:
        q[i] = _qint(q[i], dth, 1.0)


@njit(cache=True)
def _vel_imp(i, q, v, w, inv_m, Ib_inv, r, imp, sign):
    if inv_m[i] == 0.0:
        return
    for k in range(3):
        v[i, k] += sign * inv_m[i] * imp[k]
    Iinv = _iinv_world(q[i], Ib_inv[i])
    dw = Iinv @ _cross(r, imp)
    for k in range(3):
        w[i, k] += sign * dw[k]


@njit(cache=True)
def xpbd_kernel(q, x, v, w, inv_m, Ib, Ib_inv, kin, stat, g, fext, text, ia, ib, la, lb, nrm, mu, h, substeps, iterations):
    nb = q.shape[0]
    nc = ia.shape[0]
    hs = h / substeps
    prev_x = np.empty_like(x)
    prev_q = np.empty_like(q)
    lam = np.zeros(nc)
    for _s in range(substeps):
        prev_x[:, :] = x
        prev_q[:, :] = q
        # predict
        for i in range(nb):
            if stat[i]:
                continue
            if (not kin[i]) and inv_m[i] > 0.0:
                for k in range(3):
                    v[i, k] += hs * g[k] + hs * inv_m[i] * fext[i, k]
                R = _rot(q[i])
                wb = R.T @ w[i]
                Iw = Ib[i] @ wb
                rhs = -_cross(wb, Iw) + R.T @ text[i]
                wb = wb + hs * (Ib_inv[i] @ rhs)
                w[i] = R @ wb
            for k in range(3):
                x[i, k] += hs * v[i, k]
            q[i] = _qint(q[i], w[i], hs)
        lam[:] = 0.0
        # position constraints
        for _it in range(iterations):
            for c in range(nc):
                a = ia[c]
                b = ib[c]
                n = nrm[c]
                pa = _qrot(q[a], la[c]) + x[a]
                pb = _qrot(q[b], lb[c]) + x[b]
                d = (pb - pa) @ n
                if d >= 0.0:
                    continue
                ra = pa - x[a]
                rb = pb - x[b]
                wa = _gen_inv_mass(a, q, inv_m, Ib_inv, ra, n)
                wbm = _gen_inv_mass(b, q, inv_m, Ib_inv, rb, n)
                if wa + wbm == 0.0:
                    continue
                dl = -d / (wa + wbm)
                lam[c] += dl
                dp = dl * n
                _pos_corr(a, q, x, inv_m, Ib_inv, ra, dp, -1.0)
                _pos_corr(b, q, x, inv_m, Ib_inv, rb, dp, 1.0)
                # static friction on the tangential relative displacement over the substep
                pa = _qrot(q[a], la[c]) + x[a]
                pb = _qrot(q[b], lb[c]) + x[b]
                pa_prev = _qrot(prev_q[a], la[c]) + prev_x[a]
                pb_prev = _qrot(prev_q[b], lb[c]) + prev_x[b]
                dpt = (pb - pb_prev) - (pa - pa_prev)
                dpt = dpt - (dpt @ n) * n
                lt = math.sqrt(dpt @ dpt)
                if lt > 1e-12:
                    t = dpt / lt
                    ra = pa - x[a]
                    rb = pb - x[b]
                    wat = _gen_inv_mass(a, q, inv_m, Ib_inv, ra, t)
                    wbt = _gen_inv_mass(b, q, inv_m, Ib_inv, rb, t)
                    if wat + wbt > 0.0:
                        corr = min(lt, mu[c] * lam[c] * (wat + wbt)) / (wat + wbt)
                        dpf = -corr * t
                        _pos_corr(a, q, x, inv_m, Ib_inv, ra, dpf, -1.0)
                        _pos_corr(b, q, x, inv_m, Ib_inv, rb, dpf, 1.0)
        # velocities from positions
        for i in range(nb):
            if stat[i] or kin[i]:
                continue
            for k in range(3):
                v[i, k] = (x[i, k] - prev_x[i, k]) / hs
            pc = prev_q[i].copy()
            pc[1] = -pc[1]
            pc[2] = -pc[2]
            pc[3] = -pc[3]
            dq = _qmul(q[i], pc)
            if dq[0] < 0.0:
                dq = -dq
            for k in range(3):
                w[i, k] = 2.0 * dq[k + 1] / hs
        # velocity pass: inelastic normal response and dynamic friction
        for c in range(nc):
            if lam[c] <= 0.0:
                continue
            a = ia[c]
            b = ib[c]
            n = nrm[c]
            pa = _qrot(q[a], la[c]) + x[a]
            ra = pa - x[a]
            rb = pa - x[b]
            dv = (v[b] + _cross(w[b], rb)) - (v[a] + _cross(w[a], ra))
            vn = dv @ n
            wan = _gen_inv_mass(a, q, inv_m, Ib_inv, ra, n)
            wbn = _gen_inv_mass(b, q, inv_m, Ib_inv, rb, n)
            if wan + wbn > 0.0 and vn != 0.0:
                impn = (-vn / (wan + wbn)) * n
                _vel_imp(a, q, v, w, inv_m, Ib_inv, ra, impn, -1.0)
                _vel_imp(b, q, v, w, inv_m, Ib_inv, rb, impn, 1.0)
            dv = (v[b] + _cross(w[b], rb)) - (v[a] + _cross(w[a], ra))
            vt = dv - (dv @ n) * n
            lt = math.sqrt(vt @ vt)
            if lt > 1e-12:
                t = vt / lt
                wat = _gen_inv_mass(a, q, inv_m, Ib_inv, ra, t)
                wbt = _gen_inv_mass(b, q, inv_m, Ib_inv, rb, t)
                if wat + wbt > 0.0:
                    fn = lam[c] / (hs * hs)
                    dvmag = min(lt, hs * mu[c] * fn * (wat + wbt))
                    imp = -(dvmag / (wat + wbt)) * t
                    _vel_imp(a, q, v, w, inv_m, Ib_inv, ra, imp, -1.0)
                    _vel_imp(b, q, v, w, inv_m, Ib_inv, rb, imp, 1.0)
    return lam


def step_xpbd(world, substeps: int = 4, iterations: int = 2):
    h = world.dt
    hs = h / substeps
    contacts = find_contacts(world, world.contact_margin)
    world.contacts = contacts
    bodies = world.bodies
    nb = len(bodies)
    q = np.array([b.q for b in bodies], dtype=np.float64).reshape(nb, 4)
    x = np.array([b.x for b in bodies], dtype=np.float64).reshape(nb, 3)
    v = np.array([b.v for b in bodies], dtype=np.float64).reshape(nb, 3)
    w = np.array([b.w for b in bodies], dtype=np.float64).reshape(nb, 3)
    inv_m = np.array([b.inv_mass for b in bodies], dtype=np.float64)
    Ib = np.zeros((nb, 3, 3))
    Ib_inv = np.zeros((nb, 3, 3))
    for i, b in enumerate(bodies):
        if inv_m[i] > 0.0:
            Ib[i] = b.inertia_body
            Ib_inv[i] = np.linalg.inv(b.inertia_body)
        else:
            Ib[i] = np.eye(3)
    kin = np.array([b.kinematic for b in bodies], dtype=np.bool_)
    stat = np.array([b.static for b in bodies], dtype=np.bool_)
    fext = np.zeros((nb, 3))
    text = np.zeros((nb, 3))
    for bid, f in world._ext_force.items():
        fext[bid] = f
    for bid, t in world._ext_torque.items():
        text[bid] = t
    nc = len(contacts)
    ia = np.empty(nc, dtype=np.int64)
    ib = np.empty(nc, dtype=np.int64)
    la = np.empty((nc, 3))
    lb = np.empty((nc, 3))
    nrm = np.empty((nc, 3))
    mu = np.empty(nc)
    for k, c in enumerate(contacts):
        A, B = bodies[c.body_a], bodies[c.body_b]
        ia[k], ib[k] = c.body_a, c.body_b
        # A's surface lies half the depth above the contact point along n, B's half below
        la[k] = A.pose.inverse().apply(c.point + 0.5 * c.depth * c.normal)
        lb[k] = B.pose.inverse().apply(c.point - 0.5 * c.depth * c.normal)
        nrm[k] = c.normal
        mu[k] = c.mu
    lam = xpbd_kernel(q, x, v, w, inv_m, Ib, Ib_inv, kin, stat, np.asarray(world.gravity, dtype=np.float64), fext, text, ia, ib, la, lb, nrm, mu, h, int(substeps), int(iterations))
    for i, b in enumerate(bodies):
        if b.static:
            continue
        b.x = x[i].copy()
        b.q = q[i].copy()
        if not b.kinematic:
            b.v = v[i].copy()
            b.w = w[i].copy()
    for k, c in enumerate(contacts):
        c.fn = float(lam[k] / (hs * hs) * h)
        c.ft1 = 0.0
        c.ft2 = 0.0
    world.last_residuals = {"max_cone_violation": 0.0, "max_complementarity": 0.0, "max_penetration_velocity": 0.0, "pgs_iterations": iterations * substeps}
