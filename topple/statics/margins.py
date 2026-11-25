"""Stability margins (Sections 7.2 to 7.6).

* Margin 1, gravity tilt margin: the smallest rotation of gravity about a horizontal axis at
  which the equilibrium program stops being feasible on the whole interval [0, alpha].
* Margin 2, disturbance wrench margin: the largest horizontal force at the top stone's centre of
  mass, divided by its weight.
* Margin 3, support quality radius: the Ferrari and Canny radius adapted to support contacts.
* Support polygon view: the single object tipping angle arctan(d / h_c).
* Subassembly cuts: every margin evaluated on each upper subassembly, minimum over cuts.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from ..core.transforms import quat_from_axis_angle, quat_rotate, tangent_basis
from .equilibrium import ContactLP, contact_basis
from .tower import Tower


def analytic_block_tilt(mu: float, b: float, h: float) -> float:
    """Critical tilt angle in degrees of a block of width b and height h on a plane (Section 7.2)."""
    return float(np.degrees(min(np.arctan(mu), np.arctan(b / h))))


def horizontal_basis(down: np.ndarray):
    """Two unit vectors orthogonal to the gravity direction."""
    up = -np.asarray(down, dtype=np.float64)
    up = up / max(np.linalg.norm(up), 1e-300)
    t1, t2 = tangent_basis(up)
    return t1, t2


def rotated_gravity(g0: np.ndarray, azimuth: float, alpha_deg: float) -> np.ndarray:
    """Rotate g0 by alpha about the horizontal axis with azimuth ``azimuth``."""
    g0 = np.asarray(g0, dtype=np.float64)
    t1, t2 = horizontal_basis(g0 / max(np.linalg.norm(g0), 1e-300))
    axis = np.cos(azimuth) * t1 + np.sin(azimuth) * t2
    q = quat_from_axis_angle(axis, np.radians(alpha_deg))
    return quat_rotate(q, g0)


# ------------------------------------------------------------------- margin 1


def _critical_angle(lp: ContactLP, g0, azimuth, tol_deg, max_deg, coarse_deg):
    """Largest alpha such that the LP is feasible on [0, alpha] for one azimuth."""

    def ok(a):
        return lp.feasible(rotated_gravity(g0, azimuth, a))[0]

    if ok(max_deg):
        alpha = max_deg
    else:
        lo, hi = 0.0, max_deg
        while hi - lo > tol_deg:
            mid = 0.5 * (lo + hi)
            if ok(mid):
                lo = mid
            else:
                hi = mid
        alpha = lo
    # verify the whole interval on a coarse grid (feasibility need not be monotone in alpha)
    a = coarse_deg
    while a < alpha:
        if not ok(a):
            lo, hi = a - coarse_deg, a
            while hi - lo > tol_deg:
                mid = 0.5 * (lo + hi)
                if ok(mid):
                    lo = mid
                else:
                    hi = mid
            alpha = lo
            break
        a += coarse_deg
    return alpha


def tilt_margin(tower: Tower, cone_sides: int = 8, n_azimuth: int = 36, tol_deg: float = 0.05, max_deg: float = 89.0, coarse_deg: float = 5.0, refine: bool = True):
    """Gravity tilt margin (Section 7.2).

    Returns (margin_deg, azimuth_rad, per_azimuth) where ``per_azimuth`` is an array of
    critical angles at the ``n_azimuth`` regularly spaced azimuths.
    """
    lp = ContactLP(tower, cone_sides)
    g0 = tower.gravity
    azimuths = np.linspace(0.0, 2 * np.pi, n_azimuth, endpoint=False)
    if not lp.feasible(g0)[0]:
        return 0.0, 0.0, np.zeros(n_azimuth)
    per = np.array([_critical_angle(lp, g0, az, tol_deg, max_deg, coarse_deg) for az in azimuths])
    k = int(np.argmin(per))
    best, best_az = per[k], azimuths[k]
    if refine and n_azimuth >= 3 and best < max_deg:
        # golden section over the azimuth between the two neighbours of the minimum
        step = 2 * np.pi / n_azimuth
        a, b = azimuths[k] - step, azimuths[k] + step
        phi = (np.sqrt(5.0) - 1) / 2
        c = b - phi * (b - a)
        d = a + phi * (b - a)
        fc = _critical_angle(lp, g0, c, tol_deg, max_deg, coarse_deg)
        fd = _critical_angle(lp, g0, d, tol_deg, max_deg, coarse_deg)
        for _ in range(6):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - phi * (b - a)
                fc = _critical_angle(lp, g0, c, tol_deg, max_deg, coarse_deg)
            else:
                a, c, fc = c, d, fd
                d = a + phi * (b - a)
                fd = _critical_angle(lp, g0, d, tol_deg, max_deg, coarse_deg)
        for az, val in ((c, fc), (d, fd)):
            if val < best:
                best, best_az = val, az % (2 * np.pi)
    return float(best), float(best_az), per


# ------------------------------------------------------------------- margin 2


def push_margin(tower: Tower, top_body: int | None = None, cone_sides: int = 8, n_dir: int = 8):
    """Disturbance wrench margin (Section 7.3): min over horizontal directions of s* / W_top.

    Returns (s_over_weight, direction (3,)).
    """
    top_id = tower.top_body().id if top_body is None else top_body
    lp = ContactLP(tower, cone_sides)
    t1, t2 = horizontal_basis(tower.down())
    W = tower.body(top_id).mass * np.linalg.norm(tower.gravity)
    best, best_e = np.inf, None
    for th in np.linspace(0.0, 2 * np.pi, n_dir, endpoint=False):
        e = np.cos(th) * t1 + np.sin(th) * t2
        s, _, _ = lp.push(e, top_id)
        if s < best:
            best, best_e = s, e
    return float(best / W), best_e


# ------------------------------------------------------------------- margin 3


def _support_contacts(tower: Tower):
    """Contacts between a ground body and a free body, with the sign giving the force on the free body."""
    ground = tower.ground_ids()
    out = []
    for c in tower.contacts:
        if c.body_a in ground and c.body_b not in ground:
            out.append((c, +1.0))
        elif c.body_b in ground and c.body_a not in ground:
            out.append((c, -1.0))
    return out


def primitive_wrenches(tower: Tower, cone_sides: int = 8) -> np.ndarray:
    """Wrenches B_i g_ik of the friction pyramid edges of the support contacts (n, 6)."""
    rows = []
    for c, sign in _support_contacts(tower):
        B, _ = contact_basis(c, cone_sides)
        for k in range(cone_sides):
            th = 2 * np.pi * k / cone_sides
            gk = np.array([c.mu * np.cos(th), c.mu * np.sin(th), 1.0])
            rows.append(sign * (B @ gk))
    return np.array(rows).reshape(-1, 6)


def _ball_radius_in_hull(points: np.ndarray, centre: np.ndarray) -> float:
    """Largest r with ball(centre, r) inside conv(points); 0 if the centre is outside."""
    if np.linalg.matrix_rank(points - centre, tol=1e-9 * max(1.0, np.abs(points).max())) < points.shape[1]:
        return 0.0
    try:
        hull = ConvexHull(points)
    except QhullError:
        try:
            hull = ConvexHull(points, qhull_options="QJ")
        except QhullError:
            return 0.0
    eq = hull.equations  # n . x + d <= 0 inside
    dist = -(eq[:, :-1] @ centre + eq[:, -1])
    r = float(np.min(dist))
    return max(r, 0.0)


def fc_margin(tower: Tower, cone_sides: int = 8, force_budget_factor: float = 2.0, length_scale: float = 0.1) -> float:
    """Support quality radius (Section 7.4, author's proposal).

    The free bodies are treated as one rigid body supported by the contacts with the ground
    bodies. W_F is the convex hull of {0} and F w_ik with F = force_budget_factor times the total
    weight; the margin is the radius of the largest ball under the L norm around -w_g inside W_F.
    """
    W = primitive_wrenches(tower, cone_sides)
    if W.shape[0] == 0:
        return 0.0
    free = tower.free_bodies()
    if not free:
        return 0.0
    g = tower.gravity
    w_g = np.zeros(6)
    for b in free:
        w_g[:3] += b.mass * g
        w_g[3:] += np.cross(b.com, b.mass * g)
    F = force_budget_factor * tower.total_weight()
    scale = np.array([1, 1, 1, 1 / length_scale, 1 / length_scale, 1 / length_scale])
    pts = np.vstack([np.zeros(6), F * W]) * scale
    centre = -w_g * scale
    return _ball_radius_in_hull(pts, centre)


# ----------------------------------------------------------- support polygon


def _poly_signed_distance_2d(poly: np.ndarray, p: np.ndarray) -> float:
    """Signed distance from p to the boundary of the convex polygon (counter clockwise), positive inside."""
    n = len(poly)
    if n == 1:
        return -float(np.linalg.norm(p - poly[0]))
    if n == 2:
        a, b = poly
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-300), 0, 1)
        return -float(np.linalg.norm(p - (a + t * ab)))
    inside = True
    dmin = np.inf
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ab = b - a
        L = np.linalg.norm(ab)
        if L < 1e-12:
            continue
        # outward normal for a counter clockwise polygon
        nrm = np.array([ab[1], -ab[0]]) / L
        s = np.dot(p - a, nrm)
        if s > 0:
            inside = False
        t = np.clip(np.dot(p - a, ab) / (L * L), 0, 1)
        dmin = min(dmin, np.linalg.norm(p - (a + t * ab)))
    return float(dmin if inside else -dmin)


def support_polygon_margin(tower: Tower, body_id: int):
    """Single object support polygon view (Section 7.5).

    The support contacts of ``body_id`` are its contacts with lower bodies (or the ground). The
    centre of mass is projected along g onto the fitted support plane; d is the signed distance
    from the projection to the polygon boundary and h_c the height of the centre of mass above
    the plane. Returns (tipping_angle_deg, d, h_c, polygon (k, 3) world points).
    """
    body = tower.body(body_id)
    hb = tower.height(body.com)
    pts, nrms = [], []
    for c in tower.contacts:
        other = None
        if c.body_a == body_id:
            other = c.body_b
        elif c.body_b == body_id:
            other = c.body_a
        if other is None:
            continue
        ob = tower.body(other)
        if ob.is_ground or tower.height(ob.com) < hb:
            pts.append(c.point)
            nrms.append(c.normal if c.body_b == body_id else -c.normal)
    if not pts:
        return 0.0, -np.inf, 0.0, np.zeros((0, 3))
    pts = np.array(pts)
    n = np.mean(np.array(nrms), axis=0)
    n = n / max(np.linalg.norm(n), 1e-300)
    p0 = pts.mean(axis=0)
    g = tower.gravity
    gn = float(np.dot(g, n))
    if abs(gn) < 1e-9:
        return 0.0, -np.inf, 0.0, pts
    t = np.dot(p0 - body.com, n) / gn
    proj = body.com + t * g
    h_c = abs(t) * np.linalg.norm(g)
    # 2D coordinates in the plane
    u, v = tangent_basis(n)
    P2 = np.column_stack([(pts - p0) @ u, (pts - p0) @ v])
    q2 = np.array([(proj - p0) @ u, (proj - p0) @ v])
    if len(P2) >= 3:
        try:
            hull = ConvexHull(P2)
            poly2 = P2[hull.vertices]  # counter clockwise in 2D
            poly3 = pts[hull.vertices]
        except QhullError:
            # collinear points: keep the two extremes
            d = P2 - P2.mean(axis=0)
            axis = np.linalg.svd(d)[2][0]
            s = d @ axis
            idx = [int(np.argmin(s)), int(np.argmax(s))]
            poly2, poly3 = P2[idx], pts[idx]
    else:
        poly2, poly3 = P2, pts
    d = _poly_signed_distance_2d(poly2, q2)
    ang = float(np.degrees(np.arctan2(d, h_c))) if h_c > 0 else (90.0 if d > 0 else 0.0)
    return ang, float(d), float(h_c), poly3


# -------------------------------------------------------------------- cuts


def subassembly_margins(tower: Tower, kind: str = "tilt", **kw) -> list[float]:
    """Margin of each upper subassembly, bottom cut first (Section 7.6)."""
    out = []
    for k in range(tower.n_cuts()):
        sub = tower.upper_subassembly(k)
        if kind == "tilt":
            out.append(tilt_margin(sub, **kw)[0])
        elif kind == "push":
            out.append(push_margin(sub, **kw)[0])
        elif kind == "fc":
            out.append(fc_margin(sub, **kw))
        elif kind == "support":
            lowest = sub.ordered_free_bodies()[0]
            out.append(support_polygon_margin(sub, lowest.id)[0])
        else:
            raise ValueError(f"unknown margin kind {kind}")
    return out


def tower_margin(tower: Tower, kind: str = "tilt", **kw) -> float:
    """Minimum over cuts of the chosen margin (0 for an empty tower)."""
    vals = subassembly_margins(tower, kind, **kw)
    return float(min(vals)) if vals else 0.0
