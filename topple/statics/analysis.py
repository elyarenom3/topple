"""Margin model: from stone estimates and poses to a margin distribution (Sections 7, 8.8).

``MarginModel.evaluate`` takes the placed stones (estimate plus observed pose) and a
candidate (estimate plus pose), builds sample towers from the posterior, finds the contact
set of each sample tower by proximity between the sample meshes, computes the chosen margin
with ``topple.statics`` and propagates the uncertainty with one of four methods:

* ``none``: a point estimate on the posterior mean (hypothesis H2).
* ``mc``: Monte Carlo over joint posterior samples of all stones.
* ``unscented``: 2n + 1 sigma points on the reduced vector per stone (centre of mass offset,
  log friction coefficient, contact plane orientation, shape modes).
* ``first_order``: J Sigma J^T with a finite difference Jacobian on the reduced vector.

The reduced vector parametrises the posterior through a PCA of the posterior vertex
displacements (``shape_modes``), the centre of mass covariance, the log normal friction
posterior and the pose uncertainty; Monte Carlo draws from the same parametrisation so the
three propagation methods see one posterior.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from ..core.config import get
from ..core.mesh import TriMesh
from ..core.raycast import BVH
from ..core.transforms import SE3, quat_from_rotvec, quat_mul, quat_rotate
from .tower import StaticBody, StaticContact, Tower


@dataclass
class MarginResult:
    samples: np.ndarray
    mean: float
    std: float
    lo: float
    hi: float
    p_hat: float
    method: str
    kind: str
    n_contacts: int = 0
    details: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- posterior


class StoneParam:
    """Reduced parametrisation of one stone's posterior (Section 8.8, item 2)."""

    def __init__(self, est, pose: SE3, n_modes: int = 10):
        self.est = est
        self.pose = pose
        self.mean_mesh: TriMesh = est.mean_mesh
        self.density = float(est.density)
        modes, std = shape_modes_from_estimate(est, n_modes)
        self.modes = modes  # (K, nV, 3)
        self.mode_std = std  # (K,)
        self.n_modes = len(std)
        self.com_mean = np.asarray(getattr(est, "com_mean", np.zeros(3)), dtype=float)
        cov = np.asarray(getattr(est, "com_cov", np.zeros((3, 3))), dtype=float)
        w, U = np.linalg.eigh(cov + 1e-16 * np.eye(3))
        self.com_axes = U * np.sqrt(np.maximum(w, 0))[None, :]  # columns scaled by std
        self.log_mu_mean = float(getattr(est, "log_mu_mean", np.log(0.6)))
        self.log_mu_std = float(getattr(est, "log_mu_std", 0.0))
        self.rot_std = float(getattr(est, "pose_rot_std", 0.0))
        self.trans_std = float(getattr(est, "pose_trans_std", 0.0))
        self.plane_std = float(getattr(est, "plane_std", 0.0))
        # layout of the reduced vector xi: [com(3), log_mu(1), plane(2), modes(K), rot(3), trans(3)]
        self.dim = 3 + 1 + 2 + self.n_modes + 3 + 3

    def realise(self, xi: np.ndarray) -> dict:
        """Mesh (world frame), mass, com (world), mu, and the normal perturbation for xi."""
        k = self.n_modes
        c = xi[0:3]
        lmu = xi[3]
        plane = xi[4:6]
        a = xi[6 : 6 + k]
        rot = xi[6 + k : 9 + k]
        tr = xi[9 + k : 12 + k]
        V = self.mean_mesh.vertices
        if k:
            V = V + np.tensordot(a * self.mode_std, self.modes, axes=(0, 0))
        mesh_body = TriMesh(V, self.mean_mesh.faces, self.mean_mesh.name)
        vol = mesh_body.volume()
        mass = self.density * vol
        com_body = mesh_body.centroid() + self.com_mean + self.com_axes @ c
        # pose perturbation
        q = quat_mul(quat_from_rotvec(self.rot_std * rot), self.pose.q)
        T = SE3(q, self.pose.t + self.trans_std * tr)
        return {
            "mesh": mesh_body.transformed(T),
            "mass": float(mass),
            "com": T.apply(com_body),
            "mu": float(np.exp(self.log_mu_mean + self.log_mu_std * lmu)),
            "plane": self.plane_std * plane,  # small rotation (rad) of this stone's contact normals about x and y
            "pose": T,
            "name": getattr(self.est, "name", self.mean_mesh.name),
            "mu_table": getattr(self.est, "mu_table", None),
        }


def shape_modes_from_estimate(est, n_modes: int = 10):
    """PCA modes of the posterior vertex displacements. Returns (modes (K, nV, 3), std (K,))."""
    if hasattr(est, "shape_modes") and est.shape_modes is not None:
        m = np.asarray(est.shape_modes)
        s = np.asarray(est.mode_std)
        return m[:n_modes], s[:n_modes]
    samples = getattr(est, "samples", None)
    V0 = est.mean_mesh.vertices
    if not samples or len(samples) < 2:
        return np.zeros((0,) + V0.shape), np.zeros(0)
    meshes = [s.mesh for s in samples]
    if any(m.n_vertices != V0.shape[0] for m in meshes):
        return np.zeros((0,) + V0.shape), np.zeros(0)
    D = np.array([m.vertices.ravel() - V0.ravel() for m in meshes])
    D -= D.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(D, full_matrices=False)
    k = min(n_modes, len(S))
    std = S[:k] / np.sqrt(max(len(meshes) - 1, 1))
    modes = Vt[:k].reshape(k, -1, 3)
    keep = std > 1e-9
    return modes[keep], std[keep]


# ------------------------------------------------------------------------------ contacts


def farthest_point_subset(P: np.ndarray, k: int, first: int = 0) -> np.ndarray:
    n = len(P)
    if n <= k:
        return np.arange(n)
    chosen = [first]
    d = np.linalg.norm(P - P[first], axis=1)
    for _ in range(k - 1):
        j = int(np.argmax(d))
        chosen.append(j)
        d = np.minimum(d, np.linalg.norm(P - P[j], axis=1))
    return np.array(chosen)


def proximity_contacts(mesh_a: TriMesh, mesh_b: TriMesh, margin: float, max_points: int = 8, bvh_b: BVH | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Contact points of ``mesh_a`` against ``mesh_b``: vertices of a within ``margin`` of b's
    surface (and up to a small penetration), with the normal of b's closest face pointing from b
    towards a. Returns a list of (point, normal_from_b_to_a)."""
    bvh = mesh_b.bvh() if bvh_b is None else bvh_b
    dist, cp, tid = bvh.closest_points(mesh_a.vertices)
    inside = mesh_b.contains(mesh_a.vertices)
    sd = np.where(inside, -dist, dist)
    m = sd < margin
    if not m.any():
        return []
    pts = cp[m]
    normals = mesh_b.face_normals()[tid[m]]
    idx = farthest_point_subset(pts, max_points, first=int(np.argmin(sd[m])))
    return [(pts[i], normals[i]) for i in idx]


def ground_contacts(mesh: TriMesh, margin: float, max_points: int = 8) -> list[tuple[np.ndarray, np.ndarray]]:
    V = mesh.vertices
    m = V[:, 2] < margin
    if not m.any():
        return []
    pts = V[m].copy()
    pts[:, 2] = 0.0
    idx = farthest_point_subset(pts, max_points, first=int(np.argmin(V[m, 2])))
    return [(pts[i], np.array([0.0, 0.0, 1.0])) for i in idx]


def _vertical_gap(upper: TriMesh, lower: TriMesh | None) -> float:
    """Approximate vertical separation of ``upper`` above ``lower`` (negative when they
    interpenetrate); ``lower`` None means the ground plane z = 0."""
    if lower is None:
        return float(upper.vertices[:, 2].min())
    lo_u, hi_u = upper.bounds()
    lo_l, hi_l = lower.bounds()
    if np.any(lo_u[:2] > hi_l[:2]) or np.any(lo_l[:2] > hi_u[:2]):
        return np.inf
    gaps = []
    # upper vertices against the lower surface, restricted to the overlap band
    Vu = upper.vertices
    sel = Vu[:, 2] < hi_l[2] + 0.02
    if sel.any():
        d, _, _ = lower.bvh().closest_points(Vu[sel])
        inside = lower.contains(Vu[sel])
        gaps.append(float(np.where(inside, -d, d).min()))
    Vl = lower.vertices
    sel = Vl[:, 2] > lo_u[2] - 0.02
    if sel.any():
        d, _, _ = upper.bvh().closest_points(Vl[sel])
        inside = upper.contains(Vl[sel])
        gaps.append(float(np.where(inside, -d, d).min()))
    return min(gaps) if gaps else np.inf


def seat_stones(realised: list[dict], max_shift: float = 0.01, iters: int = 4, tol: float = 2e-5) -> list[dict]:
    """Translate each realised stone vertically, bottom to top in placement order, until it just
    touches the ground or a stone below it.

    A posterior sample perturbs shape and pose, so at the observed pose it can float above or sink
    into its support. Seating by vertical translation restores contact cheaply; the rotation part of
    settling is left to the margin analysis (a sample that would tip gives a small margin). Shifts
    are capped at ``max_shift`` so a sample never jumps onto a different support."""
    out = []
    for j, r in enumerate(realised):
        mesh = r["mesh"]
        com = np.asarray(r["com"], dtype=float).copy()
        total = 0.0
        for _ in range(iters):
            g = _vertical_gap(mesh, None)
            for lower in out:
                g = min(g, _vertical_gap(mesh, lower["mesh"]))
            if not np.isfinite(g) or abs(g) < tol:
                break
            shift = float(np.clip(g, -max_shift - total, max_shift - total)) if abs(total) < max_shift else 0.0
            if shift == 0.0:
                break
            mesh = mesh.translated([0.0, 0.0, -shift])
            com = com - np.array([0.0, 0.0, shift])
            total += shift
        r2 = dict(r)
        r2["mesh"] = mesh
        r2["com"] = com
        r2["seat_shift"] = total
        out.append(r2)
    return out


class HullGeom:
    """Convex hull of a point set with its face planes n . x + d <= 0 (scipy Qhull).

    All queries used by the static tower builder are closed form on the planes: vertical gaps by
    clipping vertical rays against the halfspaces, signed distances by the most violated plane
    (exact where the closest feature is a face, which holds for points near a face), and the
    contact normal as that plane's normal."""

    def __init__(self, points: np.ndarray):
        from scipy.spatial import ConvexHull

        P = np.asarray(points, dtype=float)
        h = ConvexHull(P)
        self.vertices = P[h.vertices]
        eq = h.equations
        # merge near duplicate planes (coplanar triangles)
        key = np.round(eq / np.maximum(np.abs(eq).max(axis=1, keepdims=True), 1e-12), 7)
        _, idx = np.unique(key, axis=0, return_index=True)
        eq = eq[np.sort(idx)]
        self.normals = eq[:, :3]
        self.offsets = eq[:, 3]
        self.lo = self.vertices.min(axis=0)
        self.hi = self.vertices.max(axis=0)

    def plane_distance(self, P: np.ndarray):
        """Most violated plane value per point (signed distance proxy) and that plane's index."""
        D = P @ self.normals.T + self.offsets[None, :]
        k = np.argmax(D, axis=1)
        return D[np.arange(len(P)), k], k

    def down_ray_entry(self, P: np.ndarray) -> np.ndarray:
        """Distance t >= -inf at which the ray P - t e_z enters the hull (inf when it misses)."""
        nz = self.normals[:, 2]
        num = P @ self.normals.T + self.offsets[None, :]  # n.(P - t ez) + d <= 0  <=>  t * nz >= n.P + d
        with np.errstate(divide="ignore", invalid="ignore"):
            t = num / nz[None, :]
        enter = np.where(nz[None, :] > 1e-12, t, -np.inf).max(axis=1)  # nz > 0: t >= num / nz
        exit_ = np.where(nz[None, :] < -1e-12, t, np.inf).min(axis=1)  # nz < 0: t <= num / nz
        parallel_bad = np.any((np.abs(nz)[None, :] <= 1e-12) & (num > 0), axis=1)
        miss = (enter > exit_) | parallel_bad
        return np.where(miss, np.inf, enter)

    def up_ray_entry(self, P: np.ndarray) -> np.ndarray:
        flipped = HullGeom.__new__(HullGeom)
        flipped.normals = self.normals * np.array([1.0, 1.0, -1.0])
        flipped.offsets = self.offsets
        Q = P * np.array([1.0, 1.0, -1.0])
        return flipped.down_ray_entry(Q)


def hull_vertical_gap(upper: HullGeom, lower: HullGeom | None) -> float:
    """Vertical separation of ``upper`` above ``lower`` (negative when they interpenetrate), from
    vertical rays through the vertices of both hulls."""
    if lower is None:
        return float(upper.vertices[:, 2].min())
    if np.any(upper.lo[:2] > lower.hi[:2]) or np.any(lower.lo[:2] > upper.hi[:2]):
        return np.inf
    # upper vertices moving down into the lower hull: gap = entry distance of a downward ray
    t1 = lower.down_ray_entry(upper.vertices)
    # lower vertices moving up into the upper hull
    t2 = upper.up_ray_entry(lower.vertices)
    g = min(float(np.min(t1)) if len(t1) else np.inf, float(np.min(t2)) if len(t2) else np.inf)
    if not np.isfinite(g):
        return np.inf
    # when the rays start inside the other hull the entry is negative: the overlap depth
    return g


def seat_hulls(realised: list[dict], hulls: list[HullGeom], max_shift: float = 0.01, iters: int = 3, tol: float = 2e-6):
    """``seat_stones`` on convex hulls with exact vertical gaps (fast path)."""
    out, out_h = [], []
    for r, h in zip(realised, hulls):
        com = np.asarray(r["com"], dtype=float).copy()
        V = h.vertices.copy()
        total = 0.0
        for _ in range(iters):
            g = float(V[:, 2].min())
            for lh in out_h:
                g = min(g, hull_vertical_gap(_shifted(h, V), lh))
            if not np.isfinite(g) or abs(g) < tol:
                break
            shift = float(np.clip(g, -max_shift - total, max_shift - total))
            if shift == 0.0:
                break
            V = V - np.array([0.0, 0.0, shift])
            com = com - np.array([0.0, 0.0, shift])
            total += shift
        hs = _shifted(h, V)
        r2 = dict(r)
        r2["com"] = com
        r2["seat_shift"] = total
        r2["mesh_shift"] = total
        out.append(r2)
        out_h.append(hs)
    return out, out_h


def _shifted(h: HullGeom, V: np.ndarray) -> HullGeom:
    dz = float(V[0, 2] - h.vertices[0, 2])
    g = HullGeom.__new__(HullGeom)
    g.vertices = V
    g.normals = h.normals
    g.offsets = h.offsets - h.normals[:, 2] * dz
    g.lo = h.lo + np.array([0.0, 0.0, dz])
    g.hi = h.hi + np.array([0.0, 0.0, dz])
    return g


def hull_contacts(upper: HullGeom, lower: HullGeom, band: float, max_points: int):
    """Contacts between two hulls: vertices of each within ``band`` of the other's surface, with the
    normal of the closest face of the lower hull (pointing from lower to upper)."""
    pts = []
    d, k = lower.plane_distance(upper.vertices)
    sel = (d < band) & (d > -4 * band)
    for p, kk in zip(upper.vertices[sel], k[sel]):
        pts.append((p, lower.normals[kk].copy()))
    d2, k2 = upper.plane_distance(lower.vertices)
    sel2 = (d2 < band) & (d2 > -4 * band)
    for p, kk in zip(lower.vertices[sel2], k2[sel2]):
        pts.append((p, -upper.normals[kk].copy()))
    if len(pts) > max_points:
        P = np.array([q for q, _ in pts])
        depth = np.concatenate([d[sel], d2[sel2]])
        keep = farthest_point_subset(P, max_points, first=int(np.argmin(depth)))
        pts = [pts[i] for i in keep]
    return pts


def _rotate_hull(h: HullGeom, R: np.ndarray, pivot: np.ndarray) -> HullGeom:
    """Hull after x' = R (x - pivot) + pivot."""
    g = HullGeom.__new__(HullGeom)
    g.vertices = (h.vertices - pivot) @ R.T + pivot
    n2 = h.normals @ R.T
    g.normals = n2
    g.offsets = h.offsets + h.normals @ pivot - n2 @ pivot
    g.lo = g.vertices.min(axis=0)
    g.hi = g.vertices.max(axis=0)
    return g


def _support_points(h: HullGeom, supports: list, band: float) -> np.ndarray:
    pts = []
    for s in supports:
        if s is None:
            V = h.vertices
            pts.append(V[V[:, 2] < band])
        else:
            if np.any(h.lo > s.hi + band) or np.any(s.lo > h.hi + band):
                continue
            d, _ = s.plane_distance(h.vertices)
            pts.append(h.vertices[d < band])
            d2, _ = h.plane_distance(s.vertices)
            pts.append(s.vertices[d2 < band])
    pts = [p for p in pts if len(p)]
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def _seat_vertical(h: HullGeom, com: np.ndarray, supports: list, max_shift: float, iters: int = 3, tol: float = 2e-6):
    total = 0.0
    for _ in range(iters):
        g = np.inf
        for s in supports:
            g = min(g, float(h.vertices[:, 2].min()) if s is None else hull_vertical_gap(h, s))
        if not np.isfinite(g) or abs(g) < tol:
            break
        shift = float(np.clip(g, -max_shift - total, max_shift - total))
        if shift == 0.0:
            break
        h = _shifted(h, h.vertices - np.array([0.0, 0.0, shift]))
        com = com - np.array([0.0, 0.0, shift])
        total += shift
    return h, com, total


def settle_hull_stone(h: HullGeom, com: np.ndarray, supports: list, band: float, max_tilt_deg: float = 15.0, step_deg: float = 0.5, max_shift: float = 0.01):
    """Quasi static settling of one convex stone on fixed supports (the ground is ``None``): seat it
    vertically, then while the centre of mass projects outside the support polygon, tip it about the
    nearest support edge (or vertex) by ``step_deg`` and seat again. Stops when the centre of mass is
    inside, or after ``max_tilt_deg`` of total rotation (the sample then keeps its last pose and its
    small or zero margin reports the instability). Returns (hull, com, total rotation in degrees)."""
    from ..core.transforms import quat_from_axis_angle, quat_to_rot
    from ..planning.candidates import _tip_axis

    h, com, _ = _seat_vertical(h, com, supports, max_shift)
    total = 0.0
    while total < max_tilt_deg:
        P = _support_points(h, supports, band)
        if len(P) == 0:
            break
        axis_pivot, _ = _tip_axis(P, com)
        if axis_pivot is None:
            break
        axis, pivot = axis_pivot
        R = quat_to_rot(quat_from_axis_angle(axis, np.radians(step_deg)))
        h = _rotate_hull(h, R, pivot)
        com = R @ (com - pivot) + pivot
        total += step_deg
        h, com, _ = _seat_vertical(h, com, supports, max_shift)
    return h, com, total


def seat_hulls_rotational(realised: list[dict], hulls: list[HullGeom], band: float, max_tilt_deg: float = 15.0, step_deg: float = 0.5):
    """Settle each stone in placement order on the ground and the stones placed before it."""
    out, out_h = [], []
    for r, h in zip(realised, hulls):
        supports = [None] + out_h
        h2, com, rot = settle_hull_stone(h, np.asarray(r["com"], dtype=float), supports, band, max_tilt_deg, step_deg)
        r2 = dict(r)
        r2["com"] = com
        r2["settle_rotation_deg"] = rot
        out.append(r2)
        out_h.append(h2)
    return out, out_h


def build_tower_hull(realised: list[dict], band: float, max_points: int = 8, gravity=(0, 0, -9.81), seat: bool = True, settle: bool = True) -> Tower:
    """Fast static tower on convex hulls (the engine's collision geometry for nearly convex stones).

    With ``settle`` each stone is settled quasi statically (vertical seating plus tipping about its
    support polygon) so that a posterior sample is analysed in the pose it would come to rest in;
    without it the stones are only seated vertically."""
    hulls = [HullGeom(r["mesh"].vertices) for r in realised]
    if seat and settle:
        realised, hulls = seat_hulls_rotational(realised, hulls, band)
    elif seat:
        realised, hulls = seat_hulls(realised, hulls)
    return tower_from_hulls(realised, hulls, band, max_points, gravity)


def tower_from_hulls(realised: list[dict], hulls: list[HullGeom], band: float, max_points: int = 8, gravity=(0, 0, -9.81)) -> Tower:
    """Static tower from settled hulls: ground contacts are hull vertices within ``band`` of z = 0,
    stone contacts come from ``hull_contacts`` between every pair placed one on another."""
    bodies = [StaticBody(0, 0.0, np.zeros(3), True, "ground")]
    for j, r in enumerate(realised):
        bodies.append(StaticBody(j + 1, r["mass"], np.asarray(r["com"], dtype=float), False, r.get("name", f"stone{j}")))
    contacts: list[StaticContact] = []
    for j, (r, h) in enumerate(zip(realised, hulls)):
        V = h.vertices
        sel = V[:, 2] < band
        if sel.any():
            P = V[sel].copy()
            keep = farthest_point_subset(P, max_points, first=int(np.argmin(P[:, 2])))
            mu = _pair_mu(r, {"name": "ground", "mu": 0.6})
            for i in keep:
                p = P[i].copy()
                p[2] = 0.0
                contacts.append(StaticContact(0, j + 1, p, _tilt_normal(np.array([0.0, 0.0, 1.0]), r.get("plane")), float(mu)))
    for i in range(len(realised)):
        for j in range(i + 1, len(realised)):
            hi_, hj = hulls[i], hulls[j]
            if np.any(hi_.lo > hj.hi + band) or np.any(hj.lo > hi_.hi + band):
                continue
            pts = hull_contacts(hj, hi_, band, max_points)
            mu = _pair_mu(realised[i], realised[j])
            for p, n in pts:
                contacts.append(StaticContact(i + 1, j + 1, p, _tilt_normal(n, realised[j].get("plane")), float(mu)))
    t = Tower(bodies, contacts, np.asarray(gravity, dtype=float))
    t.realised = realised
    t.hulls = hulls
    return t


def build_tower(realised: list[dict], margin: float, max_points: int = 8, gravity=(0, 0, -9.81), mu_pairs=None, seat: bool = True, geometry: str = "hull", settle: bool = True) -> Tower:
    """Static tower from realised stones (dicts with mesh, mass, com, mu, plane) in placement order
    (each stone rests on earlier ones). Body 0 is the ground. With ``seat`` the stones are first
    seated vertically on their supports and ``margin`` is the contact band around zero separation.

    ``geometry="hull"`` replaces each stone by its convex hull before seating, matching the
    collision geometry of the physics engine (which collides convex pieces, the hull for stones
    with V / V_hull of at least 0.97); ``"mesh"`` uses the surface mesh itself."""
    if geometry == "hull":
        return build_tower_hull(realised, margin, max_points, gravity, seat, settle)
    if seat:
        realised = seat_stones(realised)
    bodies = [StaticBody(0, 0.0, np.zeros(3), True, "ground")]
    for j, r in enumerate(realised):
        bodies.append(StaticBody(j + 1, r["mass"], np.asarray(r["com"], dtype=float), False, r.get("name", f"stone{j}")))
    contacts: list[StaticContact] = []
    bvhs = [r["mesh"].bvh() for r in realised]
    bounds = [r["mesh"].bounds() for r in realised]
    for j, r in enumerate(realised):
        for p, n in ground_contacts(r["mesh"], margin, max_points):
            mu = _pair_mu(r, {"name": "ground", "mu": 0.6}) if mu_pairs is None else mu_pairs(0, j + 1)
            contacts.append(StaticContact(0, j + 1, p, _tilt_normal(n, r.get("plane")), float(mu)))
    for i in range(len(realised)):
        for j in range(i + 1, len(realised)):
            lo_i, hi_i = bounds[i]
            lo_j, hi_j = bounds[j]
            if np.any(lo_i > hi_j + margin) or np.any(lo_j > hi_i + margin):
                continue
            # the earlier placed stone supports the later one
            a, b = i, j
            pts = proximity_contacts(realised[b]["mesh"], realised[a]["mesh"], margin, max_points, bvhs[a])
            pts2 = proximity_contacts(realised[a]["mesh"], realised[b]["mesh"], margin, max_points, bvhs[b])
            pts = pts + [(p, -n) for p, n in pts2]
            if len(pts) > max_points:
                P = np.array([p for p, _ in pts])
                keep = farthest_point_subset(P, max_points)
                pts = [pts[k] for k in keep]
            mu = _pair_mu(realised[a], realised[b]) if mu_pairs is None else mu_pairs(a + 1, b + 1)
            for p, n in pts:
                contacts.append(StaticContact(a + 1, b + 1, p, _tilt_normal(n, realised[b].get("plane")), float(mu)))
    t = Tower(bodies, contacts, np.asarray(gravity, dtype=float))
    t.realised = realised
    return t


def _pair_mu(ra: dict, rb: dict) -> float:
    """Pair friction: the true table when an oracle estimate carries one, else the geometric mean."""
    ta, tb = ra.get("mu_table"), rb.get("mu_table")
    if ta is not None and rb.get("name") in ta:
        return float(ta[rb["name"]])
    if tb is not None and ra.get("name") in tb:
        return float(tb[ra["name"]])
    return float(np.sqrt(ra["mu"] * rb["mu"]))


def _tilt_normal(n: np.ndarray, plane) -> np.ndarray:
    if plane is None or not np.any(plane):
        return np.asarray(n, dtype=float)
    q = quat_from_rotvec(np.array([plane[0], plane[1], 0.0]))
    return quat_rotate(q, np.asarray(n, dtype=float))


# ------------------------------------------------------------------------------- model


class MarginModel:
    def __init__(self, cfg: dict, cond, rng: np.random.Generator):
        self.cfg = cfg
        self.cond = cond
        self.rng = rng
        self.kind = cond.margin
        self.method = cond.propagation
        self.cone_sides = int(cond.cone_sides)
        self.n_samples = int(get(cfg, "propagation.mc_samples", 256))
        self.n_modes = int(get(cfg, "propagation.shape_modes", 10))
        self.tau = float(get(cfg, "propagation.survival_tilt_deg", 3.0))
        self.margin_m = float(get(cfg, "statics.contact_band", 5e-4))
        self.geometry = str(get(cfg, "statics.geometry", "hull"))
        self.settle_samples = bool(get(cfg, "statics.settle_samples", True))
        self.max_points = int(get(cfg, "world.max_manifold_points", 8))
        self.n_azimuth = int(get(cfg, "statics.n_azimuth", 36))
        self.gravity = np.asarray(get(cfg, "world.gravity", (0, 0, -9.81)), dtype=float)

    # ------------------------------------------------------------ core margin
    def margin_of_tower(self, tower: Tower, n_azimuth=None) -> float:
        from . import margins

        n_az = self.n_azimuth if n_azimuth is None else n_azimuth
        if len(tower.contacts) == 0:
            return 0.0
        if self.kind == "tilt":
            return float(margins.tower_margin(tower, kind="tilt", cone_sides=self.cone_sides, n_azimuth=n_az))
        if self.kind == "push":
            return float(margins.tower_margin(tower, kind="push", cone_sides=self.cone_sides))
        if self.kind == "fc":
            return float(margins.tower_margin(tower, kind="fc", cone_sides=self.cone_sides, force_budget_factor=float(get(self.cfg, "statics.fc_force_budget_factor", 2.0)), length_scale=float(get(self.cfg, "statics.fc_length_scale", 0.1))))
        if self.kind == "support":
            top = max(tower.bodies, key=lambda b: b.com[2]).id
            return float(margins.support_polygon_margin(tower, top)[0])
        if self.kind == "rocking":
            from .rocking import rocking_tilt_margin

            realised = getattr(tower, "realised", None)
            hulls = getattr(tower, "hulls", None)
            if realised is None or hulls is None:
                return 0.0
            h = self.cfg.get("harness", {})
            return float(rocking_tilt_margin(realised, hulls, self.gravity, self.cone_sides, n_azimuth=min(n_az, 12), fail_rot_deg=float(h.get("fail_rot_deg", 2.0)), fail_disp_m=float(h.get("fail_disp_m", 0.005)), band=self.margin_m)[0])
        if self.kind == "simulation":
            realised = getattr(tower, "realised", None)
            hulls = getattr(tower, "hulls", None)
            if realised is not None and hulls is not None:
                # simulate the settled hull geometry the static analysis used
                from scipy.spatial import ConvexHull as _CH

                sims = []
                for r, h in zip(realised, hulls):
                    ch = _CH(h.vertices)
                    m = TriMesh(h.vertices, ch.simplices).fix_orientation()
                    m = _outward(m)
                    sims.append(dict(r, mesh=m))
                realised = sims
            return float(simulation_margin(tower, self.cfg, realised))
        raise ValueError(self.kind)

    # ------------------------------------------------------------ evaluation
    def _params(self, placed: list[dict], cand_est, cand_pose: SE3) -> list[StoneParam]:
        ps = [StoneParam(p["estimate"], p["pose"], self.n_modes) for p in placed]
        ps.append(StoneParam(cand_est, cand_pose, self.n_modes))
        return ps

    def _margin_at(self, params: list[StoneParam], xi_all: np.ndarray, n_azimuth=None) -> tuple[float, int]:
        realised = []
        off = 0
        for p in params:
            realised.append(p.realise(xi_all[off : off + p.dim]))
            off += p.dim
        tower = build_tower(realised, self.margin_m, self.max_points, self.gravity, geometry=self.geometry, settle=self.settle_samples)
        return self.margin_of_tower(tower, n_azimuth), len(tower.contacts)

    def evaluate(self, placed: list[dict], cand_est, cand_pose: SE3, n_samples: int | None = None, n_azimuth: int | None = None, method: str | None = None) -> MarginResult:
        params = self._params(placed, cand_est, cand_pose)
        dim = sum(p.dim for p in params)
        method = self.method if method is None else method
        if method == "none" or dim == 0:
            m, nc = self._margin_at(params, np.zeros(dim), n_azimuth)
            return MarginResult(np.array([m]), m, 0.0, m, m, float(m > self.tau), "none", self.kind, nc)
        if method == "mc":
            S = self.n_samples if n_samples is None else int(n_samples)
            xs = self.rng.standard_normal((S, dim))
            vals = np.empty(S)
            nc = 0
            for s in range(S):
                vals[s], nc = self._margin_at(params, xs[s], n_azimuth)
            return self._from_samples(vals, nc, "mc")
        if method == "unscented":
            from ..perception.propagate import unscented

            f = lambda x: self._margin_at(params, x, n_azimuth)[0]
            # alpha = 1, kappa = 0: all weights positive and the sigma points at the typical radius of
            # a joint sample (sqrt(n) standard deviations). The margin is piecewise smooth and often
            # exactly zero, so the small alpha default (a central weight near -1 / alpha^2) is unusable.
            mean, var, pts, vals = unscented(f, np.zeros(dim), np.eye(dim), alpha=1.0, beta=2.0, kappa=0.0)
            std = float(np.sqrt(max(var, 0)))
            p = float(norm.sf((self.tau - mean) / std)) if std > 1e-9 else float(mean > self.tau)
            return MarginResult(np.asarray(vals), float(mean), std, mean - 1.96 * std, mean + 1.96 * std, p, "unscented", self.kind, details={"sigma_points": len(vals)})
        if method == "first_order":
            f = lambda x: self._margin_at(params, x, n_azimuth)[0]
            x0 = np.zeros(dim)
            m0 = f(x0)
            h = 0.5
            J = np.empty(dim)
            for i in range(dim):
                e = np.zeros(dim)
                e[i] = h
                J[i] = (f(x0 + e) - f(x0 - e)) / (2 * h)
            var = float(J @ J)  # Sigma = I in the standardised reduced vector
            std = float(np.sqrt(var))
            p = float(norm.sf((self.tau - m0) / std)) if std > 1e-9 else float(m0 > self.tau)
            return MarginResult(np.array([m0]), float(m0), std, m0 - 1.96 * std, m0 + 1.96 * std, p, "first_order", self.kind, details={"jacobian": J})
        raise ValueError(method)

    def _from_samples(self, vals: np.ndarray, nc: int, method: str) -> MarginResult:
        from .survival import survival_probability

        vals = np.asarray(vals, dtype=float)
        p = survival_probability(vals, self.tau)
        return MarginResult(vals, float(vals.mean()), float(vals.std()), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)), float(p), method, self.kind, nc)

    def quick_margin(self, placed: list[dict], cand_est, cand_pose: SE3) -> float:
        """Fast point estimate on the mean for candidate screening (support polygon or coarse tilt)."""
        params = self._params(placed, cand_est, cand_pose)
        realised = [p.realise(np.zeros(p.dim)) for p in params]
        tower = build_tower(realised, self.margin_m, self.max_points, self.gravity, geometry=self.geometry)
        if len(tower.contacts) == 0:
            return -90.0
        from . import margins

        top = tower.bodies[-1].id
        try:
            m_sp = float(margins.support_polygon_margin(tower, top)[0])
        except Exception:
            m_sp = -90.0
        if m_sp <= 0:
            return m_sp
        if self.kind == "tilt":
            return float(margins.tower_margin(tower, kind="tilt", cone_sides=self.cone_sides, n_azimuth=8))
        return self.margin_of_tower(tower)


def _outward(m: TriMesh) -> TriMesh:
    """Orient every hull face away from the centroid."""
    c = m.vertices.mean(axis=0)
    n = m.face_normals()
    tc = m.triangles().mean(axis=1)
    flip = np.einsum("ij,ij->i", n, tc - c) < 0
    F = m.faces.copy()
    F[flip] = F[flip][:, [0, 2, 1]]
    return TriMesh(m.vertices, F, m.name)


def simulation_margin(tower: Tower, cfg: dict, realised=None) -> float:
    """E8: simulation based placement score (Furrer et al. 2017, Liu et al. 2021 score placements
    by physics simulation). Delegates to ``topple.statics.simscore`` which builds a dynamic world
    from the estimated meshes and sweeps a coarse gravity tilt."""
    from .simscore import simulated_tilt_margin

    return simulated_tilt_margin(realised if realised is not None else tower, cfg)
