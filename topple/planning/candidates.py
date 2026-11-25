"""Candidate placements and quasi static settling (Section 9.1).

Positions are sampled on the tower top, each candidate is lowered along -z to first contact
with signed distance queries against the support meshes, and then tipped quasi statically about
its contact set until the centre of mass projects inside the support polygon. The dynamic
alternative ``settle_candidate`` runs a short simulation in a physics ``World``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from ..core.mesh import TriMesh
from ..core.transforms import SE3, quat_from_axis_angle, quat_mul, quat_normalize, quat_rotate

_Z = np.array([0.0, 0.0, 1.0])


@dataclass
class SettleResult:
    pose: SE3
    contact_points: np.ndarray  # (k, 3) world points on the support
    contact_normals: np.ndarray  # (k, 3) support normals
    support_index: np.ndarray  # (k,) index into support_meshes
    stable: bool
    iterations: int
    min_distance: float = 0.0


def upper_surface_points(mesh: TriMesh, n: int = 200, rng=None, min_nz: float = 0.3) -> np.ndarray:
    """World points on the upward facing part of a mesh (candidate positions)."""
    rng = np.random.default_rng(0) if rng is None else rng
    pts, nrm = mesh.sample_surface(max(4 * n, 64), rng)
    keep = nrm[:, 2] > min_nz
    pts = pts[keep]
    if len(pts) == 0:
        top = mesh.vertices[np.argsort(-mesh.vertices[:, 2])[: max(n // 4, 1)]]
        return top
    if len(pts) > n:
        pts = pts[rng.choice(len(pts), n, replace=False)]
    return pts


def sample_candidates(top_surface_points: np.ndarray, stone_spec, n: int, rng=None, resting=None, clearance: float = 0.02) -> list[SE3]:
    """Sample n candidate poses: a position from the top surface, a uniform yaw and a resting
    orientation (``resting`` list of (quat, prob) or ``stone_spec.resting``), with the stone's
    lowest point ``clearance`` above the chosen surface point."""
    rng = np.random.default_rng(0) if rng is None else rng
    mesh = stone_spec.mesh if hasattr(stone_spec, "mesh") else stone_spec
    if resting is None:
        resting = getattr(stone_spec, "resting", None)
    if not resting:
        from .resting import resting_orientations

        resting = resting_orientations(mesh, n_drops=100, rng=rng)
        if not resting:
            resting = [(np.array([1.0, 0.0, 0.0, 0.0]), 1.0)]
    quats = np.array([r[0] for r in resting])
    probs = np.array([max(r[1], 1e-3) for r in resting])
    probs /= probs.sum()
    pts = np.atleast_2d(top_surface_points)
    out = []
    for _ in range(n):
        p = pts[rng.integers(len(pts))]
        yaw = rng.uniform(0, 2 * np.pi)
        q = quat_mul(quat_from_axis_angle(_Z, yaw), quats[rng.choice(len(quats), p=probs)])
        V = quat_rotate(q, mesh.vertices)
        t = np.array([p[0], p[1], p[2] + clearance - V[:, 2].min()])
        out.append(SE3(q, t))
    return out


def _min_signed_distance(points: np.ndarray, support_meshes: list, which: bool = False, reject_gap: float = 0.3):
    """Minimum signed distance over the support meshes for each point."""
    best = np.full(len(points), np.inf)
    idx = np.full(len(points), -1)
    for k, m in enumerate(support_meshes):
        lo, hi = m.bounds()
        plo, phi = points.min(axis=0), points.max(axis=0)
        if np.any(plo > hi + reject_gap) or np.any(phi < lo - reject_gap):
            continue
        d = m.signed_distance(points)
        better = d < best
        best[better] = d[better]
        idx[better] = k
    if which:
        return best, idx
    return best


def _world_points(samples: np.ndarray, pose: SE3) -> np.ndarray:
    return pose.apply(samples)


def drop_to_contact(stone_mesh: TriMesh, pose: SE3, support_meshes: list, samples: np.ndarray | None = None, tol: float = 1e-4, max_drop: float = 0.5, rng=None, levels: int = 32):
    """Translate the stone along z so that its surface samples just touch the supports.

    The shift is found by two grid searches on the vertical offset (coarse then fine) so that
    the minimum signed distance is zero within ``tol``. The stone may move up if it starts in
    penetration. Returns (pose, min_distance).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    if samples is None:
        samples = _surface_samples(stone_mesh, rng)
    P = _world_points(samples, pose)
    d0 = float(_min_signed_distance(P, support_meshes, reject_gap=1.0).min())
    if not np.isfinite(d0):
        return pose, d0
    lo, hi = -0.03, max(d0, 0.0) + min(max_drop, max(0.2, 2 * abs(d0)))

    def scan(lo, hi, n):
        shifts = np.linspace(lo, hi, n)
        Pall = np.concatenate([P - np.array([0.0, 0.0, s]) for s in shifts])
        d = _min_signed_distance(Pall, support_meshes).reshape(n, -1).min(axis=1)
        return shifts, d

    shifts, d = scan(lo, hi, levels)
    # first index where penetration starts
    pen = np.nonzero(d < 0)[0]
    if len(pen) == 0:
        best = shifts[-1]
        return SE3(pose.q, pose.t - np.array([0.0, 0.0, best])), float(d[-1])
    k = int(pen[0])
    if k == 0:
        # already penetrating at the smallest shift: search upward
        shifts, d = scan(lo - 0.2, lo, levels)
        pen = np.nonzero(d < 0)[0]
        k = int(pen[0]) if len(pen) else levels - 1
        if k == 0:
            return SE3(pose.q, pose.t - np.array([0.0, 0.0, shifts[0]])), float(d[0])
    a, b = shifts[k - 1], shifts[k]
    for _ in range(2):
        shifts, d = scan(a, b, levels)
        pen = np.nonzero(d < 0)[0]
        if len(pen) == 0:
            break
        k = int(pen[0])
        if k == 0:
            break
        a, b = shifts[k - 1], shifts[k]
        if (b - a) / levels < tol:
            break
    best = a
    pose2 = SE3(pose.q, pose.t - np.array([0.0, 0.0, best]))
    dmin = float(_min_signed_distance(_world_points(samples, pose2), support_meshes).min())
    return pose2, dmin


def _tip_axis(contacts: np.ndarray, com: np.ndarray):
    """Rotation axis (unit, through pivot) that lowers the centre of mass, or None if stable."""
    c2 = contacts[:, :2]
    g2 = com[:2]
    n = len(contacts)
    if n >= 3:
        try:
            hull = ConvexHull(c2)
            poly = contacts[hull.vertices]
            poly2 = c2[hull.vertices]
            m = len(poly2)
            inside, best_d, best_feat = True, np.inf, None
            for i in range(m):
                a, b = poly2[i], poly2[(i + 1) % m]
                ab = b - a
                L = np.linalg.norm(ab)
                if L < 1e-12:
                    continue
                nrm = np.array([ab[1], -ab[0]]) / L
                s = np.dot(g2 - a, nrm)
                if s > 0:
                    inside = False
                t = np.clip(np.dot(g2 - a, ab) / (L * L), 0, 1)
                d = np.linalg.norm(g2 - (a + t * ab))
                if d < best_d:
                    best_d = d
                    if 1e-6 < t < 1 - 1e-6:
                        best_feat = ("edge", poly[i], poly[(i + 1) % m])
                    else:
                        best_feat = ("vertex", poly[i] if t <= 0.5 else poly[(i + 1) % m], None)
            if inside:
                return None, best_d
            feat = best_feat
        except QhullError:
            # collinear contacts: two extremes
            d = c2 - c2.mean(axis=0)
            axis = np.linalg.svd(d)[2][0]
            s = d @ axis
            feat = ("edge", contacts[int(np.argmin(s))], contacts[int(np.argmax(s))])
            ab = feat[2][:2] - feat[1][:2]
            L = np.linalg.norm(ab)
            t = np.clip(np.dot(g2 - feat[1][:2], ab) / max(L * L, 1e-18), 0, 1)
            if t <= 1e-6 or t >= 1 - 1e-6:
                feat = ("vertex", feat[1] if t <= 0.5 else feat[2], None)
    elif n == 2:
        a, b = contacts
        ab = b[:2] - a[:2]
        L = np.linalg.norm(ab)
        t = np.clip(np.dot(g2 - a[:2], ab) / max(L * L, 1e-18), 0, 1)
        if 1e-6 < t < 1 - 1e-6 and L > 1e-6:
            feat = ("edge", a, b)
        else:
            feat = ("vertex", a if t <= 0.5 else b, None)
    else:
        feat = ("vertex", contacts[0], None)
    if feat[0] == "edge":
        a, b = feat[1], feat[2]
        axis = b - a
        axis[2] = 0.0
        axis /= max(np.linalg.norm(axis), 1e-12)
        pivot = a
    else:
        pivot = feat[1]
        r = com - pivot
        rh = np.array([r[0], r[1], 0.0])
        if np.linalg.norm(rh) < 1e-9:
            return None, 0.0
        axis = np.array([-r[1], r[0], 0.0]) / np.linalg.norm(rh)
    # choose the sign that lowers the centre of mass
    vel = np.cross(axis, com - pivot)
    if vel[2] > 0:
        axis = -axis
    return (axis, pivot), 0.0


def _surface_samples(stone_mesh: TriMesh, rng, n: int = 200, max_vertices: int = 400) -> np.ndarray:
    """Vertices (subsampled when dense) plus uniform surface samples, in the body frame."""
    V = stone_mesh.vertices
    if len(V) > max_vertices:
        V = V[rng.choice(len(V), max_vertices, replace=False)]
    s, _ = stone_mesh.sample_surface(n, rng)
    return np.concatenate([V, s])


def settle_pose(stone_mesh: TriMesh, pose: SE3, support_meshes: list, rng=None, max_iter: int = 120, angle_step_deg: float = 2.0, com: np.ndarray | None = None, contact_tol: float = 1e-3, inside_margin: float = 1e-3, samples: np.ndarray | None = None) -> SettleResult:
    """Lower and tip the stone quasi statically until it rests on at least three contacts with
    the centre of mass projection inside their convex hull."""
    rng = np.random.default_rng(0) if rng is None else rng
    com_b = stone_mesh.centroid() if com is None else np.asarray(com, dtype=np.float64)
    if samples is None:
        samples = _surface_samples(stone_mesh, rng)
    pose, dmin = drop_to_contact(stone_mesh, pose, support_meshes, samples, rng=rng)
    stable = False
    it = 0
    contacts = np.zeros((0, 3))
    cidx = np.zeros(0, dtype=int)
    for it in range(1, max_iter + 1):
        P = _world_points(samples, pose)
        d, idx = _min_signed_distance(P, support_meshes, which=True)
        near = d < contact_tol
        contacts, cidx = P[near], idx[near]
        if len(contacts) == 0:
            pose, dmin = drop_to_contact(stone_mesh, pose, support_meshes, samples, rng=rng)
            continue
        com_w = pose.apply(com_b)
        axis_pivot, inside_d = _tip_axis(contacts, com_w)
        if axis_pivot is None:
            if len(contacts) >= 3 and inside_d >= inside_margin:
                stable = True
                break
            if len(contacts) >= 3 and inside_d >= 0:
                stable = True  # marginal but inside
                break
            # on a single point directly under the centre of mass: nudge
            axis = np.array([1.0, 0.0, 0.0])
            pivot = contacts[0]
        else:
            axis, pivot = axis_pivot
        dq = quat_from_axis_angle(axis, np.radians(angle_step_deg))
        q_new = quat_normalize(quat_mul(dq, pose.q))
        t_new = pivot + quat_rotate(dq, pose.t - pivot)
        pose = SE3(q_new, t_new)
        pose, dmin = drop_to_contact(stone_mesh, pose, support_meshes, samples, rng=rng)
    # final contact set and support normals
    P = _world_points(samples, pose)
    d, idx = _min_signed_distance(P, support_meshes, which=True)
    near = d < contact_tol
    contacts, cidx = P[near], idx[near]
    normals = np.zeros((len(contacts), 3))
    for k in np.unique(cidx):
        if k < 0:
            continue
        m = support_meshes[k]
        sel = cidx == k
        _, cp, tid = m.bvh().closest_points(contacts[sel])
        normals[sel] = m.face_normals()[tid]
        contacts[sel] = cp
    return SettleResult(pose, contacts, normals, cidx, stable, it, float(dmin))


def settle_candidate(world, body, pose: SE3, steps: int = 400) -> SE3:
    """Dynamic settling: clone the world, place the body at ``pose`` at rest, step, read back."""
    w = world.clone()
    b = w.bodies[body.id] if hasattr(body, "id") and body.id >= 0 else body
    b.pose = pose
    b.v[:] = 0.0
    b.w[:] = 0.0
    w.step(steps)
    return SE3(b.q.copy(), b.x.copy())
