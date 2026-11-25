"""Grasp planning (Section 9.3).

Antipodal pairs of surface points are found by casting rays from surface samples through the
stone. Grasps are ranked by the Ferrari and Canny (1992) radius of the wrench hull with a soft
finger torsional friction term, by the distance of the grasp axis from the centre of mass and by
finger clearance at the target pose, and they can be evaluated on posterior shape samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from ..core.mesh import TriMesh
from ..core.transforms import SE3, tangent_basis


@dataclass
class Grasp:
    p1: np.ndarray
    p2: np.ndarray
    n1: np.ndarray  # outward normals at the contacts
    n2: np.ndarray
    axis: np.ndarray  # unit vector from p1 to p2
    width: float
    centre: np.ndarray
    quality: float = 0.0
    com_distance: float = 0.0
    clearance: float = 1.0
    score: float = 0.0
    meta: dict = field(default_factory=dict)

    def pose(self, approach: np.ndarray | None = None) -> SE3:
        """TCP pose for this grasp in the stone frame: y along the grasp axis, z along the approach."""
        from ..core.transforms import rot_to_quat

        y = self.axis
        if approach is None:
            approach = np.array([0.0, 0.0, -1.0])
        z = approach - np.dot(approach, y) * y
        if np.linalg.norm(z) < 1e-6:
            z = tangent_basis(y)[0]
        z /= np.linalg.norm(z)
        x = np.cross(y, z)
        R = np.column_stack([x, y, z])
        return SE3(rot_to_quat(R), self.centre)


def antipodal_grasps(mesh: TriMesh, mu_g: float = 1.0, n_pairs: int = 200, rng=None, max_width: float = 0.08, min_width: float = 0.005) -> list[Grasp]:
    """Antipodal grasp candidates: the angle between the grasp axis and the inward normals at
    both contacts is at most arctan(mu_g), and the width is within the finger range."""
    rng = np.random.default_rng(0) if rng is None else rng
    pts, nrm = mesh.sample_surface(n_pairs, rng)
    o = pts - 1e-5 * nrm
    d = -nrm
    t, tid, _, _ = mesh.bvh().intersect(o, d, tmin=1e-4)
    fn = mesh.face_normals()
    out = []
    cos_lim = np.cos(np.arctan(mu_g))
    for i in range(len(pts)):
        if tid[i] < 0:
            continue
        p1, n1 = pts[i], nrm[i]
        p2 = o[i] + t[i] * d[i]
        n2 = fn[tid[i]]
        w = np.linalg.norm(p2 - p1)
        if w < min_width or w > max_width:
            continue
        l = (p2 - p1) / w
        c1 = float(np.dot(l, -n1))
        c2 = float(np.dot(-l, -n2))
        if c1 >= cos_lim and c2 >= cos_lim:
            out.append(Grasp(p1, p2, n1, n2, l, float(w), 0.5 * (p1 + p2), meta={"cos1": c1, "cos2": c2}))
    return out


def primitive_wrenches(points, normals, mu: float, cone_sides: int = 8, torsion: float = 0.0, origin=None) -> np.ndarray:
    """Primitive contact wrenches (force into the object along -normal, friction pyramid edges,
    optional soft finger torsion) about ``origin``."""
    origin = np.zeros(3) if origin is None else np.asarray(origin)
    rows = []
    for p, n in zip(points, normals):
        n = np.asarray(n) / np.linalg.norm(n)
        f_in = -n
        t1, t2 = tangent_basis(f_in)
        r = np.asarray(p) - origin
        for k in range(cone_sides):
            th = 2 * np.pi * k / cone_sides
            f = f_in + mu * (np.cos(th) * t1 + np.sin(th) * t2)
            rows.append(np.concatenate([f, np.cross(r, f)]))
        if torsion > 0:
            rows.append(np.concatenate([np.zeros(3), torsion * f_in]))
            rows.append(np.concatenate([np.zeros(3), -torsion * f_in]))
    return np.array(rows).reshape(-1, 6)


def ferrari_canny(wrenches: np.ndarray, L: float = 0.05) -> float:
    """Radius of the largest ball under the L norm around the origin inside the wrench hull."""
    if wrenches.shape[0] < 7:
        return 0.0
    scale = np.array([1, 1, 1, 1 / L, 1 / L, 1 / L])
    W = wrenches * scale
    # a wrench set that does not span all six dimensions cannot enclose the origin
    if np.linalg.matrix_rank(W, tol=1e-9 * max(1.0, np.abs(W).max())) < 6:
        return 0.0
    try:
        hull = ConvexHull(W)
    except QhullError:
        try:
            hull = ConvexHull(W, qhull_options="QJ")
        except QhullError:
            return 0.0
    eq = hull.equations
    dist = -eq[:, -1]  # distance of each facet from the origin (inside when all positive)
    r = float(np.min(dist))
    return max(r, 0.0)


def grasp_quality(grasp: Grasp, mu: float = 1.0, cone_sides: int = 8, torsion: float = 0.005, L: float = 0.05, origin=None) -> float:
    W = primitive_wrenches([grasp.p1, grasp.p2], [grasp.n1, grasp.n2], mu, cone_sides, torsion, origin=grasp.centre if origin is None else origin)
    return ferrari_canny(W, L)


def rank_grasps(grasps: list[Grasp], com: np.ndarray, target_pose: SE3 | None = None, clearance_fn=None, mu: float = 1.0, weights=(1.0, 20.0, 5.0), shape_samples: list[TriMesh] | None = None, rng=None) -> list[Grasp]:
    """Score and sort grasps (best first).

    score = w0 * quality - w1 * distance of the grasp axis from the centre of mass
            + w2 * clearance, where clearance comes from ``clearance_fn(grasp, target_pose)``.
    If ``shape_samples`` are given, the quality is averaged over the samples by re-evaluating the
    contact normals and width on each sample mesh (grasp evaluated on posterior shapes).
    """
    com = np.asarray(com, dtype=np.float64)
    for g in grasps:
        q = grasp_quality(g, mu, origin=com)
        if shape_samples:
            qs = [q]
            for m in shape_samples:
                qs.append(_quality_on_sample(g, m, mu, com))
            q = float(np.mean(qs))
        g.quality = q
        r = com - g.p1
        g.com_distance = float(np.linalg.norm(r - np.dot(r, g.axis) * g.axis))
        g.clearance = float(clearance_fn(g, target_pose)) if clearance_fn is not None else 1.0
        g.score = weights[0] * g.quality - weights[1] * g.com_distance + weights[2] * g.clearance
    return sorted(grasps, key=lambda g: -g.score)


def _quality_on_sample(g: Grasp, mesh: TriMesh, mu: float, com: np.ndarray) -> float:
    """Re-evaluate a grasp on another shape sample: cast the grasp line through the sample."""
    bvh = mesh.bvh()
    o = g.centre[None, :] - 0.2 * g.axis[None, :]
    t, tid, _, _ = bvh.intersect(o, g.axis[None, :], tmin=1e-6)
    o2 = g.centre[None, :] + 0.2 * g.axis[None, :]
    t2, tid2, _, _ = bvh.intersect(o2, -g.axis[None, :], tmin=1e-6)
    if tid[0] < 0 or tid2[0] < 0:
        return 0.0
    fn = mesh.face_normals()
    p1 = o[0] + t[0] * g.axis
    p2 = o2[0] - t2[0] * g.axis
    W = primitive_wrenches([p1, p2], [fn[tid[0]], fn[tid2[0]]], mu, 8, 0.005, origin=com)
    return ferrari_canny(W)
