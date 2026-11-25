"""Four procedural stone families (Section 5.1). Each returns a watertight TriMesh.

A: convex hull of anisotropic Gaussian points, subdivided, Taubin smoothed, optional fractal relief.
B: icosphere scaled to (a, b, c) and displaced along normals by octaves of gradient noise.
C: superquadric (Barr 1981) obtained by radial projection of an icosphere, with perturbation.
D: one Voronoi cell of a fractured block (Raghavachary 2002), rounded by subdivision and smoothing.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection

from ..core.mesh import TriMesh, icosphere
from .noise import fbm3


def _hull_mesh(points: np.ndarray, name: str) -> TriMesh:
    return TriMesh(points, np.zeros((0, 3), dtype=np.int64), name).convex_hull()


def radial_remesh_convex(hull: TriMesh, subdivisions: int = 3) -> TriMesh:
    """Resample a convex mesh with icosphere topology.

    For a convex polytope containing its centroid, the boundary point along unit direction d
    from the centroid is r(d) d with r(d) = min_i b_i / (n_i . d) over the face planes with
    n_i . d > 0. Mapping an icosphere this way gives well shaped triangles everywhere, which
    the subdivision of a few large hull faces does not.
    """
    c = hull.centroid()
    n = hull.face_normals()
    b = np.einsum("ij,ij->i", n, hull.triangles()[:, 0] - c)  # plane offsets about the centroid
    sph = icosphere(subdivisions, 1.0)
    d = sph.vertices
    nd = d @ n.T  # (n_dirs, n_faces)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(nd > 1e-12, b[None, :] / nd, np.inf).min(axis=1)
    V = c + d * r[:, None]
    return TriMesh(V, sph.faces.copy(), hull.name).fix_orientation()


def _displace_along_normals(mesh: TriMesh, amplitude: float, frequency: float, octaves: int, gain: float, seed: int) -> TriMesh:
    if amplitude <= 0:
        return mesh
    R = mesh.bounding_radius()
    d = fbm3(mesh.vertices / R * frequency, octaves=octaves, gain=gain, seed=seed)
    n = mesh.vertex_normals()
    out = TriMesh(mesh.vertices + amplitude * d[:, None] * n, mesh.faces.copy(), mesh.name)
    return out


def family_a_hull(rng, a, b, c, n_points: int = 40, subdivisions: int = 3, smooth_iters: int = 10, fractal_amp: float = 0.0, fractal_octaves: int = 3, point_distribution: str = "gaussian", seed=None) -> TriMesh:
    """Convex hull of random points then smoothing (Section 5.1, family A).

    ``point_distribution`` is "gaussian" (anisotropic Gaussian with covariance diag(a^2, b^2, c^2),
    as in the specification) or "shell" (points on the ellipsoid surface with radial jitter, which
    gives blocky, compact hulls from few points). The hull is resampled with icosphere topology
    (``radial_remesh_convex``) before Taubin smoothing and the optional fractal relief.
    """
    seed = int(rng.integers(2**31 - 1)) if seed is None else int(seed)
    axes = np.array([a, b, c], dtype=np.float64)
    if point_distribution == "shell":
        dirs = rng.normal(size=(int(n_points), 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        pts = dirs * axes * rng.uniform(0.85, 1.0, size=(int(n_points), 1))
    else:
        pts = rng.normal(size=(int(n_points), 3)) * axes
    hull = _hull_mesh(pts, "family_a")
    m = radial_remesh_convex(hull, subdivisions)
    if smooth_iters > 0:
        m = m.taubin_smooth(smooth_iters)
    m = _displace_along_normals(m, fractal_amp, 2.0, fractal_octaves, 0.5, seed)
    return m.fix_orientation()


def family_b_ellipsoid(rng, a, b, c, amplitude: float, octaves: int = 4, gain: float = 0.5, subdivisions: int = 3, frequency: float = 1.5, seed=None) -> TriMesh:
    """Noise displaced ellipsoid (Section 5.1, family B)."""
    seed = int(rng.integers(2**31 - 1)) if seed is None else int(seed)
    m = icosphere(subdivisions, 1.0)
    unit = m.vertices.copy()
    axes = np.array([a, b, c], dtype=np.float64)
    V = unit * axes
    n = unit / axes  # gradient of the ellipsoid implicit function
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    d = fbm3(unit * frequency, octaves=octaves, gain=gain, seed=seed)
    V = V + amplitude * d[:, None] * n
    return TriMesh(V, m.faces.copy(), "family_b").fix_orientation()


def superquadric_inside_outside(p: np.ndarray, a, b, c, e1, e2) -> np.ndarray:
    """Barr's inside outside function F(x); F = 1 on the surface."""
    p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
    x = np.abs(p[:, 0] / a)
    y = np.abs(p[:, 1] / b)
    z = np.abs(p[:, 2] / c)
    return (x ** (2.0 / e2) + y ** (2.0 / e2)) ** (e2 / e1) + z ** (2.0 / e1)


def family_c_superquadric(rng, a, b, c, e1, e2, perturb: float = 0.0, subdivisions: int = 3, octaves: int = 3, seed=None) -> TriMesh:
    """Superquadric with perturbation (Section 5.1, family C).

    F is homogeneous: F(r d) = r^(2/e1) F(d), so the surface point along unit direction d is
    d * F(d)^(-e1/2). Mapping an icosphere this way keeps the angular sampling uniform.
    """
    seed = int(rng.integers(2**31 - 1)) if seed is None else int(seed)
    m = icosphere(subdivisions, 1.0)
    d = m.vertices
    F = superquadric_inside_outside(d, a, b, c, e1, e2)
    r = F ** (-e1 / 2.0)
    V = d * r[:, None]
    out = TriMesh(V, m.faces.copy(), "family_c")
    out = _displace_along_normals(out, perturb, 2.0, octaves, 0.5, seed)
    return out.fix_orientation()


def family_d_voronoi(rng, block_extents=(1.0, 0.8, 0.7), n_seeds: int = 12, pick=None, subdivisions: int = 3, smooth_iters: int = 2) -> TriMesh:
    """One Voronoi fragment of a block (Section 5.1, family D).

    The cell of seed i is the intersection of the half spaces bounded by the bisector planes
    with every other seed, clipped by the block faces. The convex cell is resampled with
    icosphere topology and lightly smoothed; the erosion step of Section 5.2 rounds it further.
    """
    ext = np.asarray(block_extents, dtype=np.float64)
    seeds = (rng.random((int(n_seeds), 3)) - 0.5) * ext
    if pick is None:
        pick = int(np.argmin(np.linalg.norm(seeds, axis=1)))  # most central fragment
    s = seeds[pick]
    hs = []
    for j in range(len(seeds)):
        if j == pick:
            continue
        n = seeds[j] - s
        mid = 0.5 * (seeds[j] + s)
        hs.append([n[0], n[1], n[2], -float(n @ mid)])
    for k in range(3):
        e = np.zeros(3)
        e[k] = 1.0
        hs.append([e[0], e[1], e[2], -ext[k] / 2])
        hs.append([-e[0], -e[1], -e[2], -ext[k] / 2])
    hsi = HalfspaceIntersection(np.array(hs), s)
    pts = np.unique(np.round(hsi.intersections, 9), axis=0)
    hull = _hull_mesh(pts, "family_d")
    m = radial_remesh_convex(hull, subdivisions)
    if smooth_iters > 0:
        m = m.taubin_smooth(smooth_iters)
    return m.fix_orientation()
