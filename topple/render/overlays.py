"""Scientific overlays as scene geometry (Section 10.6).

Every function returns a ``TriMesh`` (plus per vertex colours where the overlay encodes a
scalar field) that the caller adds with ``Scene.add_overlay``. Overlays are rendered in a second
pass with unlit emissive materials and the same camera, so they share depth of field, motion
blur and occlusion with the stones.

O1 contact force arrows and friction cones (``arrow``, ``cone``), O2 thrust line (``tube``),
O3 section cap (``section_cap``), O4 centre of mass sphere, plumb line and support polygon
(``sphere``, ``bar``, ``polygon_outline``), O5 uncertainty ellipsoid (``ellipsoid``),
O6 stability landscape (``heatmap_plane``), O8 gauge (``dial``), O9 trajectory painting
(``polyline``).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

from ..core.mesh import TriMesh, concatenate, icosphere
from ..core.transforms import rotation_from_two_vectors
from .color import colormap

_Z = np.array([0.0, 0.0, 1.0])


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else _Z.copy()


def _align_z(mesh: TriMesh, axis, origin) -> TriMesh:
    R = rotation_from_two_vectors(_Z, _unit(axis))
    V = mesh.vertices @ R.T + np.asarray(origin, dtype=np.float64)
    return TriMesh(V, mesh.faces.copy(), mesh.name)


def _cylinder_z(radius: float, length: float, n: int = 16, closed: bool = True) -> TriMesh:
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([radius * np.cos(ang), radius * np.sin(ang), np.zeros(n)], 1)
    V = [ring, ring + np.array([0, 0, length])]
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [[i, j, n + j], [i, n + j, n + i]]
    if closed:
        V.append(np.array([[0, 0, 0.0], [0, 0, length]]))
        cb, ct = 2 * n, 2 * n + 1
        for i in range(n):
            j = (i + 1) % n
            F += [[cb, j, i], [ct, n + i, n + j]]
    return TriMesh(np.concatenate(V), np.array(F), "cylinder").fix_orientation()


def _cone_z(radius: float, height: float, n: int = 24, closed: bool = True) -> TriMesh:
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([radius * np.cos(ang), radius * np.sin(ang), np.full(n, height)], 1)
    V = np.concatenate([[[0, 0, 0.0]], ring, [[0, 0, height]]])
    F = []
    for i in range(n):
        j = (i + 1) % n
        F.append([0, 1 + i, 1 + j])
        if closed:
            F.append([n + 1, 1 + j, 1 + i])
    m = TriMesh(V, np.array(F), "cone")
    return m.fix_orientation() if closed else m


def arrow(p, v, length: float | None = None, radius: float = 0.002, head_fraction: float = 0.3) -> TriMesh:
    """Arrow from p along v. ``length`` defaults to |v|. Head radius is 2.5 x shaft radius."""
    p = np.asarray(p, dtype=np.float64)
    L = float(np.linalg.norm(v)) if length is None else float(length)
    L = max(L, 1e-6)
    hl = head_fraction * L
    shaft = _cylinder_z(radius, L - hl, 12)
    head = _cone_z(2.5 * radius, hl, 16).translated([0, 0, L - hl])
    m = concatenate([shaft, head])
    m.name = "arrow"
    return _align_z(m, v, p)


def cone(apex, axis, half_angle: float, height: float, n: int = 32) -> TriMesh:
    """Friction cone of half angle (radians) opening along ``axis`` from ``apex`` (O1)."""
    r = height * np.tan(half_angle)
    m = _cone_z(r, height, n, closed=True)
    return _align_z(m, axis, apex)


def sphere(c, r: float, subdivisions: int = 2) -> TriMesh:
    return icosphere(subdivisions, r).translated(c)


def ellipsoid(c, Sigma, chi2: float = 7.814727903251179, subdivisions: int = 3) -> TriMesh:
    """Level set (x - c)^T Sigma^-1 (x - c) = chi2 of a Gaussian (default 95 percent for 3 dof)."""
    w, U = np.linalg.eigh(np.asarray(Sigma, dtype=np.float64))
    w = np.maximum(w, 1e-18)
    A = U @ np.diag(np.sqrt(chi2 * w))
    s = icosphere(subdivisions, 1.0)
    V = s.vertices @ A.T + np.asarray(c, dtype=np.float64)
    m = TriMesh(V, s.faces.copy(), "ellipsoid")
    return m.fix_orientation()


def tube(points, radius: float = 0.003, n: int = 12) -> TriMesh:
    """Tube along a polyline (O2 thrust line, O9 trajectories)."""
    P = np.asarray(points, dtype=np.float64)
    if len(P) < 2:
        return sphere(P[0], radius)
    rings = []
    # parallel transport frames along the polyline
    tangents = np.gradient(P, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-12)
    t0 = tangents[0]
    a = np.array([1.0, 0, 0]) if abs(t0[0]) < 0.9 else np.array([0, 1.0, 0])
    nrm = np.cross(t0, a)
    nrm /= np.linalg.norm(nrm)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    for k in range(len(P)):
        t = tangents[k]
        nrm = nrm - np.dot(nrm, t) * t
        if np.linalg.norm(nrm) < 1e-9:
            a = np.array([1.0, 0, 0]) if abs(t[0]) < 0.9 else np.array([0, 1.0, 0])
            nrm = np.cross(t, a)
        nrm /= np.linalg.norm(nrm)
        b = np.cross(t, nrm)
        rings.append(P[k] + radius * (np.cos(ang)[:, None] * nrm + np.sin(ang)[:, None] * b))
    V = np.concatenate(rings)
    F = []
    for k in range(len(P) - 1):
        for i in range(n):
            j = (i + 1) % n
            a0, a1 = k * n + i, k * n + j
            b0, b1 = (k + 1) * n + i, (k + 1) * n + j
            F += [[a0, a1, b1], [a0, b1, b0]]
    # caps
    nv = V.shape[0]
    V = np.concatenate([V, P[:1], P[-1:]])
    for i in range(n):
        j = (i + 1) % n
        F.append([nv, j, i])
        F.append([nv + 1, (len(P) - 1) * n + i, (len(P) - 1) * n + j])
    return TriMesh(V, np.array(F), "tube").fix_orientation()


def polyline(points, radius: float = 0.002) -> TriMesh:
    return tube(points, radius, 8)


def bar(p0, p1, width: float = 0.003) -> TriMesh:
    """Straight bar between two points (O4 distance bar, plumb line)."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    L = np.linalg.norm(p1 - p0)
    return _align_z(_cylinder_z(width / 2, max(L, 1e-6), 8), p1 - p0, p0)


def polygon_outline(points, radius: float = 0.002) -> TriMesh:
    """Closed outline of a planar polygon (O4 support polygon)."""
    P = np.asarray(points, dtype=np.float64)
    return tube(np.concatenate([P, P[:1]]), radius, 8)


def section_cap(mesh: TriMesh, plane_point, plane_normal, hatch_period: float = 0.004):
    """Clip a mesh by a plane and return (clipped mesh, cap mesh, cap vertex colours).

    The cap is the convex hull of the clipped cross section (stones are nearly convex);
    the hatching is encoded as alternating vertex colours along the in plane u axis.
    """
    n = _unit(plane_normal)
    p0 = np.asarray(plane_point, dtype=np.float64)
    V = mesh.vertices
    s = (V - p0) @ n
    F = mesh.faces
    keep = (s[F] <= 0).all(axis=1)
    clipped = TriMesh(V, F[keep], mesh.name + "_clip")
    # cross section points: edge intersections
    pts = []
    for f in F[(s[F] <= 0).any(axis=1) & (s[F] > 0).any(axis=1)]:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            if (s[a] <= 0) != (s[b] <= 0):
                t = s[a] / (s[a] - s[b])
                pts.append(V[a] + t * (V[b] - V[a]))
    if len(pts) < 3:
        return clipped, None, None
    P = np.array(pts)
    u = _unit(np.cross(n, [0, 0, 1.0]) if abs(n[2]) < 0.9 else np.cross(n, [1.0, 0, 0]))
    v = np.cross(n, u)
    uv = np.stack([(P - p0) @ u, (P - p0) @ v], 1)
    hull = ConvexHull(uv)
    ring = P[hull.vertices]
    c = ring.mean(axis=0)
    Vc = np.concatenate([[c], ring])
    Fc = np.array([[0, 1 + i, 1 + (i + 1) % len(ring)] for i in range(len(ring))])
    cap = TriMesh(Vc, Fc, "cap")
    phase = ((Vc - p0) @ u) / hatch_period
    col = np.where((np.floor(phase) % 2 == 0)[:, None], np.array([[0.85, 0.85, 0.85]]), np.array([[0.45, 0.45, 0.45]]))
    return clipped, cap, col


def hatched_section_cap(mesh: TriMesh, plane_point, plane_normal, hatch_period: float = 0.004, tint=(0.7, 0.7, 0.7), resolution: int = 6):
    """O3: clip a mesh by a plane and return (clipped mesh, cap mesh, cap vertex colours) where the cap
    is finely triangulated (Delaunay over the section polygon and an interior grid) so that the
    hatching stripes, encoded as vertex colours, stay sharp. ``tint`` sets the density colour."""
    from scipy.spatial import Delaunay
    from matplotlib.path import Path as _Path

    clipped, cap, _ = section_cap(mesh, plane_point, plane_normal, hatch_period)
    if cap is None:
        return clipped, None, None
    n = _unit(plane_normal)
    p0 = np.asarray(plane_point, dtype=np.float64)
    u = _unit(np.cross(n, [0, 0, 1.0]) if abs(n[2]) < 0.9 else np.cross(n, [1.0, 0, 0]))
    v = np.cross(n, u)
    ring = cap.vertices[1:]
    uv_ring = np.stack([(ring - p0) @ u, (ring - p0) @ v], 1)
    step = hatch_period / resolution
    lo, hi = uv_ring.min(axis=0), uv_ring.max(axis=0)
    gu, gv = np.meshgrid(np.arange(lo[0], hi[0], step), np.arange(lo[1], hi[1], step), indexing="ij")
    G = np.stack([gu.ravel(), gv.ravel()], 1)
    inside = _Path(uv_ring).contains_points(G)
    P2 = np.concatenate([uv_ring, G[inside]])
    tri = Delaunay(P2)
    cent = P2[tri.simplices].mean(axis=1)
    keep = _Path(uv_ring).contains_points(cent)
    F = tri.simplices[keep]
    V = p0 + P2[:, :1] * u + P2[:, 1:2] * v
    capm = TriMesh(V, F, "cap")
    # orient the cap normal along the plane normal (facing away from the kept half)
    if np.mean(capm.face_normals() @ n) < 0:
        capm = TriMesh(V, F[:, [0, 2, 1]], "cap")
    phase = (P2[:, 0] + P2[:, 1]) / hatch_period  # 45 degree hatching
    stripe = (np.floor(phase) % 2 == 0)[:, None]
    t = np.asarray(tint, dtype=np.float64)[None, :]
    col = np.where(stripe, 0.9 * t + 0.1, 0.55 * t)
    return clipped, capm, col


def heatmap_plane(origin, u, v, values: np.ndarray, cmap: str = "sequential", vmin=None, vmax=None):
    """Planar grid spanned by u and v with ``values`` (nu, nv) as vertex colours (O6)."""
    vals = np.asarray(values, dtype=np.float64)
    nu, nv = vals.shape
    origin = np.asarray(origin, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    su = np.linspace(0, 1, nu)
    sv = np.linspace(0, 1, nv)
    V = np.array([origin + a * u + b * v for a in su for b in sv])
    F = []
    for i in range(nu - 1):
        for j in range(nv - 1):
            a = i * nv + j
            b = a + 1
            c = a + nv
            d = c + 1
            F += [[a, b, d], [a, d, c]]
    cols = colormap(vals.reshape(-1), cmap, vmin, vmax)
    return TriMesh(V, np.array(F), "heatmap"), cols


def dial(center, radius: float, value: float, interval: tuple | None = None, vmin: float = 0.0, vmax: float = 30.0, normal=(0, -1.0, 0), up=(0, 0, 1.0)):
    """Gauge for O8: an arc from vmin to vmax, a shaded interval arc and a needle at ``value``.

    Returns a list of (mesh, colour) pairs.
    """
    c = np.asarray(center, dtype=np.float64)
    n = _unit(normal)
    upv = _unit(np.asarray(up) - np.dot(up, n) * n)
    side = np.cross(n, upv)

    def pt(angle_frac, r):
        a = np.pi * (1.0 - angle_frac)  # left to right over 180 degrees
        return c + r * (np.cos(a) * side + np.sin(a) * upv)

    out = []
    arc = np.array([pt(t, radius) for t in np.linspace(0, 1, 48)])
    out.append((tube(arc, 0.04 * radius, 8), (0.9, 0.9, 0.9)))
    if interval is not None:
        a0 = (interval[0] - vmin) / (vmax - vmin)
        a1 = (interval[1] - vmin) / (vmax - vmin)
        band = np.array([pt(t, 0.9 * radius) for t in np.linspace(np.clip(a0, 0, 1), np.clip(a1, 0, 1), 24)])
        out.append((tube(band, 0.08 * radius, 8), (0.95, 0.6, 0.2)))
    fv = np.clip((value - vmin) / (vmax - vmin), 0, 1)
    out.append((bar(c, pt(fv, 0.85 * radius), 0.06 * radius), (1.0, 0.25, 0.2)))
    out.append((sphere(c, 0.08 * radius, 1), (0.9, 0.9, 0.9)))
    return out


def contact_force_overlays(contacts, forces, mu, scale: float = 0.02, cone_height: float = 0.03, log_scale: bool | None = None):
    """O1: arrows at contact points scaled by force magnitude and coloured by utilisation,
    plus translucent friction cones. Returns list of (mesh, colour, alpha, glass)."""
    forces = np.asarray(forces, dtype=np.float64)
    mags = np.linalg.norm(forces, axis=1)
    if log_scale is None:
        log_scale = mags.max() / max(mags[mags > 0].min(), 1e-12) > 20 if (mags > 0).any() else False
    out = []
    fmax = max(mags.max(), 1e-12)
    for c, f, m in zip(contacts, forces, mags):
        if m <= 0:
            continue
        L = scale * (np.log1p(m) / np.log1p(fmax) if log_scale else m / fmax)
        n = _unit(c.normal)
        fn = float(np.dot(f, n))
        ft = np.linalg.norm(f - fn * n)
        util = ft / max(mu * fn, 1e-12) if fn > 0 else 1.0
        col = colormap(np.array([np.clip(util, 0, 1)]), "sequential", 0, 1)[0]
        out.append((arrow(c.point, f, L, 0.0015), tuple(col), 1.0, False))
        out.append((cone(c.point, n, np.arctan(mu), cone_height), (0.7, 0.85, 1.0), 0.15, True))
    return out
