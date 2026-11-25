"""Stable resting orientations of a stone (Section 9.1).

A facet of the convex hull is a stable resting face if the projection of the centre of mass
along the facet normal falls inside the facet. The probability of each face is estimated by
quasi static drops: random initial orientations followed by descent of the support function
h(d) = max_v (v - c) . d on the sphere of down directions, which is what a convex body does
when it tips about its support point until a facet lies flat. A dynamic ``world_factory``
can be passed to replace the quasi static model by simulated drops.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

from ..core.mesh import TriMesh
from ..core.transforms import rotation_from_two_vectors, rot_to_quat, quat_to_rot


def _facet_groups(hull: ConvexHull, tol: float = 1e-6):
    """Group hull triangles into planar facets. Returns list of (normal, offset, vertex ids)."""
    eq = hull.equations
    key = np.round(eq / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    groups = []
    for g in range(len(first)):
        tri = hull.simplices[inv == g]
        verts = np.unique(tri)
        n = eq[first[g], :3]
        groups.append((n / np.linalg.norm(n), eq[first[g], 3], verts))
    return groups


def _point_in_facet(p: np.ndarray, verts3: np.ndarray, normal: np.ndarray) -> float:
    """Signed distance (positive inside) from the projection of p to the facet polygon boundary."""
    from ..core.transforms import tangent_basis

    u, v = tangent_basis(normal)
    c = verts3.mean(axis=0)
    P2 = np.column_stack([(verts3 - c) @ u, (verts3 - c) @ v])
    q2 = np.array([(p - c) @ u, (p - c) @ v])
    if len(P2) < 3:
        return -np.inf
    try:
        h = ConvexHull(P2)
    except Exception:
        return -np.inf
    poly = P2[h.vertices]
    n = len(poly)
    dmin, inside = np.inf, True
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ab = b - a
        L = np.linalg.norm(ab)
        if L < 1e-12:
            continue
        nrm = np.array([ab[1], -ab[0]]) / L
        s = np.dot(q2 - a, nrm)
        if s > 0:
            inside = False
        t = np.clip(np.dot(q2 - a, ab) / (L * L), 0, 1)
        dmin = min(dmin, np.linalg.norm(q2 - (a + t * ab)))
    return float(dmin if inside else -dmin)


def stable_facets(mesh: TriMesh, com: np.ndarray | None = None, min_margin: float = 1e-4):
    """Stable resting facets of the convex hull of ``mesh``.

    Returns a list of dicts with keys normal (outward, body frame), vertices (k, 3), margin
    (distance of the centre of mass projection inside the facet) and height of the centre of
    mass above the facet.
    """
    com = mesh.centroid() if com is None else np.asarray(com, dtype=np.float64)
    hull = ConvexHull(mesh.vertices)
    out = []
    for n, off, verts in _facet_groups(hull):
        V = mesh.vertices[verts]
        m = _point_in_facet(com, V, n)
        if m > min_margin:
            out.append({"normal": n, "vertices": V, "margin": float(m), "height": float(-(com @ n + off))})
    return out


def _descend_on_sphere(d: np.ndarray, verts_c: np.ndarray, iters: int = 400, step: float = 0.03) -> np.ndarray:
    """Descent of the support function on the sphere for a batch of down directions d (n, 3)."""
    for _ in range(iters):
        dots = verts_c @ d.T  # (nv, n)
        k = np.argmax(dots, axis=0)
        s = verts_c[k]  # support points (n, 3)
        h = np.einsum("ij,ij->i", s, d)
        g = s - h[:, None] * d  # tangential subgradient
        gn = np.linalg.norm(g, axis=1)
        active = gn > 1e-9
        upd = np.zeros_like(d)
        upd[active] = g[active] / gn[active, None]
        d = d - step * upd
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
    return d


def resting_orientations(body_mesh: TriMesh, n_drops: int = 300, rng=None, world_factory=None, com: np.ndarray | None = None, min_prob: float = 0.0):
    """Resting orientations of ``body_mesh`` as a list of (quaternion, probability).

    The quaternion rotates the body so that the facet normal points along world -z (the facet
    lies flat on a horizontal support). Probabilities are the fractions of random initial
    orientations that end on each facet; facets never reached keep probability 0 and are listed
    last. If ``world_factory`` is given it must return a callable drop(quat) -> settled quat
    which replaces the quasi static model.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    com = body_mesh.centroid() if com is None else np.asarray(com, dtype=np.float64)
    facets = stable_facets(body_mesh, com)
    if not facets:
        return []
    normals = np.array([f["normal"] for f in facets])
    counts = np.zeros(len(facets))
    if n_drops > 0:
        hull = ConvexHull(body_mesh.vertices)
        verts_c = body_mesh.vertices[hull.vertices] - com
        if world_factory is None:
            d0 = rng.normal(size=(n_drops, 3))
            d0 /= np.linalg.norm(d0, axis=1, keepdims=True)
            d = _descend_on_sphere(d0, verts_c)
            # assign to the nearest stable facet normal
            k = np.argmax(d @ normals.T, axis=1)
            np.add.at(counts, k, 1.0)
        else:
            drop = world_factory()
            for _ in range(n_drops):
                q0 = rng.normal(size=4)
                q0 /= np.linalg.norm(q0)
                qf = drop(q0)
                down_body = quat_to_rot(qf).T @ np.array([0.0, 0.0, -1.0])
                counts[int(np.argmax(normals @ down_body))] += 1.0
        probs = counts / max(counts.sum(), 1.0)
    else:
        probs = np.full(len(facets), 1.0 / len(facets))
    out = []
    for f, p in zip(facets, probs):
        R = rotation_from_two_vectors(f["normal"], np.array([0.0, 0.0, -1.0]))
        out.append((rot_to_quat(R), float(p)))
    out.sort(key=lambda t: -t[1])
    return [o for o in out if o[1] >= min_prob]
