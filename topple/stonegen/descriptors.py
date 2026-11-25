"""Shape descriptors (Section 5.3): elongation b/a, flatness c/b, convexity V / V_hull and sphericity."""

from __future__ import annotations

import numpy as np

from ..core.mesh import TriMesh


def principal_extents(mesh: TriMesh) -> tuple[np.ndarray, np.ndarray]:
    """Full extents (a >= b >= c) of the mesh along its principal axes of inertia, and the axes."""
    mp = mesh.mass_properties(1.0)
    w, U = np.linalg.eigh(mp["inertia_com"])
    proj = (mesh.vertices - mp["com"]) @ U
    ext = proj.max(axis=0) - proj.min(axis=0)
    order = np.argsort(ext)[::-1]
    return ext[order], U[:, order]


def descriptors(mesh: TriMesh) -> dict:
    ext, axes = principal_extents(mesh)
    a, b, c = ext
    V = mesh.volume()
    A = mesh.surface_area()
    Vh = mesh.convex_hull().volume()
    return {
        "elongation": float(b / a),
        "flatness": float(c / b),
        "convexity": float(V / Vh),
        "sphericity": float(np.pi ** (1 / 3) * (6 * V) ** (2 / 3) / A),
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "volume": float(V),
        "area": float(A),
        "n_faces": int(mesh.n_faces),
    }


def in_box(desc: dict, box: dict) -> bool:
    for k, (lo, hi) in box.items():
        if k not in desc:
            continue
        if not (lo <= desc[k] <= hi):
            return False
    return True
