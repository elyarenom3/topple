"""Curvature driven erosion (Section 5.2).

Each vertex moves inward along its normal by a step proportional to the positive part of its
discrete mean curvature H, so convex edges recede faster than flat faces:

    x_i <- x_i - eta * max(0, H_i) * n_i

This project takes ``eta`` dimensionless: the displacement is eta * R * max(0, H R) with R the
bounding radius, so a sphere shrinks by eta * R per step and an edge with curvature 20 / R
recedes 20 times faster. Each step is limited to a fraction of the local edge length and to half the
normal distance to the neighbours' centroid, so that the surface stays embedded and the flow
is monotone, and a tangential relaxation after every ``remesh_every`` steps
keeps triangle quality without changing the topology (the "periodic remeshing").
"""

from __future__ import annotations

import numpy as np

from ..core.mesh import TriMesh


def local_edge_length(mesh: TriMesh) -> np.ndarray:
    e = mesh.edges()
    L = np.linalg.norm(mesh.vertices[e[:, 0]] - mesh.vertices[e[:, 1]], axis=1)
    acc = np.zeros(mesh.n_vertices)
    cnt = np.zeros(mesh.n_vertices)
    np.add.at(acc, e[:, 0], L)
    np.add.at(acc, e[:, 1], L)
    np.add.at(cnt, e[:, 0], 1)
    np.add.at(cnt, e[:, 1], 1)
    return acc / np.maximum(cnt, 1)


def tangential_relax(mesh: TriMesh, iterations: int = 2, lam: float = 0.5) -> TriMesh:
    """Move vertices toward the centroid of their neighbours within the tangent plane."""
    m = mesh.copy()
    for _ in range(iterations):
        n = m.vertex_normals()
        lap = m.laplacian_umbrella()
        tangential = lap - np.einsum("ij,ij->i", lap, n)[:, None] * n
        m.vertices = m.vertices + lam * tangential
        m.invalidate()
    return m


def _smooth_scalar(mesh: TriMesh, s: np.ndarray, iters: int = 1) -> np.ndarray:
    e = mesh.edges()
    out = s.copy()
    for _ in range(iters):
        acc = out.copy()
        cnt = np.ones(len(out))
        np.add.at(acc, e[:, 0], out[e[:, 1]])
        np.add.at(acc, e[:, 1], out[e[:, 0]])
        np.add.at(cnt, e[:, 0], 1)
        np.add.at(cnt, e[:, 1], 1)
        out = acc / cnt
    return out


def erode(mesh: TriMesh, eta: float = 0.002, steps: int = 4, remesh_every: int = 1, step_frac: float = 0.25, relax_iters: int = 2) -> TriMesh:
    """Apply ``steps`` erosion steps and return a new mesh."""
    m = mesh.copy()
    if steps <= 0 or eta <= 0:
        return m
    R = m.bounding_radius()
    for s in range(steps):
        H = _smooth_scalar(m, m.mean_curvature(), 1)
        n = m.vertex_normals()
        disp = eta * R * np.maximum(0.0, H * R)
        # stability: a vertex never passes the centroid of its neighbours along its normal, so a
        # sharp edge is rounded instead of being pushed into a groove
        umbrella = np.einsum("ij,ij->i", m.laplacian_umbrella(), n)
        disp = np.minimum(disp, step_frac * local_edge_length(m))
        disp = np.minimum(disp, 0.5 * np.maximum(-umbrella, 0.0))
        m.vertices = m.vertices - disp[:, None] * n
        m.invalidate()
        if remesh_every > 0 and (s + 1) % remesh_every == 0:
            m = tangential_relax(m, relax_iters)
    return m.fix_orientation()
