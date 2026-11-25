"""Screened Poisson surface reconstruction on a regular grid (Kazhdan and Hoppe 2013).

Oriented surface points (outward normals) are splatted trilinearly onto the voxel centres to
give a vector field V. The implicit function phi solves the screened Poisson equation

    (-Laplacian + alpha S^T S) phi = -div V,

where the Laplacian is the 7 point finite difference operator with natural (Neumann)
boundaries and S interpolates phi trilinearly at the samples, so the screening term pins
phi = 0 on the input points. phi grows outwards (positive outside, the TSDF convention). It is
divided by its median gradient magnitude at the samples, so that near the surface it reads as a
signed distance in metres. The system is symmetric positive definite and is solved by Jacobi
preconditioned conjugate gradients.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import cg


def _trilinear(points: np.ndarray, origin: np.ndarray, spacing: float, shape) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows (point index), columns (flat voxel index) and weights of trilinear interpolation."""
    g = (np.asarray(points, dtype=float) - origin) / spacing
    i0 = np.floor(g).astype(np.int64)
    f = g - i0
    shape = np.asarray(shape)
    rows, cols, vals = [], [], []
    n = len(g)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                idx = i0 + np.array([dx, dy, dz])
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                ok = np.all((idx >= 0) & (idx < shape), axis=1)
                flat = np.ravel_multi_index(np.clip(idx, 0, shape - 1).T, tuple(shape))
                rows.append(np.arange(n)[ok])
                cols.append(flat[ok])
                vals.append(w[ok])
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def _laplacian_1d(n: int, h: float) -> sp.csr_matrix:
    main = np.full(n, 2.0)
    main[0] = main[-1] = 1.0  # Neumann ends
    off = -np.ones(n - 1)
    return sp.diags([off, main, off], [-1, 0, 1], format="csr") / (h * h)


def neg_laplacian(shape, h: float) -> sp.csr_matrix:
    nx, ny, nz = shape
    Ix, Iy, Iz = sp.identity(nx), sp.identity(ny), sp.identity(nz)
    return (sp.kron(sp.kron(_laplacian_1d(nx, h), Iy), Iz) + sp.kron(sp.kron(Ix, _laplacian_1d(ny, h)), Iz) + sp.kron(sp.kron(Ix, Iy), _laplacian_1d(nz, h))).tocsr()


def screened_poisson(points: np.ndarray, normals: np.ndarray, origin, spacing: float, shape, alpha: float = 4.0, smooth_voxels: float = 1.0, tol: float = 1e-7, maxiter: int = 3000) -> np.ndarray:
    """Implicit function (shape ``shape``) whose zero level set fits the oriented points."""
    origin = np.asarray(origin, dtype=float)
    h = float(spacing)
    shape = tuple(int(s) for s in shape)
    N = int(np.prod(shape))
    P = np.asarray(points, dtype=float)
    Nn = np.asarray(normals, dtype=float)
    Nn = Nn / np.maximum(np.linalg.norm(Nn, axis=1, keepdims=True), 1e-12)
    # voxel centres sit at origin + spacing * index
    r, c, w = _trilinear(P, origin, h, shape)
    S = sp.csr_matrix((w, (r, c)), shape=(len(P), N))
    # splat the normals: V = S^T n, per unit area (each point stands for ~h^2 of surface)
    V = np.stack([S.T @ Nn[:, k] for k in range(3)], axis=-1).reshape(shape + (3,))
    touched = np.asarray((S.T @ np.ones(len(P))) > 0).reshape(shape)
    density = len(P) / max(int(touched.sum()), 1)  # points per touched voxel
    V /= max(density, 1e-12)
    if smooth_voxels > 0:
        V = np.stack([gaussian_filter(V[..., k], smooth_voxels, mode="nearest") for k in range(3)], axis=-1)
    div = np.zeros(shape)
    for k in range(3):
        div += np.gradient(V[..., k], h, axis=k)
    A = neg_laplacian(shape, h)
    a_eff = alpha * (6.0 / (h * h)) / max(density, 1e-12) / 6.0
    A = (A + a_eff * (S.T @ S)).tocsr()
    b = -div.reshape(-1)
    Minv = sp.diags(1.0 / np.maximum(A.diagonal(), 1e-300))
    phi, info = cg(A, b, rtol=tol, maxiter=maxiter, M=Minv) if "rtol" in cg.__code__.co_varnames else cg(A, b, tol=tol, maxiter=maxiter, M=Minv)
    phi = phi.reshape(shape)
    # pin the level set at the samples exactly and scale to metres near the surface
    phi -= float(np.median(S @ phi.reshape(-1)))
    g = np.stack(np.gradient(phi, h), axis=-1).reshape(-1, 3)
    gmag = np.linalg.norm(S @ g, axis=1) / np.maximum(np.asarray(S.sum(axis=1)).ravel(), 1e-12)
    scale = float(np.median(gmag[np.isfinite(gmag) & (gmag > 0)])) if np.any(gmag > 0) else 1.0
    return phi / max(scale, 1e-12)
