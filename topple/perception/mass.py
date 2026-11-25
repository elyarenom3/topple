"""Mass properties and density priors (Section 8.6)."""

from __future__ import annotations

import numpy as np

from ..core.mesh import TriMesh
from ..stonegen.noise import fbm3


def mass_properties(mesh: TriMesh, density: float = 1.0) -> dict:
    return mesh.mass_properties(density)


def density_prior(rng: np.random.Generator, lo: float = 2500.0, hi: float = 2800.0, n: int | None = None):
    return rng.uniform(lo, hi, size=n)


def random_density_field_com(mesh: TriMesh, rng: np.random.Generator, amplitude: float = 0.3, length_frac: float = 0.5, n_points: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    """Centre of mass under a random density field rho(x) = rho0 (1 + amplitude * noise(x)).

    Returns (com_uniform, com_random) estimated by Monte Carlo over interior points, for the
    ablation of Section 8.6 in which the true centre of mass moves away from the centroid.
    """
    lo, hi = mesh.bounds()
    pts = rng.uniform(lo, hi, size=(n_points, 3))
    inside = mesh.contains(pts)
    p = pts[inside]
    R = mesh.bounding_radius()
    seed = int(rng.integers(2**31 - 1))
    f = fbm3((p - mesh.centroid()) / (length_frac * R), octaves=2, seed=seed)
    f = f / max(np.abs(f).max(), 1e-12)
    w = 1.0 + amplitude * f
    com_u = p.mean(axis=0)
    com_r = (w[:, None] * p).sum(axis=0) / w.sum()
    return com_u, com_r
