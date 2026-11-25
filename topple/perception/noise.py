"""Depth noise models (Section 8.2).

* Structured light geometry: a projector pinhole at a 75 mm baseline; pixels occluded from the
  projector are invalid (shadow bands); disparity d = f b / z quantised to 1/8 pixel.
* Khoshelham and Elberink (2012): random depth error growing with the square of the distance,
  sigma_z = 1.425e-3 z^2 (z in metres), a few millimetres near the sensor, about 4 cm at range.
* Nguyen, Izadi and Lovell (2012): axial and lateral noise as functions of distance z and
  incidence angle theta,
      sigma_z(z, theta) = 0.0012 + 0.0019 (z - 0.4)^2 + 0.0001 / sqrt(z) * theta^2 / (pi/2 - theta)^2
      sigma_L(theta) [pixels] = 0.8 + 0.035 theta / (pi/2 - theta)
* Handa et al. (2014): perturbation along normals plus quantisation (ICL-NUIM style).
"""

from __future__ import annotations

import numpy as np

THETA_MAX = np.deg2rad(85.0)


def khoshelham_sigma(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return 1.425e-3 * z * z


def nguyen_sigma(z: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Axial sigma in metres and lateral sigma in pixels."""
    z = np.asarray(z, dtype=np.float64)
    th = np.clip(np.asarray(theta, dtype=np.float64), 0.0, THETA_MAX)
    denom = np.pi / 2 - th
    sigma_z = 0.0012 + 0.0019 * (z - 0.4) ** 2 + 0.0001 / np.sqrt(np.maximum(z, 1e-3)) * th * th / (denom * denom)
    sigma_l = 0.8 + 0.035 * th / denom
    return sigma_z, sigma_l


def handa_sigma(z: np.ndarray, base: float = 0.0015) -> np.ndarray:
    """A constant perturbation scale along the surface normal, as used for synthetic benchmarks."""
    return np.full_like(np.asarray(z, dtype=np.float64), base)


def quantise_disparity(z: np.ndarray, f: float, baseline: float, step: float = 0.125) -> tuple[np.ndarray, np.ndarray]:
    """Quantise depth through disparity d = f b / z rounded to ``step`` pixels.

    Returns the quantised depth and the one sigma quantisation error (step width / sqrt 12).
    """
    z = np.asarray(z, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = f * baseline / z
        dq = np.round(d / step) * step
        zq = np.where(dq > 0, f * baseline / np.maximum(dq, 1e-9), z)
        dz = z * z / (f * baseline) * step
    return zq, dz / np.sqrt(12.0)


def lateral_jitter_indices(height: int, width: int, sigma_px: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Row and column indices of the pixel actually sampled under lateral noise (clipped to the image)."""
    V, U = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    du = rng.normal(size=(height, width)) * sigma_px
    dv = rng.normal(size=(height, width)) * sigma_px
    uu = np.clip(np.rint(U + du), 0, width - 1).astype(np.int64)
    vv = np.clip(np.rint(V + dv), 0, height - 1).astype(np.int64)
    return vv, uu


def structured_light_shadow(points_world: np.ndarray, normals_world: np.ndarray, projector_world: np.ndarray, bvh, offset: float = 1e-4) -> np.ndarray:
    """True where the surface point is occluded from the projector (shadow ray test)."""
    p = points_world + offset * normals_world
    d = projector_world[None, :] - p  # unnormalised: the ray parameter t = 1 is the projector
    return bvh.occluded(p, d, tmin=1e-6, tmax=1.0 - 1e-3)
