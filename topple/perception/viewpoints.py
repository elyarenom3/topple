"""Viewpoint planning (Section 8.3): hemisphere views and in hand views of the underside."""

from __future__ import annotations

import numpy as np

from ..core.transforms import SE3
from .camera import look_at


def hemisphere_views(center, radius: float, n: int = 12, rng: np.random.Generator | None = None, min_elev_deg: float = 20.0, max_elev_deg: float = 80.0) -> list[SE3]:
    """Fibonacci spiral of ``n`` camera poses on the upper hemisphere looking at ``center``."""
    center = np.asarray(center, dtype=np.float64)
    phase = rng.uniform(0, 2 * np.pi) if rng is not None else 0.0
    golden = np.pi * (3.0 - np.sqrt(5.0))
    lo, hi = np.sin(np.deg2rad(min_elev_deg)), np.sin(np.deg2rad(max_elev_deg))
    poses = []
    for i in range(n):
        s = lo + (hi - lo) * (i + 0.5) / n
        elev = np.arcsin(s)
        az = i * golden + phase
        eye = center + radius * np.array([np.cos(elev) * np.cos(az), np.cos(elev) * np.sin(az), np.sin(elev)])
        poses.append(look_at(eye, center))
    return poses


def in_hand_views(center, radius: float, n: int = 4, rng: np.random.Generator | None = None, elev_deg: float = -45.0) -> list[SE3]:
    """Camera poses below the stone, equivalent to rotating the held stone in front of a fixed camera."""
    center = np.asarray(center, dtype=np.float64)
    phase = rng.uniform(0, 2 * np.pi) if rng is not None else 0.0
    elev = np.deg2rad(elev_deg)
    poses = []
    for i in range(n):
        az = phase + 2 * np.pi * i / n
        eye = center + radius * np.array([np.cos(elev) * np.cos(az), np.cos(elev) * np.sin(az), np.sin(elev)])
        poses.append(look_at(eye, center))
    return poses


def stone_rotations_for_views(views: list[SE3], fixed_camera: SE3) -> list[SE3]:
    """Stone poses that present the same relative view to a fixed camera (for the held stone)."""
    out = []
    for v in views:
        # want T_cam_stone' = (fixed^-1 stone') = v^-1 stone  with stone = identity
        out.append(fixed_camera.compose(v.inverse()))
    return out
