"""Pinhole camera model (Section 8.1).

Camera frame: z along the optical axis, x to the right, y down (image rows increase
downward). A pixel (u, v) at depth z back projects to z K^-1 (u, v, 1)^T. Pixel centres sit
at (u + 0.5, v + 0.5).
"""

from __future__ import annotations

import numpy as np

from ..core.transforms import SE3, rot_to_quat


class PinholeCamera:
    def __init__(self, K: np.ndarray, width: int, height: int):
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.width = int(width)
        self.height = int(height)
        self.K_inv = np.linalg.inv(self.K)

    @staticmethod
    def from_fov(width: int, height: int, fov_deg: float = 58.0) -> "PinholeCamera":
        """Horizontal field of view in degrees, square pixels, principal point at the image centre."""
        fx = 0.5 * width / np.tan(0.5 * np.deg2rad(fov_deg))
        K = np.array([[fx, 0.0, 0.5 * width], [0.0, fx, 0.5 * height], [0.0, 0.0, 1.0]])
        return PinholeCamera(K, width, height)

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    def project(self, P_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Camera frame points to continuous pixel coordinates (u, v)."""
        P = np.atleast_2d(P_cam)
        z = P[:, 2]
        u = self.fx * P[:, 0] / z + self.cx
        v = self.fy * P[:, 1] / z + self.cy
        return u, v

    def backproject(self, u: np.ndarray, v: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Pixel coordinates and depth z to camera frame points."""
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        x = (u - self.cx) / self.fx * z
        y = (v - self.cy) / self.fy * z
        return np.stack([x, y, z], axis=-1)

    def pixel_grid(self) -> tuple[np.ndarray, np.ndarray]:
        us = np.arange(self.width) + 0.5
        vs = np.arange(self.height) + 0.5
        U, V = np.meshgrid(us, vs)  # (H, W)
        return U, V

    def ray_dirs_cam(self, u=None, v=None) -> np.ndarray:
        """Unit ray directions in the camera frame for all pixels (H*W, 3) or for given (u, v)."""
        if u is None:
            U, V = self.pixel_grid()
            u, v = U.ravel(), V.ravel()
        d = self.backproject(u, v, np.ones_like(np.asarray(u, dtype=np.float64)))
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def rays(self, T_world_cam: SE3) -> tuple[np.ndarray, np.ndarray]:
        """World frame ray origins and unit directions for every pixel, row major (H*W, 3)."""
        d = self.ray_dirs_cam()
        dirs = T_world_cam.apply_vec(d)
        origins = np.tile(T_world_cam.t, (d.shape[0], 1))
        return origins, dirs

    def depth_to_points(self, depth: np.ndarray, valid: np.ndarray | None = None, T_world_cam: SE3 | None = None) -> np.ndarray:
        """Back project a depth image to (n, 3) points, camera frame or world frame when a pose is given."""
        U, V = self.pixel_grid()
        m = np.isfinite(depth) & (depth > 0)
        if valid is not None:
            m &= valid
        P = self.backproject(U[m], V[m], depth[m])
        if T_world_cam is not None:
            P = T_world_cam.apply(P)
        return P


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> SE3:
    """Camera pose T_world_cam looking from ``eye`` toward ``target`` (z forward, x right, y down)."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    z = target - eye
    z /= np.linalg.norm(z)
    if abs(np.dot(z, up / np.linalg.norm(up))) > 0.999:
        up = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    return SE3(rot_to_quat(R), eye)
