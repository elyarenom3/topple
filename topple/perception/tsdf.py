"""Truncated signed distance fusion (Curless and Levoy 1996) with inverse variance weights (Section 8.4).

    D(x) = sum_i w_i d_i(x) / sum_i w_i,   w_i = 1 / sigma_{z,i}^2

Distances are projective (along the camera ray), truncated at ``truncation`` metres. Voxels
behind the observed surface by more than the truncation band are left untouched.
"""

from __future__ import annotations

import numpy as np

from ..core.mesh import TriMesh
from ..core.transforms import SE3
from .depth_sensor import DepthImage
from .marching_cubes import largest_component, marching_cubes


class TSDFVolume:
    def __init__(self, origin, spacing: float, shape, truncation: float):
        self.origin = np.asarray(origin, dtype=np.float64)
        self.spacing = float(spacing)
        self.shape = tuple(int(s) for s in shape)
        self.truncation = float(truncation)
        self.D = np.zeros(self.shape)
        self.W = np.zeros(self.shape)
        self.n_views = 0

    @staticmethod
    def from_bounds(lo, hi, spacing: float, truncation: float, pad: float | None = None) -> "TSDFVolume":
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        pad = 2.0 * truncation if pad is None else pad
        lo = lo - pad
        hi = hi + pad
        shape = np.ceil((hi - lo) / spacing).astype(int) + 1
        return TSDFVolume(lo, spacing, shape, truncation)

    def voxel_centres(self) -> np.ndarray:
        nx, ny, nz = self.shape
        I, J, K = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        return self.origin + self.spacing * np.stack([I.ravel(), J.ravel(), K.ravel()], 1).astype(np.float64)

    def integrate(self, img: DepthImage, sigma: np.ndarray | None = None, T_world_cam: SE3 | None = None, K: np.ndarray | None = None, mask: np.ndarray | None = None, min_sigma: float = 1e-4) -> int:
        """Fuse one depth image. Returns the number of voxels updated."""
        T = img.T_world_cam if T_world_cam is None else T_world_cam
        K = img.K if K is None else K
        sigma = img.sigma if sigma is None else sigma
        valid = img.valid if mask is None else (img.valid & mask)
        H, W = img.depth.shape
        P = self.voxel_centres()
        Pc = T.inverse().apply(P)
        z = Pc[:, 2]
        front = z > 1e-6
        u = K[0, 0] * Pc[:, 0] / np.where(front, z, 1.0) + K[0, 2]
        v = K[1, 1] * Pc[:, 1] / np.where(front, z, 1.0) + K[1, 2]
        ui = np.floor(u).astype(np.int64)
        vi = np.floor(v).astype(np.int64)
        inside = front & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        ui = np.clip(ui, 0, W - 1)
        vi = np.clip(vi, 0, H - 1)
        d_obs = img.depth[vi, ui]
        ok = inside & valid[vi, ui] & (d_obs > 0)
        sdf = d_obs - z
        ok &= sdf > -self.truncation
        tsdf = np.clip(sdf, -self.truncation, self.truncation)
        sig = np.maximum(sigma[vi, ui], min_sigma)
        w = 1.0 / (sig * sig)
        # linear weight drop behind the surface within the band
        behind = sdf < 0
        w = np.where(behind, w * np.clip(1.0 + sdf / self.truncation, 0.0, 1.0), w)
        ok &= w > 0
        idx = np.where(ok)[0]
        Df = self.D.reshape(-1)
        Wf = self.W.reshape(-1)
        Wn = Wf[idx] + w[idx]
        Df[idx] = (Wf[idx] * Df[idx] + w[idx] * tsdf[idx]) / Wn
        Wf[idx] = Wn
        self.n_views += 1
        return int(idx.size)

    def observed(self) -> np.ndarray:
        return self.W > 0

    def field(self, fill: float | None = None) -> np.ndarray:
        """Distance field with unobserved voxels set to ``fill`` (NaN by default)."""
        out = self.D.copy()
        out[~self.observed()] = np.nan if fill is None else fill
        return out

    def extract_mesh(self, fill: float | None = None, keep_largest: bool = True) -> TriMesh:
        m = marching_cubes(self.field(fill), self.origin, self.spacing, 0.0)
        return largest_component(m) if keep_largest and m.n_faces > 0 else m

    def voxel_std(self, floor_frac: float = 0.25) -> np.ndarray:
        """Per voxel standard deviation sqrt(1 / W) for observed voxels (NaN elsewhere), floored at a fraction of the voxel size."""
        with np.errstate(divide="ignore"):
            s = np.where(self.W > 0, 1.0 / np.sqrt(np.maximum(self.W, 1e-300)), np.nan)
        return np.maximum(s, floor_frac * self.spacing)
