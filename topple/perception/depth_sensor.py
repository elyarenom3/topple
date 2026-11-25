"""Simulated structured light depth camera (Sections 8.1, 8.2).

Depth images come from one primary ray per pixel cast against the exact world meshes with
the shared BVH, then corruption by the configured noise model. The believed camera pose
carries injected kinematic noise; the true pose is kept only in ``DepthSensor.true_poses``
for the test harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.mesh import TriMesh
from ..core.raycast import BVH
from ..core.transforms import SE3, quat_from_rotvec
from .camera import PinholeCamera
from .noise import THETA_MAX, handa_sigma, khoshelham_sigma, lateral_jitter_indices, nguyen_sigma, quantise_disparity, structured_light_shadow


@dataclass
class DepthImage:
    depth: np.ndarray  # (H, W) metres along the optical axis, 0 where invalid
    valid: np.ndarray  # (H, W) bool
    sigma: np.ndarray  # (H, W) one sigma depth uncertainty (metres)
    T_world_cam: SE3  # believed camera pose
    K: np.ndarray
    normals: np.ndarray  # (H, W, 3) camera frame surface normals facing the camera (as a real sensor would estimate them)
    incidence: np.ndarray  # (H, W) incidence angle (radians)
    object_id: np.ndarray  # (H, W) int, -1 for background
    meta: dict = field(default_factory=dict)

    @property
    def height(self) -> int:
        return self.depth.shape[0]

    @property
    def width(self) -> int:
        return self.depth.shape[1]

    def camera(self) -> PinholeCamera:
        return PinholeCamera(self.K, self.width, self.height)

    def points(self, world: bool = True, mask: np.ndarray | None = None) -> np.ndarray:
        m = self.valid if mask is None else (self.valid & mask)
        cam = self.camera()
        return cam.depth_to_points(self.depth, m, self.T_world_cam if world else None)

    def points_and_normals(self, mask: np.ndarray | None = None, T_world_cam: SE3 | None = None) -> tuple[np.ndarray, np.ndarray]:
        """World frame points and normals under the believed pose (or a given pose)."""
        m = self.valid if mask is None else (self.valid & mask)
        T = self.T_world_cam if T_world_cam is None else T_world_cam
        cam = self.camera()
        return cam.depth_to_points(self.depth, m, T), T.apply_vec(self.normals[m])


class DepthSensor:
    def __init__(self, camera: PinholeCamera, cfg_sensor: dict | None = None, rng: np.random.Generator | None = None):
        cfg = cfg_sensor or {}
        self.cam = camera
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.baseline = float(cfg.get("baseline_m", 0.075))
        self.quant = float(cfg.get("disparity_quant", 0.125))
        self.model = str(cfg.get("model", "nguyen"))
        self.scale = float(cfg.get("noise_scale", 1.0))
        self.shadow = bool(cfg.get("structured_light_shadow", True))
        self.pose_noise_rot = np.deg2rad(float(cfg.get("pose_noise_rot_deg", 0.0)))
        self.pose_noise_trans = float(cfg.get("pose_noise_trans_m", 0.0))
        self.max_range = float(cfg.get("max_range_m", 5.0))
        self.min_range = float(cfg.get("min_range_m", 0.1))
        self.true_poses: list[SE3] = []

    # ------------------------------------------------------------------ scene
    @staticmethod
    def build_scene(meshes_world: list[TriMesh], ids=None):
        Vs, Fs, obj = [], [], []
        off = 0
        for k, m in enumerate(meshes_world):
            Vs.append(m.vertices)
            Fs.append(m.faces + off)
            obj.append(np.full(m.n_faces, ids[k] if ids is not None else k, dtype=np.int64))
            off += m.n_vertices
        V = np.concatenate(Vs)
        F = np.concatenate(Fs)
        face_obj = np.concatenate(obj)
        bvh = BVH(V, F)
        tri = V[F]
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-300)
        return bvh, face_obj, fn

    # ----------------------------------------------------------------- render
    def render_depth(self, meshes_world: list[TriMesh], T_world_cam: SE3, mask_ids=None, ids=None, noise: bool = True) -> DepthImage:
        H, W = self.cam.height, self.cam.width
        bvh, face_obj, fn = self.build_scene(meshes_world, ids)
        origins, dirs = self.cam.rays(T_world_cam)
        t, tid, _, _ = bvh.intersect(origins, dirs, tmin=1e-6, tmax=self.max_range * 2)
        hit = tid >= 0
        tid_safe = np.maximum(tid, 0)
        points = origins + t[:, None] * dirs
        d_cam = self.cam.ray_dirs_cam()
        depth = np.where(hit, t * d_cam[:, 2], 0.0)
        normals = fn[tid_safe]
        flip = np.einsum("ij,ij->i", normals, dirs) > 0
        normals[flip] *= -1
        cos_inc = np.clip(-np.einsum("ij,ij->i", normals, dirs), 0.0, 1.0)
        incidence = np.arccos(cos_inc)
        object_id = np.where(hit, face_obj[tid_safe], -1)
        valid = hit & (depth > self.min_range) & (depth < self.max_range)
        if mask_ids is not None:
            valid &= ~np.isin(object_id, np.asarray(mask_ids))
        depth = depth.reshape(H, W)
        valid = valid.reshape(H, W)
        normals = normals.reshape(H, W, 3)
        incidence = incidence.reshape(H, W)
        object_id = object_id.reshape(H, W)
        points = points.reshape(H, W, 3)
        sigma = np.zeros((H, W))
        meta = {"model": self.model if noise else "none", "noise_scale": self.scale}
        if noise:
            depth, valid, sigma, normals, incidence, object_id = self._apply_noise(depth, valid, normals, incidence, object_id, points, T_world_cam, bvh)
        self.true_poses.append(T_world_cam.copy())
        T_believed = self._perturb_pose(T_world_cam) if noise else T_world_cam.copy()
        normals_cam = T_world_cam.inverse().apply_vec(normals.reshape(-1, 3)).reshape(H, W, 3)
        return DepthImage(depth, valid, sigma, T_believed, self.cam.K.copy(), normals_cam, incidence, object_id, meta)

    # ------------------------------------------------------------------ noise
    def _perturb_pose(self, T: SE3) -> SE3:
        if self.pose_noise_rot <= 0 and self.pose_noise_trans <= 0:
            return T.copy()
        dq = quat_from_rotvec(self.rng.normal(size=3) * self.pose_noise_rot)
        dt = self.rng.normal(size=3) * self.pose_noise_trans
        return T.compose(SE3(dq, dt))  # kinematic error expressed in the camera frame

    def _apply_noise(self, depth, valid, normals, incidence, object_id, points, T_world_cam, bvh):
        H, W = depth.shape
        rng = self.rng
        model = self.model
        if model == "none":
            return depth, valid, np.where(valid, 1e-4, 0.0), normals, incidence, object_id
        # lateral noise: sample the clean image at a jittered pixel
        if model == "nguyen":
            _, sig_l = nguyen_sigma(np.maximum(depth, 1e-3), incidence)
            vv, uu = lateral_jitter_indices(H, W, sig_l * self.scale * valid, rng)
            depth = depth[vv, uu]
            valid = valid[vv, uu]
            normals = normals[vv, uu]
            incidence = incidence[vv, uu]
            object_id = object_id[vv, uu]
            points = points[vv, uu]
        # grazing incidence is unreliable
        valid = valid & (incidence < THETA_MAX)
        # structured light shadow
        if self.shadow and self.baseline > 0:
            proj = T_world_cam.apply(np.array([self.baseline, 0.0, 0.0]))
            m = valid.ravel()
            if m.any():
                sh = structured_light_shadow(points.reshape(-1, 3)[m], normals.reshape(-1, 3)[m], proj, bvh)
                vflat = valid.ravel().copy()
                idx = np.where(m)[0]
                vflat[idx[sh]] = False
                valid = vflat.reshape(H, W)
        # axial noise
        z = np.maximum(depth, 1e-3)
        if model == "nguyen":
            sig_z, _ = nguyen_sigma(z, incidence)
        elif model == "khoshelham":
            sig_z = khoshelham_sigma(z)
        elif model == "handa":
            sig_z = handa_sigma(z)
        else:
            raise ValueError(f"unknown depth noise model {model}")
        sig_z = sig_z * self.scale
        if model == "handa":
            # perturb the 3D point along its normal and take the resulting depth
            d_cam = self.cam.ray_dirs_cam().reshape(H, W, 3)
            n_cam = T_world_cam.inverse().apply_vec(normals.reshape(-1, 3)).reshape(H, W, 3)
            dn = rng.normal(size=(H, W)) * sig_z
            cosang = np.abs(np.einsum("ijk,ijk->ij", n_cam, d_cam))
            depth = depth + dn * cosang
        else:
            depth = depth + rng.normal(size=(H, W)) * sig_z
        # disparity quantisation
        dq, sig_q = quantise_disparity(np.maximum(depth, 1e-3), self.cam.fx, self.baseline, self.quant)
        depth = np.where(valid, dq, 0.0)
        sigma = np.where(valid, np.sqrt(sig_z ** 2 + sig_q ** 2), 0.0)
        valid = valid & (depth > self.min_range) & (depth < self.max_range)
        return depth, valid, sigma, normals, incidence, object_id
