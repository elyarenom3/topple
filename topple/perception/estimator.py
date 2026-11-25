"""Stone estimator (Section 8): depth images in, posterior samples of shape, mass properties,
friction coefficient and pose out. Never imports the ground truth store."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..core.mesh import TriMesh
from ..core.transforms import SE3, quat_from_rotvec
from .depth_sensor import DepthImage
from .friction import FrictionPrior, GridPosterior
from scipy.spatial import cKDTree

from .icp import icp_point_to_plane


def _point_to_plane_rms(src_cam, T, tree, dst, dn, max_dist):
    p = T.apply(src_cam)
    d, idx = tree.query(p)
    m = d < max_dist
    if m.sum() < 12:
        return float("inf")
    r = np.einsum("ij,ij->i", dn[idx[m]], dst[idx[m]] - p[m])
    return float(np.sqrt(np.mean(r * r)))
from .posterior import ShapePosterior
from .tsdf import TSDFVolume


@dataclass
class StoneSample:
    mesh: TriMesh
    mass: float
    com: np.ndarray
    inertia: np.ndarray
    mu: float
    pose_offset: SE3
    density: float = 0.0


@dataclass
class StoneEstimate:
    mean_mesh: TriMesh
    samples: list
    density: float
    mu_samples: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def mean_com(self) -> np.ndarray:
        return np.mean([s.com for s in self.samples], axis=0)

    @property
    def com_cov(self) -> np.ndarray:
        return np.cov(np.stack([s.com for s in self.samples]).T)

    @property
    def mean_mass(self) -> float:
        return float(np.mean([s.mass for s in self.samples]))

    @property
    def mean_volume(self) -> float:
        return float(np.mean([s.mesh.volume() for s in self.samples]))

    def body_frame(self) -> tuple[TriMesh, SE3]:
        """Mean mesh in its principal frame and the pose of that frame in the world."""
        return self.mean_mesh.to_principal_frame(self.density)


class Estimator:
    def __init__(self, cfg: dict | None = None, rng: np.random.Generator | None = None):
        cfg = cfg or {}
        self.cfg = cfg
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.scfg = cfg.get("sensor", {})
        self.tcfg = cfg.get("tsdf", {})
        self.fcfg = cfg.get("friction", {})
        self.density_range = tuple(cfg.get("stones", {}).get("density_range", (2500.0, 2800.0)))
        self.voxel = float(self.tcfg.get("voxel", 0.002))
        self.truncation = float(self.tcfg.get("truncation", 4 * self.voxel))
        self.completion = str(self.tcfg.get("completion", "hull"))
        self.corr = float(self.tcfg.get("correlation_length", 0.005))
        self.n_samples_default = int(self.tcfg.get("posterior_samples", 32))
        self.icp_iters = int(self.scfg.get("icp_iterations", 10))
        # residual pose uncertainty of the fused model frame (author's choice). Registration aligns the
        # views to each other, so the absolute frame error of the fused model is the mean of the
        # independent per view kinematic errors: the sensor terms shrink as 1 / sqrt(number of views).
        # The rotation error also moves the model at the lever arm of the camera distance.
        self.sens_rot = np.deg2rad(float(self.scfg.get("pose_noise_rot_deg", 0.0)))
        self.sens_trans = float(self.scfg.get("pose_noise_trans_m", 0.0))
        self.lever_arm = float(self.tcfg.get("pose_lever_arm_m", 0.35))
        self.base_sigma_rot = np.deg2rad(float(self.tcfg.get("pose_sigma_rot_deg", 0.1)))
        self.base_sigma_trans = float(self.tcfg.get("pose_sigma_trans_m", 0.0005))
        self.pose_sigma_rot = self.base_sigma_rot + self.sens_rot
        self.pose_sigma_trans = self.base_sigma_trans + self.sens_trans + self.lever_arm * self.sens_rot
        self.friction_prior = FrictionPrior(float(self.fcfg.get("prior_median", 0.6)), float(self.fcfg.get("prior_log_sigma", 0.25)))
        self.max_voxels = int(self.tcfg.get("max_voxels", 2_000_000))
        self.max_correction = (np.deg2rad(float(self.tcfg.get("icp_max_rot_deg", 1.0))), float(self.tcfg.get("icp_max_trans_m", 0.006)))

    # ---------------------------------------------------------------- helpers
    def _masks(self, images: list[DepthImage], target_id) -> list[np.ndarray]:
        if target_id is None:
            return [im.valid for im in images]
        return [im.valid & (im.object_id == target_id) for im in images]

    def refine_poses(self, images: list[DepthImage], masks: list[np.ndarray], max_points: int = 3000) -> tuple[list[SE3], list[float]]:
        """Leave one out point to plane ICP of each view against the cloud of all other views.

        The believed (kinematic) poses are nearly right, so each view is refined against the
        cloud built from the other views under their believed poses, with normal compatibility
        and a bound on the total correction (``max_correction``: radians, metres). A view whose
        correction exceeds the bound keeps its believed pose.
        """
        poses = [im.T_world_cam.copy() for im in images]
        resid = [float("nan")] * len(images)
        if self.icp_iters <= 0 or len(images) < 2:
            return poses, resid
        rng = self.rng
        cam = images[0].camera()
        clouds, normals, srcs, src_n = [], [], [], []
        for im, m in zip(images, masks):
            pc = cam.depth_to_points(im.depth, m, None)
            nc = im.normals[m & im.valid]
            if pc.shape[0] > max_points:
                sel = rng.choice(pc.shape[0], max_points, replace=False)
                pc, nc = pc[sel], nc[sel]
            srcs.append(pc)
            src_n.append(nc)
            clouds.append(im.T_world_cam.apply(pc))
            normals.append(im.T_world_cam.apply_vec(nc))
        max_rot, max_tr = self.max_correction
        for k, im in enumerate(images):
            if srcs[k].shape[0] < 20:
                continue
            dst = np.concatenate([c for j, c in enumerate(clouds) if j != k])
            dn = np.concatenate([c for j, c in enumerate(normals) if j != k])
            if dst.shape[0] < 20:
                continue
            tree = cKDTree(dst)
            r0 = _point_to_plane_rms(srcs[k], im.T_world_cam, tree, dst, dn, 4 * self.voxel)
            T, r = icp_point_to_plane(srcs[k], dst, dn, im.T_world_cam, self.icp_iters, max_dist=4 * self.voxel, tree=tree, src_normals=src_n[k], max_step=(0.5 * max_rot, 0.5 * max_tr))
            r1 = _point_to_plane_rms(srcs[k], T, tree, dst, dn, 4 * self.voxel)
            corr = T.compose(im.T_world_cam.inverse())
            ang = 2 * np.arccos(np.clip(abs(corr.q[0]), -1, 1))
            if ang <= max_rot and np.linalg.norm(corr.t) <= max_tr and np.isfinite(r1) and r1 < r0:
                poses[k] = T
                resid[k] = r1
            else:
                resid[k] = r0
        return poses, resid

    # --------------------------------------------------------------- estimate
    def estimate(self, depth_images: list[DepthImage], n_samples: int | None = None, target_id=None, slip_angles=None, decimate_mean_to: int | None = 2000, decimate_samples_to: int | None = None, completion: str | None = None) -> StoneEstimate:
        t0 = time.time()
        n_samples = self.n_samples_default if n_samples is None else int(n_samples)
        images = list(depth_images)
        masks = self._masks(images, target_id)
        poses, resid = self.refine_poses(images, masks)
        # bounds from the fused point cloud
        pts = []
        for im, m, T in zip(images, masks, poses):
            cam = im.camera()
            pts.append(T.apply(cam.depth_to_points(im.depth, m, None)))
        P = np.concatenate(pts) if pts else np.zeros((0, 3))
        if P.shape[0] < 10:
            raise ValueError("estimator: too few valid depth pixels for the target")
        lo, hi = P.min(axis=0), P.max(axis=0)
        voxel = self.voxel
        n_vox = np.prod(np.ceil((hi - lo + 4 * self.truncation) / voxel) + 1)
        if n_vox > self.max_voxels:
            voxel = voxel * float((n_vox / self.max_voxels) ** (1 / 3))
        tsdf = TSDFVolume.from_bounds(lo, hi, voxel, self.truncation)
        for im, m, T in zip(images, masks, poses):
            tsdf.integrate(im, T_world_cam=T, mask=m)
        post = ShapePosterior(tsdf, completion or self.completion, self.corr, self.rng)
        mean_mesh = post.mean_mesh
        if decimate_mean_to and mean_mesh.n_faces > decimate_mean_to:
            from .decimate import decimate

            mean_mesh = decimate(mean_mesh, decimate_mean_to)
        meshes = post.sample(n_samples, self.rng, decimate_samples_to)
        # friction
        if slip_angles:
            gp = GridPosterior(self.friction_prior)
            for a in slip_angles:
                gp.update(a, float(self.fcfg.get("slip_noise_deg", 1.0)))
            mu_samples = np.asarray(gp.sample(self.rng, n_samples))
        else:
            mu_samples = np.asarray(self.friction_prior.sample(self.rng, n_samples))
        samples = []
        root_n = np.sqrt(max(len(images), 1))
        self.pose_sigma_rot = self.base_sigma_rot + self.sens_rot / root_n
        self.pose_sigma_trans = self.base_sigma_trans + (self.sens_trans + self.lever_arm * self.sens_rot) / root_n
        densities = self.rng.uniform(self.density_range[0], self.density_range[1], size=n_samples)
        for mesh, mu, rho in zip(meshes, mu_samples, densities):
            if mesh.n_faces < 4 or mesh.volume() <= 0:
                mesh = mean_mesh
            mp = mesh.mass_properties(rho)
            off = SE3(quat_from_rotvec(self.rng.normal(size=3) * self.pose_sigma_rot), self.rng.normal(size=3) * self.pose_sigma_trans)
            samples.append(StoneSample(mesh, mp["mass"], mp["com"], mp["inertia_com"], float(mu), off, float(rho)))
        meta = {"tsdf": tsdf, "posterior": post, "poses": poses, "icp_residuals": resid, "voxel": voxel, "n_images": len(images), "seconds": time.time() - t0, "n_points": int(P.shape[0])}
        return StoneEstimate(mean_mesh, samples, float(np.mean(densities)), mu_samples, meta)
