"""Uniform view of a stone estimate for the planner and the margin model.

``EstimateView`` wraps a ``StoneEstimate`` (world frame meshes at scan time) and exposes
everything the planner (Section 9) and the margin model (Section 8.8) read, all in the
estimate frame: the principal frame of the posterior mean mesh.

* ``frame``: pose of the estimate frame in the world at scan time.
* ``mean_mesh``: posterior mean mesh in the estimate frame; ``samples``: posterior samples.
* ``shape_modes``, ``mode_std``: PCA of the posterior surface displacement at the mean mesh
  vertices (along the vertex normals), used by the reduced vector of the unscented transform
  and by the Monte Carlo sampler; ``mode_observed`` marks modes whose energy sits mostly on
  vertices in observed voxels (surface noise) as opposed to completed regions (underside).
* ``com_mean``, ``com_cov``: centre of mass offset from the mesh centroid and its covariance
  over the posterior (density and shape).
* ``log_mu_mean``, ``log_mu_std``: friction posterior in log space.
* ``pose_rot_std``, ``pose_trans_std``: residual pose uncertainty from the samples.
* ``plane_std``: contact plane orientation uncertainty, the median surface standard
  deviation divided by half the stone's extent (a geometric proxy, author's choice).

``OracleEstimate`` carries exact geometry, mass properties and the true friction table
with zero uncertainty (condition E1).
"""

from __future__ import annotations

import numpy as np

from ..core.mesh import TriMesh
from ..core.transforms import SE3, quat_to_rotvec


class EstimateView:
    def __init__(self, est, n_modes: int = 10, n_mode_samples: int = 48, rng=None, smooth_iterations: int = 10):
        self.raw = est
        self.density = float(est.density)
        mean = est.mean_mesh
        if smooth_iterations > 0:
            # Taubin smoothing removes the millimetre spikes of the fused surface that otherwise
            # inflate the convex hull used for contact (volume preserving in the mean)
            mean = mean.taubin_smooth(int(smooth_iterations))
        mesh_body, frame = mean.to_principal_frame(self.density)
        self.frame: SE3 = frame
        self.mean_mesh: TriMesh = mesh_body
        T_inv = frame.inverse()
        self.samples = []
        for s in est.samples:
            s2 = type(s)(s.mesh.transformed(T_inv), s.mass, T_inv.apply(s.com), s.inertia, s.mu, s.pose_offset, getattr(s, "density", self.density))
            self.samples.append(s2)
        self.mu_samples = np.asarray(est.mu_samples, dtype=float)
        self.meta = dict(est.meta)
        # mass, volume, centre of mass
        self.volume = float(np.mean([s.mesh.volume() for s in self.samples])) if self.samples else self.mean_mesh.volume()
        self.mass = float(np.mean([s.mass for s in self.samples])) if self.samples else self.density * self.mean_mesh.volume()
        centroid = self.mean_mesh.centroid()
        if self.samples:
            offs = np.array([s.com - s.mesh.centroid() for s in self.samples])
            coms = np.array([s.com for s in self.samples])
            self.com_mean = offs.mean(axis=0)
            self.com_cov = np.cov(coms.T) if len(coms) > 1 else np.zeros((3, 3))
            self.com = coms.mean(axis=0)
        else:
            self.com_mean = np.zeros(3)
            self.com_cov = np.zeros((3, 3))
            self.com = centroid
        # friction
        lm = np.log(np.clip(self.mu_samples, 1e-3, None)) if len(self.mu_samples) else np.array([np.log(0.6)])
        self.log_mu_mean = float(lm.mean())
        self.log_mu_std = float(lm.std())
        self.mu = float(np.exp(self.log_mu_mean))
        # pose
        if self.samples:
            rv = np.array([quat_to_rotvec(s.pose_offset.q) for s in self.samples])
            tr = np.array([s.pose_offset.t for s in self.samples])
            self.pose_rot_std = float(np.sqrt((rv ** 2).sum(axis=1).mean() / 3))
            self.pose_trans_std = float(np.sqrt((tr ** 2).sum(axis=1).mean() / 3))
        else:
            self.pose_rot_std = 0.0
            self.pose_trans_std = 0.0
        # shape modes at the mean mesh vertices (along normals)
        self.shape_modes, self.mode_std, self.mode_observed, self.surface_std = self._modes(n_modes, n_mode_samples, rng)
        ext = float(np.max(self.mean_mesh.extents()))
        self.plane_std = float(np.median(self.surface_std) / max(0.5 * ext, 1e-3)) if self.surface_std is not None else 0.0

    def _modes(self, n_modes, n_samples, rng):
        post = self.meta.get("posterior")
        if post is None or n_modes <= 0:
            return np.zeros((0,) + self.mean_mesh.vertices.shape), np.zeros(0), np.zeros(0, dtype=bool), None
        from .posterior import sample_grid

        rng = post.rng if rng is None else rng
        Vw = self.frame.apply(self.mean_mesh.vertices)  # world frame at scan time (posterior grid frame)
        std_v = sample_grid(post.std_field, post.origin, post.spacing, Vw)
        std_v = np.where(np.isfinite(std_v), std_v, getattr(post, "prior_std_min", 0.0))
        disp = np.stack([-std_v * sample_grid(post.noise_field(rng), post.origin, post.spacing, Vw) for _ in range(n_samples)])
        disp = np.where(np.isfinite(disp), disp, 0.0)
        U, S, Vt = np.linalg.svd(disp, full_matrices=False)
        k = min(n_modes, Vt.shape[0])
        sig = S[:k] / np.sqrt(n_samples)
        keep = sig > 1e-7
        normals = self.mean_mesh.vertex_normals()
        modes = Vt[:k][keep][:, :, None] * normals[None, :, :]
        # observed mask: vertices inside voxels that received depth measurements
        tsdf = self.meta.get("tsdf")
        observed = np.ones(len(Vw), dtype=bool)
        if tsdf is not None and hasattr(tsdf, "W"):
            w = sample_grid(tsdf.W, tsdf.origin, tsdf.spacing, Vw)
            observed = np.isfinite(w) & (w > 0)
        energy = (Vt[:k][keep] ** 2)
        frac_obs = (energy * observed[None, :]).sum(axis=1) / np.maximum(energy.sum(axis=1), 1e-12)
        return modes, sig[keep], frac_obs > 0.5, std_v

    def __repr__(self):
        return f"EstimateView(V={self.mean_mesh.n_vertices}, mass={self.mass:.3f}, mu={self.mu:.2f}, modes={len(self.mode_std)})"


class OracleEstimate:
    """Exact inputs (E1): true mesh in the body frame, true mass properties, true friction table."""

    def __init__(self, spec, mu_table: dict, pose: SE3, body=None):
        """``body`` is the true RigidBody; its shape mesh defines the estimate frame so that the
        estimate frame and the true body frame coincide exactly (the principal frame of a mesh is
        defined only up to axis sign flips, so re deriving it from ``spec.mesh`` is not enough)."""
        self.raw = spec
        self.frame = pose.copy()
        mesh = spec.mesh if body is None else body.shape.mesh
        self.mean_mesh = mesh
        self.density = float(spec.density)
        self.samples = []
        self.mu_samples = np.array([mu_table.get("ground", 0.6)])
        self.meta = {"oracle": True}
        self.volume = float(spec.volume) if body is None else float(mesh.volume())
        self.mass = float(spec.mass) if body is None else float(body.mass)
        self.com = np.asarray(spec.com, dtype=float) if body is None else np.zeros(3)
        self.com_mean = self.com - mesh.centroid()
        self.com_cov = np.zeros((3, 3))
        self.mu_table = dict(mu_table)
        self.name = spec.name
        self.mu = float(mu_table.get("ground", 0.6))
        self.log_mu_mean = float(np.log(self.mu))
        self.log_mu_std = 0.0
        self.pose_rot_std = 0.0
        self.pose_trans_std = 0.0
        self.plane_std = 0.0
        self.shape_modes = np.zeros((0,) + mesh.vertices.shape)
        self.mode_std = np.zeros(0)
        self.mode_observed = np.zeros(0, dtype=bool)
        self.surface_std = None


def underside_modes(mean_mesh, modes: np.ndarray, R: np.ndarray, nz_max: float = -0.5) -> np.ndarray:
    """Shape modes concentrated on the underside of a placed stone (Sobol grouping, E3).

    A vertex faces down when its normal, rotated by the placement rotation ``R``, has a vertical
    component below ``nz_max``. A mode is an underside mode when the share of its energy on
    down facing vertices is larger than the share of down facing vertices, that is, when it is
    more concentrated on the underside than a uniform displacement. Returns a boolean per mode."""
    modes = np.asarray(modes, dtype=float)
    if modes.size == 0:
        return np.zeros(0, dtype=bool)
    nz = (np.asarray(mean_mesh.vertex_normals()) @ np.asarray(R, dtype=float).T)[:, 2]
    down = nz < nz_max
    if not down.any():
        return np.zeros(len(modes), dtype=bool)
    energy = (modes ** 2).sum(axis=2)  # (K, nV)
    frac = (energy * down[None, :]).sum(axis=1) / np.maximum(energy.sum(axis=1), 1e-300)
    return frac > down.mean()
