"""Shape posterior over the fused distance field (Section 8.5, author's design).

Each observed voxel carries an approximate Gaussian with mean D and variance equal to the
reciprocal of the summed fusion weights. Unobserved voxels receive a prior from one of

* ``hull``: the signed distance of the convex hull of the observed surface,
* ``mirror``: the observed field reflected about the best principal plane where that lands on
  observed voxels, hull elsewhere,
* ``gp``: a Gaussian process on the distance field (squared exponential kernel, hull prior
  mean) conditioned on observed surface points and on points offset along their normals.

The prior standard deviation of an unobserved voxel grows with its distance to the nearest
observed surface point (``std_growth`` times that distance, floored at a quarter voxel and
capped at half the stone radius). The outermost voxel layer is forced outside so that every
sample is a closed surface. Samples add spatially correlated perturbations (Gaussian filtered white
noise with correlation length 5 mm) scaled by the per voxel standard deviation, and one mesh
is extracted per sample.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from ..core.mesh import TriMesh
from .marching_cubes import largest_component, marching_cubes
from .tsdf import TSDFVolume


def sample_grid(field: np.ndarray, origin: np.ndarray, spacing: float, points: np.ndarray) -> np.ndarray:
    """Trilinear sampling of a grid at world points; NaN outside the grid or next to NaN voxels."""
    p = np.atleast_2d(points)
    g = (p - origin) / spacing
    n = np.array(field.shape)
    out = np.full(p.shape[0], np.nan)
    ok = np.all(g >= 0, axis=1) & np.all(g <= n - 1, axis=1)
    g = np.clip(g, 0, n - 1 - 1e-9)
    i0 = np.floor(g).astype(int)
    f = g - i0
    i1 = np.minimum(i0 + 1, n - 1)
    acc = np.zeros(p.shape[0])
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                ii = np.where(dx, i1[:, 0], i0[:, 0])
                jj = np.where(dy, i1[:, 1], i0[:, 1])
                kk = np.where(dz, i1[:, 2], i0[:, 2])
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                acc += w * field[ii, jj, kk]
    out[ok] = acc[ok]
    return out


class ShapePosterior:
    def __init__(self, tsdf: TSDFVolume, completion: str = "hull", correlation_length: float = 5e-3, rng: np.random.Generator | None = None, prior_std_min: float | None = None, gp_inducing: int = 300, std_growth: float = 0.3):
        self.tsdf = tsdf
        self.completion = completion
        self.corr = float(correlation_length)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.spacing = tsdf.spacing
        self.origin = tsdf.origin
        self.trunc = tsdf.truncation
        self.obs = tsdf.observed()
        self.D_obs = tsdf.field(None)
        self.std_obs = tsdf.voxel_std()
        self.prior_std_min = 0.25 * self.spacing if prior_std_min is None else float(prior_std_min)
        self.centres = tsdf.voxel_centres()
        self.observed_mesh = tsdf.extract_mesh(None, keep_largest=True)
        self._filter_norm = None
        self.modes = None
        self.mode_sigmas = None
        self.std_growth = std_growth
        prior, prior_std = self._completion_prior(gp_inducing)
        self.prior_field = prior
        self.mean_field = np.where(self.obs, self.D_obs, prior)
        self.std_field = np.where(self.obs, self.std_obs, prior_std)
        self._force_boundary_outside(self.mean_field, self.std_field)
        self.mean_mesh = self._extract(self.mean_field)

    # ---------------------------------------------------------------- priors
    def _surface_points(self) -> tuple[np.ndarray, np.ndarray]:
        m = self.observed_mesh
        if m.n_faces < 4:
            # fall back to zero crossings of observed voxels
            pts = self.centres[(self.obs & (np.abs(self.D_obs) < self.spacing)).ravel()]
            return pts, np.zeros_like(pts)
        return m.vertices, m.vertex_normals()

    def _hull_field(self, pts: np.ndarray) -> np.ndarray:
        if pts.shape[0] < 4:
            c = self.origin + 0.5 * self.spacing * (np.array(self.tsdf.shape) - 1)
            r = 0.25 * self.spacing * min(self.tsdf.shape)
            return np.linalg.norm(self.centres - c, axis=1).reshape(self.tsdf.shape) - r
        hull = TriMesh(pts, np.zeros((0, 3), dtype=np.int64), "hull").convex_hull()
        d = hull.signed_distance(self.centres).reshape(self.tsdf.shape)
        return d

    def _distance_std(self, pts: np.ndarray) -> np.ndarray:
        """Prior standard deviation growing with the distance to the nearest observed surface point."""
        if pts.shape[0] == 0:
            return np.full(self.tsdf.shape, self.trunc)
        tree = cKDTree(pts)
        d, _ = tree.query(self.centres)
        R = max(np.linalg.norm(pts - pts.mean(axis=0), axis=1).max(), self.spacing)
        return np.clip(self.std_growth * d, self.prior_std_min, 0.5 * R).reshape(self.tsdf.shape)

    def _force_boundary_outside(self, field: np.ndarray, std: np.ndarray) -> None:
        """The outermost voxel layer is outside with certainty, so every sample is a closed surface."""
        big = max(self.trunc, 4 * self.spacing)
        for ax in range(3):
            sl = [slice(None)] * 3
            for idx in (0, -1):
                sl[ax] = idx
                field[tuple(sl)] = np.maximum(field[tuple(sl)], big)
                std[tuple(sl)] = 0.0

    def _completion_prior(self, gp_inducing: int) -> tuple[np.ndarray, np.ndarray]:
        pts, nrm = self._surface_points()
        hull = self._hull_field(pts)  # not truncated: only the zero crossing matters for the mesh
        std = self._distance_std(pts)
        if self.completion == "hull" or pts.shape[0] < 10:
            return hull, std
        if self.completion == "mirror":
            return self._mirror_prior(pts, hull, std)
        if self.completion == "gp":
            return self._gp_prior(pts, nrm, hull, std, gp_inducing)
        if self.completion == "poisson":
            return self._poisson_prior(pts, nrm, hull, std)
        raise ValueError(f"unknown completion {self.completion}")

    def _poisson_prior(self, pts, nrm, hull, std):
        """Screened Poisson surface through the observed oriented points (Kazhdan and Hoppe 2013),
        redistanced: the prior is the signed distance to the extracted closed surface, so that it
        grows away from the surface like the TSDF and the hull prior."""
        from .poisson import screened_poisson

        if not np.any(np.linalg.norm(nrm, axis=1) > 0):
            return hull, std
        phi = screened_poisson(pts, nrm, self.origin, self.spacing, self.tsdf.shape, tol=1e-5)
        surf = marching_cubes(phi, self.origin, self.spacing, 0.0)
        surf = largest_component(surf) if surf.n_faces > 0 else surf
        if surf.n_faces < 8:
            return hull, std
        return surf.signed_distance(self.centres).reshape(self.tsdf.shape), std

    def _mirror_prior(self, pts, hull, std):
        c = pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(pts - c, full_matrices=False)
        best = None
        for k in range(3):
            u = Vt[k]
            x = self.centres
            xm = x - 2.0 * ((x - c) @ u)[:, None] * u[None, :]
            vals = sample_grid(np.where(self.obs, self.D_obs, np.nan), self.origin, self.spacing, xm).reshape(self.tsdf.shape)
            ok = np.isfinite(vals)
            both = ok & self.obs
            score = np.mean(np.abs(vals[both] - self.D_obs[both])) if both.sum() > 10 else np.inf
            gain = (ok & ~self.obs).sum()
            if best is None or score < best[0]:
                best = (score, k, vals, ok, gain)
        _, k, vals, ok, _ = best
        self.mirror_axis = int(k)
        prior = np.where(ok, vals, hull)
        std_out = np.where(ok, np.maximum(std * 0.5, self.prior_std_min), std)
        return prior, std_out

    def _gp_prior(self, pts, nrm, hull, std, n_ind):
        rng = self.rng
        n = pts.shape[0]
        sel = rng.choice(n, size=min(n_ind, n), replace=False)
        P = pts[sel]
        N = nrm[sel]
        delta = 2.0 * self.spacing
        X = np.concatenate([P, P + delta * N, P - delta * N])
        y = np.concatenate([np.zeros(len(P)), np.full(len(P), delta), np.full(len(P), -delta)])
        R = np.linalg.norm(pts - pts.mean(axis=0), axis=1).max()
        ell = 0.15 * R
        s2 = self.trunc ** 2
        m_train = sample_grid(hull, self.origin, self.spacing, X)
        m_train = np.where(np.isfinite(m_train), m_train, 0.0)

        def kern(A, B):
            d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
            return s2 * np.exp(-0.5 * d2 / (ell * ell))

        K = kern(X, X) + 1e-6 * s2 * np.eye(len(X))
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y - m_train))
        unobs = ~self.obs.ravel()
        Xq = self.centres[unobs]
        prior = hull.copy()
        std_out = std.copy()
        mean_q = np.empty(Xq.shape[0])
        var_q = np.empty(Xq.shape[0])
        for s in range(0, Xq.shape[0], 20000):
            Ks = kern(Xq[s : s + 20000], X)
            mean_q[s : s + 20000] = Ks @ alpha
            v = np.linalg.solve(L, Ks.T)
            var_q[s : s + 20000] = s2 - (v * v).sum(axis=0)
        flat_prior = prior.reshape(-1)
        flat_prior[unobs] = hull.reshape(-1)[unobs] + mean_q
        flat_std = std_out.reshape(-1)
        flat_std[unobs] = np.clip(np.sqrt(np.maximum(var_q, 0.0)), self.prior_std_min, std.reshape(-1)[unobs])
        return flat_prior.reshape(self.tsdf.shape), flat_std.reshape(self.tsdf.shape)

    # --------------------------------------------------------------- sampling
    def _extract(self, field: np.ndarray) -> TriMesh:
        m = marching_cubes(field, self.origin, self.spacing, 0.0)
        return largest_component(m) if m.n_faces > 0 else m

    def noise_field(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Unit variance Gaussian random field with the configured correlation length."""
        rng = self.rng if rng is None else rng
        sig = self.corr / self.spacing
        if self._filter_norm is None:
            delta = np.zeros(self.tsdf.shape)
            delta[tuple(np.array(self.tsdf.shape) // 2)] = 1.0
            k = gaussian_filter(delta, sig, mode="constant")
            self._filter_norm = float(np.sqrt((k * k).sum()))
        white = rng.normal(size=self.tsdf.shape)
        return gaussian_filter(white, sig, mode="nearest") / self._filter_norm

    def sample_field(self, rng: np.random.Generator | None = None) -> np.ndarray:
        return self.mean_field + self.std_field * self.noise_field(rng)

    def sample(self, n: int, rng: np.random.Generator | None = None, decimate_to: int | None = None) -> list[TriMesh]:
        rng = self.rng if rng is None else rng
        out = []
        for _ in range(int(n)):
            m = self._extract(self.sample_field(rng))
            if decimate_to and m.n_faces > decimate_to:
                from .decimate import decimate

                m = decimate(m, decimate_to)
            out.append(m)
        return out

    # ------------------------------------------------------------ shape modes
    def shape_modes(self, n_modes: int = 10, n_samples: int = 64, rng: np.random.Generator | None = None):
        """Principal modes of the surface displacement along the mean mesh normals.

        For a distance field the surface moves by minus the field perturbation along the normal,
        so displacements are the noise field sampled at the mean mesh vertices. Returns
        (modes (n_modes, n_vertices), sigmas (n_modes,)).
        """
        rng = self.rng if rng is None else rng
        V = self.mean_mesh.vertices
        std_v = sample_grid(self.std_field, self.origin, self.spacing, V)
        std_v = np.where(np.isfinite(std_v), std_v, self.prior_std_min)
        disp = np.stack([-std_v * sample_grid(self.noise_field(rng), self.origin, self.spacing, V) for _ in range(n_samples)])
        disp = np.where(np.isfinite(disp), disp, 0.0)
        U, S, Vt = np.linalg.svd(disp, full_matrices=False)
        k = min(n_modes, Vt.shape[0])
        self.modes = Vt[:k]
        self.mode_sigmas = S[:k] / np.sqrt(n_samples)
        return self.modes, self.mode_sigmas

    def mesh_from_modes(self, coeffs: np.ndarray) -> TriMesh:
        if self.modes is None:
            self.shape_modes()
        c = np.asarray(coeffs, dtype=np.float64)
        k = min(len(c), len(self.mode_sigmas))
        d = (c[:k] * self.mode_sigmas[:k]) @ self.modes[:k]
        n = self.mean_mesh.vertex_normals()
        return TriMesh(self.mean_mesh.vertices + d[:, None] * n, self.mean_mesh.faces.copy(), self.mean_mesh.name)
