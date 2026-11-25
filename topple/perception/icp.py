"""Point to plane ICP (Chen and Medioni 1992, as used by KinectFusion) with a KD tree.

Matches are rejected when farther than ``max_dist`` or when the source and target normals
disagree by more than ``normal_tol_deg``; each update is clamped to ``max_step`` (rotation in
radians, translation in metres) so that the refinement stays a refinement of a nearly correct
kinematic pose.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..core.transforms import SE3, quat_from_rotvec


def icp_point_to_plane(src_pts: np.ndarray, dst_pts: np.ndarray, dst_normals: np.ndarray, T0: SE3 | None = None, iters: int = 10, max_dist: float | None = None, tree: cKDTree | None = None, src_normals: np.ndarray | None = None, normal_tol_deg: float = 30.0, max_step: tuple[float, float] = (0.05, 0.02)) -> tuple[SE3, float]:
    """Refine T so that T(src) lies on the surface sampled by (dst_pts, dst_normals).

    Returns the refined pose and the final root mean square point to plane residual (NaN when
    no registration was possible).
    """
    T = SE3() if T0 is None else T0.copy()
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    nrm = np.asarray(dst_normals, dtype=np.float64)
    if src.shape[0] < 10 or dst.shape[0] < 10:
        return T, float("nan")
    tree = cKDTree(dst) if tree is None else tree
    cos_tol = np.cos(np.deg2rad(normal_tol_deg))
    rms = float("nan")
    for _ in range(int(iters)):
        p = T.apply(src)
        d, idx = tree.query(p)
        thr = max_dist if max_dist is not None else 3.0 * np.median(d) + 1e-6
        m = d < thr
        if src_normals is not None:
            ns = T.apply_vec(np.asarray(src_normals, dtype=np.float64))
            m &= np.einsum("ij,ij->i", ns, nrm[idx]) > cos_tol
        if m.sum() < 12:
            break
        q = dst[idx[m]]
        n = nrm[idx[m]]
        pm = p[m]
        b = np.einsum("ij,ij->i", n, q - pm)
        A = np.concatenate([np.cross(pm, n), n], axis=1)
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        rot = np.linalg.norm(x[:3])
        if rot > max_step[0]:
            x[:3] *= max_step[0] / rot
        tr = np.linalg.norm(x[3:])
        if tr > max_step[1]:
            x[3:] *= max_step[1] / tr
        dT = SE3(quat_from_rotvec(x[:3]), x[3:])
        T = dT.compose(T)
        rms = float(np.sqrt(np.mean(b * b)))
        if np.linalg.norm(x) < 1e-9:
            break
    return T, rms
