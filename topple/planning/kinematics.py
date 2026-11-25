"""Inverse kinematics (Section 9.4).

Damped least squares with a null space joint limit term (Buss 2004) and selectively damped
least squares (Buss and Kim 2005). Forward kinematics and Jacobians come from ``ArmModel``.
"""

from __future__ import annotations

import numpy as np

from ..core.transforms import SE3, quat_conj, quat_mul, quat_to_rotvec


def pose_error(target: SE3, current: SE3) -> np.ndarray:
    """6 vector (position error, rotation vector of R_t R_c^T) in the world frame."""
    dp = target.t - current.t
    dq = quat_mul(target.q, quat_conj(current.q))
    return np.concatenate([dp, quat_to_rotvec(dq)])


def _limit_gradient(arm, q, weight: float) -> np.ndarray:
    """Gradient of H(q) = -weight * sum(((q - mid) / half_range)^2), pushing joints to mid range."""
    mid = 0.5 * (arm.q_min + arm.q_max)
    half = 0.5 * (arm.q_max - arm.q_min)
    return -2.0 * weight * (q - mid) / (half ** 2)


def ik_dls(arm, target: SE3, q0, iters: int = 300, lam: float = 0.05, joint_limit_weight: float = 0.02, tol_pos: float = 1e-4, tol_rot: float = 1e-3, step_max: float = 0.3, pos_weight: float = 1.0, rot_weight: float = 0.5):
    """Damped least squares IK: dq = J^T (J J^T + lam^2 I)^-1 e + (I - J^+ J) grad H(q).

    Returns (q, err) where err is the final 6 vector pose error.
    """
    q = arm.clamp(np.asarray(q0, dtype=np.float64).copy())
    W = np.diag([pos_weight] * 3 + [rot_weight] * 3)
    err = pose_error(target, arm.fk(q))
    for _ in range(iters):
        if np.linalg.norm(err[:3]) < tol_pos and np.linalg.norm(err[3:]) < tol_rot:
            break
        J = arm.jacobian(q)
        Jw = W @ J
        ew = W @ err
        JJt = Jw @ Jw.T + lam * lam * np.eye(6)
        Jdls = Jw.T @ np.linalg.solve(JJt, np.eye(6))
        dq = Jdls @ ew
        if joint_limit_weight > 0:
            N = np.eye(7) - Jdls @ Jw
            dq = dq + N @ _limit_gradient(arm, q, joint_limit_weight)
        n = np.linalg.norm(dq)
        if n > step_max:
            dq *= step_max / n
        q = arm.clamp(q + dq)
        err = pose_error(target, arm.fk(q))
    return q, err


def ik_sdls(arm, target: SE3, q0, iters: int = 300, gamma_max: float = 0.4, tol_pos: float = 1e-4, tol_rot: float = 1e-3, joint_limit_weight: float = 0.0):
    """Selectively damped least squares IK (Buss and Kim 2005) for one 6 DoF end effector."""
    q = arm.clamp(np.asarray(q0, dtype=np.float64).copy())
    err = pose_error(target, arm.fk(q))
    for _ in range(iters):
        if np.linalg.norm(err[:3]) < tol_pos and np.linalg.norm(err[3:]) < tol_rot:
            break
        J = arm.jacobian(q)
        U, s, Vt = np.linalg.svd(J, full_matrices=False)
        dq = np.zeros(7)
        col_norms = np.linalg.norm(J, axis=0)
        for i in range(len(s)):
            if s[i] < 1e-8:
                continue
            u, v = U[:, i], Vt[i]
            alpha = float(u @ err)
            # N_i: norm of the end effector response, M_i: estimate of joint displacement per unit response
            N_i = np.linalg.norm(u[:3]) + np.linalg.norm(u[3:])
            M_i = (1.0 / s[i]) * float(np.sum(np.abs(v) * col_norms))
            gamma_i = min(1.0, N_i / max(M_i, 1e-12)) * gamma_max
            phi = (alpha / s[i]) * v
            m = np.max(np.abs(phi))
            if m > gamma_i:
                phi *= gamma_i / m
            dq += phi
        if joint_limit_weight > 0:
            dq += _limit_gradient(arm, q, joint_limit_weight)
        m = np.max(np.abs(dq))
        if m > gamma_max:
            dq *= gamma_max / m
        q = arm.clamp(q + dq)
        err = pose_error(target, arm.fk(q))
    return q, err
