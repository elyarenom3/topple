"""Rigid transforms and quaternion utilities (Section 4.2).

Conventions
-----------
* World frame W has z up, gravity g = (0, 0, -9.81) m/s^2.
* Rotations are unit quaternions in double precision, stored as (w, x, y, z).
* Transforms are elements of SE(3); SE3 stores a quaternion q and a translation t
  and maps body points p_B to world points p_W = R(q) p_B + t.
* Body frames sit at the centre of mass along the principal axes of inertia.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-15


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(n, _EPS)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a * b for (w, x, y, z) quaternions. Broadcasts."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) v by unit quaternion q. Broadcasts over leading dims."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    qv = q[..., 1:]
    w = q[..., :1]
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Shepperd's method, robust for all rotation matrices."""
    R = np.asarray(R, dtype=np.float64)
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = np.array([0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s])
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2
        q = np.array([(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s])
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2
        q = np.array([(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s])
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2
        q = np.array([(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s])
    return quat_normalize(q)


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < _EPS:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / n
    s = np.sin(0.5 * angle)
    return np.array([np.cos(0.5 * angle), axis[0] * s, axis[1] * s, axis[2] * s])


def quat_from_rotvec(rv: np.ndarray) -> np.ndarray:
    rv = np.asarray(rv, dtype=np.float64)
    angle = np.linalg.norm(rv)
    if angle < 1e-12:
        return quat_normalize(np.array([1.0, 0.5 * rv[0], 0.5 * rv[1], 0.5 * rv[2]]))
    return quat_from_axis_angle(rv / angle, angle)


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    if q[0] < 0:
        q = -q
    s = np.linalg.norm(q[1:])
    if s < 1e-12:
        return 2.0 * q[1:]
    angle = 2.0 * np.arctan2(s, q[0])
    return q[1:] / s * angle


def quat_slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    """Spherical linear interpolation; shortest arc."""
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1 = -q1
        d = -d
    if d > 0.9995:
        return quat_normalize(q0 + u * (q1 - q0))
    th0 = np.arccos(d)
    th = th0 * u
    q2 = quat_normalize(q1 - q0 * d)
    return q0 * np.cos(th) + q2 * np.sin(th)


def quat_integrate(q: np.ndarray, omega_world: np.ndarray, h: float) -> np.ndarray:
    """Exact integration of q under constant world angular velocity for time h."""
    dq = quat_from_rotvec(np.asarray(omega_world, dtype=np.float64) * h)
    return quat_normalize(quat_mul(dq, q))


def skew(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def rotation_from_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit vector a onto unit vector b (Rodrigues)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / max(np.linalg.norm(a), _EPS)
    b = b / max(np.linalg.norm(b), _EPS)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        # opposite: rotate by pi about any axis orthogonal to a
        t = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, t)
        axis /= np.linalg.norm(axis)
        return quat_to_rot(quat_from_axis_angle(axis, np.pi))
    K = skew(v)
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def tangent_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit tangents t1, t2 with (t1, t2, n) right handed (Frisvad style, branch free)."""
    n = np.asarray(n, dtype=np.float64)
    n = n / max(np.linalg.norm(n), _EPS)
    if n[2] < -0.9999999:
        return np.array([0.0, -1.0, 0.0]), np.array([-1.0, 0.0, 0.0])
    a = 1.0 / (1.0 + n[2])
    b = -n[0] * n[1] * a
    t1 = np.array([1.0 - n[0] * n[0] * a, b, -n[0]])
    t2 = np.array([b, 1.0 - n[1] * n[1] * a, -n[1]])
    return t1, t2


@dataclass
class SE3:
    """Rigid transform p_W = R(q) p + t with q = (w, x, y, z)."""

    q: np.ndarray
    t: np.ndarray

    def __init__(self, q=None, t=None):
        self.q = quat_normalize(np.array([1.0, 0.0, 0.0, 0.0]) if q is None else np.asarray(q, dtype=np.float64))
        self.t = np.zeros(3) if t is None else np.asarray(t, dtype=np.float64).copy()

    @staticmethod
    def identity() -> "SE3":
        return SE3()

    @staticmethod
    def from_matrix(T: np.ndarray) -> "SE3":
        T = np.asarray(T, dtype=np.float64)
        return SE3(rot_to_quat(T[:3, :3]), T[:3, 3])

    @staticmethod
    def from_rot(R: np.ndarray, t=None) -> "SE3":
        return SE3(rot_to_quat(R), t)

    @staticmethod
    def from_axis_angle(axis, angle: float, t=None) -> "SE3":
        return SE3(quat_from_axis_angle(axis, angle), t)

    @property
    def R(self) -> np.ndarray:
        return quat_to_rot(self.q)

    def matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    def compose(self, other: "SE3") -> "SE3":
        """self * other: apply other first, then self."""
        return SE3(quat_mul(self.q, other.q), self.t + quat_rotate(self.q, other.t))

    def __matmul__(self, other: "SE3") -> "SE3":
        return self.compose(other)

    def inverse(self) -> "SE3":
        qi = quat_conj(self.q)
        return SE3(qi, -quat_rotate(qi, self.t))

    def apply(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        return quat_rotate(self.q, p) + self.t

    def apply_vec(self, v: np.ndarray) -> np.ndarray:
        return quat_rotate(self.q, np.asarray(v, dtype=np.float64))

    def copy(self) -> "SE3":
        return SE3(self.q.copy(), self.t.copy())

    def interpolate(self, other: "SE3", u: float) -> "SE3":
        """Linear translation, slerp rotation (used by the renderer for motion blur)."""
        return SE3(quat_slerp(self.q, other.q, u), (1 - u) * self.t + u * other.t)

    def __repr__(self) -> str:
        return f"SE3(q={np.round(self.q, 6).tolist()}, t={np.round(self.t, 6).tolist()})"
