"""Arm model (Section 6.6): a 7 degree of freedom revolute chain with two prismatic fingers.

Kinematics use the product of exponentials (Murray, Li and Sastry 1994) with screw axes taken
from the modified Denavit Hartenberg chain of the Franka Panda at the zero configuration.
Dynamics in reduced coordinates: recursive Newton Euler for inverse dynamics and the articulated
body algorithm for forward dynamics (Featherstone 2008), with torque and velocity limits.

Kinematic parameters, joint limits, velocity limits and torque limits are the public Franka
Panda datasheet values. Link masses, centres of mass and inertia tensors are the values of the
public ``franka_description`` URDF (identified by Gaz et al. 2019). Fingers have friction
coefficient 1.0 (author's choice, Section 6.6) and a maximum opening of 0.08 m.
"""

from __future__ import annotations

import numpy as np

from ..core.mesh import TriMesh, box, icosphere, concatenate
from ..core.transforms import SE3, quat_from_axis_angle, quat_to_rot, rot_to_quat, skew

# modified DH: (a, d, alpha) for joints 1..7 and the flange
_DH = np.array(
    [
        [0.0, 0.333, 0.0],
        [0.0, 0.0, -np.pi / 2],
        [0.0, 0.316, np.pi / 2],
        [0.0825, 0.0, np.pi / 2],
        [-0.0825, 0.384, -np.pi / 2],
        [0.0, 0.0, np.pi / 2],
        [0.088, 0.0, np.pi / 2],
        [0.0, 0.107, 0.0],  # flange
    ]
)
_HAND_OFFSET = 0.1034  # flange to tool centre point (between the fingertips), m
_PALM_DEPTH = 0.058  # flange to the bottom of the hand body (finger base), m
_HAND_YAW = -np.pi / 4  # hand rotated about the flange z axis

Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
DQ_MAX = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])
TAU_MAX = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
Q_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

# (mass, com (3,), [ixx, ixy, ixz, iyy, iyz, izz]) in link frames, franka_description URDF values
_INERTIAL = [
    (4.970684, [3.875e-03, 2.081e-03, -0.1750], [0.70337, -0.000139, 0.006772, 0.70661, 0.019169, 0.009117]),
    (0.646926, [-3.141e-03, -2.872e-02, 3.495e-03], [0.007962, -3.925e-3, 1.0254e-02, 2.8110e-02, 7.04e-04, 2.5995e-02]),
    (3.228604, [2.7518e-02, 3.9252e-02, -6.6502e-02], [3.7242e-02, -4.761e-03, -1.1396e-02, 3.6155e-02, -1.2805e-02, 1.083e-02]),
    (3.587895, [-5.317e-02, 1.04419e-01, 2.7454e-02], [2.5853e-02, 7.796e-03, -1.332e-03, 1.9552e-02, 8.641e-03, 2.8323e-02]),
    (1.225946, [-1.1953e-02, 4.1065e-02, -3.8437e-02], [3.5549e-02, -2.117e-03, -4.037e-03, 2.9474e-02, 2.29e-04, 8.627e-03]),
    (1.666555, [6.0149e-02, -1.4117e-02, -1.0517e-02], [1.964e-03, 1.09e-04, -1.158e-03, 4.354e-03, 3.41e-04, 5.433e-03]),
    (7.35522e-01, [1.0517e-02, -4.252e-03, 6.1597e-02], [1.2516e-02, -4.28e-04, -1.196e-03, 1.0027e-02, -7.41e-04, 4.815e-03]),
]
_HAND_INERTIAL = (0.73 + 2 * 0.015, [-0.01, 0.0, 0.03 + 0.107], [0.001, 0.0, 0.0, 0.0025, 0.0, 0.0017])  # in link 7 frame


def _sym(v):
    ixx, ixy, ixz, iyy, iyz, izz = v
    return np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=np.float64)


def _combine(m1, c1, I1, m2, c2, I2):
    """Combine two rigid inertias given about their own centres of mass in one frame."""
    m = m1 + m2
    c = (m1 * c1 + m2 * c2) / m
    d1, d2 = c1 - c, c2 - c
    I = I1 + m1 * (np.dot(d1, d1) * np.eye(3) - np.outer(d1, d1)) + I2 + m2 * (np.dot(d2, d2) * np.eye(3) - np.outer(d2, d2))
    return m, c, I


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _dh_fixed(a, d, alpha) -> SE3:
    """Rot_x(alpha) Trans_x(a) Trans_z(d); the joint rotation about z is applied after it."""
    R = _rot_x(alpha)
    t = R @ np.array([a, 0.0, d]) if False else np.array([a, 0.0, 0.0]) + R @ np.array([0.0, 0.0, d])
    return SE3(rot_to_quat(R), t)


def exp_twist(S: np.ndarray, theta: float) -> SE3:
    """Matrix exponential of the twist S = (omega, v) with unit omega, as an SE3."""
    w, v = S[:3], S[3:]
    q = quat_from_axis_angle(w, theta)
    W = skew(w)
    p = (np.eye(3) * theta + (1 - np.cos(theta)) * W + (theta - np.sin(theta)) * W @ W) @ v
    return SE3(q, p)


def _crm(v):
    """Spatial cross product operator for motion vectors v = (omega, vo)."""
    out = np.zeros((6, 6))
    out[:3, :3] = skew(v[:3])
    out[3:, :3] = skew(v[3:])
    out[3:, 3:] = skew(v[:3])
    return out


def _crf(v):
    return -_crm(v).T


def _spatial_inertia(m, c, Ic):
    C = skew(c)
    I = np.zeros((6, 6))
    I[:3, :3] = Ic + m * C @ C.T
    I[:3, 3:] = m * C
    I[3:, :3] = m * C.T
    I[3:, 3:] = m * np.eye(3)
    return I


def _X_from_pose(T_parent_child: SE3) -> np.ndarray:
    """Plucker transform from parent coordinates to child coordinates."""
    E = T_parent_child.R.T
    r = T_parent_child.t
    X = np.zeros((6, 6))
    X[:3, :3] = E
    X[3:, :3] = -E @ skew(r)
    X[3:, 3:] = E
    return X


def capsule_mesh(p0, p1, radius: float, subdivisions: int = 2) -> TriMesh:
    """Watertight capsule from p0 to p1 (a split icosphere whose halves are pushed apart)."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    axis = p1 - p0
    L = np.linalg.norm(axis)
    m = icosphere(subdivisions, radius)
    V = m.vertices.copy()
    V[:, 2] += np.where(V[:, 2] >= 0, L / 2, -L / 2)
    if L > 1e-12:
        from ..core.transforms import rotation_from_two_vectors

        R = rotation_from_two_vectors(np.array([0.0, 0.0, 1.0]), axis / L)
        V = V @ R.T
    V += 0.5 * (p0 + p1)
    return TriMesh(V, m.faces, "capsule")


_S6 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # revolute joint about local z


class ArmModel:
    """7 revolute joints + 2 prismatic fingers, product of exponentials, reduced coordinates."""

    n_joints = 7

    def __init__(self, name: str = "panda7", base: SE3 | None = None):
        self.name = name
        self.base = SE3() if base is None else base.copy()
        self.q_min, self.q_max = Q_MIN.copy(), Q_MAX.copy()
        self.dq_max, self.tau_max = DQ_MAX.copy(), TAU_MAX.copy()
        self.q_home = Q_HOME.copy()
        self.finger_max = 0.08
        self.finger_mu = 1.0
        self.dh = _DH.copy()
        # fixed parent to child transforms (joint rotation applied after)
        self.X_fixed = [_dh_fixed(*row) for row in self.dh]  # 8 entries (7 joints + flange)
        # frames at zero configuration, in the arm base frame
        frames0 = []
        T = SE3()
        for i in range(8):
            T = T @ self.X_fixed[i]
            frames0.append(T.copy())
        self.frames0 = frames0  # joint frames 1..7 and flange
        T_hand = SE3(quat_from_axis_angle([0, 0, 1], _HAND_YAW), [0, 0, 0])
        self.T_flange_hand = T_hand
        self.T_flange_tcp = T_hand @ SE3(None, [0.0, 0.0, _HAND_OFFSET])
        self.M = frames0[7] @ self.T_flange_tcp  # TCP home pose in base frame
        # screw axes in the base frame
        S = np.zeros((7, 6))
        for i in range(7):
            w = frames0[i].R[:, 2]
            qpt = frames0[i].t
            S[i, :3] = w
            S[i, 3:] = -np.cross(w, qpt)
        self.S = S
        # inertial parameters per link (link frames 1..7), hand lumped into link 7
        self.link_mass, self.link_com, self.link_inertia = [], [], []
        for i, (m, c, Iv) in enumerate(_INERTIAL):
            c = np.asarray(c, dtype=np.float64)
            I = _sym(Iv)
            if i == 6:
                mh, ch, Ih = _HAND_INERTIAL
                m, c, I = _combine(m, c, I, mh, np.asarray(ch), _sym(Ih))
            self.link_mass.append(float(m))
            self.link_com.append(c)
            self.link_inertia.append(I)
        self.spatial_inertia = [_spatial_inertia(m, c, I) for m, c, I in zip(self.link_mass, self.link_com, self.link_inertia)]
        # capsule radii per link (links 1..7) and the hand box
        self.link_radius = [0.06, 0.06, 0.06, 0.06, 0.05, 0.05, 0.045]

    # ------------------------------------------------------------------ kinematics
    def clamp(self, q: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(q, dtype=np.float64), self.q_min, self.q_max)

    def _prod_exp(self, q, upto: int) -> SE3:
        T = SE3()
        for i in range(upto):
            T = T @ exp_twist(self.S[i], float(q[i]))
        return T

    def fk(self, q) -> SE3:
        """Tool centre point pose in the world frame."""
        return self.base @ self._prod_exp(q, 7) @ self.M

    def flange(self, q) -> SE3:
        return self.base @ self._prod_exp(q, 7) @ self.frames0[7]

    def link_poses(self, q) -> list[SE3]:
        """Eight link frames (joint frames 1 to 7 and the flange) in the world frame."""
        out = []
        T = SE3()
        for i in range(8):
            if i < 7:
                T = T @ exp_twist(self.S[i], float(q[i]))
            out.append(self.base @ T @ self.frames0[i])
        return out

    def joint_axes(self, q):
        """World frame joint axes (omega_i, point_i) at configuration q."""
        axes, pts = [], []
        T = SE3()
        for i in range(7):
            axes.append(self.base.apply_vec(T.apply_vec(self.S[i, :3])))
            pts.append(self.base.apply(T.apply(self.frames0[i].t)))
            T = T @ exp_twist(self.S[i], float(q[i]))
        return np.array(axes), np.array(pts)

    def jacobian(self, q) -> np.ndarray:
        """(6, 7) geometric Jacobian of the TCP in the world frame, rows [v; w]."""
        axes, pts = self.joint_axes(q)
        p = self.fk(q).t
        J = np.zeros((6, 7))
        for i in range(7):
            J[:3, i] = np.cross(axes[i], p - pts[i])
            J[3:, i] = axes[i]
        return J

    # ------------------------------------------------------------------ dynamics
    def _joint_transforms(self, q):
        """Plucker transforms X_i (parent to child coordinates) for i = 1..7."""
        Xs = []
        for i in range(7):
            T = self.X_fixed[i] @ SE3(quat_from_axis_angle([0, 0, 1], float(q[i])), None)
            Xs.append(_X_from_pose(T))
        return Xs

    def _gravity_base(self, gravity):
        g = np.asarray(gravity, dtype=np.float64)
        return self.base.R.T @ g

    def inverse_dynamics(self, q, dq, ddq, gravity=(0.0, 0.0, -9.81), external_tcp_wrench=None) -> np.ndarray:
        """Joint torques by the recursive Newton Euler algorithm.

        ``external_tcp_wrench`` (force, torque) in world coordinates applied at the TCP is
        mapped to the joints through the Jacobian transpose.
        """
        g = self._gravity_base(gravity)
        Xs = self._joint_transforms(q)
        v = [np.zeros(6)]
        a = [np.concatenate([np.zeros(3), -g])]
        f = []
        for i in range(7):
            vJ = _S6 * float(dq[i])
            vi = Xs[i] @ v[-1] + vJ
            ai = Xs[i] @ a[-1] + _S6 * float(ddq[i]) + _crm(vi) @ vJ
            I = self.spatial_inertia[i]
            f.append(I @ ai + _crf(vi) @ (I @ vi))
            v.append(vi)
            a.append(ai)
        tau = np.zeros(7)
        for i in range(6, -1, -1):
            tau[i] = _S6 @ f[i]
            if i > 0:
                f[i - 1] = f[i - 1] + Xs[i].T @ f[i]
        if external_tcp_wrench is not None:
            w = np.asarray(external_tcp_wrench, dtype=np.float64)
            Jt = self.jacobian(q)
            tau -= Jt.T @ w
        return tau

    def mass_matrix(self, q) -> np.ndarray:
        """Joint space mass matrix from inverse dynamics columns (zero velocity, zero gravity)."""
        M = np.zeros((7, 7))
        z = np.zeros(7)
        for j in range(7):
            e = np.zeros(7)
            e[j] = 1.0
            M[:, j] = self.inverse_dynamics(q, z, e, gravity=(0.0, 0.0, 0.0))
        return 0.5 * (M + M.T)

    def forward_dynamics(self, q, dq, tau, gravity=(0.0, 0.0, -9.81)) -> np.ndarray:
        """Joint accelerations by the articulated body algorithm."""
        g = self._gravity_base(gravity)
        Xs = self._joint_transforms(q)
        v = [np.zeros(6)]
        c, IA, pA = [], [], []
        for i in range(7):
            vJ = _S6 * float(dq[i])
            vi = Xs[i] @ v[-1] + vJ
            v.append(vi)
            c.append(_crm(vi) @ vJ)
            I = self.spatial_inertia[i]
            IA.append(I.copy())
            pA.append(_crf(vi) @ (I @ vi))
        U, D, u = [None] * 7, [0.0] * 7, [0.0] * 7
        for i in range(6, -1, -1):
            U[i] = IA[i] @ _S6
            D[i] = float(_S6 @ U[i])
            u[i] = float(tau[i]) - float(_S6 @ pA[i])
            if i > 0:
                Ia = IA[i] - np.outer(U[i], U[i]) / D[i]
                pa = pA[i] + Ia @ c[i] + U[i] * (u[i] / D[i])
                IA[i - 1] = IA[i - 1] + Xs[i].T @ Ia @ Xs[i]
                pA[i - 1] = pA[i - 1] + Xs[i].T @ pa
        a_prev = np.concatenate([np.zeros(3), -g])
        ddq = np.zeros(7)
        for i in range(7):
            ap = Xs[i] @ a_prev + c[i]
            ddq[i] = (u[i] - U[i] @ ap) / D[i]
            a_prev = ap + _S6 * ddq[i]
        return ddq

    def gravity_torque(self, q, gravity=(0.0, 0.0, -9.81)) -> np.ndarray:
        z = np.zeros(7)
        return self.inverse_dynamics(q, z, z, gravity)

    def integrate(self, q, dq, tau, dt: float, gravity=(0.0, 0.0, -9.81)):
        """Semi implicit Euler step under torque and velocity limits and joint limits."""
        tau = np.clip(np.asarray(tau, dtype=np.float64), -self.tau_max, self.tau_max)
        ddq = self.forward_dynamics(q, dq, tau, gravity)
        dq_new = np.clip(dq + dt * ddq, -self.dq_max, self.dq_max)
        q_new = q + dt * dq_new
        hit = (q_new < self.q_min) | (q_new > self.q_max)
        q_new = np.clip(q_new, self.q_min, self.q_max)
        dq_new = np.where(hit, 0.0, dq_new)
        return q_new, dq_new

    # ------------------------------------------------------------------ geometry
    def link_capsules(self) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """(p0, p1, radius) per link frame 1..7 plus the hand as a capsule in the flange frame."""
        caps = []
        for i in range(7):
            a, d, alpha = self.dh[i + 1]
            p1 = np.array([a, -d * np.sin(alpha), d * np.cos(alpha)])
            p0 = np.zeros(3)
            if i == 0:
                p0 = np.array([0.0, 0.0, -0.19])
            caps.append((p0, p1, self.link_radius[i]))
        # hand, in the flange frame: a short wrist capsule along the flange z axis, then the palm as
        # a bar across the hand width (hand y axis) whose underside is the finger base, 0.058 m
        # below the flange (the fingers are separate bodies and the held stone sits between them)
        caps.append((np.zeros(3), np.array([0.0, 0.0, 0.02]), 0.045))
        return caps

    def hand_extra_capsules(self) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """Palm bar capsule in the flange frame (see ``link_capsules``)."""
        r = 0.025
        y = self.T_flange_hand.apply_vec(np.array([0.0, 1.0, 0.0]))
        c = np.array([0.0, 0.0, _PALM_DEPTH - r])
        return [(c - 0.085 * y, c + 0.085 * y, r)]

    def link_meshes(self) -> list[TriMesh]:
        """Capsule meshes in link frames (7 links) plus the hand box in the flange frame."""
        meshes = []
        caps = self.link_capsules()
        for i in range(7):
            p0, p1, r = caps[i]
            meshes.append(capsule_mesh(p0, p1, r))
        hand = box((0.06, 0.20, 0.06), center=(0.0, 0.0, 0.03)).transformed(self.T_flange_hand)
        meshes.append(hand)
        return meshes

    def base_mesh(self) -> TriMesh:
        return capsule_mesh([0, 0, 0.0], [0, 0, 0.14], 0.09)

    def finger_meshes(self, opening: float) -> tuple[TriMesh, TriMesh]:
        """Two finger boxes in the TCP frame, separated by ``opening`` along the hand y axis."""
        opening = float(np.clip(opening, 0.0, self.finger_max))
        half = 0.5 * opening
        f = box((0.02, 0.012, 0.05), center=(0.0, 0.0, -0.025))  # fingertip at z = 0 (TCP)
        left = f.translated([0.0, -half - 0.006, 0.0])
        right = f.translated([0.0, half + 0.006, 0.0])
        return left, right

    def world_capsules(self, q) -> list[tuple[np.ndarray, np.ndarray, float]]:
        poses = self.link_poses(q)
        out = []
        for (p0, p1, r), T in zip(self.link_capsules(), poses):
            out.append((T.apply(p0), T.apply(p1), r))
        Tf = poses[-1]
        for p0, p1, r in self.hand_extra_capsules():
            out.append((Tf.apply(p0), Tf.apply(p1), r))
        return out
