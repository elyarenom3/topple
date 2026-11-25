"""Arm inside the world: kinematic link and finger bodies driven from joint state (Section 6.6).

The arm links and the two fingers are kinematic bodies in the contact solver (their inverse
mass is zero), so contact impulses act on the stones and not on the arm; the contact wrench
on the fingers is summed into a simulated wrist force torque sensor. Joint dynamics are
integrated with the articulated body algorithm of ``ArmModel`` under torque and velocity
limits.
"""

from __future__ import annotations

import numpy as np

from ..core.transforms import SE3
from .body import CollisionShape, RigidBody


class ArmInWorld:
    def __init__(self, world, model, finger_mu: float = 1.0, link_mu: float = 0.3):
        self.world = world
        self.model = model
        self.q = np.array(model.q_home, dtype=float).copy() if hasattr(model, "q_home") else np.zeros(model.n_joints)
        self.dq = np.zeros(model.n_joints)
        self.opening = model.finger_max
        self.d_opening = 0.0
        self.link_ids = []
        self.finger_ids = []
        for k, mesh in enumerate(model.link_meshes()):
            shape = CollisionShape(mesh=mesh, pieces=[mesh.convex_hull()], sdf=None, bounding_radius=mesh.bounding_radius(np.zeros(3)))
            b = RigidBody(f"arm_link{k}", np.inf, np.diag([np.inf] * 3), shape, friction=link_mu, kinematic=True)
            self.link_ids.append(world.add_body(b))
        # finger meshes at zero opening (inner faces on the tool plane); ``finger_frames`` shifts the
        # left finger by -opening / 2 and the right finger by +opening / 2 along the tool y axis
        for k, mesh in enumerate(model.finger_meshes(0.0)):
            shape = CollisionShape(mesh=mesh, pieces=[mesh.convex_hull()], sdf=None, bounding_radius=mesh.bounding_radius(np.zeros(3)))
            b = RigidBody(f"arm_finger{k}", np.inf, np.diag([np.inf] * 3), shape, friction=finger_mu, kinematic=True)
            self.finger_ids.append(world.add_body(b))
        self._prev_link = None
        self.attached = None  # (body_id, T_tcp_body, grasp_width) while a stone is held
        self.handover = None  # compliant hold during the release
        self.sensor_stiffness = 2.0e4  # N/m, compliance of fingers and wrist seen by the force sensor
        self.set_joint_state(self.q, self.dq, self.opening)

    # --------------------------------------------------------------- holding
    def attach(self, body_id: int, grasp_width: float):
        """Hold a stone rigidly in the tool frame (kinematic) until the fingers open past the grasp
        width. Contact forces on the held stone are reported by ``wrist_wrench``."""
        b = self.world.bodies[body_id]
        T_tcp = self.model.fk(self.q)
        self.attached = (body_id, T_tcp.inverse() @ b.pose, float(grasp_width))
        b.kinematic = True
        b.v[:] = 0.0
        b.w[:] = 0.0

    def detach(self):
        """End any hold: a kinematic attachment becomes a free stone at rest, a handover ends."""
        if self.handover is not None:
            self.handover = None
        if self.attached is None:
            return
        bid = self.attached[0]
        b = self.world.bodies[bid]
        b.kinematic = False
        b.v[:] = 0.0
        b.w[:] = 0.0
        self.attached = None

    # --------------------------------------------------------------- handover
    def begin_handover(self, stiffness: float = 2000.0, rot_stiffness: float = 5.0, f_max: float = 30.0, tau_max: float = 1.0):
        """Compliant release (Section 9.6): the held stone becomes a dynamic body held to its goal
        in the tool frame by a damped spring with gravity feed forward. ``set_handover_scale`` ramps
        the hold from 1 to 0 so that the support takes the load gradually while the fingers open."""
        if self.attached is None:
            return
        bid, T_tcp_body, _ = self.attached
        b = self.world.bodies[bid]
        b.kinematic = False
        b.v[:] = 0.0
        b.w[:] = 0.0
        I = float(np.trace(b.inertia_body)) / 3.0
        self.handover = {
            "bid": bid, "T_tcp_body": T_tcp_body, "scale": 1.0, "k": stiffness, "c": 2.0 * np.sqrt(stiffness * b.mass),
            "kr": rot_stiffness, "cr": 2.0 * np.sqrt(rot_stiffness * I), "f_max": f_max, "tau_max": tau_max,
        }
        self.attached = None

    def set_handover_scale(self, s: float):
        if self.handover is not None:
            self.handover["scale"] = float(np.clip(s, 0.0, 1.0))
            if self.handover["scale"] <= 0.0:
                self.handover = None

    def _apply_handover(self):
        h = self.handover
        if h is None:
            return
        from ..core.transforms import quat_conj, quat_mul, quat_to_rotvec

        b = self.world.bodies[h["bid"]]
        T_goal = self.model.fk(self.q) @ h["T_tcp_body"]
        s = h["scale"]
        F = s * (h["k"] * (T_goal.t - b.x) - h["c"] * b.v - b.mass * np.asarray(self.world.gravity))
        nF = float(np.linalg.norm(F))
        if nF > h["f_max"]:
            F *= h["f_max"] / nF
        rv = quat_to_rotvec(quat_mul(T_goal.q, quat_conj(b.q)))
        tau = s * (h["kr"] * rv - h["cr"] * b.w)
        nt = float(np.linalg.norm(tau))
        if nt > h["tau_max"]:
            tau *= h["tau_max"] / nt
        self.world.external_force(h["bid"], F, b.x)
        self.world.external_torque(h["bid"], tau)

    # --------------------------------------------------------------- kinematics
    def finger_frames(self, q, opening) -> list[SE3]:
        T = self.model.fk(q)
        return [T @ SE3(t=np.array([0.0, -opening / 2, 0.0])), T @ SE3(t=np.array([0.0, +opening / 2, 0.0]))]

    def set_joint_state(self, q, dq, opening, d_opening: float = 0.0):
        self.q = np.asarray(q, dtype=float).copy()
        self.dq = np.asarray(dq, dtype=float).copy()
        self.opening = float(np.clip(opening, 0.0, self.model.finger_max))
        self.d_opening = float(d_opening)
        dt = self.world.dt
        links = self.model.link_poses(self.q)
        fingers = self.finger_frames(self.q, self.opening)
        # velocities from one step finite differences of the next pose
        q_next = self.q + self.dq * dt
        links_next = self.model.link_poses(q_next)
        fingers_next = self.finger_frames(q_next, self.opening + self.d_opening * dt)
        if self.attached is not None:
            bid, T_tcp_body, width = self.attached
            if self.opening > width + 0.003:
                self.detach()
            else:
                T_now = self.model.fk(self.q) @ T_tcp_body
                T_nxt = self.model.fk(q_next) @ T_tcp_body
                links = list(links)
                b = self.world.bodies[bid]
                b.pose = T_now
                from ..core.transforms import quat_to_rotvec as _qrv

                self.world.set_kinematic_velocity(bid, (T_nxt.t - T_now.t) / dt, _qrv((T_nxt @ T_now.inverse()).q) / dt)
        self._apply_handover()
        for ids, now, nxt in ((self.link_ids, links, links_next), (self.finger_ids, fingers, fingers_next)):
            for bid, T0, T1 in zip(ids, now, nxt):
                b = self.world.bodies[bid]
                b.pose = T0
                v = (T1.t - T0.t) / dt
                dq_rel = (T1 @ T0.inverse()).q
                from ..core.transforms import quat_to_rotvec

                w = quat_to_rotvec(dq_rel) / dt
                self.world.set_kinematic_velocity(bid, v, w)

    def step_dynamics(self, tau, dt):
        """Integrate joint accelerations from ABA with gravity and clamp to the limits."""
        g = self.world.gravity
        ddq = self.model.forward_dynamics(self.q, self.dq, np.clip(tau, -self.model.tau_max, self.model.tau_max), g)
        dq = np.clip(self.dq + ddq * dt, -self.model.dq_max, self.model.dq_max)
        q = np.clip(self.q + dq * dt, self.model.q_min, self.model.q_max)
        self.set_joint_state(q, dq, self.opening, self.d_opening)

    # --------------------------------------------------------------- sensing
    def wrist_wrench(self) -> np.ndarray:
        """Force and torque (about the TCP) from the last step's finger contacts, in N and N m."""
        dt = self.world.dt
        T = self.model.fk(self.q)
        f_tot = np.zeros(3)
        tau_tot = np.zeros(3)
        held = self.attached[0] if self.attached is not None else None
        contacts = list(self.world.contacts)
        if held is not None:
            # the engine skips pairs of two non dynamic bodies, so the contacts of the kinematic held
            # stone with a static support are queried here with the stone treated as dynamic
            from .collision import find_contacts

            hb = self.world.bodies[held]
            hb.kinematic = False
            try:
                extra = [c for c in find_contacts(self.world, 5e-4) if held in (c.body_a, c.body_b)]
            finally:
                hb.kinematic = True
            known = {(c.body_a, c.body_b) for c in contacts}
            contacts += [c for c in extra if (c.body_a, c.body_b) not in known]
        for c in contacts:
            if held is not None and held in (c.body_a, c.body_b) and not (c.body_a in self.finger_ids or c.body_b in self.finger_ids):
                # reaction on the held stone from its support, transmitted to the wrist. The held stone
                # moves with the tool as a kinematic body, which exchanges no impulse with a static
                # support, so the sensor force comes from the compliance of the fingers and the wrist:
                # a stiff spring on the penetration depth (the larger of this and the impulse force)
                n = c.normal if c.body_b == held else -c.normal
                f_imp = (c.force() / dt) * (1.0 if c.body_b == held else -1.0)
                f_spring = self.sensor_stiffness * max(c.depth, 0.0) * n
                f_on = f_spring if np.linalg.norm(f_spring) > np.linalg.norm(f_imp) else f_imp
                f_tot += f_on
                tau_tot += np.cross(c.point - T.t, f_on)
                continue
            if c.body_a in self.finger_ids or c.body_b in self.finger_ids:
                f = c.force() / dt  # impulse to force
                if c.body_b in self.finger_ids:
                    f_on_finger = f
                else:
                    f_on_finger = -f
                f_tot += f_on_finger
                tau_tot += np.cross(c.point - T.t, f_on_finger)
        return np.concatenate([f_tot, tau_tot])

    def collision_checker(self, obstacles, inflate: float = 0.01, with_held: bool = False):
        """Joint space collision function (link capsules against inflated obstacles and the ground).
        With ``with_held`` the hull vertices of the held stone, in the tool frame, are checked too."""
        from ..planning.rrt import make_arm_collision_fn

        held_pts = None
        if with_held and self.attached is not None:
            bid, T_tcp_body, _ = self.attached
            hull = self.world.bodies[bid].shape.mesh.convex_hull()
            held_pts = T_tcp_body.apply(hull.vertices)
        return make_arm_collision_fn(self.model, obstacles, ground_z=0.0, inflate=inflate, attached_points=held_pts)

    def remove(self):
        self.detach()
        # remove from the highest id down: each removal shifts the ids above it
        for bid in sorted(self.link_ids + self.finger_ids, reverse=True):
            self.world.remove_body(bid)
        self.link_ids, self.finger_ids = [], []
