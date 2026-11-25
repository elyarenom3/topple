"""Placement controller (Section 9.6).

The arm moves to 3 cm above the target, descends at 1 cm/s under Cartesian impedance control
(a stiffness term on the position error and a damping term on the velocity), detects contact
from the simulated wrist force torque sensor, drops the stiffness so that the stone settles,
opens the fingers slowly and retracts. The controller is written against the ``World`` and
``ArmInWorld`` contracts; its pure parts (impedance law, phase machine) are unit tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..core.transforms import SE3, quat_conj, quat_mul, quat_to_rotvec


class PlacementPhase(Enum):
    APPROACH = 0
    DESCEND = 1
    CONTACT = 2
    RELEASE = 3
    RETRACT = 4
    DONE = 5


@dataclass
class PlacementResult:
    contact_detected: bool
    released_pose: SE3 | None
    wrist_log: list = field(default_factory=list)
    phases: list = field(default_factory=list)
    success: bool = False
    contact_time: float = -1.0


def impedance_force(x_d, x, v_d, v, K, D, q_d=None, q=None, K_rot: float = 0.0, D_rot: float = 0.0, w=None) -> np.ndarray:
    """Cartesian impedance wrench: F = K (x_d - x) + D (v_d - v), plus an orientation term
    K_rot * rotvec(q_d q^-1) - D_rot * w when quaternions are given. Returns (6,) [force; torque]."""
    x_d, x, v_d, v = (np.asarray(a, dtype=np.float64) for a in (x_d, x, v_d, v))
    F = np.zeros(6)
    F[:3] = K * (x_d - x) + D * (v_d - v)
    if q_d is not None and q is not None:
        F[3:] = K_rot * quat_to_rotvec(quat_mul(q_d, quat_conj(q))) - D_rot * (np.zeros(3) if w is None else np.asarray(w))
    return F


class PhaseMachine:
    """Pure transition logic of the placement controller."""

    def __init__(self, approach_height: float = 0.03, descend_speed: float = 0.01, contact_force_threshold: float = 0.5, release_seconds: float = 0.5, retract_height: float = 0.05):
        self.approach_height = approach_height
        self.descend_speed = descend_speed
        self.threshold = contact_force_threshold
        self.release_seconds = release_seconds
        self.retract_height = retract_height
        self.phase = PlacementPhase.APPROACH
        self.t_phase = 0.0
        self.contact_time = -1.0

    def advance(self, dt: float, at_approach: bool, wrist_force: float, descended: float, height_above_target: float) -> PlacementPhase:
        """Advance one control step. ``descended`` is the distance travelled downward since the
        start of DESCEND and ``height_above_target`` the current TCP height above the target."""
        self.t_phase += dt
        p = self.phase
        if p == PlacementPhase.APPROACH:
            # start the descent at the approach pose, or after 3 s of settling towards it
            if at_approach or self.t_phase >= 3.0:
                self._go(PlacementPhase.DESCEND)
        elif p == PlacementPhase.DESCEND:
            if wrist_force > self.threshold:
                self.contact_time = self.t_phase
                self._go(PlacementPhase.CONTACT)
            elif height_above_target < -0.01:
                # descended past the target without contact: treat as contact to avoid driving into the tower
                self._go(PlacementPhase.CONTACT)
        elif p == PlacementPhase.CONTACT:
            if self.t_phase >= 0.2:
                self._go(PlacementPhase.RELEASE)
        elif p == PlacementPhase.RELEASE:
            if self.t_phase >= self.release_seconds:
                self._go(PlacementPhase.RETRACT)
        elif p == PlacementPhase.RETRACT:
            if height_above_target >= self.retract_height:
                self._go(PlacementPhase.DONE)
        return self.phase

    def _go(self, phase: PlacementPhase):
        self.phase = phase
        self.t_phase = 0.0


class PlacementController:
    """Executes a placement in a physics ``World`` through an ``ArmInWorld`` driver."""

    def __init__(self, world, arm_in_world, cfg_arm: dict | None = None):
        cfg = cfg_arm or {}
        self.world = world
        self.aiw = arm_in_world
        self.arm = arm_in_world.model if hasattr(arm_in_world, "model") else arm_in_world.arm
        self.approach_height = float(cfg.get("approach_height", 0.03))
        self.descend_speed = float(cfg.get("descend_speed", 0.01))
        self.K = float(cfg.get("impedance_stiffness", 400.0))
        self.D = float(cfg.get("impedance_damping", 40.0))
        self.K_soft = float(cfg.get("impedance_stiffness_soft", 40.0))
        self.threshold = float(cfg.get("contact_force_threshold", 0.5))
        self.release_seconds = float(cfg.get("release_seconds", 0.5))
        self.K_rot = float(cfg.get("rot_stiffness", 20.0))
        self.D_rot = float(cfg.get("rot_damping", 2.0))
        self.max_seconds = float(cfg.get("max_seconds", 20.0))
        # "admittance" (position controlled arm, default) or "torque" (joint torques from J^T F)
        self.mode = str(cfg.get("control_mode", "admittance"))

    def _track(self, T_cmd: SE3, dt: float, gain: float = 0.5, lam: float = 0.02):
        """One step of differential IK towards ``T_cmd`` within the joint velocity limits."""
        from .kinematics import pose_error

        q = self.aiw.q
        e = pose_error(T_cmd, self.arm.fk(q)) * (gain / dt)
        J = self.arm.jacobian(q)
        dq = J.T @ np.linalg.solve(J @ J.T + lam * lam * np.eye(6), e)
        lim = np.asarray(self.arm.dq_max, dtype=float)
        scale = float(np.max(np.abs(dq) / lim)) if np.any(lim > 0) else 1.0
        if scale > 1.0:
            dq = dq / scale
        q_new = np.clip(q + dq * dt, self.arm.q_min, self.arm.q_max)
        self.aiw.set_joint_state(q_new, (q_new - q) / dt, self.aiw.opening)

    def _tcp_state(self):
        q, dq = self.aiw.q, self.aiw.dq
        T = self.arm.fk(q)
        J = self.arm.jacobian(q)
        tw = J @ dq
        return T, tw[:3], tw[3:], J

    def execute(self, trajectory, stone_body, target_pose: SE3, recorder=None) -> PlacementResult:
        """``trajectory`` is (times, q_of_t) reaching the approach pose; ``target_pose`` is the
        TCP pose at which the stone rests on the tower."""
        dt = self.world.dt
        times, q_of_t = trajectory
        pm = PhaseMachine(self.approach_height, self.descend_speed, self.threshold, self.release_seconds)
        result = PlacementResult(False, None)
        t = 0.0
        # joint space tracking of the trajectory to the approach pose (kinematic)
        while t < times[-1]:
            q_next = q_of_t(t + dt)
            dq = (q_next - self.aiw.q) / dt
            self.aiw.set_joint_state(q_next, dq, self.aiw.opening)
            self.world.step()
            t += dt
        approach_pose = SE3(target_pose.q, target_pose.t + np.array([0.0, 0.0, self.approach_height]))
        x_d = approach_pose.t.copy()
        descended = 0.0
        K = self.K
        opening0 = self.aiw.opening
        f_filt = np.zeros(3)
        adm_off = np.zeros(3)
        steps = 0
        while pm.phase != PlacementPhase.DONE and steps * dt < self.max_seconds:
            T, v, w, J = self._tcp_state()
            wrench = self.aiw.wrist_wrench()
            fmag = float(np.linalg.norm(wrench[:3]))
            result.wrist_log.append((t, fmag))
            height = float(T.t[2] - target_pose.t[2])
            phase = pm.advance(dt, np.linalg.norm(T.t - approach_pose.t) < 5e-3, fmag, descended, height)
            result.phases.append((t, phase.name))
            if phase == PlacementPhase.APPROACH:
                x_d = approach_pose.t
                v_d = np.zeros(3)
            elif phase == PlacementPhase.DESCEND:
                x_d = x_d - np.array([0.0, 0.0, self.descend_speed * dt])
                descended += self.descend_speed * dt
                v_d = np.array([0.0, 0.0, -self.descend_speed])
            elif phase == PlacementPhase.CONTACT:
                # hold the tool where contact was made; the compliance of the release is in the
                # handover spring, which starts ramping down now (Section 9.6)
                v_d = np.zeros(3)
                if not result.contact_detected:
                    result.contact_detected = True
                    result.contact_time = t
                    x_d = T.t.copy()
                    if hasattr(self.aiw, "begin_handover"):
                        self.aiw.begin_handover()
                ramp = 0.2 + self.release_seconds
                if hasattr(self.aiw, "set_handover_scale"):
                    self.aiw.set_handover_scale(1.0 - pm.t_phase / ramp)
            elif phase == PlacementPhase.RELEASE:
                v_d = np.zeros(3)
                frac = min(pm.t_phase / self.release_seconds, 1.0)
                self.aiw.opening = opening0 + frac * (self.arm.finger_max - opening0)
                ramp = 0.2 + self.release_seconds
                if hasattr(self.aiw, "set_handover_scale"):
                    self.aiw.set_handover_scale(1.0 - (0.2 + pm.t_phase) / ramp)
            elif phase == PlacementPhase.RETRACT:
                K = self.K
                if hasattr(self.aiw, "set_handover_scale"):
                    self.aiw.set_handover_scale(0.0)
                x_d = x_d + np.array([0.0, 0.0, 0.05 * dt])
                v_d = np.array([0.0, 0.0, 0.05])
            if self.mode == "admittance":
                # admittance realisation of the Cartesian impedance law on a position controlled arm:
                # the commanded pose is the setpoint shifted by the low pass filtered contact force
                # over the stiffness (offset limited to 5 mm and to 2 cm/s), tracked by damped least
                # squares velocity IK with the orientation held exactly
                f_filt += (dt / 0.02) * (wrench[:3] - f_filt)
                target_off = np.clip(f_filt / max(K, 1e-9), -0.005, 0.005)
                adm_off += np.clip(target_off - adm_off, -0.02 * dt, 0.02 * dt)
                self._track(SE3(target_pose.q, x_d + adm_off), dt)
            else:
                F = impedance_force(x_d, T.t, v_d, v, K, self.D, target_pose.q, T.q, self.K_rot, self.D_rot, w)
                tau = J.T @ F + self.arm.gravity_torque(self.aiw.q, self.world.gravity)
                self.aiw.step_dynamics(tau, dt)
            self.world.step()
            if recorder is not None:
                recorder.append(self.world.state())
            t += dt
            steps += 1
        result.released_pose = SE3(stone_body.q.copy(), stone_body.x.copy())
        result.success = pm.phase == PlacementPhase.DONE
        return result
