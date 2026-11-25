"""Placement execution in the true world: kinematic lowering and the arm pipeline.

``lower_and_release`` is the oracle placement used by the calibration experiments:
the stone is held 3 cm above its target, lowered at 1 cm/s as a kinematic body until a
contact impulse is seen, then released as a dynamic body and left to settle.

``arm_place`` runs the full chain of Section 9: grasp selection, inverse kinematics to a
pre grasp and to the approach pose, RRT-Connect in joint space with the tower inflated,
and the Cartesian impedance placement controller. The stone is carried by friction between
the two fingers in the contact solver.
"""

from __future__ import annotations

import numpy as np

from ..core.config import get
from ..core.transforms import SE3, quat_normalize


def _min_separation(world, body, margin: float = 0.01) -> float:
    """Smallest separation between ``body`` and any other body (inf beyond ``margin``)."""
    from ..physics.collision import find_contacts

    was_k, was_s = body.kinematic, body.static
    body.kinematic = False
    body.static = False
    try:
        cs = find_contacts(world, margin)
    finally:
        body.kinematic, body.static = was_k, was_s
    seps = [-c.depth for c in cs if body.id in (c.body_a, c.body_b)]
    return min(seps) if seps else np.inf


def lower_and_release(world, body, T_target: SE3, cfg: dict, settle_s: float = 0.6, touch_tol: float = 2e-4, max_below: float = 0.05) -> bool:
    """Oracle placement: hold the stone 3 cm above its target with its target orientation, lower
    it along -z until the first contact, release it as a dynamic body with zero velocity and let
    it settle.

    The descent at 1 cm/s of Section 9.6 is quasi static, so it is realised by conservative
    advancement: the stone moves down by its current separation from the other bodies until the
    separation is below ``touch_tol``. Returns False when no contact is reached within
    ``max_below`` under the target height."""
    approach = float(get(cfg, "arm.approach_height", 0.03))
    margin = 0.01
    z0 = float(T_target.t[2]) + approach
    z = z0
    body.v[:] = 0
    body.w[:] = 0
    touched = False
    for _ in range(200):
        body.pose = SE3(T_target.q, np.array([T_target.t[0], T_target.t[1], z]))
        sep = _min_separation(world, body, margin)
        if sep <= touch_tol:
            touched = sep > -5e-3
            break
        step = 0.9 * margin if not np.isfinite(sep) else max(sep - 0.5 * touch_tol, 0.25 * touch_tol)
        z -= step
        if z < float(T_target.t[2]) - max_below:
            break
    body.kinematic = False
    world._warm = {}
    info = world.damped_settle(seconds=settle_s, damping=float(get(cfg, "planner.settle_damping", 60.0)))
    body.meta["release_check"] = info
    return touched


def _move_joints(world, arm, q_target, dq_max, dt, collision_fn=None):
    """Kinematic joint space motion to q_target under the velocity limits."""
    from .rrt import time_parameterise

    times, q_of_t = time_parameterise([arm.q.copy(), np.asarray(q_target, dtype=float)], dq_max)
    t = 0.0
    while t < times[-1]:
        q_next = q_of_t(t + dt)
        arm.set_joint_state(q_next, (q_next - arm.q) / dt, arm.opening, 0.0)
        world.step(1)
        t += dt


def _move_cartesian(world, arm, T_goal: SE3, speed: float, dt: float, max_s: float = 6.0, tol: float = 5e-4):
    """Straight line tool motion to ``T_goal`` (position at ``speed`` m/s, orientation held at the
    goal's) by damped least squares velocity IK, with the world stepped at every control step."""
    from .kinematics import pose_error

    model = arm.model
    T0 = model.fk(arm.q)
    dist = float(np.linalg.norm(T_goal.t - T0.t))
    n = max(1, int(np.ceil(dist / max(speed * dt, 1e-9))))
    for k in range(int(max_s / dt)):
        u = min(1.0, (k + 1) / n)
        T_cmd = SE3(T_goal.q, T0.t + u * (T_goal.t - T0.t))
        q = arm.q
        e = pose_error(T_cmd, model.fk(q)) * (0.5 / dt)
        J = model.jacobian(q)
        dq = J.T @ np.linalg.solve(J @ J.T + 4e-4 * np.eye(6), e)
        scale = float(np.max(np.abs(dq) / model.dq_max))
        if scale > 1.0:
            dq = dq / scale
        q_new = np.clip(q + dq * dt, model.q_min, model.q_max)
        arm.set_joint_state(q_new, (q_new - q) / dt, arm.opening, 0.0)
        world.step(1)
        if u >= 1.0 and np.linalg.norm(model.fk(arm.q).t - T_goal.t) < tol:
            break


def _stone_tcp_offset(body, arm) -> np.ndarray:
    T = arm.model.fk(arm.q)
    return T.inverse().apply(body.x)


def arm_place(world, body, T_true: SE3, cfg: dict, rng, grasp=None, T_est_body: SE3 | None = None, recorder=None) -> bool:
    """Full arm placement (Section 9): grasp, pick, carry, place with the impedance controller.

    ``grasp`` is a ``Grasp`` in the stone's estimated frame; ``T_est_body`` maps body frame points
    to estimated frame points (identity when the estimate frame is the body frame). The stone is
    carried by friction between the fingers (``arm.grasp_mode: friction``) or by a kinematic weld
    (``weld``); with friction, slip beyond 5 mm during the carry switches to the weld and the
    event is logged in ``body.meta``. Any failing stage falls back to kinematic lowering and the
    reason is stored in ``body.meta['arm_fallback']``.
    """
    from ..physics.arm import ArmModel
    from ..physics.arm_world import ArmInWorld
    from .controller import PlacementController
    from .grasp import antipodal_grasps, rank_grasps
    from .kinematics import ik_dls
    from .rrt import RRTConnect, shortcut, time_parameterise

    arm_cfg = dict(cfg.get("arm", {}))
    T_est_body = SE3() if T_est_body is None else T_est_body
    model = ArmModel(arm_cfg.get("model", "panda7"), base=SE3(t=np.asarray(arm_cfg.get("base", [-0.45, 0, 0]), dtype=float)))
    model.finger_max = float(arm_cfg.get("finger_max", model.finger_max))
    arm = ArmInWorld(world, model, finger_mu=float(get(cfg, "friction.finger_mu", 1.0)))
    dt = world.dt
    grasp_mode = str(arm_cfg.get("grasp_mode", "friction"))
    body.meta["arm_fallback"] = None
    try:
        mesh_est = body.shape.mesh.transformed(T_est_body)  # stone in the estimate frame
        grasps = antipodal_grasps(mesh_est, mu_g=float(get(cfg, "friction.finger_mu", 1.0)), n_pairs=300, rng=rng, max_width=model.finger_max - 0.004)
        ranked = rank_grasps(grasps, mesh_est.centroid(), None, None)
        if grasp is not None:
            ranked = [grasp] + ranked
        if not ranked:
            raise RuntimeError("no antipodal grasp")
        T_world_est_pick = body.pose @ T_est_body.inverse()
        T_world_est_place = T_true @ T_est_body.inverse()
        approach_est = T_world_est_pick.R.T @ np.array([0.0, 0.0, -1.0])
        others = [b.world_mesh().convex_hull() for b in world.bodies if b.id != body.id and not b.static and not b.kinematic]
        coll = arm.collision_checker(others, inflate=float(get(cfg, "planner.inflate_m", 0.01)))
        chosen = None
        reasons = []
        for g in ranked[:12]:
            # the grasp axis must be close to horizontal at the pick and at the place pose
            if abs(float(T_world_est_pick.apply_vec(g.axis)[2])) > 0.5 or abs(float(T_world_est_place.apply_vec(g.axis)[2])) > 0.5:
                reasons.append("axis")
                continue
            T_body_tcp = T_est_body.inverse() @ g.pose(approach_est)
            T_pick = body.pose @ T_body_tcp
            T_place = T_true @ T_body_tcp
            found = None
            # retry with more clearance above the pick and the place when a pose collides or is out of reach
            for pre_h, app_h in ((0.08, float(arm_cfg.get("approach_height", 0.03))), (0.12, 0.05), (0.12, 0.08)):
                T_pre = SE3(T_pick.q, T_pick.t + np.array([0.0, 0.0, pre_h]))
                T_approach = SE3(T_place.q, T_place.t + np.array([0.0, 0.0, app_h]))
                q_pre, e0 = ik_dls(model, T_pre, model.q_home)
                q_pick, e1 = ik_dls(model, T_pick, q_pre)
                q_app, e2 = ik_dls(model, T_approach, q_pre)
                if any(np.linalg.norm(e[:3]) > 5e-3 or np.linalg.norm(e[3:]) > 5e-2 for e in (e0, e1, e2)):
                    reasons.append("ik")
                    continue
                if coll(q_pre) or coll(q_app):
                    reasons.append("collision")
                    continue
                found = (q_pre, q_pick, q_app, app_h)
                break
            if found is None:
                continue
            q_pre, q_pick, q_app, app_h = found
            arm_cfg["approach_height"] = app_h
            chosen = (g, T_body_tcp, q_pre, q_pick, q_app)
            break
        if chosen is None:
            raise RuntimeError("no feasible grasp: " + ",".join(reasons))
        grasp, T_body_tcp, q_pre, q_pick, q_app = chosen
        if arm_cfg.get("debug"):
            Ta = model.fk(q_app)
            print("debug grasp axis world", np.round(T_world_est_pick.apply_vec(grasp.axis), 3), "tcp z at approach", np.round(Ta.R[:, 2], 3), "tcp y", np.round(Ta.R[:, 1], 3), "tcp pos", np.round(Ta.t, 3))
        T_place = T_true @ T_body_tcp
        body.meta["grasp"] = {"centre": grasp.centre.tolist(), "axis": grasp.axis.tolist(), "width": float(grasp.width)}
        sampler = lambda r: r.uniform(model.q_min, model.q_max)
        # 1. home to pre grasp
        planner = RRTConnect(sampler, coll, model.q_min, model.q_max, step=0.15, rng=rng)
        path = planner.plan(arm.q.copy(), q_pre)
        if path is None:
            raise RuntimeError("no path to the pre grasp pose")
        path = shortcut(path, coll, rng)
        times, q_of_t = time_parameterise(path, model.dq_max)
        t = 0.0
        while t < times[-1]:
            q_next = q_of_t(t + dt)
            arm.set_joint_state(q_next, (q_next - arm.q) / dt, arm.opening, 0.0)
            world.step(1)
            t += dt
        # 2. straight vertical descent to the pick pose, then close the fingers
        pick_pose_before = body.pose.copy()
        _move_cartesian(world, arm, body.pose @ T_body_tcp, 0.05, dt)
        moved = float(np.linalg.norm(body.x - pick_pose_before.t))
        if moved > 0.005:
            raise RuntimeError(f"stone pushed by {1000 * moved:.0f} mm during the pick descent")
        squeeze = float(grasp.width) + 0.0005  # close to the stone surface (the stone is then held by attachment)
        n_close = int(0.5 / dt)
        for k in range(n_close):
            arm.set_joint_state(arm.q, np.zeros_like(arm.q), arm.opening + (squeeze - arm.opening) / (n_close - k), (squeeze - arm.opening) / ((n_close - k) * dt))
            world.step(1)
        # the closed fingers hold the stone; it is carried attached to the tool frame
        arm.attach(body.id, float(grasp.width))
        if arm_cfg.get("debug"):
            Tp = T_body_tcp.inverse()
            Ta = arm.attached[1]
            print("debug attach: planned tcp->body t", np.round(Tp.t, 4), "actual", np.round(Ta.t, 4), "tcp at attach", np.round(model.fk(arm.q).t, 4), "target pick tcp", np.round((body.pose @ T_body_tcp).t, 4), "q err", np.round(np.abs(arm.q - q_pick).max(), 4))
        # the stone can shift while the fingers close: re plan the place and approach poses from the
        # held offset (the in hand pose, observed by the fixed camera during in hand scanning)
        T_tcp_body = arm.attached[1]
        T_place = T_true @ T_tcp_body.inverse()
        T_approach = SE3(T_place.q, T_place.t + np.array([0.0, 0.0, float(arm_cfg.get("approach_height", 0.03))]))
        q_app2, e3 = ik_dls(model, T_approach, q_app)
        if np.linalg.norm(e3[:3]) < 5e-3 and np.linalg.norm(e3[3:]) < 5e-2:
            q_app = q_app2
        else:
            body.meta["arm_replan_failed"] = True
        # 3. straight vertical lift, then carry to the approach pose
        T_lift = model.fk(arm.q)
        _move_cartesian(world, arm, SE3(T_lift.q, T_lift.t + np.array([0.0, 0.0, 0.08])), 0.05, dt)
        obstacles = [b.world_mesh().convex_hull() for b in world.bodies if b.id != body.id and not b.static and not b.kinematic]
        coll2 = arm.collision_checker(obstacles, inflate=float(get(cfg, "planner.inflate_m", 0.01)), with_held=True)
        planner2 = RRTConnect(sampler, coll2, model.q_min, model.q_max, step=0.15, rng=rng)
        path2 = planner2.plan(arm.q.copy(), q_app)
        if path2 is None:
            raise RuntimeError(f"no path to the approach pose (start in collision {bool(coll2(arm.q))}, goal in collision {bool(coll2(q_app))}, iterations {planner2.iterations})")
        path2 = shortcut(path2, coll2, rng)
        times2, q_of_t2 = time_parameterise(path2, model.dq_max * 0.5)
        trajectory = (times2, q_of_t2)
        # 4. placement controller: approach, descend at 1 cm/s, contact, release, retract
        ctrl = PlacementController(world, arm, arm_cfg)
        res = ctrl.execute(trajectory, body, T_place, recorder=recorder)
        body.meta["wrist_log"] = res.wrist_log
        body.meta["phases"] = res.phases
        ok = bool(res.contact_detected)
        arm.detach()
        if not ok:
            raise RuntimeError("no contact detected during descent")
        body.meta["release_check"] = world.damped_settle(seconds=0.6, damping=float(get(cfg, "planner.settle_damping", 60.0)))
    except Exception as exc:  # any stage failing falls back to the kinematic placement
        body.meta["arm_fallback"] = repr(exc)
        body.kinematic = False
        ok = lower_and_release(world, body, T_true, cfg)
    finally:
        arm.remove()
    return ok
