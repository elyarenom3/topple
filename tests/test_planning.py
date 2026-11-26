"""Planning and arm model tests (Section 9, Section 6.6)."""

import numpy as np
import pytest

from topple.core.mesh import box, icosphere
from topple.core.transforms import SE3, quat_conj, quat_from_axis_angle, quat_mul, quat_to_rotvec
from topple.physics.arm import ArmModel
from topple.planning import (
    Grasp,
    PlacementPhase,
    RRTConnect,
    RRTStar,
    antipodal_grasps,
    chance_ok,
    cma_es,
    ferrari_canny,
    gradient_ascent,
    ik_dls,
    ik_sdls,
    impedance_force,
    make_arm_collision_fn,
    nelder_mead,
    objective,
    path_collision_free,
    pose_error,
    rank_grasps,
    resting_orientations,
    settle_pose,
    shortcut,
    stable_facets,
    time_parameterise,
)
from topple.planning.controller import PhaseMachine
from topple.planning.grasp import primitive_wrenches
from topple.planning.planner import Planner, PlacedStone, TowerState, build_static_tower


@pytest.fixture(scope="module")
def arm():
    return ArmModel()


def test_fk_zero_configuration_matches_hand_computation(arm):
    T = arm.fk(np.zeros(7))
    # flange at (0.088, 0, 0.926) pointing down; TCP 0.1034 further along the flange z axis
    assert np.allclose(T.t, [0.088, 0.0, 0.926 - 0.1034], atol=1e-9)
    assert np.allclose(T.R[:, 2], [0.0, 0.0, -1.0], atol=1e-9)
    Tf = arm.flange(np.zeros(7))
    assert np.allclose(Tf.t, [0.088, 0.0, 0.926], atol=1e-9)
    # the standard home configuration reaches the known TCP height
    Th = arm.fk(arm.q_home)
    assert abs(Th.t[2] - 0.487) < 2e-3 and abs(Th.t[0] - 0.307) < 2e-3


def test_jacobian_matches_finite_differences(arm):
    rng = np.random.default_rng(0)
    for _ in range(3):
        q = rng.uniform(arm.q_min, arm.q_max)
        J = arm.jacobian(q)
        Jfd = np.zeros((6, 7))
        eps = 1e-6
        for i in range(7):
            qp, qm = q.copy(), q.copy()
            qp[i] += eps
            qm[i] -= eps
            Tp, Tm = arm.fk(qp), arm.fk(qm)
            Jfd[:3, i] = (Tp.t - Tm.t) / (2 * eps)
            Jfd[3:, i] = quat_to_rotvec(quat_mul(Tp.q, quat_conj(Tm.q))) / (2 * eps)
        assert np.max(np.abs(J - Jfd)) < 1e-6


def test_ik_reaches_reachable_target(arm):
    rng = np.random.default_rng(1)
    reached = 0
    for _ in range(4):
        q_goal = np.clip(arm.q_home + rng.uniform(-0.4, 0.4, size=7), arm.q_min, arm.q_max)
        target = arm.fk(q_goal)
        q, err = ik_dls(arm, target, arm.q_home)
        if np.linalg.norm(err[:3]) < 1e-3 and np.degrees(np.linalg.norm(err[3:])) < 0.1:
            reached += 1
        q2, err2 = ik_sdls(arm, target, arm.q_home)
        assert np.linalg.norm(err2[:3]) < 5e-3
    assert reached >= 3
    # joint limits are respected
    assert np.all(q >= arm.q_min - 1e-12) and np.all(q <= arm.q_max + 1e-12)


def test_dynamics_consistency(arm):
    rng = np.random.default_rng(2)
    q = arm.q_home + rng.uniform(-0.3, 0.3, size=7)
    M = arm.mass_matrix(q)
    assert np.allclose(M, M.T, atol=1e-10)
    assert np.linalg.eigvalsh(M).min() > 0
    dq = rng.normal(size=7)
    ddq = rng.normal(size=7)
    tau = arm.inverse_dynamics(q, dq, ddq)
    assert np.allclose(arm.forward_dynamics(q, dq, tau), ddq, atol=1e-8)
    # at rest the gravity torque holds the arm: zero acceleration under the gravity torque
    tau_g = arm.gravity_torque(q)
    assert np.allclose(arm.forward_dynamics(q, np.zeros(7), tau_g), 0.0, atol=1e-8)
    assert np.all(np.abs(tau_g) < arm.tau_max)


def test_rrt_connect_and_shortcut_around_obstacle(arm):
    obstacle = box((0.3, 0.04, 0.6), center=(0.45, 0.0, 0.3))
    collides = make_arm_collision_fn(arm, [obstacle], ground_z=0.0, inflate=0.01)
    start_T = SE3(arm.fk(arm.q_home).q, [0.45, 0.25, 0.25])
    goal_T = SE3(arm.fk(arm.q_home).q, [0.45, -0.25, 0.25])
    q0, e0 = ik_dls(arm, start_T, arm.q_home)
    q1, e1 = ik_dls(arm, goal_T, arm.q_home)
    assert np.linalg.norm(e0[:3]) < 2e-3 and np.linalg.norm(e1[:3]) < 2e-3
    assert not collides(q0) and not collides(q1)
    # the straight joint space line passes through the obstacle
    rng = np.random.default_rng(0)
    planner = RRTConnect(lambda r: r.uniform(arm.q_min, arm.q_max), collides, arm.q_min, arm.q_max, step=0.25, max_iters=3000, rng=rng)
    path = planner.plan(q0, q1)
    assert path is not None
    assert np.allclose(path[0], q0) and np.allclose(path[-1], q1)
    assert path_collision_free(path, collides, 0.05)
    short = shortcut(path, collides, rng, iters=60)
    assert len(short) <= len(path)
    assert path_collision_free(short, collides, 0.05)
    times, q_of_t = time_parameterise(short, arm.dq_max)
    assert times[-1] > 0 and np.allclose(q_of_t(0.0), q0) and np.allclose(q_of_t(times[-1]), q1)
    # velocity limit respected on a fine sampling
    ts = np.linspace(0, times[-1], 400)
    Q = np.array([q_of_t(t) for t in ts])
    v = np.abs(np.diff(Q, axis=0)) / np.diff(ts)[:, None]
    assert np.all(v <= 1.5 * arm.dq_max + 1e-9)


def test_rrt_star_finds_a_path(arm):
    collides = make_arm_collision_fn(arm, [], ground_z=-1.0)
    rng = np.random.default_rng(1)
    q0 = arm.q_home
    q1 = np.clip(arm.q_home + np.array([0.6, 0.2, -0.3, 0.2, 0.1, -0.2, 0.3]), arm.q_min, arm.q_max)
    planner = RRTStar(lambda r: r.uniform(arm.q_min, arm.q_max), collides, arm.q_min, arm.q_max, step=0.3, max_iters=400, rng=rng, goal_bias=0.3, goal_tol=0.3)
    path = planner.plan(q0, q1)
    assert path is not None and np.allclose(path[-1], q1)


def _smooth_f(x):
    c = np.array([0.3, -0.2, 0.5])
    return -np.sum((x - c) ** 2) + 2.0


def test_optimisers_find_the_maximum():
    x, fx, _ = cma_es(_smooth_f, np.zeros(3), sigma0=0.5, iters=150, rng=np.random.default_rng(0))
    assert np.allclose(x, [0.3, -0.2, 0.5], atol=1e-3) and abs(fx - 2.0) < 1e-5
    x, fx, _ = nelder_mead(_smooth_f, np.zeros(3), step=0.2, iters=400)
    assert np.allclose(x, [0.3, -0.2, 0.5], atol=1e-3)
    x, fx, _ = gradient_ascent(_smooth_f, lambda x: -2 * (x - np.array([0.3, -0.2, 0.5])), np.zeros(3), step=0.3, iters=200)
    assert np.allclose(x, [0.3, -0.2, 0.5], atol=1e-3)


def test_antipodal_grasps_on_a_box():
    b = box((0.06, 0.04, 0.03))
    grasps = antipodal_grasps(b, mu_g=0.3, n_pairs=400, rng=np.random.default_rng(0), max_width=0.08)
    assert len(grasps) > 10
    for g in grasps:
        # opposite faces: normals anti parallel and the width equals an extent of the box
        assert np.dot(g.n1, g.n2) < -0.99
        assert min(abs(g.width - w) for w in (0.06, 0.04, 0.03)) < 1e-6
        assert abs(abs(np.dot(g.axis, g.n1)) - 1.0) < 1e-6
    ranked = rank_grasps(grasps, np.zeros(3))
    assert ranked[0].score >= ranked[-1].score
    assert all(g.quality > 0 for g in ranked[:5])


def test_ferrari_canny_radius():
    # symmetric two finger grasp on a sphere: positive radius with soft finger torsion
    pts = [np.array([0.03, 0, 0]), np.array([-0.03, 0, 0])]
    nrm = [np.array([1.0, 0, 0]), np.array([-1.0, 0, 0])]
    W = primitive_wrenches(pts, nrm, mu=0.8, cone_sides=8, torsion=0.01)
    assert ferrari_canny(W) > 0
    # one point grasp cannot resist all wrenches: zero radius
    W1 = primitive_wrenches(pts[:1], nrm[:1], mu=0.8, cone_sides=8, torsion=0.01)
    assert ferrari_canny(W1) == 0.0
    # without torsion a hard finger two point grasp cannot resist the torque about the axis
    W2 = primitive_wrenches(pts, nrm, mu=0.8, cone_sides=8, torsion=0.0)
    assert ferrari_canny(W2) == 0.0


def test_objective_and_chance_constraint():
    m = np.array([10.0, 12.0, 8.0, 10.0])
    J = objective(m, dh=0.05, kappa=1.0, lambda_h=2.0)
    assert J == pytest.approx(10.0 - np.std(m, ddof=1) + 0.1)
    assert objective(m, 0.0, kappa=0.0) == pytest.approx(10.0)
    assert chance_ok(0.96, 0.05) and not chance_ok(0.9, 0.05)
    assert objective([], 0.0) == -np.inf


def test_resting_orientations_of_a_box():
    b = box((0.1, 0.06, 0.04))
    facets = stable_facets(b)
    assert len(facets) == 6
    ro = resting_orientations(b, n_drops=400, rng=np.random.default_rng(0))
    assert len(ro) == 6
    probs = np.array([p for _, p in ro])
    assert probs.sum() == pytest.approx(1.0)
    assert probs[0] > probs[-1]  # the largest face is the most likely resting face
    # each orientation puts one face normal along -z
    normals = np.array([f["normal"] for f in facets])
    for q, _ in ro:
        from topple.core.transforms import quat_rotate

        down = quat_rotate(quat_conj(q), np.array([0.0, 0.0, -1.0]))
        assert np.max(normals @ down) > 0.999
    # a sphere like shape has no uniquely stable facets below the margin but never errors
    s = icosphere(2, 0.03)
    assert isinstance(resting_orientations(s, n_drops=20), list)


def test_settle_pose_tips_a_block_flat():
    b = box((0.1, 0.06, 0.04))
    state = TowerState()
    pose = SE3(quat_from_axis_angle([1, 0, 0], 0.4), [0, 0, 0.1])
    res = settle_pose(b, pose, state.support_meshes(), rng=np.random.default_rng(0))
    assert res.stable
    V = b.transformed(res.pose).vertices
    assert abs(V[:, 2].min()) < 5e-4
    assert len(res.contact_points) >= 3
    # the block lies flat: all four bottom corners near z = 0
    assert np.sum(V[:, 2] < 2e-3) == 4


def test_impedance_law_and_phase_machine():
    F = impedance_force([0, 0, 0.1], [0, 0, 0.09], [0, 0, 0], [0, 0, 0.02], K=400.0, D=40.0)
    assert np.allclose(F[:3], [0, 0, 400 * 0.01 - 40 * 0.02])
    pm = PhaseMachine(approach_height=0.03, descend_speed=0.01, contact_force_threshold=0.5, release_seconds=0.5)
    assert pm.phase == PlacementPhase.APPROACH
    pm.advance(0.001, at_approach=False, wrist_force=0.0, descended=0.0, height_above_target=0.03)
    assert pm.phase == PlacementPhase.APPROACH
    pm.advance(0.001, at_approach=True, wrist_force=0.0, descended=0.0, height_above_target=0.03)
    assert pm.phase == PlacementPhase.DESCEND
    pm.advance(0.001, True, 0.1, 0.01, 0.02)
    assert pm.phase == PlacementPhase.DESCEND
    pm.advance(0.001, True, 1.0, 0.03, 0.0)
    assert pm.phase == PlacementPhase.CONTACT
    for _ in range(250):
        pm.advance(0.001, True, 1.0, 0.03, 0.0)
    assert pm.phase == PlacementPhase.RELEASE
    for _ in range(600):
        pm.advance(0.001, True, 0.0, 0.03, 0.0)
    assert pm.phase == PlacementPhase.RETRACT
    pm.advance(0.001, True, 0.0, 0.03, 0.06)
    assert pm.phase == PlacementPhase.DONE


def test_planner_chooses_a_feasible_placement():
    from topple.statics import tilt_margin

    base = box((0.12, 0.10, 0.04), center=(0, 0, 0.02))
    state = TowerState([PlacedStone("base", base, 1.2, base.centroid(), 0.6)])
    stone = icosphere(2, 0.03).scaled([1.2, 1.0, 0.6])

    class Est:
        mean_mesh = stone
        mass = 0.25
        mu = 0.6

    planner = Planner({"planner": {"n_candidates": 3, "refine_iters": 3, "kappa": 1.0}}, rng=np.random.default_rng(0))
    margin_fn = lambda tower, cand: np.array([tilt_margin(tower, n_azimuth=8)[0] + d for d in (-0.5, 0.0, 0.5)])
    decision = planner.choose(state, [Est()], margin_fn)
    assert decision is not None and decision.stone_index == 0
    assert decision.p_hat > 0.5 and decision.J > 0
    assert decision.grasp is not None
    assert len(decision.candidates) == 3
    # the chosen stone rests on the base: its lowest point is near the base top
    V = stone.transformed(decision.pose).vertices
    assert abs(V[:, 2].min() - 0.04) < 2e-3
    tower = build_static_tower(state, stone.transformed(decision.pose), 0.25, None, 0.6)
    assert tower.n_contacts >= 3


def test_planner_chance_constraint_uses_the_recalibrated_probability():
    from topple.statics import tilt_margin

    base = box((0.12, 0.10, 0.04), center=(0, 0, 0.02))
    state = TowerState([PlacedStone("base", base, 1.2, base.centroid(), 0.6)])
    stone = icosphere(2, 0.03).scaled([1.2, 1.0, 0.6])

    class Est:
        mean_mesh = stone
        mass = 0.25
        mu = 0.6

    class FixedRecalibrator:
        seen = []

        def predict(self, records):
            self.seen.extend(records)
            return np.full(len(records), 0.2)

    planner = Planner({"planner": {"n_candidates": 3, "refine_iters": 3, "kappa": 1.0, "delta": 0.05}}, rng=np.random.default_rng(0))
    planner.recalibrator = FixedRecalibrator()
    planner.recal_context = {"classes": ["flat_river"], "settings": {"delta": 0.05}}
    margin_fn = lambda tower, cand: np.array([tilt_margin(tower, n_azimuth=8)[0] + d for d in (-0.5, 0.0, 0.5)])
    decision = planner.choose(state, [Est()], margin_fn)
    assert decision is not None
    assert decision.p_hat == 0.2 and decision.p_raw > 0.5
    assert not decision.feasible  # 0.2 misses p >= 0.95 although the raw probability passes
    rec = FixedRecalibrator.seen[0]
    assert rec["placement"] == 1 and rec["meta"]["stone_class"] == "flat_river" and rec["meta"]["n_contacts"] >= 1
