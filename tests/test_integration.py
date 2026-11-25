"""Integration tests: hull statics, sample settling, world subsets and settling, the estimate view,
arm attachment, and one short end to end tower."""

import numpy as np
import pytest

from topple.core.config import load_config
from topple.core.mesh import box, icosphere
from topple.core.transforms import SE3, quat_from_axis_angle
from topple.physics.body import RigidBody
from topple.physics.world import World
from topple.statics import margins
from topple.statics.analysis import HullGeom, build_tower, hull_vertical_gap, settle_hull_stone


def _r(mesh, mass, name):
    return {"mesh": mesh, "mass": mass, "com": mesh.centroid(), "mu": 0.6, "name": name}


def test_hull_tower_matches_mesh_tower():
    a = box((0.1, 0.1, 0.05), center=(0, 0, 0.025))
    b = box((0.06, 0.06, 0.1), center=(0.01, 0, 0.1003))
    r = [_r(a, 1.3, "a"), _r(b, 0.9, "b")]
    th = build_tower(r, 5e-4, 8, geometry="hull")
    tm = build_tower(r, 5e-4, 8, geometry="mesh")
    mh = margins.tower_margin(th, kind="tilt", cone_sides=8, n_azimuth=36)
    mm = margins.tower_margin(tm, kind="tilt", cone_sides=8, n_azimuth=36)
    assert abs(mh - mm) < 0.3
    # the upper block slides first: atan(mu cos(pi / 8)) for the inscribed octagon
    assert abs(mh - np.degrees(np.arctan(0.6 * np.cos(np.pi / 8)))) < 0.5


def test_hull_vertical_gap_boxes():
    lower = HullGeom(box((0.1, 0.1, 0.05), center=(0, 0, 0.025)).vertices)
    upper = HullGeom(box((0.05, 0.05, 0.05), center=(0.02, 0.0, 0.0775)).vertices)
    assert abs(hull_vertical_gap(upper, lower) - 0.0025) < 1e-9
    upper2 = HullGeom(box((0.05, 0.05, 0.05), center=(0.02, 0.0, 0.074)).vertices)
    assert abs(hull_vertical_gap(upper2, lower) + 0.001) < 1e-9
    far = HullGeom(box((0.05, 0.05, 0.05), center=(0.3, 0.0, 0.0775)).vertices)
    assert hull_vertical_gap(far, lower) == np.inf


def test_settle_hull_stone_tips_a_tilted_block_onto_its_face():
    m = box((0.06, 0.04, 0.02)).transformed(SE3(quat_from_axis_angle([1, 0, 0], np.radians(4.0)), [0, 0, 0.0125]))
    h = HullGeom(m.vertices)
    h2, com, rot = settle_hull_stone(h, m.centroid(), [None], 5e-4, max_tilt_deg=15.0, step_deg=0.5)
    assert 3.0 <= rot <= 6.0
    assert abs(h2.vertices[:, 2].min()) < 1e-5
    flat = np.sum(h2.vertices[:, 2] < 5e-4)
    assert flat >= 4


def test_world_subset_and_damped_settle_bring_a_stack_to_rest():
    w = World(load_config("quick"))
    w.add_body(RigidBody.ground(0.6))
    for k in range(3):
        b = RigidBody.from_mesh(f"c{k}", box((0.05, 0.05, 0.05)), 2600.0, 0.6, sdf_spacing=None, pose=SE3(t=[0.002 * k, 0, 0.026 + 0.0505 * k]))
        w.add_body(b)
    w.set_pair_friction(1, 2, 0.41)
    info = w.damped_settle(seconds=0.4)
    assert info["still"]
    sub, m = w.subset([0, 1, 2])
    assert len(sub.bodies) == 3 and np.isclose(sub.pair_friction(m[1], m[2]), 0.41)
    ref = sub.state()
    sub.step(300)
    assert np.abs(sub.state().x - ref.x).max() < 2e-4


def test_estimate_view_from_a_synthetic_estimate():
    from topple.perception.estimate_view import EstimateView
    from topple.perception.estimator import StoneEstimate, StoneSample

    rng = np.random.default_rng(0)
    mean = icosphere(2, 0.04)
    samples = []
    for _ in range(8):
        V = mean.vertices * (1 + 0.01 * rng.standard_normal((mean.n_vertices, 1)))
        from topple.core.mesh import TriMesh

        sm = TriMesh(V, mean.faces)
        mp = sm.mass_properties(2600.0)
        samples.append(StoneSample(sm, mp["mass"], mp["com"], mp["inertia_com"], 0.6, SE3(t=rng.normal(size=3) * 1e-3), 2600.0))
    est = StoneEstimate(mean, samples, 2600.0, np.full(8, 0.6), {})
    ev = EstimateView(est, n_modes=0, smooth_iterations=0)
    assert abs(ev.mass - mean.volume() * 2600.0) / ev.mass < 0.05
    assert np.linalg.norm(ev.mean_mesh.centroid()) < 1e-6
    assert ev.pose_trans_std > 0 and ev.log_mu_std < 1e-9


def test_arm_attach_carries_the_stone_with_the_tool():
    from topple.physics.arm import ArmModel
    from topple.physics.arm_world import ArmInWorld

    w = World(load_config("quick"))
    w.add_body(RigidBody.ground(0.6))
    model = ArmModel("panda7", base=SE3(t=np.array([-0.45, 0.0, 0.0])))
    stone = RigidBody.from_mesh("s", box((0.04, 0.04, 0.04)), 2600.0, 0.6, sdf_spacing=None)
    T_tcp = model.fk(model.q_home)
    stone.pose = T_tcp
    w.add_body(stone)
    arm = ArmInWorld(w, model)
    arm.attach(stone.id, 0.04)
    q2 = model.q_home + np.array([0.2, -0.1, 0.0, 0.1, 0.0, 0.1, 0.0])
    arm.set_joint_state(q2, np.zeros(7), 0.04)
    assert np.allclose(w.bodies[stone.id].x, model.fk(q2).t, atol=1e-9)
    arm.set_joint_state(q2, np.zeros(7), 0.06)  # fingers open past the grasp width: released
    assert arm.attached is None and not w.bodies[stone.id].kinematic
    arm.remove()
    assert len(w.bodies) == 2


@pytest.mark.slow
def test_one_oracle_tower_end_to_end():
    from topple.pipeline import Condition, Experiment

    cfg = load_config("quick")
    cfg["experiment"]["max_stones"] = 1
    cfg["experiment"]["probes_per_step"] = 1
    cfg["stones"]["n_pile"] = 4
    cfg["harness"]["tilt_azimuths"] = 2
    exp = Experiment(cfg, Condition(name="t", oracle=True, propagation="none", tests=("tilt", "fixed"), extra={"eval_cones": [4]}), verbose=False)
    recs = exp.run_tower(0)
    assert recs, "no records"
    main = [r for r in recs if not r.meta.get("probe")]
    for r in recs:
        assert np.isfinite(r.m_hat) and 0.0 <= r.p_hat <= 1.0 and r.y in (0.0, 1.0)
    if main:
        assert np.isfinite(main[0].m_true)
        # a single stone on flat ground: exact inputs give a static margin near the measured one
        assert abs(main[0].m_hat - main[0].m_true) < 10.0
        assert "cone:4" in main[0].meta["preds"]


def test_fingers_open_to_the_commanded_width():
    from topple.physics.arm import ArmModel
    from topple.physics.arm_world import ArmInWorld

    w = World(load_config("quick"))
    w.add_body(RigidBody.ground(0.6))
    model = ArmModel("panda7", base=SE3(t=np.array([-0.45, 0.0, 0.0])))
    arm = ArmInWorld(w, model)
    T = model.fk(model.q_home)
    for opening in (0.0, 0.04, 0.08):
        arm.set_joint_state(model.q_home, np.zeros(7), opening)
        # inner faces of the two fingers, measured along the tool y axis
        ys = [T.inverse().apply(w.bodies[i].world_mesh().vertices)[:, 1] for i in arm.finger_ids]
        inner_left, inner_right = ys[0].max(), ys[1].min()
        assert abs((inner_right - inner_left) - opening) < 1e-9


@pytest.mark.slow
def test_arm_places_a_box_on_a_box_accurately():
    from topple.core.transforms import quat_conj, quat_mul, quat_rotate, quat_to_rotvec
    from topple.planning.placement import arm_place

    cfg = load_config("quick")
    cfg["arm"]["finger_max"] = 0.12
    w = World(cfg)
    w.add_body(RigidBody.ground(0.6))
    base = RigidBody.from_mesh("base", box((0.1, 0.1, 0.05)), 2600.0, 0.6, sdf_spacing=None)
    base.x = np.array([0.0, 0.0, 0.025])
    w.add_body(base)
    st = RigidBody.from_mesh("stone", box((0.07, 0.05, 0.04)), 2600.0, 0.6, sdf_spacing=None)
    V = quat_rotate(st.q, st.shape.mesh.vertices)
    st.x = np.array([0.0, 0.42, -V[:, 2].min()])
    w.add_body(st)
    w.damped_settle(seconds=0.3)
    V = quat_rotate(st.q, st.shape.mesh.vertices)
    target = SE3(st.q, np.array([0.0, 0.0, 0.05 - V[:, 2].min()]))
    base0 = base.x.copy()
    assert arm_place(w, st, target, cfg, np.random.default_rng(0))
    assert st.meta["arm_fallback"] is None
    assert np.linalg.norm(st.x - target.t) < 0.005
    assert np.degrees(np.linalg.norm(quat_to_rotvec(quat_mul(st.q, quat_conj(target.q))))) < 2.0
    assert np.linalg.norm(base.x - base0) < 0.001
    assert len(w.bodies) == 3  # the arm bodies are removed again
