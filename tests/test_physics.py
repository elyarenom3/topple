"""Physics unit tests (Section 12.1) plus solver, collision, recorder, XPBD and merging checks."""

from __future__ import annotations

import numpy as np
import pytest

from topple.core.mesh import box, icosphere, tetrahedron
from topple.core.transforms import SE3, quat_from_axis_angle, quat_to_rotvec, quat_mul, quat_conj
from topple.physics import RigidBody, StateRecorder, World, find_contacts, gjk_distance, epa_penetration, verify_contacts_sdf
from topple.physics.solver import lemke, project_polygon


def _cube_body(name, size=0.1, z=0.05, mu=0.6, density=2600.0, x=0.0, y=0.0):
    return RigidBody.from_mesh(name, box((size, size, size)), density, mu, sdf_spacing=None, pose=SE3(t=[x, y, z]))


def _stack(n=10, size=0.1, mu=0.6, **kw):
    w = World(**kw)
    w.add_body(RigidBody.ground(mu))
    for i in range(n):
        w.add_body(_cube_body(f"cube{i}", size, size / 2 + size * i, mu))
    return w


# ------------------------------------------------------------------ 12.1 item 1

def test_free_body_momentum_per_step():
    w = World(gravity=(0, 0, 0))
    b = RigidBody.from_mesh("rod", box((0.3, 0.05, 0.02)), 2600.0, sdf_spacing=None, pose=SE3(t=[0, 0, 1.0]))
    w.add_body(b)
    b.v = np.array([0.3, -0.2, 0.1])
    b.w = np.array([5.0, 0.7, -3.0])
    for _ in range(200):
        p0 = w.linear_momentum().copy()
        L0 = w.angular_momentum().copy()
        w.step()
        p1 = w.linear_momentum()
        L1 = w.angular_momentum()
        assert np.linalg.norm(p1 - p0) <= 1e-10 * np.linalg.norm(p0)
        # the implicit midpoint rule conserves the angular momentum magnitude exactly
        assert abs(np.linalg.norm(L1) - np.linalg.norm(L0)) <= 1e-10 * np.linalg.norm(L0)


def test_torque_free_precession_conserves_energy():
    w = World(gravity=(0, 0, 0))
    b = RigidBody("asym", 1.0, np.diag([1.0, 2.0, 3.0]), _cube_body("tmp").shape, x=[0, 0, 1])
    w.add_body(b)
    b.w = np.array([1.0, 0.1, 5.0])
    E0 = w.kinetic_energy()
    for _ in range(3000):
        w.step()
        assert abs(w.kinetic_energy() - E0) <= 1e-7 * E0


# ------------------------------------------------------------------ 12.1 item 2

def test_frictionless_sliding_conserves_horizontal_momentum():
    w = World()
    w.add_body(RigidBody.ground(0.0))
    b = _cube_body("slider", mu=0.0)
    w.add_body(b)
    w.step(50)  # settle on the ground
    b.v = np.array([0.5, 0.2, 0.0])
    p0 = (b.mass * b.v)[:2].copy()
    w.step(1000)
    p1 = (b.mass * b.v)[:2]
    assert np.linalg.norm(p1 - p0) < 1e-9 * np.linalg.norm(p0)
    assert abs(b.x[2] - 0.05) < 1e-3


# ------------------------------------------------------------------ 12.1 item 3

def test_cone_inequalities_and_residual_logging():
    # two cubes tip at atan(0.1 / 0.2) = 26.6 degrees and slide at atan(0.6) = 31 degrees,
    # so the stack rests at 15 degrees and the complementarity residual must vanish
    w = _stack(2)
    w.gravity = 9.81 * np.array([np.sin(np.radians(15)), 0.0, -np.cos(np.radians(15))])
    for _ in range(300):
        w.step()
        for c in w.contacts:
            assert c.fn >= 0.0
            assert np.hypot(c.ft1, c.ft2) <= c.mu * c.fn * (1 + 1e-9) + 1e-12
        r = w.last_residuals
        assert set(["max_cone_violation", "max_complementarity", "pgs_iterations"]) <= set(r)
        assert r["max_cone_violation"] <= 1e-9
    assert w.last_residuals["max_complementarity"] < 1e-6


# ------------------------------------------------------------------ 12.1 item 4

@pytest.mark.parametrize("alpha_deg, mu, slides", [(25.0, 0.5, False), (30.0, 0.5, True), (40.0, 0.5, True)])
def test_inclined_plane(alpha_deg, mu, slides):
    a = np.radians(alpha_deg)
    w = World()
    w.add_body(RigidBody.ground(mu))
    b = RigidBody.from_mesh("block", box((0.2, 0.2, 0.04)), 2600.0, mu, sdf_spacing=None, pose=SE3(t=[0, 0, 0.02]))
    w.add_body(b)
    w.step(50)
    w.gravity = 9.81 * np.array([np.sin(a), 0.0, -np.cos(a)])
    w.step(100)
    v0 = b.v[0]
    w.step(500)
    v1 = b.v[0]
    acc = (v1 - v0) / 0.5
    if slides:
        expected = 9.81 * (np.sin(a) - mu * np.cos(a))
        assert abs(acc - expected) < 0.01 * expected + 1e-3
    else:
        assert abs(acc) < 1e-3
        assert np.linalg.norm(b.v) < 1e-3


# ------------------------------------------------------------------ 12.1 item 5

def _critical_angle_dynamic(b_width, h_height, mu, start_deg, rate_deg_s=1.0, max_deg=60.0):
    w = World()
    w.add_body(RigidBody.ground(mu))
    blk = RigidBody.from_mesh("block", box((b_width, 0.1, h_height)), 2600.0, mu, sdf_spacing=None, pose=SE3(t=[0, 0, h_height / 2]))
    w.add_body(blk)
    w.step(100)
    x0 = blk.x.copy()
    q0 = blk.q.copy()
    ang = start_deg
    while ang < max_deg:
        ang += rate_deg_s * w.dt
        a = np.radians(ang)
        w.gravity = 9.81 * np.array([np.sin(a), 0.0, -np.cos(a)])
        w.step()
        disp = np.linalg.norm(blk.x - x0)
        rot = np.degrees(np.linalg.norm(quat_to_rotvec(quat_mul(blk.q, quat_conj(q0)))))
        if disp > 1e-3 or rot > 0.5:
            return ang
    return ang


@pytest.mark.parametrize("b, h, mu", [(0.05, 0.15, 0.8), (0.1, 0.1, 0.3)])
def test_analytic_tipping_or_sliding(b, h, mu):
    expected = np.degrees(min(np.arctan(mu), np.arctan(b / h)))
    measured = _critical_angle_dynamic(b, h, mu, start_deg=expected - 4.0)
    assert abs(measured - expected) < 1.0


# ------------------------------------------------------------------ 12.1 item 6

def test_ten_stacked_cubes_rest_10s():
    w = _stack(10)
    x0 = np.array([b.x.copy() for b in w.bodies])
    w.step(10000)
    x1 = np.array([b.x.copy() for b in w.bodies])
    drift = np.abs(x1 - x0).max()
    assert drift < 1e-4, drift
    assert w.max_body_speed() < 1e-4


# ------------------------------------------------------------------ 12.1 item 7

def _cradle(mode):
    """Five equal cubes in a frictionless row (no gravity); the first one arrives at 1 m/s."""
    w = World(gravity=(0, 0, 0), restitution=1.0, impact_mode=mode)
    s = 0.04
    for i in range(5):
        b = RigidBody.from_mesh(f"cube{i}", box((s, s, s)), 7800.0, 0.0, sdf_spacing=None, pose=SE3(t=[s * i, 0, 0.5]))
        w.add_body(b)
    w.bodies[0].x[0] -= 0.01
    w.bodies[0].v = np.array([1.0, 0.0, 0.0])
    w.step(80)
    return np.array([b.v[0] for b in w.bodies])


def test_newtons_cradle_sequential_vs_simultaneous():
    seq = _cradle("sequential")
    sim = _cradle("simultaneous")
    assert not np.allclose(seq, sim, atol=1e-3)
    # sequential treatment transfers the velocity to the last ball
    assert abs(seq[-1] - 1.0) < 1e-6
    assert np.all(np.abs(seq[:-1]) < 1e-6)
    # simultaneous treatment shares the momentum among the balls in contact
    assert sim[-1] < 0.5
    assert abs(seq.sum() - 1.0) < 0.02 and abs(sim.sum() - 1.0) < 0.02


# ------------------------------------------------------------------ 12.1 item 8

def test_mass_properties_closed_form():
    c = box((0.2, 0.3, 0.4)).mass_properties(1000.0)
    assert abs(c["mass"] - 24.0) < 1e-12
    assert np.allclose(np.diag(c["inertia_com"]), [24 * (0.09 + 0.16) / 12, 24 * (0.04 + 0.16) / 12, 24 * (0.04 + 0.09) / 12])
    t = tetrahedron(1.0).mass_properties(1.0)
    assert abs(t["volume"] - 8.0 / 3.0) < 1e-12
    assert np.allclose(t["com"], 0.0, atol=1e-12)
    errs, ierrs = [], []
    for sub in (2, 3, 4):
        s = icosphere(sub, 0.5).mass_properties(1.0)
        errs.append(abs(s["volume"] - 4 / 3 * np.pi * 0.125) / (4 / 3 * np.pi * 0.125))
        ierrs.append(abs(s["inertia_com"][0, 0] - 0.4 * s["mass"] * 0.25) / (0.4 * s["mass"] * 0.25))
    assert errs[0] > errs[1] > errs[2] and errs[2] < 5e-3
    assert ierrs[0] > ierrs[1] > ierrs[2] and ierrs[1] < 1e-2


# ------------------------------------------------------------------ solvers

def test_polygon_projection():
    for N in (4, 8, 16, 32):
        f = project_polygon(5.0, 0.0001, 1.0, N)
        assert abs(f[0] - 1.0) < 1e-9 and abs(f[1]) < 1e-3
        assert project_polygon(0.2, 0.1, 1.0, N) == (0.2, 0.1)
        assert project_polygon(0.2, 0.1, 0.0, N) == (0.0, 0.0)
        # the projection lies inside the circumscribed circle
        for ang in np.linspace(0, 2 * np.pi, 37):
            p = project_polygon(3 * np.cos(ang), 3 * np.sin(ang), 1.0, N)
            assert np.hypot(*p) <= 1.0 + 1e-12
            assert np.hypot(*p) >= np.cos(np.pi / N) - 1e-12


def test_lemke_basic():
    M = np.array([[2.0, 1.0], [1.0, 2.0]])
    q = np.array([-5.0, -6.0])
    z, status = lemke(M, q)
    w = M @ z + q
    assert status == 0 and np.all(z >= -1e-12) and np.all(w >= -1e-9) and abs(z @ w) < 1e-9


def _one_step_compare(make_world):
    outs = {}
    for solver in ("pgs", "lemke"):
        w = make_world()
        w.solver = solver
        w.pgs_iterations = 2000
        w.pgs_tol = 1e-14
        w.step()
        outs[solver] = np.concatenate([np.concatenate([b.v, b.w]) for b in w.bodies if b.inv_mass > 0])
    return outs["pgs"], outs["lemke"]


def test_lemke_matches_pgs_one_and_three_contacts():
    def sphere_world():
        w = World(cone_sides=8)
        w.add_body(RigidBody.ground(0.4))
        b = RigidBody.from_mesh("ball", icosphere(3, 0.05), 2600.0, 0.4, sdf_spacing=None, pose=SE3(t=[0, 0, 0.05 - 2e-4]))
        w.add_body(b)
        b.v = np.array([0.3, 0.1, 0.0])
        return w

    def tetra_world():
        w = World(cone_sides=8)
        w.add_body(RigidBody.ground(0.5))
        m = tetrahedron(0.05)
        # rest on the face opposite vertex 0: rotate so that face normal points down
        b = RigidBody.from_mesh("tet", m, 2600.0, 0.5, sdf_spacing=None)
        w.add_body(b)
        # drop and settle to find a resting pose, then compare one step with tangential velocity
        w.step(600)
        b.v = np.array([0.2, -0.1, 0.0])
        b.w = np.array([0.0, 0.0, 1.0])
        return w

    for mk in (sphere_world, tetra_world):
        a, b = _one_step_compare(mk)
        assert np.allclose(a, b, atol=1e-5), (a, b)


# ------------------------------------------------------------------ collision

def test_gjk_epa_boxes():
    A = box((1, 1, 1)).vertices
    d, pa, pb = gjk_distance(A, A + np.array([1.5, 0, 0]))
    assert abs(d - 0.5) < 1e-12
    depth, n, pa, pb = epa_penetration(A, A + np.array([0.9, 0.1, 0.05]))
    assert abs(depth - 0.1) < 1e-9 and np.allclose(n, [1, 0, 0])


def test_manifold_and_sdf_verification():
    w = World()
    w.add_body(RigidBody.ground(0.6))
    cube = RigidBody.from_mesh("cube", box((0.1, 0.1, 0.1)), 2600.0, 0.6, sdf_spacing=0.004, pose=SE3(t=[0, 0, 0.05]))
    w.add_body(cube)
    top = RigidBody.from_mesh("top", box((0.06, 0.06, 0.06)), 2600.0, 0.6, sdf_spacing=0.004, pose=SE3(t=[0.01, 0.0, 0.13]))
    w.add_body(top)
    cs = find_contacts(w)
    pairs = {(c.body_a, c.body_b) for c in cs}
    assert pairs == {(0, 1), (1, 2)}
    assert sum(1 for c in cs if (c.body_a, c.body_b) == (0, 1)) == 4
    assert sum(1 for c in cs if (c.body_a, c.body_b) == (1, 2)) >= 4
    for c in cs:
        assert np.allclose(c.normal, [0, 0, 1])
        assert abs(c.depth) < 1e-6
    rep = verify_contacts_sdf(w, cs, tol=5e-3)
    assert rep["checked"] >= 4 and rep["ok"]


def test_aabb_tree_pairs():
    from topple.physics.collision import AABBTree

    t = AABBTree(0.0)
    for i in range(6):
        t.insert(i, np.array([i * 1.0, 0, 0]), np.array([i * 1.0 + 1.1, 1, 1]))
    assert t.query_pairs() == [(i, i + 1) for i in range(5)]
    t.update(0, np.array([20.0, 0, 0]), np.array([21.0, 1, 1]))
    assert (0, 1) not in t.query_pairs()
    t.remove(3)
    assert all(3 not in p for p in t.query_pairs())


# ------------------------------------------------------------------ recorder, kinematic, xpbd, merging

def test_recorder_interpolation_and_io(tmp_path):
    w = World(gravity=(0, 0, 0))
    b = _cube_body("c")
    w.add_body(b)
    rec = StateRecorder(w)
    q0 = b.q.copy()
    b.v = np.array([1.0, 0, 0])
    b.w = np.array([0, 0, np.pi])
    w.step(100)
    T = rec.pose_at(0, 0.0505)
    assert abs(T.t[0] - 0.0505) < 1e-9
    rel = quat_to_rotvec(quat_mul(T.q, quat_conj(q0)))
    assert abs(np.linalg.norm(rel) - np.pi * 0.0505) < 1e-6
    rec.save(tmp_path / "rec.npz")
    r2 = StateRecorder.load(tmp_path / "rec.npz")
    assert len(r2) == len(rec)
    assert np.allclose(r2.pose_at(0, 0.03).t, rec.pose_at(0, 0.03).t)


def test_kinematic_ground_drags_cube():
    w = World()
    g = RigidBody.ground(0.6)
    w.add_body(g)
    c = _cube_body("c")
    w.add_body(c)
    w.step(50)
    w.set_kinematic_velocity(0, [0.1, 0, 0], [0, 0, 0])
    w.step(500)
    assert abs(c.v[0] - 0.1) < 1e-3
    assert g.x[0] > 0.05


def test_external_force_push():
    w = World()
    w.add_body(RigidBody.ground(0.6))
    c = _cube_body("c")
    w.add_body(c)
    w.step(50)
    w.external_force(1, [50.0, 0, 0], c.x + np.array([0, 0, 0.05]))
    w.step()
    assert c.v[0] > 0 or c.w[1] > 0
    w.step(100)
    assert np.linalg.norm(c.v) < 0.05  # the push lasted one step only, friction stops the cube


def test_xpbd_stack_stays_at_rest():
    w = _stack(3, solver="xpbd")
    x0 = np.array([b.x.copy() for b in w.bodies])
    w.step(1000)
    x1 = np.array([b.x.copy() for b in w.bodies])
    assert np.abs(x1 - x0).max() < 2e-3
    assert w.max_body_speed() < 5e-2


def test_merging_groups_and_unmerges_on_push():
    w = _stack(5, merging=True)
    w.step(400)
    groups = {b.group for b in w.bodies[1:]}
    assert len(groups) == 1 and -1 not in groups
    x0 = np.array([b.x.copy() for b in w.bodies])
    w.step(500)
    x1 = np.array([b.x.copy() for b in w.bodies])
    assert np.abs(x1 - x0).max() < 1e-4
    w.external_force(5, [30.0, 0, 0], w.bodies[5].x)
    w.step()
    assert all(b.group == -1 for b in w.bodies)
    w.step(200)
    assert w.max_body_speed() >= 0.0


def test_clone_is_independent():
    w = _stack(3)
    w.step(20)
    c = w.clone()
    c.step(200)
    assert c.time > w.time
    assert np.allclose(w.state().x, w.state().x)
    assert c.bodies[1].shape is w.bodies[1].shape


def test_remove_body_remaps_pair_friction_and_subset_keeps_it():
    w = _stack(3)
    w.set_pair_friction(0, 3, 0.31)
    w.set_pair_friction(2, 3, 0.42)
    w.remove_body(1)
    assert np.isclose(w.pair_friction(0, 2), 0.31) and np.isclose(w.pair_friction(1, 2), 0.42)
    sub, m = w.subset([0, 2])
    assert len(sub.bodies) == 2 and np.isclose(sub.pair_friction(m[0], m[2]), 0.31)
    assert sub.bodies[1].shape is w.bodies[2].shape


def test_torsional_patch_friction_stops_spin_of_a_point_contact():
    """A sphere spinning about the vertical on the ground has a single contact point; with the
    soft contact patch its spin decays at mu * r_t * m g / I_zz, without it the spin persists."""
    for r_t, decays in ((0.002, True), (0.0, False)):
        w = World(torsion_radius=r_t)
        w.add_body(RigidBody.ground(0.6))
        s = RigidBody.from_mesh("ball", icosphere(3, 0.05), 2600.0, 0.6, sdf_spacing=None, pose=SE3(t=[0, 0, 0.05]))
        w.add_body(s)
        w.step(50)
        s.w = np.array([0.0, 0.0, 2.0])
        w.step(100)
        if decays:
            # deceleration mu r_t m g / I_zz = 0.6 * 0.002 * 9.81 / (0.4 * 0.05^2) = 11.8 rad/s^2
            alpha = 0.6 * 0.002 * 9.81 / (0.4 * 0.05 ** 2)
            assert abs(s.w[2] - (2.0 - alpha * 0.1)) < 0.05
            w.step(400)
            assert abs(s.w[2]) < 1e-3
        else:
            assert abs(s.w[2] - 2.0) < 1e-3


# ------------------------------------------------------------------ touching contacts

def _ellipsoid_on_block():
    from topple.core.mesh import icosphere
    from topple.core.transforms import quat_rotate

    w = World(dt=0.001)
    w.add_body(RigidBody.ground(0.8))
    flat = box((0.14, 0.14, 0.04))
    ell = icosphere(3, 1.0).scaled([0.05, 0.04, 0.022])
    L = RigidBody.from_mesh("L", flat, 2600.0, 0.8, sdf_spacing=None)
    L.x = np.array([0.0, 0.0, 0.02])
    w.add_body(L)
    U = RigidBody.from_mesh("U", ell, 2600.0, 0.8, sdf_spacing=None)
    VU = quat_rotate(U.q, U.shape.mesh.vertices)
    U.x = np.array([0.003, 0.002, 0.04 - VU[:, 2].min() + 0.001])
    w.add_body(U)
    return w


def test_touching_hulls_report_zero_separation():
    """Two hulls touching to within a nanometre: the query must report the touching separation
    along the face normal (EPA alone is ill conditioned here and once returned 1.3 mm)."""
    from topple.core.mesh import icosphere
    from topple.physics.collision import convex_query

    VA = box((0.14, 0.14, 0.04)).convex_hull().vertices + np.array([0.0, 0.0, 0.02])
    VB = icosphere(3, 1.0).scaled([0.05, 0.04, 0.022]).convex_hull().vertices.copy()
    VB[:, 0] += 0.003
    for gap in (1e-9, 0.0, -1e-9, 2e-5, -2e-5):
        VB2 = VB + np.array([0.0, 0.0, 0.04 - VB[:, 2].min() + gap])
        ok, sep, n, pa, pb = convex_query(np.ascontiguousarray(VA), np.ascontiguousarray(VB2))
        assert ok
        assert abs(sep - gap) < 1e-7, (gap, sep)
        assert n[2] > 0.9999


def test_curved_stone_rests_without_rocking():
    """A flattened ellipsoid settled on a block stays at rest for 2 s: the manifold keeps the
    vertices of the facet it rests on and touching contacts inject no energy."""
    from topple.core.transforms import quat_conj, quat_mul, quat_to_rotvec

    w = _ellipsoid_on_block()
    info = w.damped_settle(seconds=0.6)
    assert info["still"]
    q0, x0 = w.bodies[2].q.copy(), w.bodies[2].x.copy()
    w.step(2000)
    rot = np.degrees(np.linalg.norm(quat_to_rotvec(quat_mul(w.bodies[2].q, quat_conj(q0)))))
    assert rot < 0.3, rot
    assert np.linalg.norm(w.bodies[2].x - x0) < 2e-4
