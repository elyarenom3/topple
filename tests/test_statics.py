"""Static analyser tests (Section 12.3)."""

import numpy as np
import pytest

from topple.statics import (
    StaticBody,
    StaticContact,
    Tower,
    analytic_block_tilt,
    fc_margin,
    feasible,
    margin_jacobian_fd,
    min_norm_forces,
    push_margin,
    push_margin_gradient,
    subassembly_margins,
    support_polygon_margin,
    survival_probability,
    tilt_margin,
    tower_margin,
)
from topple.statics.equilibrium import ContactLP, equilibrium_residual, cone_matrix
from topple.statics.survival import margin_summary


def block_tower(mu, b, h, m=1.0, depth=None, centre=(0.0, 0.0)):
    depth = b if depth is None else depth
    cx, cy = centre
    bodies = [StaticBody(0, 0.0, np.zeros(3), True, "ground"), StaticBody(1, m, np.array([cx, cy, h / 2]), False, "block")]
    pts = [[cx + b / 2, cy + depth / 2, 0], [cx - b / 2, cy + depth / 2, 0], [cx - b / 2, cy - depth / 2, 0], [cx + b / 2, cy - depth / 2, 0]]
    return Tower(bodies, [StaticContact(0, 1, p, [0, 0, 1], mu) for p in pts])


def stacked_tower(n=4, mu=0.6, rng=None):
    """n blocks with slightly offset centres and 8 contacts per interface (generic geometry)."""
    rng = np.random.default_rng(3) if rng is None else rng
    bodies = [StaticBody(0, 0.0, np.zeros(3), True, "ground")]
    cs = []
    z = 0.0
    for k in range(n):
        h = 0.05 + 0.01 * k
        off = rng.uniform(-0.005, 0.005, size=2)
        bodies.append(StaticBody(k + 1, 1.0 + 0.3 * k, np.array([off[0], off[1], z + h / 2]), False, f"s{k}"))
        r = 0.05 - 0.004 * k
        for a in np.linspace(0, 2 * np.pi, 8, endpoint=False) + rng.uniform(0, 0.3):
            cs.append(StaticContact(k, k + 1, np.array([off[0] + r * np.cos(a), off[1] + r * np.sin(a), z]), np.array([0, 0, 1.0]), mu))
        z += h
    return Tower(bodies, cs)


@pytest.mark.parametrize("mu,b,h", [(0.6, 0.1, 0.1), (0.3, 0.1, 0.1), (0.8, 0.05, 0.2), (0.2, 0.2, 0.1), (0.5, 0.08, 0.12)])
def test_block_tilt_matches_analytic(mu, b, h):
    t = block_tower(mu, b, h)
    m, az, per = tilt_margin(t, cone_sides=64, n_azimuth=12, tol_deg=0.02)
    assert abs(m - analytic_block_tilt(mu, b, h)) < 0.15
    # the 8 sided cone under estimates the sliding branch by about 1 - cos(pi / 8)
    m8, _, _ = tilt_margin(t, cone_sides=8, n_azimuth=16)
    sliding = np.arctan(mu) < np.arctan(b / h)
    if sliding:
        assert abs(m8 - np.degrees(np.arctan(mu * np.cos(np.pi / 8)))) < 0.2
    else:
        assert abs(m8 - analytic_block_tilt(mu, b, h)) < 0.15


def test_block_regimes_cover_sliding_and_tipping():
    assert analytic_block_tilt(0.3, 0.1, 0.1) == pytest.approx(np.degrees(np.arctan(0.3)))
    assert analytic_block_tilt(0.8, 0.05, 0.2) == pytest.approx(np.degrees(np.arctan(0.25)))


def wedge_tower(mu, beta_deg, b=0.06, h=0.03):
    """Block on a plane inclined by beta about the y axis (downhill is +x)."""
    beta = np.radians(beta_deg)
    R = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    n = R @ np.array([0, 0, 1.0])
    com = R @ np.array([0, 0, h / 2])
    corners = [R @ np.array([sx * b / 2, sy * b / 2, 0.0]) for sx in (-1, 1) for sy in (-1, 1)]
    bodies = [StaticBody(0, 0.0, np.zeros(3), True), StaticBody(1, 1.0, com, False)]
    return Tower(bodies, [StaticContact(0, 1, p, n, mu) for p in corners])


def test_block_on_wedge():
    mu = 0.5
    lim = np.degrees(np.arctan(mu))
    assert feasible(wedge_tower(mu, lim - 2.0), cone_sides=64)[0]
    assert not feasible(wedge_tower(mu, lim + 2.0), cone_sides=64)[0]
    # the remaining tilt margin downhill is the analytic angle minus the slope
    t = wedge_tower(mu, 10.0)
    m, _, _ = tilt_margin(t, cone_sides=64, n_azimuth=12, tol_deg=0.02)
    assert abs(m - (lim - 10.0)) < 0.2


def arch_tower(mu_ground, mu_key=0.6):
    s = np.sqrt(0.5)
    bodies = [
        StaticBody(0, 0.0, np.zeros(3), True, "ground"),
        StaticBody(1, 2.0, np.array([-0.09, 0.0, 0.07]), False, "leg_l"),
        StaticBody(2, 2.0, np.array([0.09, 0.0, 0.07]), False, "leg_r"),
        StaticBody(3, 1.0, np.array([0.0, 0.0, 0.15]), False, "keystone"),
    ]
    cs = []
    for sx in (-1, 1):
        leg = 1 if sx < 0 else 2
        for x in (0.05, 0.15):
            for y in (-0.03, 0.03):
                cs.append(StaticContact(0, leg, np.array([sx * x, y, 0.0]), np.array([0, 0, 1.0]), mu_ground))
        for (x, z) in ((0.04, 0.13), (0.06, 0.15)):
            for y in (-0.02, 0.02):
                cs.append(StaticContact(leg, 3, np.array([sx * x, y, z]), np.array([-sx * s, 0.0, s]), mu_key))
    return Tower(bodies, cs)


def test_three_block_arch_needs_friction():
    assert feasible(arch_tower(0.6, 0.6))[0]
    assert not feasible(arch_tower(0.0, 0.0))[0]
    assert tilt_margin(arch_tower(0.6, 0.6), n_azimuth=8)[0] > 1.0


def test_mass_scaling_invariance():
    t = stacked_tower(3)
    t2 = t.scaled_mass(3.0)
    assert feasible(t)[0] and feasible(t2)[0]
    m1, _, per1 = tilt_margin(t, n_azimuth=12)
    m2, _, per2 = tilt_margin(t2, n_azimuth=12)
    assert abs(m1 - m2) < 0.06
    assert np.allclose(per1, per2, atol=0.06)
    s1, e1 = push_margin(t)
    s2, e2 = push_margin(t2)
    assert abs(s1 - s2) < 1e-6  # normalised by weight
    lp1, lp2 = ContactLP(t), ContactLP(t2)
    f1 = lp1.push(e1, t.top_body().id)[0]
    f2 = lp2.push(e1, t.top_body().id)[0]
    assert f2 == pytest.approx(3.0 * f1, rel=1e-6)


def test_rotation_about_vertical_invariance():
    """The N sided friction polygon is symmetric under rotations by 2 pi / N, so a rotation of the
    whole tower by 45 degrees (N = 8) leaves every margin unchanged; for other angles the
    polygon discretisation moves with the tower and the margins agree only for large N."""
    t = stacked_tower(3)
    n_az = 8
    t2 = t.rotated_about_vertical(2 * np.pi / 8)
    m1, _, per1 = tilt_margin(t, n_azimuth=n_az, refine=False)
    m2, _, per2 = tilt_margin(t2, n_azimuth=n_az, refine=False)
    assert abs(m1 - m2) < 0.06
    assert np.allclose(np.roll(per1, 1), per2, atol=0.06)
    assert abs(push_margin(t, n_dir=8)[0] - push_margin(t2, n_dir=8)[0]) < 1e-6
    assert abs(fc_margin(t) - fc_margin(t2)) < 1e-6
    # a 30 degree rotation with a 64 sided cone
    t3 = t.rotated_about_vertical(np.pi / 6)
    m1, _, _ = tilt_margin(t, cone_sides=64, n_azimuth=12, refine=False)
    m3, _, _ = tilt_margin(t3, cone_sides=64, n_azimuth=12, refine=False)
    assert abs(m1 - m3) < 0.2


def test_min_norm_forces_satisfy_equilibrium_and_cones():
    t = stacked_tower(3)
    f = min_norm_forces(t, cone_sides=8)
    assert equilibrium_residual(t, f) < 1e-5
    for j, c in enumerate(t.contacts):
        assert f[j, 2] >= -1e-7
        assert np.all(cone_matrix(c.mu, 8) @ f[j] <= 1e-6)
    # the symmetric block shares the weight equally between its four corners
    tb = block_tower(0.6, 0.1, 0.1)
    fb = min_norm_forces(tb)
    assert np.allclose(fb[:, 2], 9.81 / 4, atol=1e-4)
    assert np.allclose(fb[:, :2], 0.0, atol=1e-5)


def tripod_tower():
    """One block on three generic contacts: a non degenerate LP with unique duals."""
    bodies = [StaticBody(0, 0.0, np.zeros(3), True), StaticBody(1, 1.0, np.array([0.004, -0.003, 0.06]), False)]
    pts = [[0.05, 0.01, 0], [-0.03, 0.045, 0], [-0.025, -0.05, 0]]
    cs = [StaticContact(0, 1, p, [0.02, -0.01, 1.0], 0.5 + 0.1 * i) for i, p in enumerate(pts)]
    return Tower(bodies, cs)


def test_push_gradient_matches_finite_differences():
    t = tripod_tower()
    direction = np.array([0.6, 0.8, 0.0])

    def s_star(tower):
        return ContactLP(tower).push(direction, 1)[0]

    assert s_star(t) > 0
    for wrt in ("com", "mu", "points"):
        g = push_margin_gradient(t, direction, wrt=wrt)
        g_fd = margin_jacobian_fd(s_star, t, wrt=wrt, eps=1e-6)
        scale = max(np.max(np.abs(g_fd)), 1e-6)
        assert np.max(np.abs(g - g_fd)) / scale < 1e-3, (wrt, g, g_fd)
    # the tilt margin gradient is available by finite differences
    g_tilt = margin_jacobian_fd(lambda tw: tilt_margin(tw, n_azimuth=8, tol_deg=0.01)[0], t, wrt="com", eps=2e-3)
    assert g_tilt.shape == (1, 3)
    assert g_tilt[0, 2] < 0  # raising the centre of mass lowers the tilt margin


def test_subassembly_cuts_bound_the_tower_margin():
    t = stacked_tower(4)
    whole, _, _ = tilt_margin(t, n_azimuth=12)
    cuts = subassembly_margins(t, "tilt", n_azimuth=12)
    assert len(cuts) == 4
    assert min(cuts) >= whole - 0.1
    assert tower_margin(t, "tilt", n_azimuth=12) == pytest.approx(min(cuts))
    push_cuts = subassembly_margins(t, "push")
    assert all(p >= 0 for p in push_cuts)
    fc_cuts = subassembly_margins(t, "fc")
    assert all(f >= 0 for f in fc_cuts)
    sp = subassembly_margins(t, "support")
    assert all(0 < a < 90 for a in sp)


def test_support_polygon_view_matches_tipping():
    t = block_tower(5.0, 0.05, 0.2)  # tipping regime (huge mu)
    ang, d, h_c, poly = support_polygon_margin(t, 1)
    assert ang == pytest.approx(np.degrees(np.arctan(0.025 / 0.1)))
    assert d == pytest.approx(0.025)
    assert h_c == pytest.approx(0.1)
    assert len(poly) == 4
    m, _, _ = tilt_margin(t, cone_sides=16, n_azimuth=12, tol_deg=0.02)
    assert abs(m - ang) < 0.1


def test_fc_margin_is_positive_for_a_stable_block_and_zero_when_unsupported():
    t = block_tower(0.6, 0.1, 0.1)
    assert fc_margin(t) > 0
    # centre of mass outside the support polygon
    bodies = [StaticBody(0, 0.0, np.zeros(3), True), StaticBody(1, 1.0, np.array([0.2, 0.0, 0.05]), False)]
    cs = [StaticContact(0, 1, p, [0, 0, 1], 0.6) for p in [[0.05, 0.05, 0], [-0.05, 0.05, 0], [-0.05, -0.05, 0], [0.05, -0.05, 0]]]
    assert fc_margin(Tower(bodies, cs)) == 0.0
    assert not feasible(Tower(bodies, cs))[0]


def test_survival_probability_is_fraction_above_tau():
    m = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert survival_probability(m, 3.0) == pytest.approx(2 / 5)
    assert survival_probability(m, 0.0) == 1.0
    assert survival_probability([], 3.0) == 0.0
    s = margin_summary(m)
    assert s["mean"] == pytest.approx(3.0) and s["n"] == 5


def test_from_estimates_and_interfaces():
    t = stacked_tower(3)
    t2 = Tower.from_estimates([{"id": b.id, "mass": b.mass, "com": b.com, "is_ground": b.is_ground} for b in t.bodies], t.contacts)
    assert t2.n_bodies == t.n_bodies and t2.n_contacts == t.n_contacts
    ifs = t.interfaces()
    assert len(ifs) == 3 and all(len(g) == 8 for g in ifs)
    sub = t.upper_subassembly(1)
    assert len(sub.free_bodies()) == 2 and sub.n_contacts == 16


def test_from_meshes_builds_contacts():
    from topple.core.mesh import box

    b1 = box((0.1, 0.1, 0.04), center=(0, 0, 0.02))
    b2 = box((0.06, 0.06, 0.04), center=(0.01, 0, 0.06))
    t = Tower.from_meshes([b1, b2], [1.0, 0.5])
    assert t.n_contacts >= 6
    assert feasible(t)[0]
    m, _, _ = tilt_margin(t, n_azimuth=8)
    assert m > 5.0


# ------------------------------------------------------------------ rocking aware margin (Margin 4)

def _hull_tower_from_world(w, mu):
    from topple.statics.analysis import build_tower

    realised = []
    for b in w.bodies[1:]:
        realised.append({"mesh": b.world_mesh(), "mass": b.mass, "com": b.x.copy(), "mu": mu, "name": b.name, "mu_table": {"ground": mu, **{bb.name: mu for bb in w.bodies[1:]}}})
    return build_tower(realised, 5e-4, 8, geometry="hull", settle=False)


def test_rocking_margin_of_a_block_matches_the_analytic_angle():
    from topple.core.mesh import box
    from topple.physics.body import RigidBody
    from topple.physics.world import World
    from topple.statics.rocking import rocking_tilt_margin

    mu, b, h = 0.8, 0.06, 0.12
    w = World(dt=0.001)
    w.add_body(RigidBody.ground(mu))
    blk = RigidBody.from_mesh("block", box((b, b, h)), 2600.0, mu, sdf_spacing=None)
    blk.x = np.array([0.0, 0.0, h / 2])
    w.add_body(blk)
    t = _hull_tower_from_world(w, mu)
    m, _ = rocking_tilt_margin(t.realised, t.hulls, n_azimuth=4)
    assert abs(m - analytic_block_tilt(mu, b, h)) < 1.0, m


def test_rocking_margin_is_below_the_tilt_margin_on_a_curved_support():
    """A stone resting on the facets of a curved mesh pivots onto the next facet (more than 2
    degrees) long before the contact force program becomes infeasible."""
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from test_physics import _ellipsoid_on_block

    from topple.statics.rocking import rocking_tilt_margin

    w = _ellipsoid_on_block()
    w.damped_settle(seconds=0.6)
    t = _hull_tower_from_world(w, 0.8)
    m_tilt = tower_margin(t, kind="tilt", cone_sides=8, n_azimuth=12)
    m_rock, _ = rocking_tilt_margin(t.realised, t.hulls, n_azimuth=8)
    assert m_rock < m_tilt
    assert m_rock < 5.0
