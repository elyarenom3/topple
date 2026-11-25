"""Stone generator tests (Sections 5, 12.1 item 8)."""

import numpy as np
import pytest

from topple.core.config import load_config
from topple.core.mesh import TriMesh, box, icosphere, tetrahedron
from topple.stonegen import StoneGenerator, descriptors, erode, fbm3, perlin3, worley3
from topple.stonegen.descriptors import in_box
from topple.stonegen.families import family_a_hull, family_b_ellipsoid, family_c_superquadric, family_d_voronoi, superquadric_inside_outside


def test_mass_properties_closed_form():
    b = box((2.0, 3.0, 4.0))
    mp = b.mass_properties(1.0)
    assert np.isclose(mp["volume"], 24.0)
    assert np.allclose(mp["com"], 0.0, atol=1e-12)
    assert np.allclose(np.diag(mp["inertia_com"]), [(9 + 16) / 12 * 24, (4 + 16) / 12 * 24, (4 + 9) / 12 * 24])
    t = tetrahedron(1.0)
    assert np.isclose(t.volume(), 8.0 / 3.0)
    errs = []
    for sub in (2, 3, 4):
        s = icosphere(sub, 0.5)
        mp = s.mass_properties(1.0)
        errs.append(abs(mp["inertia_com"][0, 0] - 0.4 * mp["mass"] * 0.25) / (0.4 * mp["mass"] * 0.25))
    assert errs[-1] < 2e-3 and errs[0] > errs[-1]  # convergence under refinement


def test_noise_properties():
    rng = np.random.default_rng(0)
    x = rng.uniform(-10, 10, size=(2000, 3))
    n = perlin3(x, seed=3)
    assert n.min() > -1.0 and n.max() < 1.0
    lattice = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [-4.0, 5.0, 2.0]])
    assert np.allclose(perlin3(lattice, seed=3), 0.0)
    assert np.allclose(fbm3(x, 3, seed=1), fbm3(x, 3, seed=1))
    assert not np.allclose(fbm3(x, 3, seed=1), fbm3(x, 3, seed=2))
    f1, f2 = worley3(x, seed=0)
    assert np.all(f1 <= f2) and np.all(f1 >= 0)


def test_superquadric_surface():
    rng = np.random.default_rng(0)
    m = family_c_superquadric(rng, 1.0, 0.8, 0.6, 0.5, 0.5, perturb=0.0)
    F = superquadric_inside_outside(m.vertices, 1.0, 0.8, 0.6, 0.5, 0.5)
    assert np.allclose(F, 1.0, atol=1e-9)
    assert all(m.is_valid().values())


def test_families_valid():
    rng = np.random.default_rng(1)
    meshes = [
        family_a_hull(rng, 1.0, 0.8, 0.7, n_points=10, point_distribution="shell"),
        family_a_hull(rng, 1.0, 0.8, 0.4, n_points=50),
        family_b_ellipsoid(rng, 1.0, 0.8, 0.4, amplitude=0.03),
        family_c_superquadric(rng, 1.0, 0.8, 0.7, 0.4, 0.4, perturb=0.02),
        family_d_voronoi(rng, (1.0, 0.8, 0.7), n_seeds=12),
    ]
    for m in meshes:
        v = m.is_valid()
        assert all(v.values()), (m.name, v)
        assert m.volume() > 0


def test_erosion_reduces_positive_curvature():
    rng = np.random.default_rng(2)
    m = family_d_voronoi(rng, (1.0, 0.8, 0.7), n_seeds=10, smooth_iters=0)
    H0 = m.mean_curvature() * m.bounding_radius()
    me = erode(m, 0.002, 4)
    H1 = me.mean_curvature() * me.bounding_radius()
    # the sharpest convex features (99th percentile of the dimensionless curvature) recede and the volume shrinks
    assert np.percentile(np.maximum(H1, 0), 99) < np.percentile(np.maximum(H0, 0), 99)
    assert me.volume() < m.volume()
    assert all(me.is_valid().values())


@pytest.mark.parametrize("stone_class", ["flat_river", "blocky_quarry", "rounded_cobble"])
def test_generator_classes(stone_class):
    cfg = load_config("default")
    g = StoneGenerator(cfg)
    rng = np.random.default_rng(10)
    for _ in range(2):
        s = g.sample(rng, stone_class)
        assert all(s.mesh.is_valid().values())
        assert in_box(s.descriptors, g.boxes[stone_class]), s.descriptors
        assert g.size_range[0] - 1e-9 <= s.descriptors["a"] <= g.size_range[1] + 1e-9
        assert 300 <= s.mesh.n_faces <= 6000
        assert s.mass > 0 and np.allclose(s.mesh.centroid(), 0.0, atol=1e-6)  # body frame at the centre of mass
        assert np.allclose(s.inertia, np.diag(np.diag(s.inertia)))
        assert s.material in {"slate", "granite", "sandstone", "quartzite", "river_basalt", "limestone"}


def test_generator_deterministic():
    cfg = load_config("default")
    g1, g2 = StoneGenerator(cfg), StoneGenerator(cfg)
    s1 = g1.sample(np.random.default_rng(5), "rounded_cobble")
    s2 = g2.sample(np.random.default_rng(5), "rounded_cobble")
    assert np.allclose(s1.mesh.vertices, s2.mesh.vertices) and s1.density == s2.density


def test_stone_to_body():
    from topple.stonegen import stone_to_body

    cfg = load_config("quick")
    g = StoneGenerator(cfg)
    s = g.sample(np.random.default_rng(3), "blocky_quarry")
    b = stone_to_body(s, friction=0.7, sdf_spacing=None)
    assert np.isclose(b.mass, s.mass, rtol=1e-6)
    assert b.friction == 0.7
    assert b.shape.pieces and b.meta["stone_class"] == "blocky_quarry"
