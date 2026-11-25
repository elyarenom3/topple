import numpy as np
import pytest

from topple.core import SE3, quat_from_axis_angle, quat_integrate, quat_mul, quat_rotate, quat_slerp, quat_to_rot, rot_to_quat, skew, tangent_basis
from topple.core.config import config_hash, load_config, merge
from topple.core.mesh import SDF, TriMesh, box, cylinder, icosphere, tetrahedron
from topple.core.raycast import BVH, brute_force_intersect
from topple.core.rng import Stage, make_rng


def test_quaternion_rotation_matches_matrix():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        v = rng.normal(size=3)
        assert np.allclose(quat_rotate(q, v), quat_to_rot(q) @ v)
        assert np.allclose(np.abs(rot_to_quat(quat_to_rot(q)) @ q), 1.0)


def test_quat_mul_matches_matrix_product():
    rng = np.random.default_rng(1)
    a = rng.normal(size=4)
    a /= np.linalg.norm(a)
    b = rng.normal(size=4)
    b /= np.linalg.norm(b)
    assert np.allclose(quat_to_rot(quat_mul(a, b)), quat_to_rot(a) @ quat_to_rot(b))


def test_se3_compose_inverse():
    rng = np.random.default_rng(2)
    T1 = SE3(rng.normal(size=4), rng.normal(size=3))
    T2 = SE3(rng.normal(size=4), rng.normal(size=3))
    p = rng.normal(size=3)
    assert np.allclose((T1 @ T2).apply(p), T1.apply(T2.apply(p)))
    assert np.allclose((T1 @ T1.inverse()).matrix(), np.eye(4), atol=1e-12)
    assert np.allclose(SE3.from_matrix(T1.matrix()).matrix(), T1.matrix())


def test_quat_integrate_exact_for_constant_rate():
    q0 = np.array([1.0, 0, 0, 0])
    w = np.array([0.0, 0.0, 2.0])
    q = q0
    for _ in range(100):
        q = quat_integrate(q, w, 0.01)
    expected = quat_from_axis_angle([0, 0, 1], 2.0)
    assert np.allclose(np.abs(q @ expected), 1.0, atol=1e-10)


def test_slerp_endpoints_and_midpoint():
    q0 = np.array([1.0, 0, 0, 0])
    q1 = quat_from_axis_angle([0, 0, 1], np.pi / 2)
    assert np.allclose(quat_slerp(q0, q1, 0), q0)
    assert np.allclose(quat_slerp(q0, q1, 1), q1)
    assert np.allclose(quat_slerp(q0, q1, 0.5), quat_from_axis_angle([0, 0, 1], np.pi / 4))


def test_tangent_basis_right_handed():
    rng = np.random.default_rng(3)
    for _ in range(20):
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        t1, t2 = tangent_basis(n)
        assert abs(t1 @ n) < 1e-12 and abs(t2 @ n) < 1e-12 and abs(t1 @ t2) < 1e-12
        assert np.allclose(np.cross(t1, t2), n)


def test_skew():
    a, b = np.array([1.0, 2, 3]), np.array([-2.0, 0.5, 4])
    assert np.allclose(skew(a) @ b, np.cross(a, b))


def test_mass_properties_closed_form():
    b = box((2.0, 3.0, 4.0))
    mp = b.mass_properties(2.0)
    assert np.isclose(mp["mass"], 48.0)
    assert np.allclose(np.diag(mp["inertia_com"]), 48.0 / 12 * np.array([9 + 16, 4 + 16, 4 + 9]))
    assert np.allclose(mp["com"], 0, atol=1e-12)
    t = tetrahedron()
    assert np.isclose(t.volume(), 8.0 / 3.0)
    # sphere converges under refinement
    errs = []
    for sub in (2, 3, 4):
        s = icosphere(sub, 0.5)
        errs.append(abs(s.mass_properties(1.0)["inertia_com"][0, 0] - 0.4 * s.volume() * 0.25) / (0.4 * s.volume() * 0.25))
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 2e-3


def test_mesh_validity_checks():
    for m in (box(), icosphere(2), tetrahedron(), cylinder(0.5, 1.0, 16)):
        v = m.is_valid()
        assert all(v.values()), v
    bad = TriMesh(box().vertices, box().faces[:-1])
    assert not bad.is_watertight()


def test_mean_curvature_sphere():
    s = icosphere(4, 0.5)
    H = s.mean_curvature()
    assert abs(H.mean() - 2.0) < 0.02


def test_bvh_matches_brute_force():
    s = icosphere(3, 1.0)
    rng = np.random.default_rng(4)
    o = rng.normal(size=(500, 3)) * 3
    d = rng.normal(size=(500, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    t, tid, u, v = s.bvh().intersect(o, d)
    tb, idb = brute_force_intersect(s.vertices, s.faces, o, d)
    assert np.allclose(np.where(tid >= 0, t, -1), tb)
    assert (tid >= 0).sum() > 10


def test_contains_and_sdf():
    s = icosphere(3, 0.5)
    assert s.contains(np.array([[0, 0, 0]]))[0]
    assert not s.contains(np.array([[1, 0, 0]]))[0]
    sdf = SDF.from_mesh(s, 0.02, 0.1)
    q = sdf.query(np.array([[0, 0, 0], [0.5, 0, 0], [1.0, 0, 0]]))
    assert q[0] < -0.45 and abs(q[1]) < 0.02 and abs(q[2] - 0.5) < 0.03


def test_subdivide_and_taubin_keep_validity():
    s = icosphere(3, 1.0).subdivide().taubin_smooth(5)
    assert all(s.is_valid().values())
    assert abs(s.volume() - 4 / 3 * np.pi) < 0.1


def test_config_and_hash():
    cfg = load_config("default")
    assert cfg["world"]["dt"] == 0.001
    q = load_config("quick")
    assert q["experiment"]["n_towers"] == 3 and q["world"]["dt"] == 0.001
    assert config_hash(cfg) != config_hash(q)
    assert merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 5}}) == {"a": {"b": 5, "c": 2}}


def test_rng_streams_are_keyed():
    a = make_rng(0, 1, Stage.SENSOR).random(3)
    b = make_rng(0, 1, Stage.SENSOR).random(3)
    c = make_rng(0, 2, Stage.SENSOR).random(3)
    assert np.allclose(a, b) and not np.allclose(a, c)


def test_mesh_io_roundtrip(tmp_path):
    m = icosphere(2, 0.3)
    m.save(tmp_path / "m.obj")
    m2 = TriMesh.load(tmp_path / "m.obj")
    assert np.allclose(m.vertices, m2.vertices, atol=1e-7) and np.array_equal(m.faces, m2.faces)
    m.save(tmp_path / "m.npz")
    m3 = TriMesh.load(tmp_path / "m.npz")
    assert np.allclose(m.vertices, m3.vertices)
