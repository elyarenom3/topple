"""Perception tests (Section 12.4)."""

import numpy as np
import pytest

from topple.core.config import load_config
from topple.core.mesh import TriMesh, box, icosphere
from topple.core.transforms import SE3, quat_from_rotvec
from topple.perception import (
    DepthSensor,
    Estimator,
    FrictionPrior,
    GridPosterior,
    PinholeCamera,
    TSDFVolume,
    first_order,
    hemisphere_views,
    icp_point_to_plane,
    in_hand_views,
    marching_cubes,
    monte_carlo,
    simulate_slip_angle,
    unscented,
)
from topple.perception.camera import look_at
from topple.perception.posterior import ShapePosterior
from topple.perception.propagate import finite_difference_gradient
from topple.stonegen import StoneGenerator


def _stone_scene(rng, stone_class="rounded_cobble"):
    cfg = load_config("quick")
    g = StoneGenerator(cfg)
    spec = g.sample(rng, stone_class)
    mesh = spec.mesh.translated([0, 0, 0.03 - spec.mesh.bounds()[0][2]])
    ground = box((1.0, 1.0, 0.02), center=(0, 0, -0.01))
    return cfg, spec, mesh, ground


def _images(cfg, mesh, ground, rng, n_views, model="none", pose_noise=False, width=120, height=90, in_hand=2):
    cam = PinholeCamera.from_fov(width, height, 58)
    scfg = dict(cfg["sensor"])
    scfg["model"] = model
    if not pose_noise:
        scfg["pose_noise_rot_deg"] = 0.0
        scfg["pose_noise_trans_m"] = 0.0
    sensor = DepthSensor(cam, scfg, rng)
    c = mesh.centroid()
    imgs = [sensor.render_depth([mesh, ground], T, ids=[7, 0]) for T in hemisphere_views(c, 0.45, n_views, rng)]
    # in hand scanning: the lifted stone is the only object in front of the camera (fingers masked out)
    imgs += [sensor.render_depth([mesh], T, ids=[7]) for T in in_hand_views(c, 0.45, in_hand, rng)]
    return imgs, sensor


def test_camera_roundtrip():
    cam = PinholeCamera.from_fov(64, 48, 60)
    rng = np.random.default_rng(0)
    P = np.stack([rng.uniform(-0.2, 0.2, 50), rng.uniform(-0.2, 0.2, 50), rng.uniform(0.3, 1.0, 50)], 1)
    u, v = cam.project(P)
    Q = cam.backproject(u, v, P[:, 2])
    assert np.allclose(P, Q)
    T = look_at([1.0, 0.0, 0.5], [0.0, 0.0, 0.0])
    o, d = cam.rays(T)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)
    # the optical axis points at the target
    centre = d.reshape(48, 64, 3)[24, 32]
    assert np.dot(centre, np.array([-1.0, 0.0, -0.5]) / np.sqrt(1.25)) > 0.999


def test_depth_sensor_exact_and_noise():
    rng = np.random.default_rng(0)
    cfg, spec, mesh, ground = _stone_scene(rng)
    imgs, sensor = _images(cfg, mesh, ground, rng, 1, "none", in_hand=0)
    im = imgs[0]
    m = im.valid & (im.object_id == 7)
    assert m.sum() > 50
    P = im.points(True, m)
    d, _, _ = mesh.bvh().closest_points(P)
    assert d.max() < 1e-9  # exact depth lands on the surface
    imgs_n, _ = _images(cfg, mesh, ground, rng, 1, "nguyen", in_hand=0)
    imn = imgs_n[0]
    mn = imn.valid & (imn.object_id == 7)
    Pn = imn.points(True, mn)
    dn, _, _ = mesh.bvh().closest_points(Pn)
    assert 0.0005 < np.median(dn) < 0.01  # noisy but close
    assert np.all(imn.sigma[imn.valid] > 0)
    assert imn.valid.sum() <= im.valid.sum()  # shadows and grazing pixels drop out


def test_marching_cubes_sphere_and_skimage():
    r, s = 0.05, 0.0025
    xs = np.arange(-0.065, 0.0651, s)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    D = np.sqrt(X * X + Y * Y + Z * Z) - r
    m = marching_cubes(D, (xs[0],) * 3, s)
    v_true = 4 / 3 * np.pi * r**3
    assert abs(m.volume() / v_true - 1) < 0.01
    assert m.is_watertight() and m.is_manifold()
    skimage = pytest.importorskip("skimage")
    from skimage import measure

    v, f, _, _ = measure.marching_cubes(D, level=0.0, spacing=(s, s, s))
    ms = TriMesh(v.astype(np.float64) + xs[0], f.astype(np.int64))
    assert abs(abs(ms.volume()) / m.volume() - 1) < 0.01
    d, _, _ = ms.bvh().closest_points(m.vertices)
    assert d.max() < 0.5 * s


def test_icp_recovers_perturbation():
    rng = np.random.default_rng(0)
    s = icosphere(4, 0.05)
    pts, nrm = s.sample_surface(4000, rng)
    T_true = SE3(quat_from_rotvec([0.02, -0.01, 0.03]), [0.003, -0.002, 0.001])
    src = T_true.inverse().apply(pts[:1500])
    T, rms = icp_point_to_plane(src, pts, nrm, SE3(), iters=20)
    assert np.allclose(T.t, T_true.t, atol=2e-4)
    assert np.dot(T.q, T_true.q) > 0.99999
    assert rms < 1e-4


def test_noise_free_depth_recovers_mesh_below_voxel():
    rng = np.random.default_rng(1)
    cfg, spec, mesh, ground = _stone_scene(rng, "rounded_cobble")
    imgs, _ = _images(cfg, mesh, ground, rng, 10, "none", in_hand=3)
    est = Estimator(cfg, rng)
    est.icp_iters = 0
    E = est.estimate(imgs, 4, target_id=7, decimate_mean_to=None)
    voxel = E.meta["voxel"]
    d, _, _ = mesh.bvh().closest_points(E.mean_mesh.vertices)
    assert np.median(d) < voxel
    assert abs(E.mean_mesh.volume() / spec.volume - 1) < 0.15
    assert np.linalg.norm(E.mean_com - mesh.centroid()) < 2 * voxel


def test_errors_fall_with_view_count():
    rng = np.random.default_rng(2)
    cfg, spec, mesh, ground = _stone_scene(rng, "rounded_cobble")
    errs = {}
    for n in (2, 10):
        imgs, _ = _images(cfg, mesh, ground, np.random.default_rng(5), n, "none", in_hand=0 if n == 2 else 3)
        est = Estimator(cfg, np.random.default_rng(3))
        est.icp_iters = 0
        E = est.estimate(imgs, 4, target_id=7, decimate_mean_to=None)
        errs[n] = (abs(E.mean_mesh.volume() - spec.volume), np.linalg.norm(E.mean_com - mesh.centroid()))
    assert errs[10][0] < errs[2][0]
    assert errs[10][1] < errs[2][1]


def test_com_posterior_simulation_based_calibration():
    """Rank of the true centre of mass height among posterior samples should spread over [0, 1]."""
    ranks = []
    for trial in range(8):
        rng = np.random.default_rng(100 + trial)
        cfg, spec, mesh, ground = _stone_scene(rng, "rounded_cobble")
        imgs, _ = _images(cfg, mesh, ground, rng, 6, "nguyen", pose_noise=True, in_hand=2)
        est = Estimator(cfg, rng)
        E = est.estimate(imgs, 12, target_id=7, decimate_mean_to=None)
        com_true = mesh.centroid()
        zs = np.array([(s.pose_offset.apply(s.com))[2] for s in E.samples])
        ranks.append(np.mean(zs < com_true[2]))
    ranks = np.array(ranks)
    # not degenerate at either end and roughly centred
    assert ranks.min() < 0.9 and ranks.max() > 0.1
    assert 0.2 < ranks.mean() < 0.8


def test_shape_posterior_samples_closed_and_varied():
    rng = np.random.default_rng(4)
    cfg, spec, mesh, ground = _stone_scene(rng, "flat_river")
    imgs, _ = _images(cfg, mesh, ground, rng, 5, "none", in_hand=0)  # underside unobserved
    est = Estimator(cfg, rng)
    est.icp_iters = 0
    for comp in ("hull", "mirror", "gp", "poisson"):
        E = est.estimate(imgs, 4, target_id=7, decimate_mean_to=None, completion=comp)
        assert E.mean_mesh.is_watertight(), comp
        vols = [s.mesh.volume() for s in E.samples]
        assert all(s.mesh.is_watertight() for s in E.samples), comp
        assert np.std(vols) > 0, comp
        post = E.meta["posterior"]
        assert np.nanmax(post.std_field[~post.obs]) > np.nanmin(post.std_field[post.obs])  # unobserved regions are more uncertain
    modes, sig = post.shape_modes(5, 16)
    assert modes.shape[0] == 5 and np.all(sig[:-1] >= sig[1:])


def test_friction_grid_posterior_concentrates():
    rng = np.random.default_rng(0)
    prior = FrictionPrior(0.6, 0.25)
    mu_true = 0.85
    gp = GridPosterior(prior)
    s0 = gp.std()
    for _ in range(20):
        gp.update(simulate_slip_angle(mu_true, rng, 1.0), 1.0)
    assert gp.std() < 0.2 * s0
    assert abs(gp.mean() - mu_true) < 0.03
    samples = prior.sample(rng, 20000)
    assert abs(np.median(samples) - 0.6) < 0.02


def test_propagation_methods_agree_on_quadratic():
    rng = np.random.default_rng(0)
    A = np.array([[2.0, 0.3, 0.0], [0.3, 1.0, 0.1], [0.0, 0.1, 0.5]])
    b = np.array([1.0, -2.0, 0.5])

    def fn(x):
        x = np.asarray(x)
        return float(b @ x + 0.5 * x @ A @ x)

    mean = np.array([0.2, -0.1, 0.3])
    cov = np.diag([0.01, 0.02, 0.005])
    samples = rng.multivariate_normal(mean, cov, size=40000)
    mc = monte_carlo(fn, samples)
    mu_ut, var_ut, _, _ = unscented(fn, mean, cov)
    mu_fo, var_fo = first_order(fn, lambda x: finite_difference_gradient(fn, x), mean, cov)
    assert abs(mu_ut - mc.mean()) < 0.01 * max(1, abs(mc.mean()))
    assert abs(var_ut - mc.var()) < 0.1 * mc.var()
    assert abs(var_fo - mc.var()) < 0.15 * mc.var()  # first order ignores the quadratic term in the variance


def test_tsdf_integrate_updates_weights():
    rng = np.random.default_rng(0)
    cfg, spec, mesh, ground = _stone_scene(rng)
    imgs, _ = _images(cfg, mesh, ground, rng, 2, "none", in_hand=0)
    lo, hi = mesh.bounds()
    t = TSDFVolume.from_bounds(lo, hi, 0.004, 0.008)
    n = t.integrate(imgs[0], mask=imgs[0].valid & (imgs[0].object_id == 7))
    assert n > 0 and t.observed().sum() == n
    m = t.extract_mesh()
    d, _, _ = mesh.bvh().closest_points(m.vertices)
    assert np.median(d) < 0.004


def test_underside_modes_follow_the_placed_orientation():
    from topple.core.mesh import icosphere
    from topple.perception.estimate_view import underside_modes

    m = icosphere(2, 0.05)
    n = m.vertex_normals()
    bottom = np.where(n[:, 2] < -0.5, 1.0, 0.0)
    top = np.where(n[:, 2] > 0.5, 1.0, 0.0)
    modes = np.stack([bottom[:, None] * n, top[:, None] * n, np.ones(len(n))[:, None] * n])
    assert underside_modes(m, modes, np.eye(3)).tolist() == [True, False, False]
    flip = np.diag([1.0, -1.0, -1.0])  # upside down: the top mode now faces the support
    assert underside_modes(m, modes, flip).tolist() == [False, True, False]


def test_screened_poisson_sphere_volume_within_one_percent():
    from topple.core.mesh import icosphere
    from topple.perception.marching_cubes import largest_component, marching_cubes
    from topple.perception.poisson import screened_poisson

    r, h = 0.05, 0.003
    m = icosphere(4, r)
    origin = np.full(3, -0.08)
    shape = (int(0.16 / h) + 1,) * 3
    phi = screened_poisson(m.vertices, m.vertex_normals(), origin, h, shape, tol=1e-5)
    surf = largest_component(marching_cubes(phi, origin, h, 0.0))
    assert surf.is_watertight()
    assert abs(surf.volume() / (4.0 / 3.0 * np.pi * r ** 3) - 1.0) < 0.01
    # positive outside, negative inside
    assert phi[0, 0, 0] > 0 and phi[tuple(s // 2 for s in shape)] < 0
