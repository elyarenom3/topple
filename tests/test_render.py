"""Renderer tests (Sections 10.9 and 12.5)."""

import os
import tempfile

import numpy as np
import pytest

from topple.core.mesh import box, icosphere, TriMesh
from topple.core.raycast import BVH, brute_force_intersect
from topple.core.transforms import SE3, quat_from_axis_angle
from topple.render import Camera, Material, Scene, render
from topple.render.color import D50_XYZ, M_RGB2XYZ, M_XYZ2RGB, aces_rrt_odt, bradford_adapt, srgb_encode, colormap_table, linear_srgb_to_oklab
from topple.render.flip import flip_error
from topple.render.io import read_exr, write_exr
from topple.render.lights import AreaLight, SunLight
from topple.render.sampling import sobol_owen_points
from topple.render import overlays


def _disc(radius, half, n=48):
    """Half disc (angles in [0, pi) if half == 0 else [pi, 2 pi)) in the z = 0 plane."""
    a0 = 0.0 if half == 0 else np.pi
    ang = np.linspace(a0, a0 + np.pi, n)
    V = np.concatenate([[[0, 0, 0.0]], np.stack([radius * np.cos(ang), radius * np.sin(ang), np.zeros(n)], 1)])
    F = np.array([[0, 1 + i, 2 + i] for i in range(n - 1)])
    return TriMesh(V, F, "halfdisc")


def test_furnace_closed_sphere():
    """Inside a closed sphere with albedo a and emission Le the radiance is Le / (1 - a)."""
    sc = Scene()
    sc.add(icosphere(3, 1.0), Material(base_color=(0.5, 0.5, 0.5), roughness=1.0, specular=0.0, emission=(1.0, 1.0, 1.0)))
    cam = Camera(position=[0, 0, 0], look_at=[1, 0, 0], focal_mm=35, f_number=1e9, width=12, height=8)
    out = render(sc, cam, spp=64, max_depth_diffuse=14, max_depth_specular=14, rr_start_depth=5, seed=1)
    mean = out["beauty"].mean()
    assert abs(mean - 2.0) / 2.0 < 0.03, mean


def test_sphere_light_irradiance():
    """A diffuse plane under a spherical emitter: E = pi L (r / d)^2, radiance = albedo E / pi."""
    r, d, L, albedo = 0.2, 0.5, 10.0, 0.5
    sc = Scene()
    sc.add(box((2, 2, 0.02), center=(0, 0, -0.01)), Material(base_color=(albedo,) * 3, roughness=1.0, specular=0.0))
    sc.add(icosphere(4, r), Material(emission=(L, L, L), base_color=(0, 0, 0), specular=0.0), SE3(t=[0, 0, d]))
    cam = Camera(position=[0.6, 0, 0.05], look_at=[0, 0, 0], focal_mm=400, f_number=1e9, width=8, height=8)
    out = render(sc, cam, spp=512, max_depth_diffuse=2, max_depth_specular=2, seed=3)
    expected = albedo * np.pi * L * (r / d) ** 2 / np.pi
    got = out["beauty"][2:6, 2:6, :].mean()
    assert abs(got - expected) / expected < 0.06, (got, expected)


def test_bvh_matches_brute_force():
    m = icosphere(3, 0.5)
    rng = np.random.default_rng(0)
    o = rng.normal(size=(500, 3)) * 2
    dd = rng.normal(size=(500, 3)) * 0.3 - o  # aimed near the sphere
    dd /= np.linalg.norm(dd, axis=1, keepdims=True)
    t, tid, u, v = m.bvh().intersect(o, dd)
    tb, idb = brute_force_intersect(m.vertices, m.faces, o, dd)
    assert np.allclose(np.where(tid >= 0, t, -1.0), tb)
    assert np.all((tid >= 0) == (idb >= 0))
    assert (tid >= 0).sum() > 10


def test_motion_blur_rotating_disc():
    """A half white, half black disc rotating by 90 degrees during the shutter: the time average
    at polar angle phi is piecewise linear in phi."""
    R = 0.2
    sc = Scene()
    white = Material(unlit=True, emission=(1.0, 1.0, 1.0))
    black = Material(unlit=True, emission=(0.0, 0.0, 0.0))
    T1 = SE3(quat_from_axis_angle([0, 0, 1], np.pi / 2))
    sc.add(_disc(R, 0), white, SE3(), T1)
    sc.add(_disc(R, 1), black, SE3(), T1)
    n = 32
    cam = Camera(position=[0, 0, 1.0], look_at=[0, 0, 0], up=[0, 1, 0], ortho=True, ortho_width=0.5, width=n, height=n)
    out = render(sc, cam, spp=256, seed=2)
    img = out["beauty"][:, :, 0]
    err = []
    for py in range(n):
        for px in range(n):
            o, _ = cam.pixel_ray(px + 0.5, py + 0.5)
            x, y = o[0], o[1]
            rad = np.hypot(x, y)
            if rad > 0.17 or rad < 0.02:
                continue
            phi = np.arctan2(y, x) % (2 * np.pi)
            if phi < np.pi / 2:
                frac = 2 * phi / np.pi
            elif phi < np.pi:
                frac = 1.0
            elif phi < 1.5 * np.pi:
                frac = 1 - 2 * (phi - np.pi) / np.pi
            else:
                frac = 0.0
            err.append(img[py, px] - frac)
    err = np.array(err)
    assert len(err) > 100
    assert np.abs(err).mean() < 0.04, np.abs(err).mean()
    assert np.abs(err).max() < 0.2, np.abs(err).max()


def test_overlay_occlusion_and_alpha():
    sc = Scene()
    sc.add(box((0.4, 0.02, 0.4), center=(0, 0.1, 0.0)), Material(base_color=(0.5, 0.5, 0.5)))
    sc.add_light(AreaLight.rect([0, -1, 1], [0, 1, -1], [0, 0, 1], 1, 1, 5.0))
    sc.add_overlay(overlays.sphere([0.0, 0.4, 0.0], 0.05), color=(1, 0, 0), alpha=1.0)  # behind the wall
    sc.add_overlay(overlays.sphere([0.0, -0.4, 0.0], 0.05), color=(0, 1, 0), alpha=0.5)  # in front
    cam = Camera(position=[0, -2, 0], look_at=[0, 0, 0], focal_mm=50, f_number=1e9, width=32, height=32)
    out = render(sc, cam, spp=4, seed=0)
    ov = out["overlay"]
    assert ov[:, :, 3].max() > 0.4
    assert np.all(np.abs(ov[:, :, 0] - ov[:, :, 2]) < 1e-6)  # red overlay fully hidden (outline is grey)
    centre = ov[16, 16]
    assert abs(centre[3] - 0.5) < 0.05 and abs(centre[1] - 0.5) < 0.05
    comp = out["composite"]
    assert comp.shape == (32, 32, 3)


def test_flip_identity_and_shift():
    rng = np.random.default_rng(0)
    img = rng.random((48, 48, 3))
    m0, _ = flip_error(img, img)
    m1, emap = flip_error(img, np.roll(img, 3, axis=1))
    assert m0 == 0.0
    assert m1 > 0.05
    assert emap.shape == (48, 48) and emap.min() >= 0 and emap.max() <= 1


def test_exr_round_trip():
    rng = np.random.default_rng(1)
    passes = {"beauty": rng.random((9, 7, 3)).astype(np.float32) * 10, "depth": rng.random((9, 7)).astype(np.float32), "overlay": rng.random((9, 7, 4)).astype(np.float32)}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.exr")
        write_exr(p, passes, half=True)
        r = read_exr(p)
        for k in passes:
            assert r[k].shape == passes[k].shape
            assert np.abs(r[k] - passes[k]).max() < 1e-2 * max(1.0, passes[k].max())
        write_exr(p, passes, half=False)
        r = read_exr(p)
        assert all(np.abs(r[k] - passes[k]).max() == 0 for k in passes)


def test_aces_monotone_and_mid_grey():
    x = np.linspace(0, 16, 500)
    y = aces_rrt_odt(np.stack([x, x, x], -1))[:, 1]
    assert np.all(np.diff(y) >= -1e-9)
    grey = srgb_encode(aces_rrt_odt(np.array([0.18, 0.18, 0.18])))
    assert 0.3 < grey[0] < 0.5
    assert y[-1] <= 1.0 and y[0] >= 0.0


def test_bradford_adapts_white():
    d50_rgb = D50_XYZ @ M_XYZ2RGB.T
    adapted = bradford_adapt(d50_rgb, D50_XYZ)
    assert np.allclose(adapted, [1, 1, 1], atol=2e-3)


def test_dof_limits_textbook():
    cam = Camera(position=[0, -3, 0], look_at=[0, 0, 0], focal_mm=50, f_number=8, focus_distance=3.0, circle_of_confusion_mm=0.03)
    H, near, far = cam.dof_limits()
    assert abs(H - 10.4667) < 0.01
    assert abs(near - 2.3405) < 0.005
    assert abs(far - 4.1774) < 0.005


def test_sampler_stratified():
    p = sobol_owen_points(256, 6, 5)
    for d in range(6):
        h, _ = np.histogram(p[:, d], bins=256, range=(0, 1))
        assert h.max() == 1
    h, _, _ = np.histogram2d(p[:, 0], p[:, 1], bins=16, range=[[0, 1], [0, 1]])
    assert h.max() == 1 and h.min() == 1


def test_colormaps_perceptual_lightness():
    L = linear_srgb_to_oklab(colormap_table("sequential"))[:, 0]
    assert np.all(np.diff(L) > 0)
    Ld = linear_srgb_to_oklab(colormap_table("diverging"))[:, 0]
    assert Ld[32] > Ld[0] and Ld[32] > Ld[-1]


def test_sun_and_sky_render_finite():
    sc = Scene()
    sc.add(box((2, 2, 0.02), center=(0, 0, -0.01)), Material(base_color=(0.4, 0.4, 0.4)))
    sc.add(icosphere(2, 0.1), Material(base_color=(0.6, 0.3, 0.2), coat=0.5), SE3(t=[0, 0, 0.1]))
    from topple.render.presets import lighting_preset

    lights, sky, bg = lighting_preset("dusk")
    for l in lights:
        sc.add_light(l)
    sc.set_sky(sky)
    cam = Camera(position=[0.6, -0.6, 0.3], look_at=[0, 0, 0.1], focal_mm=50, f_number=4, width=24, height=16)
    out = render(sc, cam, spp=8, seed=0)
    assert np.isfinite(out["beauty"]).all() and out["beauty"].max() > 0
    assert out["depth"][8, 12] > 0


# ------------------------------------------------------------------ Hošek Wilkie sky

def test_hosek_wilkie_kernel_matches_the_reference_formula():
    from topple.render.sky import _M, hosek_wilkie_configuration, hosek_wilkie_xyz, sky_params, sky_radiance

    for T, a, el in ((3.0, 0.1, 30.0), (2.0, 0.0, 10.0), (6.5, 0.3, 55.0), (10.0, 1.0, 5.0)):
        p = sky_params(el, T, 0.0, 1.0, model="hosek", albedo=a)
        cfg, L = hosek_wilkie_configuration(T, a, np.radians(el))
        sun = p[0:3]
        for th, ph in ((0.1, 0.5), (0.8, 0.3), (1.4, 1.2), (1.2, 2.5)):
            d = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
            gamma = np.arccos(np.clip(d @ sun, -1, 1))
            ref = _M @ (hosek_wilkie_xyz(th, gamma, cfg, L) * 6.83)
            got = np.array(sky_radiance(d[0], d[1], d[2], p, _M))
            np.testing.assert_allclose(got, np.maximum(ref, 0), rtol=1e-9, atol=1e-9)


def test_hosek_wilkie_is_continuous_in_turbidity_and_bright_around_the_sun():
    from topple.render.sky import _M, sky_params, sky_radiance

    d = (0.3, 0.2, np.sqrt(1 - 0.13))
    a = np.array(sky_radiance(*d, sky_params(30.0, 3.9999, model="hosek"), _M))
    b = np.array(sky_radiance(*d, sky_params(30.0, 4.0, model="hosek"), _M))
    np.testing.assert_allclose(a, b, rtol=1e-3)
    p = sky_params(30.0, 3.0, model="hosek")
    el = np.radians(30.0)

    def lum(gamma_deg):
        e = el + np.radians(gamma_deg)
        return float(np.array(sky_radiance(np.cos(e), 0.0, np.sin(e), p, _M)) @ [0.2126, 0.7152, 0.0722])

    assert lum(3.0) > lum(10.0) > lum(30.0)
    # clear sky: blue zenith
    z = sky_radiance(0.0, 0.0, 1.0, p, _M)
    assert z[2] > z[0]


# ------------------------------------------------------------------ traced lens (Kolb et al. 1995)

def test_double_gauss_paraxial_focal_length_and_focus():
    from topple.render.lens import LensSystem

    lens = LensSystem()
    assert abs(lens.effective_focal_length() - 0.050) < 0.001
    for s in (5.0, 1.0, 0.4):
        d = lens.focus(s)
        assert abs(lens.object_distance(d) - s) / s < 1e-6


def test_traced_lens_matches_thin_lens_exposure_in_the_furnace():
    """Inside an emitting sphere every ray sees the same radiance: the traced lens weight is
    normalised to the thin lens at the image centre and falls off towards the corners."""
    sc = Scene()
    sc.add(icosphere(3, 1.0), Material(base_color=(0.0, 0.0, 0.0), roughness=1.0, specular=0.0, emission=(1.0, 1.0, 1.0)))
    cam = Camera(position=[0, 0, 0], look_at=[1, 0, 0], focal_mm=50, f_number=2.8, focus_distance=0.8, width=24, height=16, lens="dgauss50")
    out = render(sc, cam, spp=256, max_depth_diffuse=1, max_depth_specular=1, seed=4)
    b = out["beauty"][..., 1]
    centre = b[6:10, 10:14].mean()
    corner = np.mean([b[0, 0], b[0, -1], b[-1, 0], b[-1, -1]])
    assert abs(centre - 1.0) < 0.05, centre
    assert 0.2 < corner < 0.8 * centre, corner


# ------------------------------------------------------------------ modal sound

def test_surface_modes_are_mass_orthonormal_and_scale_with_size():
    from topple.render.sound import surface_modes

    m = icosphere(2, 0.05).scaled([1.0, 0.8, 0.5])
    a = surface_modes(m, 1.0, n_modes=10)
    b = surface_modes(m.scaled([2.0, 2.0, 2.0]), 8.0, n_modes=10)
    assert np.all(np.diff(a["freq"]) >= 0) and a["freq"][0] > 1000.0
    np.testing.assert_allclose(b["freq"] / a["freq"], 0.5, rtol=1e-6)
    # mass normalised: phi^T M phi = I (lumped masses from vertex areas)
    S = a["shapes"].reshape(len(a["freq"]), -1)
    Mdiag = np.repeat(a["vertex_mass"], 3)
    np.testing.assert_allclose(S @ (Mdiag[:, None] * S.T), np.eye(len(S)), atol=1e-8)


def test_impact_sound_peaks_at_a_modal_frequency(tmp_path):
    from scipy.io import wavfile

    from topple.core.transforms import SE3
    from topple.render.sound import surface_modes, synthesize, write_wav

    m = icosphere(2, 0.08)
    md = surface_modes(m, 5.0, n_modes=8)
    y = synthesize([(0.01, 0, np.array([0.0, 0.0, 0.05]))], {0: md}, {0: lambda t: SE3()}, 0.3, pulse_gain=0.0)
    spec = np.abs(np.fft.rfft(y))
    f = np.fft.rfftfreq(len(y), 1.0 / 48000)
    fpk = f[np.argmax(spec)]
    audible = md["freq"][md["freq"] < 0.45 * 48000]
    assert np.min(np.abs(audible - fpk)) < 20.0
    write_wav(tmp_path / "a.wav", y)
    sr, data = wavfile.read(tmp_path / "a.wav")
    assert sr == 48000 and len(data) == len(y)


# ------------------------------------------------------------------ spectral mode, dielectric, MNEE

def test_spectral_furnace_equals_rgb():
    """Grey furnace: the spectral estimate equals the RGB one (energy conservation is exact for a
    white or grey reflectance), and a coloured emitter renders as its own RGB."""
    sc = Scene()
    sc.add(icosphere(3, 1.0), Material(base_color=(0.5, 0.5, 0.5), roughness=1.0, specular=0.0, emission=(1.0, 1.0, 1.0)))
    cam = Camera(position=[0, 0, 0], look_at=[1, 0, 0], focal_mm=35, f_number=1e9, width=12, height=8)
    out = render(sc, cam, spp=128, max_depth_diffuse=14, max_depth_specular=14, rr_start_depth=5, seed=1, spectral=True)
    np.testing.assert_allclose(out["beauty"].reshape(-1, 3).mean(axis=0), 2.0, rtol=0.03)
    sc = Scene()
    c = (0.8, 0.4, 0.2)
    sc.add(icosphere(3, 1.0), Material(base_color=(0.0, 0.0, 0.0), roughness=1.0, specular=0.0, emission=c))
    out = render(sc, cam, spp=256, max_depth_diffuse=2, max_depth_specular=2, seed=2, spectral=True)
    np.testing.assert_allclose(out["beauty"].reshape(-1, 3).mean(axis=0), c, rtol=0.03)


@pytest.mark.parametrize("spectral", [False, True])
def test_dielectric_sphere_in_a_furnace_is_invisible(spectral):
    """A clear (dispersive) dielectric sphere inside a uniformly emitting enclosure: refraction,
    Fresnel, the radiance scaling across the interface and the hero wavelength collapse keep the
    radiance at the enclosure value."""
    sc = Scene()
    sc.add(icosphere(3, 2.0), Material(base_color=(0.0, 0.0, 0.0), roughness=1.0, specular=0.0, emission=(1.0, 1.0, 1.0)))
    sc.add(icosphere(3, 0.4), Material(dielectric=True, ior=1.5, cauchy_b_um2=0.0042), SE3(t=[1.0, 0.0, 0.0]))
    cam = Camera(position=[-0.5, 0, 0], look_at=[1, 0, 0], focal_mm=50, f_number=1e9, width=12, height=8)
    out = render(sc, cam, spp=256, max_depth_diffuse=4, max_depth_specular=64, rr_start_depth=32, seed=3, spectral=spectral)
    np.testing.assert_allclose(out["beauty"].reshape(-1, 3).mean(axis=0), 1.0, rtol=0.04)


def _water_scene(bump_amp=0.0, elevation=45.0):
    from topple.core.mesh import plane
    from topple.render.lights import SunLight

    sc = Scene()
    a = 0.5
    sc.add(box((4, 4, 0.1), center=(0, 0, -0.05)), Material(base_color=(a, a, a), roughness=1.0, specular=0.0))
    sc.add(plane(4.0, 0.05), Material(dielectric=True, ior=1.333, bump_amp=bump_amp, bump_scale=40.0))
    sun = SunLight.from_elevation(elevation, 0.0, radiance=np.full(3, 1.0e4))
    sc.add_light(sun)
    cam = Camera(position=[0.0, 0.0, 1.0], look_at=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0], ortho=True, ortho_width=0.4, width=16, height=16)
    return sc, cam, sun, a


def test_mnee_matches_the_analytic_irradiance_under_flat_water():
    from topple.render.materials import fresnel_dielectric

    sc, cam, sun, a = _water_scene(0.0, 45.0)
    out = render(sc, cam, spp=64, max_depth_diffuse=1, max_depth_specular=8, seed=5, mnee=True)
    n = 1.333
    E_perp = float(sun.irradiance()[0])
    cos_s = np.sin(np.radians(45.0))
    expected = (1.0 - fresnel_dielectric(1.0, n)) / n ** 2 * a / np.pi * E_perp * cos_s * (1.0 - fresnel_dielectric(cos_s, n))
    got = out["beauty"][..., 1].mean()
    assert abs(got - expected) / expected < 0.03, (got, expected)


def test_mnee_caustics_conserve_the_mean_irradiance():
    """Ripples focus the sun into caustics; averaged over the floor the irradiance stays that of
    flat water (to the small change of the Fresnel transmittance)."""
    res = {}
    for amp in (0.0, 0.15):  # 0.15: the ripple strength of the river preset
        sc, cam, _, _ = _water_scene(amp, 60.0)
        cam.ortho_width = 0.1
        cam.width = cam.height = 32
        res[amp] = render(sc, cam, spp=32, max_depth_diffuse=1, max_depth_specular=8, seed=6, mnee=True)["beauty"][..., 1]
    flat, rip = res[0.0], res[0.15]
    assert rip.std() > 0.3 * rip.mean() and rip.max() > 2.5 * rip.mean()  # caustic pattern
    assert abs(rip.mean() - flat.mean()) / flat.mean() < 0.06
