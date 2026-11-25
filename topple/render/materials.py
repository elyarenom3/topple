"""Stone materials and the BSDF kernels (Section 10.3).

Base model: a diffuse lobe (Lambert, or Oren and Nayar 1994 for rough surfaces) under a GGX
microfacet lobe with Smith height correlated masking (Walter et al. 2007, Heitz 2014) and the
Kulla and Conty (2017) multiple scattering compensation from precomputed directional albedo
tables. Parameters follow the principled convention of Burley (2012): base colour, roughness,
metallic, specular, anisotropy.

Wet stone (Lekner and Dorf 1988): a smooth dielectric coating of index 1.33. Light that enters
the film is scattered by the base with albedo a, and only the fraction leaving within the
critical angle exits; the rest returns to the surface, which raises the number of absorbing
bounces. The wet diffuse lobe is a (1 - F_i)(1 - F_o) / (pi n^2 (1 - a r_i)) with r_i the
hemispherical internal reflectance of the water air interface.

Sparkle: a hashed stochastic micro flake layer. Each small object space cell either holds a
flake with a random tilt or none; a flake adds a sharp GGX lobe about its own normal, so the low
sun produces per pixel glints that move with the viewpoint. This is a simplified form of the
discrete stochastic microfacet model of Jakob et al. (2014).

Subsurface scattering: a random walk inside the mesh with exponential free flights and
isotropic scattering (a simplified form of Wrenninge, Villemin and Hery 2017, whose anisotropic
phase function and non exponential flights are not implemented).

The material table has one row per material with the columns listed below; the kernels read it
through the ``M_*`` constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numba import njit

from .textures import banding, fbm3, fbm3_grad, hash01, triplanar, worley3

# ----------------------------------------------------------------- table layout
M_TYPE = 0  # 0 principled, 1 unlit (overlay), 2 glass shell (overlay), 3 emitter (light geometry), 4 dielectric
M_BASE = 1
M_ROUGH = 4
M_METAL = 5
M_SPEC = 6
M_ANISO = 7
M_COAT = 8
M_COAT_IOR = 9
M_FLAKE_DENS = 10
M_FLAKE_INT = 11
M_SSS_MFP = 12
M_SSS_W = 15
M_EMIT = 16
M_FBM_AMP = 19
M_FBM_SCALE = 20
M_WORLEY_AMP = 21
M_WORLEY_SCALE = 22
M_BAND_AMP = 23
M_BAND_FREQ = 24
M_BAND_DIR = 25
M_LICHEN_THR = 28
M_LICHEN_SCALE = 29
M_WEATHER_AMP = 30
M_BUMP_AMP = 31
M_BUMP_SCALE = 32
M_ALPHA = 33
M_BLEND = 34
M_SEED = 35
M_USE_VCOL = 36
M_DIFFUSE_MODEL = 37
M_ANISO_DIR = 38
M_OUTLINE = 41
M_FLAKE_SIZE = 42
M_ROUGH_VAR = 43
M_WET_MASK_SCALE = 44
M_LIGHT = 45
M_IOR = 48  # dielectric: refractive index at 589 nm
M_CAUCHY_B = 49  # dielectric: Cauchy B coefficient (um^2), n(lambda) = A + B / lambda^2
M_ABSORB = 50  # dielectric: absorption coefficient per channel (1/m), 50..52
M_NCOLS = 53

# shading parameter vector produced by shade_material
S_BASE = 0
S_ROUGH = 3
S_METAL = 4
S_SPEC = 5
S_AX = 6
S_AY = 7
S_COAT = 8
S_COAT_IOR = 9
S_FLAKE_INT = 10
S_FLAKE_ON = 11
S_FLAKE_N = 12  # 12..14 local flake normal
S_SSS_MFP = 15
S_SSS_W = 18
S_EMIT = 19
S_DIFFUSE_MODEL = 22
S_N = 24  # 24..26 perturbed shading normal (world)
S_NCOLS = 28


@dataclass
class Material:
    name: str = "grey"
    base_color: tuple = (0.5, 0.5, 0.5)
    roughness: float = 0.5
    metallic: float = 0.0
    specular: float = 0.5
    anisotropy: float = 0.0
    anisotropy_dir: tuple = (1.0, 0.0, 0.0)
    coat: float = 0.0
    coat_ior: float = 1.33
    flakes_density: float = 0.0
    flakes_intensity: float = 0.0
    flakes_size: float = 0.0005
    subsurface_mfp: tuple = (0.0, 0.0, 0.0)
    subsurface_weight: float = 0.0
    emission: tuple = (0.0, 0.0, 0.0)
    diffuse_model: int = 0  # 0 Lambert, 1 Oren Nayar
    fbm_amp: float = 0.0
    fbm_scale: float = 30.0
    roughness_variation: float = 0.0
    worley_amp: float = 0.0
    worley_scale: float = 200.0
    banding_amp: float = 0.0
    banding_freq: float = 40.0
    banding_dir: tuple = (0.0, 0.0, 1.0)
    lichen_threshold: float = 0.0
    lichen_scale: float = 15.0
    weathering_amp: float = 0.0
    bump_amp: float = 0.0
    bump_scale: float = 2000.0
    wet_mask_scale: float = 0.0
    alpha: float = 1.0
    blend: int = 0  # overlays: 0 over, 1 add
    unlit: bool = False
    glass_shell: bool = False
    emitter: bool = False
    outline: float = 0.0  # overlays: |n.v| below which the outline colour is used
    use_vertex_colors: bool = False
    seed: int = 1
    # smooth dielectric (water, quartz): refraction with Fresnel, Beer absorption inside, and
    # Cauchy dispersion in the spectral mode; the bump parameters perturb its normal (ripples)
    dielectric: bool = False
    ior: float = 1.333
    cauchy_b_um2: float = 0.0
    absorption: tuple = (0.0, 0.0, 0.0)

    def row(self) -> np.ndarray:
        r = np.zeros(M_NCOLS)
        r[M_TYPE] = 4 if self.dielectric else (3 if self.emitter else (2 if self.glass_shell else (1 if self.unlit else 0)))
        r[M_BASE : M_BASE + 3] = self.base_color
        r[M_ROUGH] = self.roughness
        r[M_METAL] = self.metallic
        r[M_SPEC] = self.specular
        r[M_ANISO] = self.anisotropy
        r[M_COAT] = self.coat
        r[M_COAT_IOR] = self.coat_ior
        r[M_FLAKE_DENS] = self.flakes_density
        r[M_FLAKE_INT] = self.flakes_intensity
        r[M_SSS_MFP : M_SSS_MFP + 3] = self.subsurface_mfp
        r[M_SSS_W] = self.subsurface_weight
        r[M_EMIT : M_EMIT + 3] = self.emission
        r[M_FBM_AMP] = self.fbm_amp
        r[M_FBM_SCALE] = self.fbm_scale
        r[M_WORLEY_AMP] = self.worley_amp
        r[M_WORLEY_SCALE] = self.worley_scale
        r[M_BAND_AMP] = self.banding_amp
        r[M_BAND_FREQ] = self.banding_freq
        d = np.asarray(self.banding_dir, dtype=np.float64)
        r[M_BAND_DIR : M_BAND_DIR + 3] = d / max(np.linalg.norm(d), 1e-12)
        r[M_LICHEN_THR] = self.lichen_threshold
        r[M_LICHEN_SCALE] = self.lichen_scale
        r[M_WEATHER_AMP] = self.weathering_amp
        r[M_BUMP_AMP] = self.bump_amp
        r[M_BUMP_SCALE] = self.bump_scale
        r[M_ALPHA] = self.alpha
        r[M_BLEND] = self.blend
        r[M_SEED] = self.seed
        r[M_USE_VCOL] = 1.0 if self.use_vertex_colors else 0.0
        r[M_DIFFUSE_MODEL] = self.diffuse_model
        a = np.asarray(self.anisotropy_dir, dtype=np.float64)
        r[M_ANISO_DIR : M_ANISO_DIR + 3] = a / max(np.linalg.norm(a), 1e-12)
        r[M_OUTLINE] = self.outline
        r[M_FLAKE_SIZE] = self.flakes_size
        r[M_ROUGH_VAR] = self.roughness_variation
        r[M_WET_MASK_SCALE] = self.wet_mask_scale
        r[M_LIGHT] = -1
        r[M_IOR] = self.ior
        r[M_CAUCHY_B] = self.cauchy_b_um2
        r[M_ABSORB: M_ABSORB + 3] = self.absorption
        return r


# ------------------------------------------------------- Kulla Conty tables


def _kulla_conty_tables(n_mu: int = 32, n_alpha: int = 32, n_samples: int = 4096, seed: int = 0):
    """Directional albedo E(mu, alpha) of the GGX lobe with F = 1 and its cosine average."""
    rng = np.random.default_rng(seed)
    mus = (np.arange(n_mu) + 0.5) / n_mu
    alphas = np.maximum((np.arange(n_alpha) + 0.5) / n_alpha, 0.02) ** 2 * 0 + ((np.arange(n_alpha) + 0.5) / n_alpha)
    E = np.zeros((n_mu, n_alpha))
    u1 = rng.random(n_samples)
    u2 = rng.random(n_samples)
    for j, a in enumerate(alphas):
        a = max(a, 0.01)
        for i, mu in enumerate(mus):
            wo = np.array([np.sqrt(1 - mu * mu), 0.0, mu])
            # VNDF sampling (Heitz 2018), isotropic alpha
            Vh = np.array([a * wo[0], a * wo[1], wo[2]])
            Vh /= np.linalg.norm(Vh)
            lensq = Vh[0] ** 2 + Vh[1] ** 2
            T1 = np.array([-Vh[1], Vh[0], 0.0]) / np.sqrt(lensq) if lensq > 0 else np.array([1.0, 0, 0])
            T2 = np.cross(Vh, T1)
            r = np.sqrt(u1)
            phi = 2 * np.pi * u2
            t1 = r * np.cos(phi)
            t2 = r * np.sin(phi)
            s = 0.5 * (1 + Vh[2])
            t2 = (1 - s) * np.sqrt(np.maximum(0, 1 - t1 ** 2)) + s * t2
            Nh = t1[:, None] * T1 + t2[:, None] * T2 + np.sqrt(np.maximum(0, 1 - t1 ** 2 - t2 ** 2))[:, None] * Vh
            h = np.stack([a * Nh[:, 0], a * Nh[:, 1], np.maximum(0, Nh[:, 2])], 1)
            h /= np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)
            wi = 2 * (h @ wo)[:, None] * h - wo
            valid = wi[:, 2] > 0
            # weight = G2 / G1(wo)
            def lam(w):
                return (-1 + np.sqrt(1 + (a * a * (w[:, 0] ** 2 + w[:, 1] ** 2)) / np.maximum(w[:, 2] ** 2, 1e-12))) / 2

            lo = lam(wo[None, :])[0]
            li = lam(wi)
            wgt = np.where(valid, (1 + lo) / (1 + lo + li), 0.0)
            E[i, j] = wgt.mean()
    Eavg = (E * (2 * mus[:, None]) / n_mu).sum(axis=0)
    return E, Eavg


_KC_PATH = Path(__file__).parent / "data" / "kulla_conty.npz"
_KC = None


def kulla_conty_tables():
    global _KC
    if _KC is None:
        if _KC_PATH.exists():
            d = np.load(_KC_PATH)
            _KC = (d["E"], d["Eavg"])
        else:
            E, Eavg = _kulla_conty_tables()
            try:
                _KC_PATH.parent.mkdir(parents=True, exist_ok=True)
                np.savez(_KC_PATH, E=E, Eavg=Eavg)
            except OSError:
                pass
            _KC = (E, Eavg)
    return _KC


# ------------------------------------------------------------- numba helpers
PI = np.pi
R_INTERNAL_WATER = 0.472  # hemispherical internal reflectance for n = 1.33


@njit(cache=True, inline="always")
def luminance3(r, g, b):
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


@njit(cache=True, inline="always")
def lerp(a, b, t):
    return a + (b - a) * t


@njit(cache=True)
def fresnel_dielectric(cos_i, eta):
    """Unpolarised Fresnel reflectance entering a medium of relative index eta."""
    cos_i = min(max(cos_i, 0.0), 1.0)
    sin2_t = (1.0 - cos_i * cos_i) / (eta * eta)
    if sin2_t >= 1.0:
        return 1.0
    cos_t = np.sqrt(1.0 - sin2_t)
    rs = (cos_i - eta * cos_t) / (cos_i + eta * cos_t)
    rp = (eta * cos_i - cos_t) / (eta * cos_i + cos_t)
    return 0.5 * (rs * rs + rp * rp)


@njit(cache=True, inline="always")
def schlick(f0, cos):
    m = 1.0 - min(max(cos, 0.0), 1.0)
    m2 = m * m
    return f0 + (1.0 - f0) * m2 * m2 * m


@njit(cache=True, inline="always")
def ggx_D(hx, hy, hz, ax, ay):
    if hz <= 0.0:
        return 0.0
    v = (hx / ax) ** 2 + (hy / ay) ** 2 + hz * hz
    return 1.0 / (PI * ax * ay * v * v)


@njit(cache=True, inline="always")
def ggx_lambda(wx, wy, wz, ax, ay):
    if abs(wz) < 1e-7:
        return 1e7
    return (-1.0 + np.sqrt(1.0 + ((ax * wx) ** 2 + (ay * wy) ** 2) / (wz * wz))) * 0.5


@njit(cache=True)
def ggx_sample_vndf(wox, woy, woz, ax, ay, u1, u2):
    """Visible normal sampling (Heitz 2018). wo must be in the upper hemisphere."""
    vx = ax * wox
    vy = ay * woy
    vz = woz
    n = np.sqrt(vx * vx + vy * vy + vz * vz)
    vx /= n
    vy /= n
    vz /= n
    lensq = vx * vx + vy * vy
    if lensq > 1e-12:
        il = 1.0 / np.sqrt(lensq)
        t1x, t1y, t1z = -vy * il, vx * il, 0.0
    else:
        t1x, t1y, t1z = 1.0, 0.0, 0.0
    t2x = vy * t1z - vz * t1y
    t2y = vz * t1x - vx * t1z
    t2z = vx * t1y - vy * t1x
    r = np.sqrt(u1)
    phi = 2.0 * PI * u2
    p1 = r * np.cos(phi)
    p2 = r * np.sin(phi)
    s = 0.5 * (1.0 + vz)
    p2 = (1.0 - s) * np.sqrt(max(0.0, 1.0 - p1 * p1)) + s * p2
    p3 = np.sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2))
    nx = p1 * t1x + p2 * t2x + p3 * vx
    ny = p1 * t1y + p2 * t2y + p3 * vy
    nz = p1 * t1z + p2 * t2z + p3 * vz
    hx = ax * nx
    hy = ay * ny
    hz = max(0.0, nz)
    n = np.sqrt(hx * hx + hy * hy + hz * hz)
    if n < 1e-12:
        return 0.0, 0.0, 1.0
    return hx / n, hy / n, hz / n


@njit(cache=True, inline="always")
def kc_lookup(E, Eavg, mu, alpha):
    n_mu = E.shape[0]
    n_a = E.shape[1]
    i = int(min(max(mu, 0.0), 0.999999) * n_mu)
    j = int(min(max(alpha, 0.0), 0.999999) * n_a)
    return E[i, j], Eavg[j]


@njit(cache=True)
def _lobe_probs(sp, cos_o):
    """Lobe selection probabilities (coat, specular, diffuse)."""
    coat = sp[S_COAT]
    fc = coat * fresnel_dielectric(cos_o, sp[S_COAT_IOR]) if coat > 0.0 else 0.0
    metal = sp[S_METAL]
    f0 = lerp(0.08 * sp[S_SPEC], luminance3(sp[S_BASE], sp[S_BASE + 1], sp[S_BASE + 2]), metal)
    fs = schlick(f0, cos_o)
    fs = min(max(fs, 0.0), 1.0)
    p_coat = fc
    p_spec = (1.0 - fc) * max(fs, 0.04)
    p_diff = (1.0 - fc) * (1.0 - max(fs, 0.04)) * (1.0 - metal) * (1.0 - sp[S_SSS_W])
    s = p_coat + p_spec + p_diff
    if s <= 0.0:
        return 0.0, 1.0, 0.0
    return p_coat / s, p_spec / s, p_diff / s


@njit(cache=True)
def bsdf_eval(sp, wo, wi, E, Eavg, out_f):
    """Evaluate f(wo, wi) (into out_f, 3) and return the sampling pdf. Local frame, z = normal."""
    cos_o = wo[2]
    cos_i = wi[2]
    out_f[0] = 0.0
    out_f[1] = 0.0
    out_f[2] = 0.0
    if cos_o <= 0.0 or cos_i <= 0.0:
        return 0.0
    ax = sp[S_AX]
    ay = sp[S_AY]
    hx = wo[0] + wi[0]
    hy = wo[1] + wi[1]
    hz = wo[2] + wi[2]
    hn = np.sqrt(hx * hx + hy * hy + hz * hz)
    if hn < 1e-12:
        return 0.0
    hx /= hn
    hy /= hn
    hz /= hn
    cos_oh = wo[0] * hx + wo[1] * hy + wo[2] * hz
    p_coat, p_spec, p_diff = _lobe_probs(sp, cos_o)
    pdf = 0.0
    metal = sp[S_METAL]
    coat = sp[S_COAT]
    ior = sp[S_COAT_IOR]
    fc_o = fresnel_dielectric(cos_o, ior) if coat > 0.0 else 0.0
    fc_i = fresnel_dielectric(cos_i, ior) if coat > 0.0 else 0.0
    # specular base lobe
    D = ggx_D(hx, hy, hz, ax, ay)
    lo = ggx_lambda(wo[0], wo[1], wo[2], ax, ay)
    li = ggx_lambda(wi[0], wi[1], wi[2], ax, ay)
    G2 = 1.0 / (1.0 + lo + li)
    G1o = 1.0 / (1.0 + lo)
    spec_atten = (1.0 - coat * fc_o) * (1.0 - coat * fc_i)
    for k in range(3):
        f0 = lerp(0.08 * sp[S_SPEC], sp[S_BASE + k], metal)
        F = schlick(f0, cos_oh)
        out_f[k] += spec_atten * D * G2 * F / (4.0 * cos_o * cos_i)
    if p_spec > 0.0:
        pdf += p_spec * G1o * max(cos_oh, 0.0) * D / cos_o / (4.0 * max(cos_oh, 1e-9))
    # Kulla Conty multiple scattering lobe (sampled with the diffuse lobe)
    alpha = np.sqrt(ax * ay)
    Eo, Ea = kc_lookup(E, Eavg, cos_o, alpha)
    Ei, _ = kc_lookup(E, Eavg, cos_i, alpha)
    if Ea < 0.999:
        fms = (1.0 - Eo) * (1.0 - Ei) / (PI * (1.0 - Ea))
        for k in range(3):
            f0 = lerp(0.08 * sp[S_SPEC], sp[S_BASE + k], metal)
            favg = (20.0 * f0 + 1.0) / 21.0
            Fms = favg * favg * Ea / (1.0 - favg * (1.0 - Ea))
            out_f[k] += spec_atten * fms * Fms
    # coat lobe
    if coat > 0.0:
        ac = 0.05
        Dc = ggx_D(hx, hy, hz, ac, ac)
        loc = ggx_lambda(wo[0], wo[1], wo[2], ac, ac)
        lic = ggx_lambda(wi[0], wi[1], wi[2], ac, ac)
        G2c = 1.0 / (1.0 + loc + lic)
        Fc = fresnel_dielectric(cos_oh, ior)
        val = coat * Dc * G2c * Fc / (4.0 * cos_o * cos_i)
        for k in range(3):
            out_f[k] += val
        if p_coat > 0.0:
            pdf += p_coat * (1.0 / (1.0 + loc)) * max(cos_oh, 0.0) * Dc / cos_o / (4.0 * max(cos_oh, 1e-9))
    # diffuse lobe (dry and wet)
    wd = (1.0 - metal) * (1.0 - sp[S_SSS_W])
    if wd > 0.0:
        if sp[S_DIFFUSE_MODEL] > 0.5:
            sig = 0.6 * sp[S_ROUGH]
            s2 = sig * sig
            A = 1.0 - 0.5 * s2 / (s2 + 0.33)
            B = 0.45 * s2 / (s2 + 0.09)
            s = wo[0] * wi[0] + wo[1] * wi[1] + wo[2] * wi[2] - cos_i * cos_o
            t = max(cos_i, cos_o) if s > 0.0 else 1.0
            on = A + B * s / t
        else:
            on = 1.0
        for k in range(3):
            a = sp[S_BASE + k]
            dry = a * on / PI
            if coat > 0.0:
                wet = a * (1.0 - fc_i) * (1.0 - fc_o) / (PI * ior * ior * (1.0 - a * R_INTERNAL_WATER))
                out_f[k] += wd * ((1.0 - coat) * dry + coat * wet)
            else:
                out_f[k] += wd * dry
    if p_diff > 0.0:
        pdf += p_diff * cos_i / PI
    # flakes (evaluated, not sampled)
    if sp[S_FLAKE_ON] > 0.5 and sp[S_FLAKE_INT] > 0.0:
        fx = sp[S_FLAKE_N]
        fy = sp[S_FLAKE_N + 1]
        fz = sp[S_FLAKE_N + 2]
        # D of a sharp lobe about the flake normal: angle between h and flake normal
        c = hx * fx + hy * fy + hz * fz
        if c > 0.0:
            af = 0.03
            t2 = (1.0 - c * c) / max(c * c, 1e-12)
            Df = 1.0 / (PI * af * af * c ** 4 * (1.0 + t2 / (af * af)) ** 2)
            Ff = schlick(0.6, cos_oh)
            val = sp[S_FLAKE_INT] * Df * Ff / (4.0 * cos_o * cos_i)
            for k in range(3):
                out_f[k] += val * (0.5 + 0.5 * sp[S_BASE + k])
    return pdf


@njit(cache=True)
def bsdf_sample(sp, wo, u1, u2, u3, E, Eavg, out_wi, out_f):
    """Sample wi (local frame). Returns (pdf, is_specular). f written to out_f."""
    cos_o = wo[2]
    if cos_o <= 0.0:
        return 0.0, False
    p_coat, p_spec, p_diff = _lobe_probs(sp, cos_o)
    is_spec = False
    if u3 < p_coat:
        hx, hy, hz = ggx_sample_vndf(wo[0], wo[1], wo[2], 0.05, 0.05, u1, u2)
        d = wo[0] * hx + wo[1] * hy + wo[2] * hz
        out_wi[0] = 2.0 * d * hx - wo[0]
        out_wi[1] = 2.0 * d * hy - wo[1]
        out_wi[2] = 2.0 * d * hz - wo[2]
        is_spec = True
    elif u3 < p_coat + p_spec:
        hx, hy, hz = ggx_sample_vndf(wo[0], wo[1], wo[2], sp[S_AX], sp[S_AY], u1, u2)
        d = wo[0] * hx + wo[1] * hy + wo[2] * hz
        out_wi[0] = 2.0 * d * hx - wo[0]
        out_wi[1] = 2.0 * d * hy - wo[1]
        out_wi[2] = 2.0 * d * hz - wo[2]
        is_spec = sp[S_ROUGH] < 0.25
    else:
        r = np.sqrt(u1)
        phi = 2.0 * PI * u2
        out_wi[0] = r * np.cos(phi)
        out_wi[1] = r * np.sin(phi)
        out_wi[2] = np.sqrt(max(0.0, 1.0 - u1))
    if out_wi[2] <= 0.0:
        out_f[0] = 0.0
        out_f[1] = 0.0
        out_f[2] = 0.0
        return 0.0, is_spec
    pdf = bsdf_eval(sp, wo, out_wi, E, Eavg, out_f)
    return pdf, is_spec


# -------------------------------------------------------- material shading


@njit(cache=True)
def shade_material(mat, px, py, pz, nx, ny, nz, gx, gy, gz, curv, occ, vr, vg, vb, sp):
    """Evaluate textures at object space point p with shading normal n (object space) and
    geometric normal g, writing the shading parameter vector ``sp`` (local object space normal
    in S_N).

    curv, occ: interpolated vertex attributes; vr, vg, vb: interpolated vertex colour.
    """
    seed = int(mat[M_SEED])
    br = mat[M_BASE]
    bg = mat[M_BASE + 1]
    bb = mat[M_BASE + 2]
    if mat[M_USE_VCOL] > 0.5:
        br *= vr
        bg *= vg
        bb *= vb
    rough = mat[M_ROUGH]
    coat = mat[M_COAT]
    # albedo and roughness variation by fBm
    if mat[M_FBM_AMP] > 0.0:
        s = mat[M_FBM_SCALE]
        v = fbm3(px * s, py * s, pz * s, 4, 0.5, 2.0, seed)
        m = 1.0 + mat[M_FBM_AMP] * v
        br *= m
        bg *= m
        bb *= m
        rough += mat[M_ROUGH_VAR] * v
    # Worley grain and crystal boundaries
    if mat[M_WORLEY_AMP] > 0.0:
        s = mat[M_WORLEY_SCALE]
        f1, f2, cell = worley3(px * s, py * s, pz * s, seed + 11)
        edge = f2 - f1
        m = 1.0 - mat[M_WORLEY_AMP] * (0.6 * (cell - 0.5) + 0.5 * max(0.0, 0.08 - edge) / 0.08)
        br *= m
        bg *= m
        bb *= m
    # sedimentary banding
    if mat[M_BAND_AMP] > 0.0:
        b = banding(px, py, pz, mat[M_BAND_DIR], mat[M_BAND_DIR + 1], mat[M_BAND_DIR + 2], mat[M_BAND_FREQ], 2.5, seed + 23)
        m = 1.0 + mat[M_BAND_AMP] * b
        br *= m
        bg *= m
        bb *= m
    # weathering by curvature and occlusion
    if mat[M_WEATHER_AMP] > 0.0:
        wa = mat[M_WEATHER_AMP]
        wear = max(curv, 0.0)
        dirt = (1.0 - occ) + max(-curv, 0.0) * 0.5
        m = 1.0 + wa * (0.35 * wear - 0.5 * min(dirt, 1.0))
        br *= m
        bg *= m
        bb *= m
        rough += wa * (-0.25 * wear + 0.3 * min(dirt, 1.0))
    # lichen patches on upward facing surfaces
    if mat[M_LICHEN_THR] > 0.0 and gz > 0.3:
        s = mat[M_LICHEN_SCALE]
        v = fbm3(px * s, py * s, pz * s, 3, 0.5, 2.0, seed + 37)
        if v > mat[M_LICHEN_THR]:
            k = hash01(int(np.floor(px * s)), int(np.floor(py * s)), int(np.floor(pz * s)), seed + 41)
            if k < 0.4:
                br, bg, bb = 0.30, 0.33, 0.12  # olive
            elif k < 0.7:
                br, bg, bb = 0.70, 0.40, 0.10  # orange
            else:
                br, bg, bb = 0.60, 0.62, 0.58  # pale grey
            rough = max(rough, 0.9)
    # wetness mask
    if coat > 0.0 and mat[M_WET_MASK_SCALE] > 0.0:
        s = mat[M_WET_MASK_SCALE]
        v = fbm3(px * s, py * s, pz * s, 3, 0.5, 2.0, seed + 53)
        coat *= min(max(0.5 + 1.5 * v + 0.5 * (-gz), 0.0), 1.0)
    # bump mapping (micro relief at 0.1 mm scale)
    sx, sy, sz = nx, ny, nz
    if mat[M_BUMP_AMP] > 0.0:
        s = mat[M_BUMP_SCALE]
        dx, dy, dz = fbm3_grad(px * s, py * s, pz * s, 2, 0.5, 2.0, seed + 61, 0.05)
        t = triplanar(px, py, pz, nx, ny, nz, s * 0.5, seed + 67)
        dx = dx * s + t * 0.5 * s
        # remove the normal component
        dn = dx * nx + dy * ny + dz * nz
        dx -= dn * nx
        dy -= dn * ny
        dz -= dn * nz
        a = mat[M_BUMP_AMP] / s
        sx = nx - a * dx
        sy = ny - a * dy
        sz = nz - a * dz
        n = np.sqrt(sx * sx + sy * sy + sz * sz)
        sx /= n
        sy /= n
        sz /= n
    rough = min(max(rough, 0.02), 1.0)
    sp[S_BASE] = min(max(br, 0.0), 1.0)
    sp[S_BASE + 1] = min(max(bg, 0.0), 1.0)
    sp[S_BASE + 2] = min(max(bb, 0.0), 1.0)
    sp[S_ROUGH] = rough
    sp[S_METAL] = mat[M_METAL]
    sp[S_SPEC] = mat[M_SPEC]
    aspect = np.sqrt(1.0 - 0.9 * min(mat[M_ANISO], 0.99))
    a2 = max(rough * rough, 0.0004)
    sp[S_AX] = max(a2 / aspect, 0.001)
    sp[S_AY] = max(a2 * aspect, 0.001)
    sp[S_COAT] = min(max(coat, 0.0), 1.0)
    sp[S_COAT_IOR] = mat[M_COAT_IOR]
    sp[S_FLAKE_INT] = mat[M_FLAKE_INT]
    sp[S_FLAKE_ON] = 0.0
    if mat[M_FLAKE_DENS] > 0.0:
        fs = max(mat[M_FLAKE_SIZE], 1e-5)
        cx = int(np.floor(px / fs))
        cy = int(np.floor(py / fs))
        cz = int(np.floor(pz / fs))
        if hash01(cx, cy, cz, seed + 71) < mat[M_FLAKE_DENS]:
            sp[S_FLAKE_ON] = 1.0
            # random tilt within a 25 degree cone about the shading normal, stored in object space
            u1 = hash01(cx, cy, cz, seed + 73)
            u2 = hash01(cx, cy, cz, seed + 79)
            ct = 1.0 - u1 * (1.0 - np.cos(np.radians(25.0)))
            st = np.sqrt(max(0.0, 1.0 - ct * ct))
            ph = 2.0 * PI * u2
            # frame around (sx, sy, sz)
            if abs(sx) < 0.9:
                tx, ty, tz = 0.0, -sz, sy
            else:
                tx, ty, tz = sz, 0.0, -sx
            tn = np.sqrt(tx * tx + ty * ty + tz * tz)
            tx /= tn
            ty /= tn
            tz /= tn
            bx = sy * tz - sz * ty
            by = sz * tx - sx * tz
            bz = sx * ty - sy * tx
            sp[S_FLAKE_N] = st * np.cos(ph) * tx + st * np.sin(ph) * bx + ct * sx
            sp[S_FLAKE_N + 1] = st * np.cos(ph) * ty + st * np.sin(ph) * by + ct * sy
            sp[S_FLAKE_N + 2] = st * np.cos(ph) * tz + st * np.sin(ph) * bz + ct * sz
    sp[S_SSS_MFP] = mat[M_SSS_MFP]
    sp[S_SSS_MFP + 1] = mat[M_SSS_MFP + 1]
    sp[S_SSS_MFP + 2] = mat[M_SSS_MFP + 2]
    sp[S_SSS_W] = mat[M_SSS_W]
    sp[S_EMIT] = mat[M_EMIT]
    sp[S_EMIT + 1] = mat[M_EMIT + 1]
    sp[S_EMIT + 2] = mat[M_EMIT + 2]
    sp[S_DIFFUSE_MODEL] = mat[M_DIFFUSE_MODEL]
    sp[S_N] = sx
    sp[S_N + 1] = sy
    sp[S_N + 2] = sz
