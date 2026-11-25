"""Analytic sky and sun (Section 10.4).

Two sky models share one parameter array and one kernel entry point:

* Hošek and Wilkie (2012), the model named by the specification (default). Its CIE XYZ
  coefficient tables are the authors' published data (3-clause BSD licence, see
  ``data/LICENSE_hosek_wilkie.txt``), evaluated with the reference implementation's
  quintic Bezier interpolation in solar elevation and linear interpolation in turbidity and
  ground albedo.
* Preetham, Shirley and Smits (1999), kept as an option (``model="preetham"``).

The sun is a separate disc of angular diameter 0.53 degrees whose radiance comes from the
Preetham paper's atmospheric transmittance (Rayleigh, aerosol, ozone, water vapour and mixed
gases) for both sky models.

Units: Preetham's sky luminance Y is in kcd/m^2. The Hošek Wilkie XYZ tables give Y in
W m^-2 sr^-1 weighted by the photopic curve, so they are multiplied by 683 lm/W / 1000 to give
kcd/m^2 as well. Both are converted to scene radiance by the fixed factor ``SKY_UNIT``, so that
the ratio of sky to sun matches the models; the absolute level is handled by the exposure
(Section 10.2).
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .color import M_XYZ2RGB

SUN_ANGULAR_DIAMETER_DEG = 0.53
SKY_UNIT = 10.0  # W m^-2 sr^-1 per kcd m^-2 (scene units)


def perez_coefficients(T: float):
    """Perez function coefficients for Y, x and y as linear functions of turbidity."""
    Y = np.array([0.1787 * T - 1.4630, -0.3554 * T + 0.4275, -0.0227 * T + 5.3251, 0.1206 * T - 2.5771, -0.0670 * T + 0.3703])
    x = np.array([-0.0193 * T - 0.2592, -0.0665 * T + 0.0008, -0.0004 * T + 0.2125, -0.0641 * T - 0.8989, -0.0033 * T + 0.0452])
    y = np.array([-0.0167 * T - 0.2608, -0.0950 * T + 0.0092, -0.0079 * T + 0.2102, -0.0441 * T - 1.6537, -0.0109 * T + 0.0529])
    return Y, x, y


def zenith_values(T: float, theta_s: float):
    """Zenith luminance (kcd/m^2) and chromaticity for turbidity T and sun zenith angle theta_s."""
    chi = (4.0 / 9.0 - T / 120.0) * (np.pi - 2 * theta_s)
    Yz = (4.0453 * T - 4.9710) * np.tan(chi) - 0.2155 * T + 2.4192
    t2, t = T * T, T
    th = theta_s
    xz = (t2 * 0.00166 - t * 0.02903 + 0.11763) * th ** 3 + (t2 * -0.00375 + t * 0.06377 - 0.21196) * th ** 2 + (t2 * 0.00209 - t * 0.03202 + 0.06052) * th + (t * 0.00394 + 0.25886)
    yz = (t2 * 0.00275 - t * 0.04214 + 0.15346) * th ** 3 + (t2 * -0.00610 + t * 0.08970 - 0.26756) * th ** 2 + (t2 * 0.00317 - t * 0.04153 + 0.06670) * th + (t * 0.00516 + 0.26688)
    return max(Yz, 1e-3), xz, yz


def sun_radiance_rgb(theta_s: float, T: float) -> np.ndarray:
    """Sun disc radiance (scene units) from the Preetham transmittance model at RGB wavelengths."""
    # wavelengths in micrometres for R, G, B
    lam = np.array([0.68, 0.55, 0.44])
    beta = 0.04608 * T - 0.04586
    alpha = 1.3
    cos = np.cos(theta_s)
    theta_deg = np.degrees(theta_s)
    m = 1.0 / (cos + 0.15 * (93.885 - theta_deg) ** -1.253) if theta_deg < 90 else 40.0
    tau_r = np.exp(-0.008735 * m * lam ** -4.08)
    tau_a = np.exp(-beta * m * lam ** -alpha)
    k_o = np.array([0.021, 0.095, 0.005])  # ozone absorption, approximate per channel
    tau_o = np.exp(-k_o * 0.35 * m)
    # water vapour and mixed gases are small at these wavelengths; include a 2 percent loss
    tau_wg = np.array([0.98, 0.99, 1.0])
    irradiance = 1000.0 * tau_r * tau_a * tau_o * tau_wg  # W m^-2 at normal incidence
    half = np.radians(SUN_ANGULAR_DIAMETER_DEG / 2)
    solid_angle = 2 * np.pi * (1 - np.cos(half))
    return np.maximum(irradiance / solid_angle, 0.0)


_HW = None


def hosek_wilkie_tables():
    """(params (3, 2, 10, 6, 9), radiance (3, 2, 10, 6)) for X, Y, Z; albedo 0 and 1; turbidity
    1 to 10; six Bezier control points in solar elevation; nine model coefficients."""
    global _HW
    if _HW is None:
        from pathlib import Path

        d = np.load(Path(__file__).parent / "data" / "hosek_wilkie_xyz.npz")
        _HW = (d["params"].astype(np.float64), d["radiance"].astype(np.float64))
    return _HW


def _bezier5(ctrl: np.ndarray, t: float) -> np.ndarray:
    """Quintic Bezier over the first axis of ``ctrl`` (6, ...) at t (reference implementation)."""
    w = np.array([(1 - t) ** 5, 5 * (1 - t) ** 4 * t, 10 * (1 - t) ** 3 * t ** 2, 10 * (1 - t) ** 2 * t ** 3, 5 * (1 - t) * t ** 4, t ** 5])
    return np.tensordot(w, ctrl, axes=(0, 0))


def hosek_wilkie_configuration(turbidity: float, albedo: float, elevation_rad: float):
    """Model coefficients (3, 9) and radiance scale (3,) for X, Y and Z, as in the reference
    ``ArHosekSkyModel_CookConfiguration`` and ``CookRadianceConfiguration``."""
    params, rad = hosek_wilkie_tables()
    T = float(np.clip(turbidity, 1.0, 10.0))
    a = float(np.clip(albedo, 0.0, 1.0))
    t = (max(elevation_rad, 0.0) / (np.pi / 2.0)) ** (1.0 / 3.0)
    ti = int(T)
    tr = T - ti
    cfg = np.zeros((3, 9))
    L = np.zeros(3)
    for alb_i, wa in ((0, 1.0 - a), (1, a)):
        for t_i, wt in ((ti - 1, 1.0 - tr), (ti, tr)):
            if wt == 0.0 or t_i > 9:
                continue
            for c in range(3):
                cfg[c] += wa * wt * _bezier5(params[c, alb_i, t_i], t)
                L[c] += wa * wt * _bezier5(rad[c, alb_i, t_i], t)
    return cfg, L


def hosek_wilkie_xyz(theta: float, gamma: float, cfg: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Reference evaluation ``ArHosekSkyModel_GetRadianceInternal`` times the radiance scale."""
    out = np.zeros(3)
    for c in range(3):
        k = cfg[c]
        expM = np.exp(k[4] * gamma)
        rayM = np.cos(gamma) ** 2
        mieM = (1.0 + np.cos(gamma) ** 2) / (1.0 + k[8] ** 2 - 2.0 * k[8] * np.cos(gamma)) ** 1.5
        zen = np.sqrt(np.cos(theta))
        out[c] = (1.0 + k[0] * np.exp(k[1] / (np.cos(theta) + 0.01))) * (k[2] + k[3] * expM + k[5] * rayM + k[6] * mieM + k[7] * zen) * L[c]
    return out


HW_TO_KCD = 683.0 / 1000.0  # photopic Y in W m^-2 sr^-1 to kcd m^-2


def sky_params(elevation_deg: float, turbidity: float = 3.0, azimuth_deg: float = 0.0, scale: float = 1.0, model: str = "hosek", albedo: float = 0.2) -> np.ndarray:
    """Pack the sky model into a flat array for the kernels.

    Layout: [0:3] sun direction, [3] theta_s, [4:9] Perez Y, [9:14] Perez x, [14:19] Perez y,
    [19:22] zenith (Yz, xz, yz), [22:25] sun radiance rgb, [25] cos half angle, [26] scale,
    [27] model (0 Preetham, 1 Hošek Wilkie), [32:59] Hošek Wilkie coefficients (X, Y, Z by 9),
    [59:62] Hošek Wilkie radiance scales. ``albedo`` is the ground albedo of the Hošek Wilkie
    model.
    """
    el = np.radians(elevation_deg)
    az = np.radians(azimuth_deg)
    sun = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    theta_s = np.pi / 2 - el
    Y, x, y = perez_coefficients(turbidity)
    Yz, xz, yz = zenith_values(turbidity, theta_s)
    sunL = sun_radiance_rgb(theta_s, turbidity) * scale
    p = np.zeros(64)
    p[0:3] = sun
    p[3] = theta_s
    p[4:9] = Y
    p[9:14] = x
    p[14:19] = y
    p[19:22] = [Yz, xz, yz]
    p[22:25] = sunL
    p[25] = np.cos(np.radians(SUN_ANGULAR_DIAMETER_DEG / 2))
    p[26] = scale
    if model == "hosek":
        cfg, L = hosek_wilkie_configuration(turbidity, albedo, el)
        p[27] = 1.0
        p[32:59] = cfg.reshape(-1)
        p[59:62] = L
    elif model != "preetham":
        raise ValueError(model)
    return p


_M = M_XYZ2RGB.copy()


@njit(cache=True, inline="always")
def _perez(A, B, C, D, E, theta, gamma):
    return (1.0 + A * np.exp(B / max(np.cos(theta), 0.01))) * (1.0 + C * np.exp(D * gamma) + E * np.cos(gamma) ** 2)


@njit(cache=True, inline="always")
def _hw_channel(p, o, theta, gamma):
    expM = np.exp(p[o + 4] * gamma)
    cgm = np.cos(gamma)
    rayM = cgm * cgm
    mieM = (1.0 + cgm * cgm) / (1.0 + p[o + 8] * p[o + 8] - 2.0 * p[o + 8] * cgm) ** 1.5
    zen = np.sqrt(np.cos(theta))
    return (1.0 + p[o] * np.exp(p[o + 1] / (np.cos(theta) + 0.01))) * (p[o + 2] + p[o + 3] * expM + p[o + 5] * rayM + p[o + 6] * mieM + p[o + 7] * zen)


@njit(cache=True)
def sky_radiance(dx, dy, dz, p, M):
    """Linear RGB radiance of the sky in direction d (world, z up), without the sun disc."""
    if dz < 0.0:
        # below the horizon: use the horizon value, slightly darkened
        dz = 0.0
    theta = np.arccos(min(dz, 1.0))
    cg = dx * p[0] + dy * p[1] + dz * p[2]
    if cg > 1.0:
        cg = 1.0
    if cg < -1.0:
        cg = -1.0
    gamma = np.arccos(cg)
    if p[27] > 0.5:
        # Hošek Wilkie: theta clamped just above the horizon as in the reference renderers
        th = min(theta, 0.5 * np.pi - 0.001)
        k = 683.0 / 1000.0 * 10.0 * p[26]
        X = _hw_channel(p, 32, th, gamma) * p[59] * k
        Y = _hw_channel(p, 41, th, gamma) * p[60] * k
        Z = _hw_channel(p, 50, th, gamma) * p[61] * k
        r = M[0, 0] * X + M[0, 1] * Y + M[0, 2] * Z
        g = M[1, 0] * X + M[1, 1] * Y + M[1, 2] * Z
        b = M[2, 0] * X + M[2, 1] * Y + M[2, 2] * Z
        return max(r, 0.0), max(g, 0.0), max(b, 0.0)
    ths = p[3]
    fy = _perez(p[4], p[5], p[6], p[7], p[8], theta, gamma) / _perez(p[4], p[5], p[6], p[7], p[8], 0.0, ths)
    fx = _perez(p[9], p[10], p[11], p[12], p[13], theta, gamma) / _perez(p[9], p[10], p[11], p[12], p[13], 0.0, ths)
    fyy = _perez(p[14], p[15], p[16], p[17], p[18], theta, gamma) / _perez(p[14], p[15], p[16], p[17], p[18], 0.0, ths)
    Y = p[19] * fy * 10.0 * p[26]
    x = p[20] * fx
    y = p[21] * fyy
    if y < 1e-4:
        y = 1e-4
    X = x * Y / y
    Z = (1.0 - x - y) * Y / y
    r = M[0, 0] * X + M[0, 1] * Y + M[0, 2] * Z
    g = M[1, 0] * X + M[1, 1] * Y + M[1, 2] * Z
    b = M[2, 0] * X + M[2, 1] * Y + M[2, 2] * Z
    return max(r, 0.0), max(g, 0.0), max(b, 0.0)


def sky_image(p: np.ndarray, width: int = 256, height: int = 128) -> np.ndarray:
    """Lat long image of the sky (for inspection and for HDR comparison)."""
    out = np.zeros((height, width, 3))
    for j in range(height):
        th = np.pi * (j + 0.5) / height
        for i in range(width):
            ph = 2 * np.pi * (i + 0.5) / width
            d = (np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th))
            out[j, i] = sky_radiance(d[0], d[1], d[2], p, _M)
    return out
