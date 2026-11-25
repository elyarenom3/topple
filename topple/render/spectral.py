"""Spectral rendering support: hero wavelength sampling with three stratified wavelengths.

Each camera sample carries three wavelengths, lambda_k = 380 + 350 frac(u + k / 3) nm, in the
three throughput channels of the RGB kernel (Wilkie et al. 2014 use four; three keep the kernel's
channel count). RGB material and light values are upsampled to spectra with a smooth three band
basis that is a partition of unity, so a white reflectance is 1 at every wavelength (energy
conservation is exact) and an RGB value maps back to itself under the white illuminant:

    r(lambda) = sum_j w_j b_j(lambda),  w = P^-1 c,

where P_ij is the RGB of basis j under the illuminant. Emission is the illuminant spectrum times
the upsampled reflectance, so an emitter of RGB c renders as c. The illuminant is a 6504 K
blackbody (close to D65); the output matrix is normalised so that it maps to RGB (1, 1, 1).
The CIE 1931 colour matching functions use the multi lobe fit of Wyman, Sloan and Shirley (2013).

Dispersive dielectrics (Cauchy n = A + B / lambda^2) collapse the sample to its hero wavelength
on a refraction: the other two channels drop and the hero channel is weighted by 3.

The constants are packed into the integrator's ``settings`` array (``pack_settings``).
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .color import M_XYZ2RGB

LAMBDA_MIN = 380.0
LAMBDA_RANGE = 350.0
S_SPECTRAL = 4  # settings slot of the spectral flag
S_PINV = 5  # 9 values: inverse of the basis to RGB matrix
S_CMAT = 14  # 9 values: XYZ to output RGB matrix (white normalised)
S_YN = 23  # normalisation: integral of ybar times the illuminant
S_MNEE = 24  # manifold next event estimation of the sun through dielectric surfaces
SETTINGS_SIZE = 25


@njit(cache=True, inline="always")
def _g(x, mu, s1, s2):
    t = (x - mu) / (s1 if x < mu else s2)
    return np.exp(-0.5 * t * t)


@njit(cache=True)
def cie_xyz(lam):
    """CIE 1931 2 degree colour matching functions (Wyman et al. 2013, multi lobe fit)."""
    x = 1.056 * _g(lam, 599.8, 37.9, 31.0) + 0.362 * _g(lam, 442.0, 16.0, 26.7) - 0.065 * _g(lam, 501.1, 20.4, 26.2)
    y = 0.821 * _g(lam, 568.8, 46.9, 40.5) + 0.286 * _g(lam, 530.9, 16.3, 31.1)
    z = 1.217 * _g(lam, 437.0, 11.8, 36.0) + 0.681 * _g(lam, 459.0, 26.0, 13.8)
    return x, y, z


@njit(cache=True, inline="always")
def _smooth(x, a, b):
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


@njit(cache=True)
def basis(lam):
    """Three smooth bands (red, green, blue) that sum to one at every wavelength."""
    bb = 1.0 - _smooth(lam, 450.0, 530.0)
    br = _smooth(lam, 560.0, 630.0)
    return br, 1.0 - bb - br, bb


@njit(cache=True)
def illuminant(lam):
    """Relative spectral radiance of a 6504 K blackbody, 1 at 560 nm."""
    c2 = 1.4388e-2
    T = 6504.0
    l = lam * 1e-9
    l0 = 560e-9
    return (l0 / l) ** 5 * (np.exp(c2 / (l0 * T)) - 1.0) / (np.exp(c2 / (l * T)) - 1.0)


def _integrate(fn, n=2048):
    lam = np.linspace(LAMBDA_MIN, LAMBDA_MIN + LAMBDA_RANGE, n)
    vals = np.array([fn(l) for l in lam])
    return np.trapezoid(vals, lam, axis=0) if hasattr(np, "trapezoid") else np.trapz(vals, lam, axis=0)


def tables():
    """(P_inv (3, 3), C (3, 3), Yn): basis weights from RGB, XYZ to RGB, and normalisation."""
    Yn = float(_integrate(lambda l: cie_xyz(l)[1] * illuminant(l)))
    XYZ_w = _integrate(lambda l: np.array(cie_xyz(l)) * illuminant(l)) / Yn
    rgb_w = M_XYZ2RGB @ XYZ_w
    C = np.diag(1.0 / rgb_w) @ M_XYZ2RGB
    P = np.zeros((3, 3))
    for j in range(3):
        xyz = _integrate(lambda l, j=j: np.array(cie_xyz(l)) * illuminant(l) * basis(l)[j]) / Yn
        P[:, j] = C @ xyz
    return np.linalg.inv(P), C, Yn


def pack_settings(settings: np.ndarray, spectral: bool, mnee: bool = False) -> np.ndarray:
    out = np.zeros(max(SETTINGS_SIZE, len(settings)))
    out[: len(settings)] = settings
    out[S_SPECTRAL] = 1.0 if spectral else 0.0
    out[S_MNEE] = 1.0 if mnee else 0.0
    if spectral:
        Pinv, C, Yn = tables()
        out[S_PINV: S_PINV + 9] = Pinv.reshape(-1)
        out[S_CMAT: S_CMAT + 9] = C.reshape(-1)
        out[S_YN] = Yn
    return out


@njit(cache=True)
def upsample(settings, c0, c1, c2, lam, emission):
    """Spectral value at ``lam`` of the RGB colour c (reflectance clipped to [0, 1], or emission
    = illuminant times the non negative upsampled reflectance)."""
    p = settings
    w0 = p[S_PINV] * c0 + p[S_PINV + 1] * c1 + p[S_PINV + 2] * c2
    w1 = p[S_PINV + 3] * c0 + p[S_PINV + 4] * c1 + p[S_PINV + 5] * c2
    w2 = p[S_PINV + 6] * c0 + p[S_PINV + 7] * c1 + p[S_PINV + 8] * c2
    br, bg, bb = basis(lam)
    r = w0 * br + w1 * bg + w2 * bb
    if emission:
        return max(r, 0.0) * illuminant(lam)
    return min(max(r, 0.0), 1.0)


@njit(cache=True)
def upsample_positive(settings, c0, c1, c2, lam):
    """Upsampled non negative quantity without the [0, 1] clip (absorption, mean free path)."""
    p = settings
    w0 = p[S_PINV] * c0 + p[S_PINV + 1] * c1 + p[S_PINV + 2] * c2
    w1 = p[S_PINV + 3] * c0 + p[S_PINV + 4] * c1 + p[S_PINV + 5] * c2
    w2 = p[S_PINV + 6] * c0 + p[S_PINV + 7] * c1 + p[S_PINV + 8] * c2
    br, bg, bb = basis(lam)
    return max(w0 * br + w1 * bg + w2 * bb, 0.0)


@njit(cache=True)
def to_rgb(settings, L, lam, out):
    """RGB of the three spectral radiance samples L at wavelengths lam (uniform pdf)."""
    X = 0.0
    Y = 0.0
    Z = 0.0
    for k in range(3):
        x, y, z = cie_xyz(lam[k])
        X += L[k] * x
        Y += L[k] * y
        Z += L[k] * z
    s = LAMBDA_RANGE / 3.0 / settings[S_YN]
    X *= s
    Y *= s
    Z *= s
    C = settings
    for i in range(3):
        out[i] = C[S_CMAT + 3 * i] * X + C[S_CMAT + 3 * i + 1] * Y + C[S_CMAT + 3 * i + 2] * Z


@njit(cache=True, inline="always")
def cauchy_ior(n_d, B, lam_nm):
    """Refractive index at lam (nm) from the index at 589.3 nm and the Cauchy B (um^2)."""
    lu = lam_nm * 1e-3
    return n_d - B / (0.5893 * 0.5893) + B / (lu * lu)
