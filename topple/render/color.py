"""Colour management (Section 10.2, 10.5, 10.10).

* Radiance stays in linear scene referred units. Exposure is set so that an 18 percent grey
  card under the key light maps to 0.18 (``exposure_from_grey_card``).
* White balance uses the Bradford chromatic adaptation transform to D65 (Fairchild 2013).
* Tone mapping applies the ACES reference rendering transform and output transform through
  the Hill fit (sRGB primaries), which is the "ACES filmic" curve used by OpenColorIO configs.
* Colour maps are perceptually uniform by construction: they are sampled paths in the Oklab
  colour space with monotone lightness (sequential) or symmetric lightness (diverging),
  following the design rules of Crameri, Shephard and Heron (2020) and Kovesi (2015). The
  tables hold 64 control points each and are interpolated linearly in linear sRGB.
"""

from __future__ import annotations

import numpy as np

# sRGB (D65) matrices
M_RGB2XYZ = np.array(
    [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]]
)
M_XYZ2RGB = np.linalg.inv(M_RGB2XYZ)
D65_XYZ = np.array([0.95047, 1.0, 1.08883])
D50_XYZ = np.array([0.96422, 1.0, 0.82521])

M_BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])


def luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2126729 * rgb[..., 0] + 0.7151522 * rgb[..., 1] + 0.0721750 * rgb[..., 2]


def srgb_encode(linear: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(linear, dtype=np.float64), 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def srgb_decode(encoded: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(encoded, dtype=np.float64), 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))


def xy_to_xyz(x: float, y: float, Y: float = 1.0) -> np.ndarray:
    return np.array([x * Y / y, Y, (1 - x - y) * Y / y])


def bradford_matrix(src_white_xyz: np.ndarray, dst_white_xyz: np.ndarray = D65_XYZ) -> np.ndarray:
    """3x3 matrix in XYZ that adapts colours seen under ``src_white`` to ``dst_white``."""
    rho_s = M_BRADFORD @ src_white_xyz
    rho_d = M_BRADFORD @ dst_white_xyz
    D = np.diag(rho_d / rho_s)
    return np.linalg.inv(M_BRADFORD) @ D @ M_BRADFORD


def bradford_adapt(rgb_linear: np.ndarray, src_white_xyz: np.ndarray, dst_white_xyz: np.ndarray = D65_XYZ) -> np.ndarray:
    """Adapt a linear sRGB image rendered under ``src_white`` so that white maps to D65."""
    M = M_XYZ2RGB @ bradford_matrix(src_white_xyz, dst_white_xyz) @ M_RGB2XYZ
    return np.asarray(rgb_linear) @ M.T


def exposure_from_grey_card(key_irradiance: float, albedo: float = 0.18, target: float = 0.18) -> float:
    """Scale factor so that a Lambertian card of ``albedo`` under irradiance ``key_irradiance``
    (radiance albedo * E / pi) maps to ``target`` in linear display space."""
    radiance = albedo * key_irradiance / np.pi
    return target / max(radiance, 1e-12)


# ACES RRT + ODT fit (Stephen Hill), sRGB in, sRGB out, linear values
_ACES_IN = np.array([[0.59719, 0.35458, 0.04823], [0.07600, 0.90834, 0.01566], [0.02840, 0.13383, 0.83777]])
_ACES_OUT = np.array([[1.60475, -0.53108, -0.07367], [-0.10208, 1.10813, -0.00605], [-0.00327, -0.07276, 1.07602]])


def _rrt_odt_fit(v: np.ndarray) -> np.ndarray:
    a = v * (v + 0.0245786) - 0.000090537
    b = v * (0.983729 * v + 0.4329510) + 0.238081
    return a / b


def aces_rrt_odt(rgb_linear: np.ndarray) -> np.ndarray:
    """ACES reference rendering transform and sRGB output transform (fit). Returns linear
    display values in [0, 1]; apply ``srgb_encode`` afterwards for 8 bit output."""
    x = np.asarray(rgb_linear, dtype=np.float64)
    v = x @ _ACES_IN.T
    v = _rrt_odt_fit(np.maximum(v, 0.0))
    v = v @ _ACES_OUT.T
    return np.clip(v, 0.0, 1.0)


def tonemap(rgb_linear: np.ndarray, exposure: float = 1.0, src_white_xyz: np.ndarray | None = None) -> np.ndarray:
    """Full display pipeline: exposure, optional Bradford adaptation, ACES, sRGB encoding."""
    x = np.asarray(rgb_linear, dtype=np.float64) * exposure
    if src_white_xyz is not None:
        x = bradford_adapt(x, src_white_xyz)
    return srgb_encode(aces_rrt_odt(x))


# ------------------------------------------------------------------------ Oklab


def linear_srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = np.cbrt(l), np.cbrt(m), np.cbrt(s)
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    bb = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return np.stack([L, a, bb], axis=-1)


def oklab_to_linear_srgb(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    return np.stack([r, g, bb], axis=-1)


def _oklch_to_srgb_in_gamut(L, C, h_deg):
    """Reduce chroma until the colour fits the sRGB gamut."""
    h = np.deg2rad(h_deg)
    for _ in range(40):
        rgb = oklab_to_linear_srgb(np.array([L, C * np.cos(h), C * np.sin(h)]))
        if np.all(rgb >= -1e-6) and np.all(rgb <= 1 + 1e-6):
            return np.clip(rgb, 0, 1)
        C *= 0.9
    return np.clip(oklab_to_linear_srgb(np.array([L, 0.0, 0.0])), 0, 1)


def _build_sequential(n=64):
    """Dark blue, teal, green, yellow, orange, pink with linear lightness (batlow style path)."""
    t = np.linspace(0, 1, n)
    L = 0.28 + 0.65 * t
    hue = 265 - 300 * t  # 265 (blue) down through 160 (teal), 110 (green), 70 (yellow), 30 (orange), -35 (pink)
    C = 0.11 * np.sin(np.pi * np.clip(t, 0, 1)) ** 0.5 + 0.03
    return np.array([_oklch_to_srgb_in_gamut(L[i], C[i], hue[i]) for i in range(n)])


def _build_diverging(n=64):
    """Blue, near white, red with symmetric lightness and a neutral centre (vik style)."""
    t = np.linspace(-1, 1, n)
    L = 0.92 - 0.55 * np.abs(t) ** 1.1
    C = 0.15 * np.abs(t) ** 0.9
    hue = np.where(t < 0, 255.0, 25.0)
    return np.array([_oklch_to_srgb_in_gamut(L[i], C[i], hue[i]) for i in range(n)])


def _build_cyclic(n=64):
    t = np.linspace(0, 1, n, endpoint=False)
    L = 0.6 + 0.15 * np.cos(2 * np.pi * t)
    hue = 360 * t
    return np.array([_oklch_to_srgb_in_gamut(L[i], 0.11, hue[i]) for i in range(n)])


_TABLES: dict[str, np.ndarray] = {}


def colormap_table(name: str) -> np.ndarray:
    if name not in _TABLES:
        if name in ("sequential", "batlow_like"):
            _TABLES[name] = _build_sequential()
        elif name in ("diverging", "vik_like"):
            _TABLES[name] = _build_diverging()
        elif name in ("cyclic", "roma_like"):
            _TABLES[name] = _build_cyclic()
        else:
            raise KeyError(f"unknown colour map {name}")
    return _TABLES[name]


def colormap(values: np.ndarray, name: str = "sequential", vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    """Map scalar values to linear sRGB colours through a perceptually uniform table."""
    v = np.asarray(values, dtype=np.float64)
    vmin = float(np.nanmin(v)) if vmin is None else vmin
    vmax = float(np.nanmax(v)) if vmax is None else vmax
    t = np.clip((v - vmin) / max(vmax - vmin, 1e-12), 0, 1)
    table = colormap_table(name)
    n = table.shape[0]
    x = t * (n - 1)
    i0 = np.clip(np.floor(x).astype(int), 0, n - 1)
    i1 = np.clip(i0 + 1, 0, n - 1)
    f = (x - i0)[..., None]
    return table[i0] * (1 - f) + table[i1] * f


def diverging_colormap(values: np.ndarray, center: float = 0.0, span: float | None = None) -> np.ndarray:
    """Diverging map centred at ``center`` (Margin tint O7): negative values take the blue side."""
    v = np.asarray(values, dtype=np.float64)
    span = float(np.nanmax(np.abs(v - center))) if span is None else span
    return colormap(v, "diverging", center - span, center + span)
