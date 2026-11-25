"""FLIP: a difference evaluator for alternating images (Andersson et al. 2020), LDR variant.

The implementation follows the published algorithm step by step:

1. sRGB inputs are converted to the YCxCz opponent space.
2. Each channel is filtered by the spatial contrast sensitivity function of its opponent
   channel (achromatic, red green, blue yellow), expressed as a sum of Gaussians in the spatial
   domain at the given pixels per degree, then clamped to the sRGB gamut.
3. The colour difference is the HyAB distance in CIELAB after the Hunt adjustment, normalised by
   the green to blue distance and redistributed with the (p_c, p_t) = (0.4, 0.95) curve.
4. Features (edges and points) are detected on the achromatic channel with first and second
   derivative of Gaussian filters (width 0.082 degrees); the feature difference is the maximum
   of the edge and point differences, scaled by 1/sqrt(2) and raised to q_f = 0.5.
5. The FLIP error is deltaE_c ** (1 - deltaE_f) per pixel; the mean over pixels is reported.

Simplifications: none in the metric itself. The default viewing condition is the paper's:
0.7 m from a 0.7 m wide 3840 pixel display, 67 pixels per degree.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve

from .color import M_RGB2XYZ, M_XYZ2RGB, srgb_decode

_REF_WHITE = np.array([0.950428545377181, 1.0, 1.088900370798128])


def pixels_per_degree(distance_m: float = 0.7, width_m: float = 0.7, width_px: int = 3840) -> float:
    return distance_m * (width_px / width_m) * (np.pi / 180.0)


def _lin_to_xyz(rgb):
    return rgb @ M_RGB2XYZ.T


def _xyz_to_lin(xyz):
    return xyz @ M_XYZ2RGB.T


def _xyz_to_ycxcz(xyz):
    r = xyz / _REF_WHITE
    y = 116.0 * r[..., 1] - 16.0
    cx = 500.0 * (r[..., 0] - r[..., 1])
    cz = 200.0 * (r[..., 1] - r[..., 2])
    return np.stack([y, cx, cz], -1)


def _ycxcz_to_xyz(ycc):
    yr = (ycc[..., 0] + 16.0) / 116.0
    xr = ycc[..., 1] / 500.0 + yr
    zr = yr - ycc[..., 2] / 200.0
    return np.stack([xr, yr, zr], -1) * _REF_WHITE


def _xyz_to_lab(xyz):
    r = xyz / _REF_WHITE
    delta = 6.0 / 29.0
    f = np.where(r > delta ** 3, np.cbrt(r), r / (3 * delta ** 2) + 4.0 / 29.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], -1)


def _spatial_filter(ppd: float, channel: str) -> np.ndarray:
    params = {"A": (1.0, 0.0047, 0.0, 1e-5), "RG": (1.0, 0.0053, 0.0, 1e-5), "BY": (34.1, 0.04, 13.5, 0.025)}
    a1, b1, a2, b2 = params[channel]
    max_scale = 0.04
    r = int(np.ceil(3 * np.sqrt(max_scale / (2 * np.pi ** 2)) * ppd))
    dx = 1.0 / ppd
    x, y = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
    z = (x * dx) ** 2 + (y * dx) ** 2
    g = a1 * np.sqrt(np.pi / b1) * np.exp(-np.pi ** 2 * z / b1) + a2 * np.sqrt(np.pi / b2) * np.exp(-np.pi ** 2 * z / b2)
    return g / g.sum()


def _filter_ycxcz(img, fa, frg, fby):
    out = np.empty_like(img)
    out[..., 0] = convolve(img[..., 0], fa, mode="nearest")
    out[..., 1] = convolve(img[..., 1], frg, mode="nearest")
    out[..., 2] = convolve(img[..., 2], fby, mode="nearest")
    lin = np.clip(_xyz_to_lin(_ycxcz_to_xyz(out)), 0.0, 1.0)
    return lin


def _hunt(lab):
    L = lab[..., 0]
    return np.stack([L, 0.01 * L * lab[..., 1], 0.01 * L * lab[..., 2]], -1)


def _hyab(a, b):
    d = a - b
    return np.abs(d[..., 0]) + np.linalg.norm(d[..., 1:], axis=-1)


def _feature_filters(ppd: float):
    w = 0.082
    sd = 0.5 * w * ppd
    r = int(np.ceil(3 * sd))
    x, y = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
    g = np.exp(-(x ** 2 + y ** 2) / (2 * sd ** 2))
    ex = -x * g
    px = (x ** 2 / sd ** 2 - 1.0) * g

    def normalise(f):
        f = f.copy()
        neg = f < 0
        pos = f > 0
        if neg.any():
            f[neg] /= -f[neg].sum()
        if pos.any():
            f[pos] /= f[pos].sum()
        return f

    ex = normalise(ex)
    px = normalise(px)
    return ex, ex.T.copy(), px, px.T.copy()


def flip_error(reference: np.ndarray, test: np.ndarray, ppd: float | None = None) -> tuple[float, np.ndarray]:
    """FLIP between two sRGB encoded images in [0, 1] of shape (H, W, 3).

    Returns (mean error, per pixel error map in [0, 1]).
    """
    ppd = pixels_per_degree() if ppd is None else ppd
    ref = np.asarray(reference, dtype=np.float64)[..., :3]
    tst = np.asarray(test, dtype=np.float64)[..., :3]
    if ref.shape != tst.shape:
        raise ValueError("images must have the same shape")
    ref_ycc = _xyz_to_ycxcz(_lin_to_xyz(srgb_decode(ref)))
    tst_ycc = _xyz_to_ycxcz(_lin_to_xyz(srgb_decode(tst)))
    fa, frg, fby = _spatial_filter(ppd, "A"), _spatial_filter(ppd, "RG"), _spatial_filter(ppd, "BY")
    ref_lin = _filter_ycxcz(ref_ycc, fa, frg, fby)
    tst_lin = _filter_ycxcz(tst_ycc, fa, frg, fby)
    # colour difference
    ref_lab = _hunt(_xyz_to_lab(_lin_to_xyz(ref_lin)))
    tst_lab = _hunt(_xyz_to_lab(_lin_to_xyz(tst_lin)))
    green = _hunt(_xyz_to_lab(_lin_to_xyz(np.array([0.0, 1.0, 0.0]))))
    blue = _hunt(_xyz_to_lab(_lin_to_xyz(np.array([0.0, 0.0, 1.0]))))
    cmax = _hyab(green, blue)
    pc, pt = 0.4, 0.95
    de = _hyab(ref_lab, tst_lab) / cmax
    de_c = np.where(de < pc, (pt / pc) * de, pt + ((de - pc) / (1.0 - pc)) * (1.0 - pt))
    # feature difference on the achromatic channel in [0, 1]
    ex, ey, px, py = _feature_filters(ppd)
    yr = (ref_ycc[..., 0] + 16.0) / 116.0
    yt = (tst_ycc[..., 0] + 16.0) / 116.0

    def feats(yimg):
        e = np.sqrt(convolve(yimg, ex, mode="nearest") ** 2 + convolve(yimg, ey, mode="nearest") ** 2)
        p = np.sqrt(convolve(yimg, px, mode="nearest") ** 2 + convolve(yimg, py, mode="nearest") ** 2)
        return e, p

    er, pr = feats(yr)
    et, pt_ = feats(yt)
    de_f = np.maximum(np.abs(er - et), np.abs(pr - pt_))
    de_f = np.clip((1.0 / np.sqrt(2.0)) * de_f, 0.0, 1.0) ** 0.5
    err = np.clip(de_c, 0, 1) ** (1.0 - de_f)
    return float(err.mean()), err


def temporal_flip(frames: list[np.ndarray], ppd: float | None = None) -> float:
    """Mean FLIP between consecutive frames (temporal stability measure, Section 10.1)."""
    vals = [flip_error(frames[i], frames[i + 1], ppd)[0] for i in range(len(frames) - 1)]
    return float(np.mean(vals)) if vals else 0.0
