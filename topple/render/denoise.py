"""Denoising of path traced frames (Section 10.1).

Intel Open Image Denoise is used when its Python bindings are importable (``oidn``). Otherwise
a joint bilateral filter guided by the albedo and normal buffers is applied, with the colour
range term on the demodulated radiance (beauty divided by albedo), which keeps texture detail.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def _joint_bilateral(img, albedo, normal, radius, sigma_s, sigma_a, sigma_n, sigma_r):
    H, W = img.shape[0], img.shape[1]
    out = np.zeros_like(img)
    for y in prange(H):
        for x in range(W):
            wsum = 0.0
            acc0 = 0.0
            acc1 = 0.0
            acc2 = 0.0
            a0, a1, a2 = albedo[y, x, 0], albedo[y, x, 1], albedo[y, x, 2]
            n0, n1, n2 = normal[y, x, 0], normal[y, x, 1], normal[y, x, 2]
            c0, c1, c2 = img[y, x, 0], img[y, x, 1], img[y, x, 2]
            for dy in range(-radius, radius + 1):
                yy = y + dy
                if yy < 0 or yy >= H:
                    continue
                for dx in range(-radius, radius + 1):
                    xx = x + dx
                    if xx < 0 or xx >= W:
                        continue
                    ds = (dx * dx + dy * dy) / (2.0 * sigma_s * sigma_s)
                    da = ((albedo[yy, xx, 0] - a0) ** 2 + (albedo[yy, xx, 1] - a1) ** 2 + (albedo[yy, xx, 2] - a2) ** 2) / (2.0 * sigma_a * sigma_a)
                    dn = ((normal[yy, xx, 0] - n0) ** 2 + (normal[yy, xx, 1] - n1) ** 2 + (normal[yy, xx, 2] - n2) ** 2) / (2.0 * sigma_n * sigma_n)
                    dr = ((img[yy, xx, 0] - c0) ** 2 + (img[yy, xx, 1] - c1) ** 2 + (img[yy, xx, 2] - c2) ** 2) / (2.0 * sigma_r * sigma_r)
                    w = np.exp(-ds - da - dn - dr)
                    wsum += w
                    acc0 += w * img[yy, xx, 0]
                    acc1 += w * img[yy, xx, 1]
                    acc2 += w * img[yy, xx, 2]
            out[y, x, 0] = acc0 / wsum
            out[y, x, 1] = acc1 / wsum
            out[y, x, 2] = acc2 / wsum
    return out


def denoise(beauty: np.ndarray, albedo: np.ndarray | None = None, normal: np.ndarray | None = None, radius: int = 5, sigma_s: float = 2.5, sigma_a: float = 0.08, sigma_n: float = 0.25, sigma_r: float = 0.6) -> np.ndarray:
    beauty = np.ascontiguousarray(beauty, dtype=np.float64)
    if albedo is None:
        albedo = np.ones_like(beauty)
    if normal is None:
        normal = np.zeros_like(beauty)
    try:  # pragma: no cover
        import oidn  # type: ignore

        dev = oidn.NewDevice()
        oidn.CommitDevice(dev)
        f = oidn.NewFilter(dev, "RT")
        col = beauty.astype(np.float32)
        out = np.zeros_like(col)
        oidn.SetSharedFilterImage(f, "color", col, oidn.FORMAT_FLOAT3, col.shape[1], col.shape[0])
        oidn.SetSharedFilterImage(f, "albedo", albedo.astype(np.float32), oidn.FORMAT_FLOAT3, col.shape[1], col.shape[0])
        oidn.SetSharedFilterImage(f, "normal", normal.astype(np.float32), oidn.FORMAT_FLOAT3, col.shape[1], col.shape[0])
        oidn.SetSharedFilterImage(f, "output", out, oidn.FORMAT_FLOAT3, col.shape[1], col.shape[0])
        oidn.CommitFilter(f)
        oidn.ExecuteFilter(f)
        return out.astype(np.float64)
    except Exception:
        pass
    alb = np.ascontiguousarray(np.clip(albedo, 1e-3, None), dtype=np.float64)
    demod = beauty / alb
    scale = max(np.percentile(demod, 90), 1e-6)
    filtered = _joint_bilateral(np.ascontiguousarray(demod / scale), alb, np.ascontiguousarray(normal, dtype=np.float64), radius, sigma_s, sigma_a, sigma_n, sigma_r)
    return filtered * scale * alb
