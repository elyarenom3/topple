"""Gradient noise (Perlin 1985), fractal Brownian motion and Worley cellular noise.

All functions are vectorised over an (n, 3) array of sample points and are deterministic
for a given integer seed.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

_GRAD = np.array(
    [
        [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
        [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
        [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
        [1, 1, 0], [-1, 1, 0], [0, -1, 1], [0, -1, -1],
    ],
    dtype=np.float64,
)


@lru_cache(maxsize=64)
def _perm(seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    p = rng.permutation(256)
    return np.concatenate([p, p]).astype(np.int64)


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def perlin3(x: np.ndarray, seed: int = 0) -> np.ndarray:
    """Improved Perlin noise in three dimensions, values roughly in [-1, 1], zero at lattice points."""
    x = np.asarray(x, dtype=np.float64).reshape(-1, 3)
    perm = _perm(seed)
    xi = np.floor(x).astype(np.int64)
    xf = x - xi
    xi &= 255
    u, v, w = _fade(xf[:, 0]), _fade(xf[:, 1]), _fade(xf[:, 2])

    def h(i, j, k):
        return perm[perm[perm[(xi[:, 0] + i) & 255] + ((xi[:, 1] + j) & 255)] + ((xi[:, 2] + k) & 255)]

    def grad(hv, dx, dy, dz):
        g = _GRAD[hv & 15]
        return g[:, 0] * dx + g[:, 1] * dy + g[:, 2] * dz

    fx, fy, fz = xf[:, 0], xf[:, 1], xf[:, 2]
    n000 = grad(h(0, 0, 0), fx, fy, fz)
    n100 = grad(h(1, 0, 0), fx - 1, fy, fz)
    n010 = grad(h(0, 1, 0), fx, fy - 1, fz)
    n110 = grad(h(1, 1, 0), fx - 1, fy - 1, fz)
    n001 = grad(h(0, 0, 1), fx, fy, fz - 1)
    n101 = grad(h(1, 0, 1), fx - 1, fy, fz - 1)
    n011 = grad(h(0, 1, 1), fx, fy - 1, fz - 1)
    n111 = grad(h(1, 1, 1), fx - 1, fy - 1, fz - 1)

    def lerp(a, b, t):
        return a + t * (b - a)

    x00 = lerp(n000, n100, u)
    x10 = lerp(n010, n110, u)
    x01 = lerp(n001, n101, u)
    x11 = lerp(n011, n111, u)
    y0 = lerp(x00, x10, v)
    y1 = lerp(x01, x11, v)
    return lerp(y0, y1, w)


def fbm3(x: np.ndarray, octaves: int = 4, gain: float = 0.5, lacunarity: float = 2.0, seed: int = 0) -> np.ndarray:
    """Sum of octaves gain^o * n(lacunarity^o x) (Section 5.1, family B)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1, 3)
    out = np.zeros(x.shape[0])
    amp = 1.0
    freq = 1.0
    for o in range(octaves):
        out += amp * perlin3(x * freq, seed + 1000 * o)
        amp *= gain
        freq *= lacunarity
    return out


def worley3(x: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Cellular noise (Worley 1996): distances F1 and F2 to the nearest feature points."""
    x = np.asarray(x, dtype=np.float64).reshape(-1, 3)
    cell = np.floor(x).astype(np.int64)
    f1 = np.full(x.shape[0], np.inf)
    f2 = np.full(x.shape[0], np.inf)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                c = cell + np.array([di, dj, dk])
                hsh = _hash3(c, seed)
                off = np.stack([(hsh & 1023) / 1023.0, ((hsh >> 10) & 1023) / 1023.0, ((hsh >> 20) & 1023) / 1023.0], 1)
                fp = c + off
                d = np.linalg.norm(x - fp, axis=1)
                m = d < f1
                f2 = np.where(m, f1, np.minimum(f2, d))
                f1 = np.where(m, d, f1)
    return f1, f2


def _hash3(c: np.ndarray, seed: int) -> np.ndarray:
    h = (c[:, 0] * 73856093) ^ (c[:, 1] * 19349663) ^ (c[:, 2] * 83492791) ^ (int(seed) * 2654435761)
    h = h & 0x7FFFFFFF
    h = (h ^ (h >> 13)) * 1274126177 & 0x7FFFFFFF
    return h ^ (h >> 16)
