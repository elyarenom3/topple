"""Procedural textures evaluated in object space (Section 10.3), as numba functions.

* Gradient noise (Perlin 1985) and fractal Brownian motion (Musgrave, Kolb and Mace 1989).
* Worley cellular noise (Worley 1996) for grain and crystal boundaries.
* Sedimentary banding: a sine of distance along a bedding direction, phase warped by fBm.
* Lichen: thresholded low frequency noise on upward facing surfaces.
* Weathering by curvature: convex edges lighter and smoother, concave regions darker and
  rougher, from per vertex curvature (Rusinkiewicz 2004 style estimate in the mesh module) and a
  cosine weighted hemisphere occlusion computed by the renderer.
* Triplanar projection (Geiss 2007) for 2D patterns such as the micro relief.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, inline="always")
def _hash3(ix, iy, iz, seed):
    h = np.uint64(ix) * np.uint64(0x8DA6B343) ^ np.uint64(iy) * np.uint64(0xD8163841) ^ np.uint64(iz) * np.uint64(0xCB1AB31F) ^ np.uint64(seed) * np.uint64(0x9E3779B1)
    h = (h ^ (h >> np.uint64(13))) * np.uint64(0x5BD1E995)
    h = h ^ (h >> np.uint64(15))
    return h & np.uint64(0xFFFFFFFF)


@njit(cache=True, inline="always")
def hash01(ix, iy, iz, seed):
    return float(_hash3(ix, iy, iz, seed)) / 4294967296.0


@njit(cache=True, inline="always")
def _grad(h, x, y, z):
    # 12 gradient directions of Perlin's improved noise
    k = int(h & np.uint64(15))
    u = x if k < 8 else y
    v = y if k < 4 else (x if (k == 12 or k == 14) else z)
    return (u if (k & 1) == 0 else -u) + (v if (k & 2) == 0 else -v)


@njit(cache=True, inline="always")
def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@njit(cache=True)
def perlin3(x, y, z, seed):
    """Improved Perlin gradient noise in roughly [-1, 1]."""
    ix = int(np.floor(x))
    iy = int(np.floor(y))
    iz = int(np.floor(z))
    fx = x - ix
    fy = y - iy
    fz = z - iz
    u = _fade(fx)
    v = _fade(fy)
    w = _fade(fz)
    n000 = _grad(_hash3(ix, iy, iz, seed), fx, fy, fz)
    n100 = _grad(_hash3(ix + 1, iy, iz, seed), fx - 1, fy, fz)
    n010 = _grad(_hash3(ix, iy + 1, iz, seed), fx, fy - 1, fz)
    n110 = _grad(_hash3(ix + 1, iy + 1, iz, seed), fx - 1, fy - 1, fz)
    n001 = _grad(_hash3(ix, iy, iz + 1, seed), fx, fy, fz - 1)
    n101 = _grad(_hash3(ix + 1, iy, iz + 1, seed), fx - 1, fy, fz - 1)
    n011 = _grad(_hash3(ix, iy + 1, iz + 1, seed), fx, fy - 1, fz - 1)
    n111 = _grad(_hash3(ix + 1, iy + 1, iz + 1, seed), fx - 1, fy - 1, fz - 1)
    x00 = n000 + u * (n100 - n000)
    x10 = n010 + u * (n110 - n010)
    x01 = n001 + u * (n101 - n001)
    x11 = n011 + u * (n111 - n011)
    y0 = x00 + v * (x10 - x00)
    y1 = x01 + v * (x11 - x01)
    return y0 + w * (y1 - y0)


@njit(cache=True)
def fbm3(x, y, z, octaves, gain, lacunarity, seed):
    """Fractal Brownian motion, normalised to roughly [-1, 1]."""
    amp = 1.0
    freq = 1.0
    total = 0.0
    norm = 0.0
    for o in range(octaves):
        total += amp * perlin3(x * freq, y * freq, z * freq, seed + o)
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / norm


@njit(cache=True)
def fbm3_grad(x, y, z, octaves, gain, lacunarity, seed, h):
    """Central difference gradient of fbm3 (used for bump mapping)."""
    gx = (fbm3(x + h, y, z, octaves, gain, lacunarity, seed) - fbm3(x - h, y, z, octaves, gain, lacunarity, seed)) / (2 * h)
    gy = (fbm3(x, y + h, z, octaves, gain, lacunarity, seed) - fbm3(x, y - h, z, octaves, gain, lacunarity, seed)) / (2 * h)
    gz = (fbm3(x, y, z + h, octaves, gain, lacunarity, seed) - fbm3(x, y, z - h, octaves, gain, lacunarity, seed)) / (2 * h)
    return gx, gy, gz


@njit(cache=True)
def worley3(x, y, z, seed):
    """Cellular noise: returns (F1, F2, cell hash in [0, 1)) with one feature point per cell."""
    ix = int(np.floor(x))
    iy = int(np.floor(y))
    iz = int(np.floor(z))
    f1 = 1e30
    f2 = 1e30
    cell = 0.0
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                cx = ix + dx
                cy = iy + dy
                cz = iz + dz
                px = cx + hash01(cx, cy, cz, seed)
                py = cy + hash01(cx, cy, cz, seed + 1)
                pz = cz + hash01(cx, cy, cz, seed + 2)
                d = (px - x) ** 2 + (py - y) ** 2 + (pz - z) ** 2
                if d < f1:
                    f2 = f1
                    f1 = d
                    cell = hash01(cx, cy, cz, seed + 3)
                elif d < f2:
                    f2 = d
    return np.sqrt(f1), np.sqrt(f2), cell


@njit(cache=True)
def triplanar_weights(nx, ny, nz, sharpness):
    wx = abs(nx) ** sharpness
    wy = abs(ny) ** sharpness
    wz = abs(nz) ** sharpness
    s = wx + wy + wz
    if s <= 0:
        return 0.0, 0.0, 1.0
    return wx / s, wy / s, wz / s


@njit(cache=True)
def pattern2d(u, v, seed):
    """A 2D high frequency pattern (two octave value noise) used through triplanar projection."""
    iu = int(np.floor(u))
    iv = int(np.floor(v))
    fu = u - iu
    fv = v - iv
    fu = fu * fu * (3 - 2 * fu)
    fv = fv * fv * (3 - 2 * fv)
    a = hash01(iu, iv, 0, seed)
    b = hash01(iu + 1, iv, 0, seed)
    c = hash01(iu, iv + 1, 0, seed)
    d = hash01(iu + 1, iv + 1, 0, seed)
    return (a + fu * (b - a)) + fv * ((c + fu * (d - c)) - (a + fu * (b - a))) - 0.5


@njit(cache=True)
def triplanar(px, py, pz, nx, ny, nz, scale, seed):
    """Triplanar projection of ``pattern2d`` evaluated at object space point p with normal n."""
    wx, wy, wz = triplanar_weights(nx, ny, nz, 4.0)
    return wx * pattern2d(py * scale, pz * scale, seed) + wy * pattern2d(px * scale, pz * scale, seed + 7) + wz * pattern2d(px * scale, py * scale, seed + 13)


@njit(cache=True)
def banding(px, py, pz, dx, dy, dz, freq, warp, seed):
    """Sedimentary banding in [-1, 1]: sine of distance along the bedding direction with a warped phase."""
    s = px * dx + py * dy + pz * dz
    ph = warp * fbm3(px * freq * 0.5, py * freq * 0.5, pz * freq * 0.5, 3, 0.5, 2.0, seed)
    return np.sin(2.0 * np.pi * freq * s + ph)


def hemisphere_occlusion(mesh, n_samples: int = 64, radius: float | None = None, seed: int = 0) -> np.ndarray:
    """Cosine weighted hemisphere occlusion per vertex, computed with the mesh BVH.

    Returns values in [0, 1] (1 = fully open). The ray length is limited to ``radius`` (default
    a tenth of the bounding radius) so that only local cavities count.
    """
    from ..core.raycast import BVH

    V = mesh.vertices
    N = mesh.vertex_normals()
    bvh = mesh.bvh()
    r = 0.1 * mesh.bounding_radius() if radius is None else radius
    rng = np.random.default_rng(seed)
    occ = np.zeros(mesh.n_vertices)
    u1 = rng.random((n_samples,))
    u2 = rng.random((n_samples,))
    rr = np.sqrt(u1)
    phi = 2 * np.pi * u2
    local = np.stack([rr * np.cos(phi), rr * np.sin(phi), np.sqrt(np.maximum(0, 1 - u1))], 1)
    for k in range(n_samples):
        # build a frame per vertex
        n = N
        a = np.where(np.abs(n[:, 0:1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
        t = np.cross(a, n)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
        b = np.cross(n, t)
        d = local[k, 0] * t + local[k, 1] * b + local[k, 2] * n
        o = V + 1e-4 * n
        hit = bvh.occluded(o, d, 1e-6, r)
        occ += hit
    return 1.0 - occ / n_samples


def vertex_attributes(mesh, n_occlusion_samples: int = 32) -> np.ndarray:
    """Per vertex (normalised curvature in [-1, 1], occlusion in [0, 1]) for weathering."""
    H = mesh.mean_curvature()
    scale = np.percentile(np.abs(H), 90) + 1e-12
    c = np.clip(H / scale, -1, 1)
    o = hemisphere_occlusion(mesh, n_occlusion_samples)
    return np.stack([c, o], 1)
