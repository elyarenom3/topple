"""Traced lens system (Kolb, Mitchell and Hanrahan 1995), an option of the camera (Section 10.2).

Rays start on the film, pass through every spherical interface and the aperture stop of a real
lens prescription by Snell refraction, and leave the front element into the scene. Rays that hit
an element outside its clear aperture are blocked, which gives the lens's own vignetting, field
curvature and bokeh shape. The default prescription is the double Gauss 50 mm F/2 of Kolb et al.
(US patent 2,673,491, scaled to 50 mm), as distributed with the pbrt scenes.

Lens frame: film at z = 0, the lens towards +z (the camera's forward axis), x right and y down
as in ``scene.Camera``. Each element row is (radius, vertex z, refractive index on the film side,
clear aperture radius) in metres, ordered from the scene to the film; radius 0 is the stop.

Rays are importance sampled inside the exit pupil: for film points along +x at a set of radii,
the bounding box of the rear plane points whose rays pass the whole system is precomputed, and
the box is rotated to each film point's azimuth. The ray weight is cos^4 of the film angle times
the box area, normalised so that the centre of the image matches the thin lens exposure.
"""

from __future__ import annotations

import numpy as np
from numba import njit

# radius (mm), thickness to the next surface (mm), index after the surface, clear aperture (mm)
DOUBLE_GAUSS_50 = np.array(
    [
        [29.475, 3.76, 1.67, 25.2],
        [84.83, 0.12, 1.0, 25.2],
        [19.275, 4.025, 1.67, 23.0],
        [40.77, 3.275, 1.699, 23.0],
        [12.75, 5.705, 1.0, 18.0],
        [0.0, 4.5, 0.0, 17.1],
        [-14.495, 1.18, 1.603, 17.0],
        [40.77, 6.065, 1.658, 20.0],
        [-20.385, 0.19, 1.0, 20.0],
        [437.065, 3.22, 1.717, 20.0],
        [-39.73, 0.0, 1.0, 20.0],
    ]
)
DESIGN_F_NUMBER = 2.0
LENS_BASE = 24  # first camera array slot of the lens data
MAX_ELEMENTS = 16
PUPIL_BASE = LENS_BASE + 4 * MAX_ELEMENTS  # 88: n_bins, film radius, normalisation, rear z
N_PUPIL_BINS = 32
CAM_ARRAY_SIZE = PUPIL_BASE + 4 + 4 * N_PUPIL_BINS


@njit(cache=True)
def lens_trace(cam, ox, oy, oz, dx, dy, dz, out):
    """Trace a ray from the film through the lens elements stored in ``cam``. Writes the exit
    origin and direction (lens frame) into out[0:6]; returns False if the ray is blocked."""
    n = int(cam[LENS_BASE - 1])
    for k in range(n - 1, -1, -1):
        b = LENS_BASE + 4 * k
        R = cam[b]
        zv = cam[b + 1]
        eta_i = cam[b + 2]
        ap = cam[b + 3]
        if R == 0.0:
            if dz <= 0.0:
                return False
            t = (zv - oz) / dz
            zc = 0.0
        else:
            zc = zv - R
            qz = oz - zc
            B = ox * dx + oy * dy + qz * dz
            C = ox * ox + oy * oy + qz * qz - R * R
            disc = B * B - C
            if disc < 0.0:
                return False
            sq = np.sqrt(disc)
            t = -B - sq if R < 0.0 else -B + sq
        if t < 0.0:
            return False
        hx = ox + t * dx
        hy = oy + t * dy
        hz = oz + t * dz
        if hx * hx + hy * hy > ap * ap:
            return False
        ox, oy, oz = hx, hy, hz
        if R != 0.0:
            nx, ny, nz = hx, hy, hz - zc
            ln = np.sqrt(nx * nx + ny * ny + nz * nz)
            nx /= ln
            ny /= ln
            nz /= ln
            if nx * dx + ny * dy + nz * dz > 0.0:
                nx, ny, nz = -nx, -ny, -nz
            eta_t = 1.0
            if k > 0 and cam[LENS_BASE + 4 * (k - 1) + 2] != 0.0:
                eta_t = cam[LENS_BASE + 4 * (k - 1) + 2]
            if eta_i == 0.0:
                eta_i = 1.0
            eta = eta_i / eta_t
            cosi = -(nx * dx + ny * dy + nz * dz)
            sin2t = eta * eta * max(0.0, 1.0 - cosi * cosi)
            if sin2t >= 1.0:
                return False
            cost = np.sqrt(1.0 - sin2t)
            dx = eta * dx + (eta * cosi - cost) * nx
            dy = eta * dy + (eta * cosi - cost) * ny
            dz = eta * dz + (eta * cosi - cost) * nz
            ln = np.sqrt(dx * dx + dy * dy + dz * dz)
            dx /= ln
            dy /= ln
            dz /= ln
    out[0], out[1], out[2] = ox, oy, oz
    out[3], out[4], out[5] = dx, dy, dz
    return True


@njit(cache=True)
def lens_sample(cam, fx, fy, u2, u3, out):
    """Sample a ray from film point (fx, fy, 0) through the exit pupil. Returns the weight
    (0 when blocked); the exit ray (lens frame) is in out[0:6]."""
    nb = int(cam[PUPIL_BASE])
    rmax = cam[PUPIL_BASE + 1]
    norm = cam[PUPIL_BASE + 2]
    zr = cam[PUPIL_BASE + 3]
    r = np.sqrt(fx * fx + fy * fy)
    bi = min(int(r / rmax * nb), nb - 1)
    b = PUPIL_BASE + 4 + 4 * bi
    x0, x1, y0, y1 = cam[b], cam[b + 1], cam[b + 2], cam[b + 3]
    area = (x1 - x0) * (y1 - y0)
    if area <= 0.0:
        return 0.0
    px = x0 + u2 * (x1 - x0)
    py = y0 + u3 * (y1 - y0)
    if r > 0.0:
        c = fx / r
        s = fy / r
    else:
        c = 1.0
        s = 0.0
    qx = c * px - s * py
    qy = s * px + c * py
    dx = qx - fx
    dy = qy - fy
    dz = zr
    ln = np.sqrt(dx * dx + dy * dy + dz * dz)
    dx /= ln
    dy /= ln
    dz /= ln
    if not lens_trace(cam, fx, fy, 0.0, dx, dy, dz, out):
        return 0.0
    c4 = dz * dz * dz * dz
    return c4 * area / norm


@njit(cache=True)
def _pupil_box(cam, radii, xs, zr):
    """Bounding box of the rear plane grid points whose rays from film points (r, 0, 0) pass."""
    out = np.zeros(6)
    x0, x1, y0, y1 = 1e300, -1e300, 1e300, -1e300
    for rr in radii:
        for x in xs:
            for y in xs:
                dx = x - rr
                dy = y
                dz = zr
                ln = np.sqrt(dx * dx + dy * dy + dz * dz)
                if lens_trace(cam, rr, 0.0, 0.0, dx / ln, dy / ln, dz / ln, out):
                    x0 = min(x0, x)
                    x1 = max(x1, x)
                    y0 = min(y0, y)
                    y1 = max(y1, y)
    return x0, x1, y0, y1


class LensSystem:
    """A spherical lens prescription focused at a given distance."""

    def __init__(self, rows=DOUBLE_GAUSS_50, focal_mm: float = 50.0, f_number: float = 2.0, design_f_number: float = DESIGN_F_NUMBER):
        rows = np.asarray(rows, dtype=float).copy()
        scale = focal_mm / 50.0 * 1e-3  # mm to m, prescription scaled to the focal length
        self.radius = rows[:, 0] * scale
        self.thickness = rows[:, 1] * scale
        self.eta = rows[:, 2]
        self.aperture = rows[:, 3] * 0.5 * scale
        stop = np.where(self.radius == 0.0)[0]
        if len(stop):
            self.aperture[stop] *= min(1.0, design_f_number / max(f_number, 1e-6))
        self.film_distance = 0.05 * focal_mm / 50.0

    # -------------------------------------------------------------- geometry
    def vertices_z(self, film_distance=None) -> np.ndarray:
        d = self.film_distance if film_distance is None else film_distance
        z = np.zeros(len(self.radius))
        z[-1] = d
        for k in range(len(z) - 2, -1, -1):
            z[k] = z[k + 1] + self.thickness[k]
        return z

    def pack(self, cam: np.ndarray, film_distance=None) -> None:
        n = len(self.radius)
        if n > MAX_ELEMENTS:
            raise ValueError("too many lens elements")
        z = self.vertices_z(film_distance)
        cam[LENS_BASE - 1] = n
        for k in range(n):
            cam[LENS_BASE + 4 * k: LENS_BASE + 4 * k + 4] = (self.radius[k], z[k], self.eta[k], self.aperture[k])

    def _cam(self, film_distance=None) -> np.ndarray:
        cam = np.zeros(CAM_ARRAY_SIZE)
        self.pack(cam, film_distance)
        return cam

    def trace(self, o, d, film_distance=None):
        out = np.zeros(6)
        ok = lens_trace(self._cam(film_distance), float(o[0]), float(o[1]), float(o[2]), float(d[0]), float(d[1]), float(d[2]), out)
        return (out[:3], out[3:]) if ok else None

    # -------------------------------------------------------------- paraxial properties
    def effective_focal_length(self) -> float:
        """A ray parallel to the axis from the film side converges to the front focal point; the
        focal length is its height over the tangent of its exit angle."""
        h = 1e-4 * self.aperture.max() / 0.01
        res = self.trace([h, 0.0, 0.0], [0.0, 0.0, 1.0])
        o, d = res
        return float(h / (-d[0] / d[2]))

    def object_distance(self, film_distance: float) -> float:
        """Distance from the film to the on axis point imaged at the film centre (paraxial)."""
        zr = self.vertices_z(film_distance)[-1]
        h = 1e-3 * self.aperture[-1]
        res = self.trace([0.0, 0.0, 0.0], np.array([h, 0.0, zr]) / np.hypot(h, zr), film_distance)
        if res is None:
            return np.nan
        o, d = res
        if abs(d[0]) < 1e-15:
            return np.inf
        return float(o[2] - o[0] * d[2] / d[0])

    def focus(self, distance: float) -> float:
        """Film distance that brings ``distance`` (from the film) into focus, by bisection on the
        inverse object distance."""
        f = self.effective_focal_length()

        def g(dfilm):
            s = self.object_distance(dfilm)
            return 1.0 / s - 1.0 / distance

        lo, hi = 0.2 * f, 3.0 * f
        glo, ghi = g(lo), g(hi)
        if not (np.isfinite(glo) and np.isfinite(ghi)) or glo * ghi > 0:
            # scan for a bracket
            xs = np.linspace(0.2 * f, 3.0 * f, 200)
            gs = np.array([g(x) for x in xs])
            idx = np.where(np.isfinite(gs[:-1]) & np.isfinite(gs[1:]) & (np.sign(gs[:-1]) != np.sign(gs[1:])))[0]
            if len(idx) == 0:
                raise RuntimeError("lens cannot focus at this distance")
            lo, hi = xs[idx[0]], xs[idx[0] + 1]
            glo = g(lo)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            gm = g(mid)
            if np.sign(gm) == np.sign(glo):
                lo, glo = mid, gm
            else:
                hi = mid
        self.film_distance = 0.5 * (lo + hi)
        return self.film_distance

    # -------------------------------------------------------------- exit pupil
    def pack_pupil(self, cam: np.ndarray, film_half_diagonal: float, grid: int = 96) -> None:
        """Exit pupil boxes for N_PUPIL_BINS film radii and the centre normalisation."""
        self.pack(cam)
        zr = self.vertices_z()[-1]
        A = 1.5 * self.aperture[-1]
        xs = np.linspace(-A, A, grid)
        cell = xs[1] - xs[0]
        out = np.zeros(6)
        cam[PUPIL_BASE] = N_PUPIL_BINS
        cam[PUPIL_BASE + 1] = film_half_diagonal
        cam[PUPIL_BASE + 3] = zr
        for b in range(N_PUPIL_BINS):
            # the bin box covers the radial interval: union of the boxes at its ends and middle
            radii = film_half_diagonal * np.array([b, b + 0.5, b + 1.0]) / N_PUPIL_BINS
            x0, x1, y0, y1 = _pupil_box(cam, radii, xs, zr)
            k = PUPIL_BASE + 4 + 4 * b
            cam[k: k + 4] = (x0 - cell, x1 + cell, y0 - cell, y1 + cell) if x1 >= x0 else 0.0
        # normalisation: mean weight at the image centre equals 1
        cam[PUPIL_BASE + 2] = 1.0
        rng = np.random.default_rng(0)
        u = rng.random((8192, 2))
        w = np.array([lens_sample(cam, 0.0, 0.0, a, b_, out) for a, b_ in u])
        cam[PUPIL_BASE + 2] = max(float(w.mean()), 1e-12)


def pack_camera_lens(cam: np.ndarray, focal_mm: float, f_number: float, focus_distance: float, sensor_w: float, sensor_h: float) -> np.ndarray:
    """Camera array with the double Gauss lens focused at ``focus_distance`` (from the film)."""
    lens = LensSystem(DOUBLE_GAUSS_50, focal_mm, f_number)
    lens.focus(focus_distance)
    out = np.zeros(CAM_ARRAY_SIZE)
    out[: len(cam)] = cam
    lens.pack_pupil(out, 0.5 * float(np.hypot(sensor_w, sensor_h)))
    return out
