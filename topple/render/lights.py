"""Light sources (Section 10.4).

* ``AreaLight``: a one sided rectangular emitter (origin, edge vectors u and v, radiance).
* ``SunLight``: a directional disc of angular diameter 0.53 degrees with constant radiance.
* ``Sky``: the analytic sky of ``sky.py`` or a measured lat long HDR environment map
  importance sampled by luminance.

The light table passed to the kernels has one row per light:
col 0 type (0 quad, 1 sun, 2 sky, 3 hdr), 1:4 origin or direction, 4:7 u, 7:10 v, 10:13 radiance,
13 area or cos half angle, 14:17 normal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .sky import sky_params, sun_radiance_rgb, SUN_ANGULAR_DIAMETER_DEG

LIGHT_COLS = 20


@dataclass
class AreaLight:
    origin: np.ndarray
    u: np.ndarray
    v: np.ndarray
    radiance: np.ndarray
    name: str = "area"

    @staticmethod
    def rect(center, normal, up, width, height, radiance, name="area") -> "AreaLight":
        """Rectangle of given size centred at ``center`` facing ``normal``."""
        n = np.asarray(normal, dtype=np.float64)
        n = n / np.linalg.norm(n)
        up = np.asarray(up, dtype=np.float64)
        t = np.cross(up, n)
        if np.linalg.norm(t) < 1e-9:
            t = np.cross([1.0, 0, 0], n)
        t /= np.linalg.norm(t)
        b = np.cross(n, t)
        u = t * width
        v = b * height
        origin = np.asarray(center, dtype=np.float64) - 0.5 * u - 0.5 * v
        # ensure u x v points along n (emission side)
        if np.dot(np.cross(u, v), n) < 0:
            u, v = v, u
        return AreaLight(origin, u, v, np.asarray(radiance, dtype=np.float64) * np.ones(3), name)

    @property
    def normal(self) -> np.ndarray:
        n = np.cross(self.u, self.v)
        return n / np.linalg.norm(n)

    @property
    def area(self) -> float:
        return float(np.linalg.norm(np.cross(self.u, self.v)))

    def row(self) -> np.ndarray:
        r = np.zeros(LIGHT_COLS)
        r[0] = 0
        r[1:4] = self.origin
        r[4:7] = self.u
        r[7:10] = self.v
        r[10:13] = self.radiance
        r[13] = self.area
        r[14:17] = self.normal
        return r

    def mesh(self):
        from ..core.mesh import TriMesh

        V = np.array([self.origin, self.origin + self.u, self.origin + self.u + self.v, self.origin + self.v])
        F = np.array([[0, 1, 2], [0, 2, 3]])
        return TriMesh(V, F, self.name)


@dataclass
class SunLight:
    direction: np.ndarray  # unit vector towards the sun
    radiance: np.ndarray
    angular_diameter_deg: float = SUN_ANGULAR_DIAMETER_DEG
    name: str = "sun"

    @staticmethod
    def from_elevation(elevation_deg, azimuth_deg=0.0, turbidity=3.0, scale=1.0, radiance=None) -> "SunLight":
        el, az = np.radians(elevation_deg), np.radians(azimuth_deg)
        d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        if radiance is None:
            radiance = sun_radiance_rgb(np.pi / 2 - el, turbidity) * scale
        return SunLight(d, np.asarray(radiance, dtype=np.float64) * np.ones(3))

    @property
    def cos_half(self) -> float:
        return float(np.cos(np.radians(self.angular_diameter_deg / 2)))

    def irradiance(self) -> np.ndarray:
        """Irradiance on a surface facing the sun."""
        return self.radiance * 2 * np.pi * (1 - self.cos_half)

    def row(self) -> np.ndarray:
        r = np.zeros(LIGHT_COLS)
        r[0] = 1
        r[1:4] = self.direction / np.linalg.norm(self.direction)
        r[10:13] = self.radiance
        r[13] = self.cos_half
        return r


@dataclass
class Sky:
    """Analytic sky (``params`` from ``sky.sky_params``) or HDR environment (``image`` lat long)."""

    params: np.ndarray | None = None
    image: np.ndarray | None = None
    scale: float = 1.0
    rotation_deg: float = 0.0
    name: str = "sky"

    @staticmethod
    def analytic(elevation_deg, turbidity=3.0, azimuth_deg=0.0, scale=1.0, model: str = "hosek", albedo: float = 0.2) -> "Sky":
        """Hošek Wilkie (default) or Preetham sky; ``albedo`` is the Hošek Wilkie ground albedo."""
        return Sky(params=sky_params(elevation_deg, turbidity, azimuth_deg, scale, model=model, albedo=albedo))

    @staticmethod
    def hdr(image: np.ndarray, scale: float = 1.0, rotation_deg: float = 0.0) -> "Sky":
        return Sky(image=np.ascontiguousarray(image, dtype=np.float64), scale=scale, rotation_deg=rotation_deg)

    @staticmethod
    def hdr_from_file(path, scale: float = 1.0, rotation_deg: float = 0.0) -> "Sky":
        import imageio.v3 as iio

        img = np.asarray(iio.imread(str(path))).astype(np.float64)
        if img.max() > 1.5 and img.dtype == np.uint8:
            img = img / 255.0
        return Sky.hdr(img[..., :3], scale, rotation_deg)

    def row(self) -> np.ndarray:
        r = np.zeros(LIGHT_COLS)
        r[0] = 2 if self.image is None else 3
        r[13] = self.scale
        r[14] = np.radians(self.rotation_deg)
        return r

    def env_tables(self):
        """Marginal and conditional CDFs for luminance importance sampling of the HDR map."""
        if self.image is None:
            return np.zeros((1, 1, 3)), np.zeros(1), np.zeros((1, 1))
        img = self.image
        H, W = img.shape[:2]
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        theta = (np.arange(H) + 0.5) / H * np.pi
        w = lum * np.sin(theta)[:, None] + 1e-8
        cond = np.cumsum(w, axis=1)
        marg_w = cond[:, -1].copy()
        cond = cond / cond[:, -1:]
        marg = np.cumsum(marg_w)
        marg = marg / marg[-1]
        return np.ascontiguousarray(img), marg, np.ascontiguousarray(cond)
