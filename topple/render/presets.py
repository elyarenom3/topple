"""Stored presets (Sections 10.3, 10.4, 10.7): lighting, stone materials, ground, cameras."""

from __future__ import annotations

import numpy as np

from ..core.mesh import TriMesh, box, plane
from .lights import AreaLight, Sky, SunLight
from .materials import Material

# -------------------------------------------------------------------- lighting


def lighting_preset(name: str, center=(0.0, 0.0, 0.2), size: float = 0.6, sun_azimuth_deg: float = 210.0):
    """Return (lights, sky, background) for a named preset of Section 10.4.

    ``center`` and ``size`` place the lights relative to the subject (the tower).
    """
    c = np.asarray(center, dtype=np.float64)
    if name == "dusk":
        sky = Sky.analytic(8.0, 3.0, sun_azimuth_deg, 1.0)
        sun = SunLight.from_elevation(8.0, sun_azimuth_deg, 3.0)
        return [sun], sky, np.zeros(3)
    if name == "studio":
        # key: 1 x 1 m at 45 degrees elevation, 30 degrees azimuth, 1.5 m away
        d = 1.5
        el, az = np.radians(45.0), np.radians(30.0)
        kp = c + d * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        key = AreaLight.rect(kp, c - kp, [0, 0, 1], 1.0, 1.0, 6.0, "key")
        # fill: 2 x 1 m at one quarter intensity from the other side, lower
        el, az = np.radians(20.0), np.radians(-70.0)
        fp = c + 2.0 * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        fill = AreaLight.rect(fp, c - fp, [0, 0, 1], 2.0, 1.0, 1.5, "fill")
        # rim: thin strip behind the tower
        rp = c + np.array([-0.3, 1.4, 0.9])
        rim = AreaLight.rect(rp, c - rp, [0, 0, 1], 1.2, 0.08, 20.0, "rim")
        return [key, fill, rim], None, np.array([0.02, 0.02, 0.022])
    if name == "macro":
        el = np.radians(5.0)
        lp = c + 0.4 * np.array([np.cos(el), 0.0, np.sin(el)])
        light = AreaLight.rect(lp, c - lp, [0, 0, 1], 0.1, 0.1, 40.0, "grazing")
        return [light], None, np.zeros(3)
    raise KeyError(f"unknown lighting preset {name}")


# -------------------------------------------------------------------- materials

_STONE_PRESETS = {
    "river_basalt": dict(base_color=(0.10, 0.10, 0.105), roughness=0.42, coat=0.8, wet_mask_scale=20.0, fbm_amp=0.25, fbm_scale=40.0, roughness_variation=0.08, weathering_amp=0.6, bump_amp=0.25, diffuse_model=1),
    "granite": dict(base_color=(0.40, 0.38, 0.36), roughness=0.6, fbm_amp=0.2, fbm_scale=60.0, worley_amp=0.6, worley_scale=400.0, flakes_density=0.06, flakes_intensity=0.4, flakes_size=0.0006, weathering_amp=0.4, bump_amp=0.3, diffuse_model=1),
    "sandstone": dict(base_color=(0.48, 0.38, 0.26), roughness=0.85, banding_amp=0.18, banding_freq=60.0, banding_dir=(0.1, 0.0, 1.0), fbm_amp=0.15, fbm_scale=80.0, lichen_threshold=0.45, lichen_scale=18.0, weathering_amp=0.5, bump_amp=0.35, diffuse_model=1),
    "slate": dict(base_color=(0.13, 0.14, 0.16), roughness=0.45, anisotropy=0.7, anisotropy_dir=(1.0, 0.0, 0.0), banding_amp=0.12, banding_freq=150.0, banding_dir=(0.0, 0.0, 1.0), fbm_amp=0.15, fbm_scale=50.0, weathering_amp=0.4, bump_amp=0.3),
    "limestone": dict(base_color=(0.52, 0.50, 0.45), roughness=0.7, worley_amp=0.35, worley_scale=150.0, fbm_amp=0.12, fbm_scale=40.0, lichen_threshold=0.5, lichen_scale=14.0, weathering_amp=0.5, bump_amp=0.3, diffuse_model=1),
    "quartzite": dict(base_color=(0.62, 0.60, 0.58), roughness=0.4, subsurface_mfp=(0.002, 0.002, 0.002), subsurface_weight=0.5, fbm_amp=0.1, fbm_scale=70.0, weathering_amp=0.3, bump_amp=0.3),
}

STONE_CLASS_MATERIALS = {
    "rounded_cobble": ["river_basalt", "limestone"],
    "blocky_quarry": ["granite", "sandstone", "quartzite"],
    "flat_river": ["slate", "river_basalt"],
}


def material_preset(name: str, rng: np.random.Generator | None = None, seed: int | None = None) -> Material:
    """Stone material from the table of Section 10.3, with per stone variation of albedo and
    roughness inside the stated ranges when ``rng`` is given."""
    if name not in _STONE_PRESETS:
        raise KeyError(f"unknown material preset {name}")
    p = dict(_STONE_PRESETS[name])
    if rng is not None:
        bc = np.asarray(p["base_color"], dtype=np.float64) * rng.uniform(0.85, 1.15)
        p["base_color"] = tuple(np.clip(bc, 0, 1))
        p["roughness"] = float(np.clip(p["roughness"] + rng.uniform(-0.08, 0.08), 0.05, 1.0))
        p["seed"] = int(rng.integers(1, 10_000))
    if seed is not None:
        p["seed"] = int(seed)
    return Material(name=name, **p)


def material_for_class(stone_class: str, rng: np.random.Generator) -> Material:
    names = STONE_CLASS_MATERIALS.get(stone_class, list(_STONE_PRESETS))
    return material_preset(names[int(rng.integers(len(names)))], rng)


# ----------------------------------------------------------------------- ground


def ground_preset(name: str, size: float = 6.0, rng: np.random.Generator | None = None):
    """Return a list of (mesh, material, name) for the ground option (Section 10.3)."""
    if name == "studio_slab":
        m = box((size, size, 0.05), center=(0, 0, -0.025))
        return [(m, Material(name="concrete", base_color=(0.42, 0.42, 0.41), roughness=0.2, fbm_amp=0.1, fbm_scale=8.0, roughness_variation=0.05), "ground")]
    if name == "granite_table":
        m = box((size, size, 0.05), center=(0, 0, -0.025))
        return [(m, Material(name="dark_granite", base_color=(0.06, 0.06, 0.065), roughness=0.35, worley_amp=0.3, worley_scale=300.0, flakes_density=0.03, flakes_intensity=0.3), "ground")]
    if name == "riverbank":
        rng = np.random.default_rng(0) if rng is None else rng
        n = 96
        xs = np.linspace(-size / 2, size / 2, n)
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        from .textures import fbm3

        Z = np.zeros_like(X)
        for i in range(n):
            for j in range(n):
                Z[i, j] = 0.03 * fbm3(X[i, j] * 1.3, Y[i, j] * 1.3, 0.0, 4, 0.5, 2.0, 5) - 0.01 * (X[i, j] ** 2 + Y[i, j] ** 2) / (size * size) * 4
        # keep the centre flat for the tower
        r2 = X ** 2 + Y ** 2
        Z = Z * np.clip((r2 - 0.09) / 0.4, 0, 1)
        Z -= Z.max()
        V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
        F = []
        for i in range(n - 1):
            for j in range(n - 1):
                a = i * n + j
                F += [[a, a + 1, a + n + 1], [a, a + n + 1, a + n]]
        soil = TriMesh(V, np.array(F), "riverbank")
        water_level = float(np.percentile(Z, 35))
        water = plane(size, water_level)
        out = [
            (soil, Material(name="wet_soil", base_color=(0.22, 0.19, 0.15), roughness=0.75, coat=0.5, wet_mask_scale=3.0, fbm_amp=0.25, fbm_scale=6.0, worley_amp=0.3, worley_scale=40.0, bump_amp=0.6, bump_scale=400.0, diffuse_model=1), "ground"),
            # refractive water with ripples (bump), absorption of slightly turbid river water and
            # Cauchy dispersion (author's choices); rendered with MNEE for the sun caustics
            (water, Material(name="water", dielectric=True, ior=1.333, cauchy_b_um2=0.0035, absorption=(1.2, 0.35, 0.3), bump_amp=0.15, bump_scale=60.0), "water"),
        ]
        return out
    raise KeyError(f"unknown ground preset {name}")


# ---------------------------------------------------------------------- cameras


def camera_preset(name: str, bounds: tuple, width: int = 1024, height: int = 683):
    """Camera for one of the hero stills H1 to H12 given the tower bounds (lo, hi)."""
    from .scene import Camera

    lo, hi = np.asarray(bounds[0], dtype=np.float64), np.asarray(bounds[1], dtype=np.float64)
    c = 0.5 * (lo + hi)
    top = hi[2]
    h = max(hi[2] - lo[2], 0.05)
    base = np.array([c[0], c[1], lo[2]])
    aspect = width / height
    presets = {
        "H1": dict(position=base + np.array([1.2, -0.9, 0.4 * h]), look_at=base + np.array([0, 0, 0.42 * h]), focal_mm=85, f_number=2.8, focus_distance=1.5),
        "H2": dict(position=base + np.array([0.2, -0.15, 0.45 * h]), look_at=base + np.array([0, 0, 0.45 * h]), focal_mm=100, f_number=8, focus_distance=0.25),
        "H3": dict(position=base + np.array([0.9, -0.7, h + 0.3]), look_at=base + np.array([0, 0, h]), focal_mm=35, f_number=4, shutter=(0.0, 1 / 60)),
        "H4": dict(position=base + np.array([0, 0, 3.0]), look_at=base, up=np.array([0, 1.0, 0]), ortho=True, ortho_width=1.6 * aspect),
        "H5": dict(position=base + np.array([0.8, -0.6, 0.5 * h]), look_at=base + np.array([0, 0, 0.5 * h]), focal_mm=85, f_number=5.6),
        "H6": dict(position=base + np.array([1.0, -1.2, 0.5 * h]), look_at=base + np.array([0, 0, 0.45 * h]), focal_mm=50, f_number=4, shutter=(0.0, 1 / 30)),
        "H7": dict(position=base + np.array([0.0, -1.6, 0.5 * h]), look_at=base + np.array([0, 0, 0.5 * h]), focal_mm=85, f_number=8),
        "H8": dict(position=base + np.array([0.9, -0.9, h + 0.9]), look_at=base + np.array([0, 0, h]), focal_mm=50, f_number=5.6),
        "H9": dict(position=base + np.array([1.2, -1.2, 0.5 * h]), look_at=base + np.array([0, 0, 0.5 * h]), focal_mm=50, f_number=8, shutter=(0.0, 2.0)),
        "H10": dict(position=base + np.array([0.0, -1.6, 0.5 * h]), look_at=base + np.array([0, 0, 0.5 * h]), focal_mm=85, f_number=8),
        "H11": dict(position=base + np.array([0, 0, 6.0]), look_at=base, up=np.array([0, 1.0, 0]), ortho=True, ortho_width=4.0 * aspect),
        "H12": dict(position=base + np.array([1.5, -1.3, 0.5 * h]), look_at=base + np.array([0, 0, 0.5 * h]), focal_mm=50, f_number=8),
    }
    if name not in presets:
        raise KeyError(f"unknown camera preset {name}")
    p = presets[name]
    return Camera(width=width, height=height, name=name, **p)


def material_table() -> dict:
    """The stone material presets with their parameter ranges (for documentation and F2)."""
    return {k: dict(v) for k, v in _STONE_PRESETS.items()}
