"""Stone specification, class conditional sampling and rejection on descriptor boxes (Section 5.3).

Classes ("flat_river", "blocky_quarry", "rounded_cobble") are boxes in descriptor space
(configs/default.yaml, ``stones.class_boxes``). Each class draws a family and its parameters
from a class conditional distribution, erodes the mesh (Section 5.2), scales the longest
axis to the target size, moves the mesh to its principal frame and keeps it if the
descriptors fall in the class box and the mesh passes the quality checks (Section 5.4).
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np

from ..core.mesh import TriMesh
from .descriptors import descriptors, in_box, principal_extents
from .erosion import erode
from .families import family_a_hull, family_b_ellipsoid, family_c_superquadric, family_d_voronoi

MATERIALS_BY_CLASS = {
    "flat_river": ["slate"],
    "blocky_quarry": ["granite", "sandstone", "quartzite"],
    "rounded_cobble": ["river_basalt", "limestone"],
}

DEFAULT_BOXES = {
    "flat_river": {"elongation": [0.55, 0.95], "flatness": [0.25, 0.55], "convexity": [0.93, 1.0], "sphericity": [0.70, 0.92]},
    "blocky_quarry": {"elongation": [0.55, 1.0], "flatness": [0.55, 1.0], "convexity": [0.90, 1.0], "sphericity": [0.78, 0.93]},
    "rounded_cobble": {"elongation": [0.65, 1.0], "flatness": [0.65, 1.0], "convexity": [0.96, 1.0], "sphericity": [0.90, 1.0]},
}


@dataclass
class StoneSpec:
    """A generated stone. ``mesh`` is in the body frame (centre of mass at the origin, principal axes)."""

    name: str
    mesh: TriMesh
    density: float
    volume: float
    mass: float
    com: np.ndarray
    inertia: np.ndarray
    stone_class: str
    family: str
    descriptors: dict
    material: str
    seed: int
    params: dict = field(default_factory=dict)

    def summary(self) -> dict:
        d = {"name": self.name, "class": self.stone_class, "family": self.family, "material": self.material, "density": self.density, "mass": self.mass, "volume": self.volume, "seed": self.seed}
        d.update({k: v for k, v in self.descriptors.items()})
        return d


class StoneGenerator:
    """Class conditional stone sampler with rejection on descriptor boxes."""

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.scfg = cfg.get("stones", cfg)
        self.size_range = tuple(self.scfg.get("size_range", (0.05, 0.15)))
        self.density_range = tuple(self.scfg.get("density_range", (2500.0, 2800.0)))
        self.classes = list(self.scfg.get("classes", list(DEFAULT_BOXES)))
        self.boxes = {k: {kk: list(vv) for kk, vv in v.items()} for k, v in self.scfg.get("class_boxes", DEFAULT_BOXES).items()}
        self.erosion_steps = int(self.scfg.get("erosion_steps", 4))
        self.erosion_eta = float(self.scfg.get("erosion_eta", 0.002))
        self.subdivisions = int(self.scfg.get("subdivisions", 3))
        self.max_tries = int(self.scfg.get("max_tries", 30))
        self.counter = 0

    # ----------------------------------------------------------------- families
    def _draw_family(self, rng, stone_class: str) -> tuple[str, dict, TriMesh]:
        sd = max(1, self.subdivisions)
        if stone_class == "flat_river":
            b = rng.uniform(0.6, 0.95)
            c = b * rng.uniform(0.3, 0.55)
            if rng.random() < 0.6:
                p = dict(a=1.0, b=b, c=c, amplitude=rng.uniform(0.01, 0.04), octaves=3, subdivisions=sd)
                return "B", p, family_b_ellipsoid(rng, **p)
            p = dict(a=1.0, b=b, c=c, n_points=int(rng.integers(30, 80)), subdivisions=sd, smooth_iters=10, fractal_amp=rng.uniform(0.0, 0.02))
            return "A", p, family_a_hull(rng, **p)
        if stone_class == "blocky_quarry":
            u = rng.random()
            b = rng.uniform(0.6, 1.0)
            c = b * rng.uniform(0.6, 1.0)
            if u < 0.4:
                p = dict(a=1.0, b=b, c=c, n_points=int(rng.integers(8, 14)), subdivisions=sd, smooth_iters=int(rng.integers(3, 7)))
                return "A", p, family_a_hull(rng, **p)
            if u < 0.7:
                p = dict(block_extents=(1.0, rng.uniform(0.6, 1.0), rng.uniform(0.6, 1.0)), n_seeds=int(rng.integers(8, 16)), subdivisions=sd, smooth_iters=int(rng.integers(2, 5)))
                return "D", p, family_d_voronoi(rng, **p)
            p = dict(a=1.0, b=b, c=c, e1=rng.uniform(0.25, 0.6), e2=rng.uniform(0.25, 0.6), perturb=rng.uniform(0.0, 0.03), subdivisions=sd)
            return "C", p, family_c_superquadric(rng, **p)
        if stone_class == "rounded_cobble":
            u = rng.random()
            b = rng.uniform(0.7, 1.0)
            c = b * rng.uniform(0.7, 1.0)
            if u < 0.5:
                p = dict(a=1.0, b=b, c=c, amplitude=rng.uniform(0.005, 0.02), octaves=3, subdivisions=sd)
                return "B", p, family_b_ellipsoid(rng, **p)
            if u < 0.8:
                p = dict(a=1.0, b=b, c=c, e1=rng.uniform(0.8, 1.2), e2=rng.uniform(0.8, 1.2), perturb=rng.uniform(0.0, 0.015), subdivisions=sd)
                return "C", p, family_c_superquadric(rng, **p)
            p = dict(a=1.0, b=b, c=c, n_points=int(rng.integers(100, 300)), subdivisions=sd, smooth_iters=15)
            return "A", p, family_a_hull(rng, **p)
        raise ValueError(f"unknown stone class {stone_class}")

    # ------------------------------------------------------------------- sample
    def sample(self, rng: np.random.Generator, stone_class: str | None = None, name: str | None = None) -> StoneSpec:
        if stone_class is None:
            stone_class = self.classes[int(rng.integers(len(self.classes)))]
        box = self.boxes.get(stone_class, DEFAULT_BOXES[stone_class])
        size = rng.uniform(*self.size_range)
        density = rng.uniform(*self.density_range)
        best = None
        for attempt in range(self.max_tries):
            seed = int(rng.integers(2**31 - 1))
            fam, params, mesh = self._draw_family(rng, stone_class)
            steps = self.erosion_steps if fam == "D" else max(1, self.erosion_steps // 2)
            mesh = erode(mesh, self.erosion_eta, steps)
            ext, _ = principal_extents(mesh)
            mesh = mesh.scaled(size / ext[0])
            mesh, _ = mesh.to_principal_frame(density)
            d = descriptors(mesh)
            valid = mesh.is_valid()
            ok_box = in_box(d, box)
            score = _box_distance(d, box) + (0.0 if all(valid.values()) else 10.0)
            if best is None or score < best[0]:
                best = (score, fam, params, mesh, d, seed, valid)
            if ok_box and all(valid.values()):
                best = (0.0, fam, params, mesh, d, seed, valid)
                break
        else:
            warnings.warn(f"stone class {stone_class}: no sample in box after {self.max_tries} tries, using closest")
        _, fam, params, mesh, d, seed, valid = best
        mp = mesh.mass_properties(density)
        self.counter += 1
        mats = MATERIALS_BY_CLASS.get(stone_class, ["granite"])
        material = mats[int(rng.integers(len(mats)))]
        nm = name or f"{stone_class}_{self.counter:03d}"
        mesh.name = nm
        return StoneSpec(
            name=nm,
            mesh=mesh,
            density=float(density),
            volume=float(mp["volume"]),
            mass=float(mp["mass"]),
            com=np.zeros(3),
            inertia=np.diag(np.diag(mp["inertia_com"])),
            stone_class=stone_class,
            family=fam,
            descriptors=d,
            material=material,
            seed=seed,
            params=params,
        )

    def sample_pile(self, rng: np.random.Generator, n: int, stone_class: str | None = None) -> list[StoneSpec]:
        return [self.sample(rng, stone_class) for _ in range(int(n))]


def _box_distance(d: dict, box: dict) -> float:
    s = 0.0
    for k, (lo, hi) in box.items():
        v = d.get(k)
        if v is None:
            continue
        if v < lo:
            s += (lo - v) ** 2
        elif v > hi:
            s += (v - hi) ** 2
    return s


def stone_to_body(spec: StoneSpec, friction: float = 0.6, sdf_spacing: float | None = 0.001):
    """Build a RigidBody for the stone (body frame = principal frame of the spec mesh)."""
    from ..physics.body import RigidBody

    b = RigidBody.from_mesh(spec.name, spec.mesh, spec.density, friction=friction, sdf_spacing=sdf_spacing)
    b.meta.update({"stone_class": spec.stone_class, "family": spec.family, "material": spec.material, "descriptors": dict(spec.descriptors), "seed": spec.seed})
    return b
