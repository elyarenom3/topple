"""Hidden ground truth store (Section 4.1, item 1).

Holds the true stone meshes, densities and the true friction coefficient per stone pair.
The physics engine is built from this store; the estimator only ever sees depth images.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.rng import Stage, make_rng
from ..physics.body import RigidBody


@dataclass
class GroundTruthStore:
    stones: list  # list[StoneSpec] from stonegen, with true density
    mu_table: dict = field(default_factory=dict)  # (name_a, name_b) sorted -> mu
    mu_ground: float = 0.6
    density: dict = field(default_factory=dict)  # name -> kg/m^3

    @staticmethod
    def build(stones: list, cfg: dict, seed: int, tower_index: int = 0) -> "GroundTruthStore":
        """Draw the true friction coefficient per stone pair from the log normal prior
        (median and log sigma of ``cfg['friction']``) so the truth is a draw from the prior
        that the estimator uses, and store the true densities."""
        rng = make_rng(seed, tower_index, Stage.STONEGEN, sub=999)
        f = cfg["friction"]
        med, ls = f["prior_median"], f["prior_log_sigma"]
        names = [s.name for s in stones]
        table = {}
        for i, a in enumerate(names):
            for b in names[i:]:
                table[tuple(sorted((a, b)))] = float(np.exp(np.log(med) + ls * rng.standard_normal()))
        for a in names:
            table[tuple(sorted((a, "ground")))] = float(np.exp(np.log(f.get("ground_mu", med)) + ls * rng.standard_normal()))
        dens = {s.name: float(s.density) for s in stones}
        return GroundTruthStore(stones=stones, mu_table=table, mu_ground=f.get("ground_mu", 0.6), density=dens)

    def mu(self, name_a: str, name_b: str) -> float:
        key = tuple(sorted((name_a, name_b)))
        if key in self.mu_table:
            return self.mu_table[key]
        return self.mu_ground if "ground" in key else 0.6

    def spec(self, name: str):
        for s in self.stones:
            if s.name == name:
                return s
        raise KeyError(name)

    def make_body(self, name: str, sdf_spacing: float | None = 0.001) -> RigidBody:
        """True rigid body for the physics engine (principal frame mesh, true density)."""
        s = self.spec(name)
        b = RigidBody.from_mesh(name, s.mesh, s.density, friction=0.6, sdf_spacing=sdf_spacing)
        b.meta["true"] = True
        b.meta["stone_class"] = s.stone_class
        b.meta["material"] = s.material
        return b

    def apply_friction(self, world) -> None:
        """Write the true pair friction coefficients into a World's pair table."""
        for i, a in enumerate(world.bodies):
            for b in world.bodies[i + 1 :]:
                world.set_pair_friction(a.id, b.id, self.mu(a.name, b.name))
