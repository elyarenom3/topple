"""Shared pieces of the engine adapters."""

from __future__ import annotations

import importlib
import os
import tempfile
from dataclasses import dataclass

import numpy as np

from ..physics.body import WorldState

ENGINES = {"mujoco": "mujoco", "pybullet": "pybullet", "drake": "pydrake"}


def available_engines() -> list[str]:
    out = []
    for name, mod in ENGINES.items():
        try:
            importlib.import_module(mod)
            out.append(name)
        except Exception:
            pass
    return out


@dataclass
class BodyView:
    """Light view of a body inside an external engine, matching the attributes the harness reads."""

    id: int
    name: str
    mass: float
    static: bool
    kinematic: bool
    x: np.ndarray
    q: np.ndarray


class EngineWorld:
    """Base class: subclasses implement _sync_from_engine, _apply_gravity, step, clone, external_force."""

    def __init__(self, bodies, gravity, dt):
        self.bodies: list[BodyView] = bodies
        self._gravity = np.asarray(gravity, dtype=float).copy()
        self.dt = float(dt)
        self.time = 0.0

    @property
    def gravity(self):
        return self._gravity

    @gravity.setter
    def gravity(self, g):
        self._gravity = np.asarray(g, dtype=float).copy()
        self._apply_gravity()

    def state(self) -> WorldState:
        n = len(self.bodies)
        q = np.array([b.q for b in self.bodies]).reshape(n, 4)
        x = np.array([b.x for b in self.bodies]).reshape(n, 3)
        return WorldState(self.time, q, x, np.zeros((n, 3)), np.zeros((n, 3)))

    # to be implemented by subclasses
    def _apply_gravity(self):
        raise NotImplementedError

    def step(self, n: int = 1):
        raise NotImplementedError

    def clone(self):
        raise NotImplementedError

    def external_force(self, body_id, f, p_world=None):
        raise NotImplementedError


def export_hull_objs(world, directory: str) -> dict[int, list[str]]:
    """Write the convex pieces of each dynamic body (body frame) as OBJ files."""
    out = {}
    for b in world.bodies:
        if b.static:
            continue
        paths = []
        for k, piece in enumerate(b.shape.pieces):
            p = os.path.join(directory, f"{b.name}_{k}.obj")
            piece.save(p)
            paths.append(p)
        out[b.id] = paths
    return out


def build_engine_world(world, engine: str, tmpdir: str | None = None, **options):
    """Build the same scene (bodies, poses, mass properties, pair friction, gravity, dt) in ``engine``.
    ``options`` pass engine specific settings (Drake: approx, penetration_allowance,
    stiction_tolerance, substeps)."""
    if engine == "mujoco":
        from .mujoco_adapter import MuJoCoWorld

        return MuJoCoWorld.from_world(world, tmpdir)
    if engine == "pybullet":
        from .pybullet_adapter import PyBulletWorld

        return PyBulletWorld.from_world(world, tmpdir, **options)
    if engine == "drake":
        from .drake_adapter import DrakeWorld

        return DrakeWorld.from_world(world, tmpdir, **options)
    raise ValueError(engine)


def compare_engines(world, cfg: dict, engines=None, tests=("settle", "tilt", "fixed"), top_id=None, settle_s: float = 1.0) -> dict:
    """Run settle and the selected harness tests on the developer engine and each external engine.

    Returns dict engine -> dict(settled_x (n,3), settled_q (n,4), tilt, fixed, ...).
    """
    from ..harness.tests import fixed_perturbation, push_test, shake_test, tilt_test

    engines = available_engines() if engines is None else [e for e in engines if e in available_engines()]
    results = {}
    worlds = {"topple": world.clone()}
    with tempfile.TemporaryDirectory() as td:
        for e in engines:
            try:
                worlds[e] = build_engine_world(world, e, td)
            except Exception as exc:  # pragma: no cover
                results[e] = {"error": repr(exc)}
        for name, w in worlds.items():
            r = {}
            if "settle" in tests:
                w.step(int(settle_s / w.dt))
                s = w.state()
                r["settled_x"] = s.x.copy()
                r["settled_q"] = s.q.copy()
            if "tilt" in tests:
                r["tilt"] = tilt_test(w, cfg).value
            if "fixed" in tests:
                r["fixed"] = fixed_perturbation(w, cfg).value
            if "shake" in tests:
                r["shake"] = shake_test(w, cfg).value
            if "push" in tests and top_id is not None:
                r["push"] = push_test(w, cfg, top_id).value
            results[name] = r
    return results
