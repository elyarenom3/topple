"""Deterministic random streams (Section 13.3).

Philox counter based generators are keyed by (experiment seed, tower index, stage)
so that any placement can be replayed in isolation.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class Stage(IntEnum):
    STONEGEN = 1
    SENSOR = 2
    ESTIMATOR = 3
    PLANNER = 4
    CONTROLLER = 5
    HARNESS = 6
    RENDER = 7
    EVAL = 8
    PHYSICS = 9
    MISC = 10


def make_rng(experiment_seed: int, tower_index: int = 0, stage: int | Stage = Stage.MISC, sub: int = 0) -> np.random.Generator:
    """Return a Generator on a Philox stream keyed by the tuple (seed, tower, stage, sub)."""
    key = (int(experiment_seed) & 0xFFFFFFFFFFFFFFFF, int(tower_index) & 0xFFFFFFFFFFFFFFFF, int(stage) & 0xFFFFFFFFFFFFFFFF, int(sub) & 0xFFFFFFFFFFFFFFFF)
    bits = np.random.Philox(key=np.array(key[:2], dtype=np.uint64), counter=np.array([key[2], key[3], 0, 0], dtype=np.uint64))
    return np.random.Generator(bits)
