"""Cross engine validation adapters (Section 12.2).

Each adapter wraps an established engine (MuJoCo, PyBullet, Drake) behind the small
subset of the ``World`` interface that the harness tests use: ``bodies`` with ``x``,
``q``, ``static``, ``kinematic``, ``mass``; ``gravity``; ``dt``; ``step(n)``; ``clone()``;
``state()``; ``external_force(id, f, p)``. The same tilt, fixed, shake and push tests
therefore run unchanged on every engine.
"""

from .common import available_engines, build_engine_world, compare_engines  # noqa: F401
