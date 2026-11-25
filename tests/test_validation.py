"""Cross engine adapters (Section 12.2): the same harness test on a block in each engine."""

import numpy as np
import pytest

from topple.core.config import load_config
from topple.core.mesh import box
from topple.harness.tests import tilt_test
from topple.physics.body import RigidBody
from topple.physics.world import World
from topple.statics.margins import analytic_block_tilt


def block_world(mu=0.8, b=0.1, h=0.2):
    cfg = load_config("quick")
    w = World(cfg)
    g = RigidBody.ground(friction=mu)
    w.add_body(g)
    blk = RigidBody.from_mesh("block", box((b, b, h)), 2600.0, friction=mu, sdf_spacing=None)
    blk.x = np.array([0.0, 0.0, h / 2 + 1e-4])
    w.add_body(blk)
    w.set_pair_friction(g.id, blk.id, mu)
    cfg["harness"].update({"tilt_azimuths": 1, "coarse_rate_deg_s": 20.0, "tilt_rate_deg_s": 2.0})
    return w, cfg


@pytest.mark.parametrize("engine", ["mujoco", "drake", "pybullet"])
def test_engine_tilt_of_a_block_matches_the_analytic_angle(engine):
    from topple.validation.common import ENGINES

    pytest.importorskip(ENGINES[engine])
    from topple.validation import build_engine_world

    w, cfg = block_world()
    ew = build_engine_world(w, engine)
    ew.step(200)
    out = tilt_test(ew, cfg, azimuths=[0.0])
    expected = analytic_block_tilt(0.8, 0.1, 0.2)  # tipping at atan(b / h) = 26.6 degrees
    assert abs(out.value - expected) < 3.0, (engine, out.value, expected)


@pytest.mark.parametrize("engine", ["topple", "mujoco", "drake", "pybullet"])
def test_settled_three_stone_tower_holds_at_zero_tilt(engine):
    """A tower settled in the developer's engine stays within the failure thresholds for 2 s at
    zero tilt in every engine (no creep from contact model differences)."""
    import sys
    from pathlib import Path

    from topple.core.transforms import quat_conj, quat_mul, quat_to_rotvec

    if engine != "topple":
        from topple.validation.common import ENGINES

        pytest.importorskip(ENGINES[engine])
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from fig_engine_comparison import random_tower

    cfg = load_config("quick")
    w = random_tower(cfg, 0)
    if engine == "topple":
        ew = w
    else:
        from topple.validation import build_engine_world

        ew = build_engine_world(w, engine)
    s0 = ew.state()
    ew.step(2000)
    s1 = ew.state()
    for i in range(1, len(w.bodies)):
        disp = float(np.linalg.norm(s1.x[i] - s0.x[i]))
        rot = float(np.degrees(np.linalg.norm(quat_to_rotvec(quat_mul(s1.q[i], quat_conj(s0.q[i]))))))
        assert disp < 0.005 and rot < 2.0, (engine, i, disp, rot)
