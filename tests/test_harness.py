"""Harness tests on simple towers (need the physics engine)."""
import numpy as np
import pytest

from topple.core.config import load_config
from topple.core.mesh import box
from topple.harness.tests import failure_check, fixed_perturbation, push_test, shake_test, tilt_test
from topple.statics.margins import analytic_block_tilt

physics = pytest.importorskip("topple.physics.world")


def make_block_world(mu=0.6, b=0.1, h=0.2, cfg=None):
    from topple.physics.body import RigidBody
    from topple.physics.world import World

    cfg = cfg or load_config("quick")
    w = World(cfg["world"])
    g = RigidBody.ground(friction=mu)
    w.add_body(g)
    blk = RigidBody.from_mesh("block", box((b, b, h)), 2600.0, friction=mu, sdf_spacing=None)
    blk.x = np.array([0.0, 0.0, h / 2])
    w.add_body(blk)
    w.set_pair_friction(g.id, blk.id, mu)
    return w, cfg, blk


def test_tilt_test_matches_analytic_block():
    w, cfg, blk = make_block_world(mu=0.6, b=0.1, h=0.2)
    cfg["harness"]["tilt_azimuths"] = 4
    cfg["harness"]["coarse_rate_deg_s"] = 20.0
    cfg["harness"]["tilt_rate_deg_s"] = 2.0
    w.step(200)
    out = tilt_test(w, cfg)
    expected = analytic_block_tilt(0.6, 0.1, 0.2)
    assert abs(out.value - expected) < 1.5, (out.value, expected)
    assert out.failed


def test_fixed_perturbation_survives_small_tilt():
    w, cfg, blk = make_block_world()
    cfg["harness"]["tilt_azimuths"] = 2
    cfg["harness"]["fixed_hold_s"] = 0.5
    out = fixed_perturbation(w, cfg)
    assert out.value == 1.0 and not out.failed


def test_push_and_shake_run():
    w, cfg, blk = make_block_world(mu=0.6, b=0.1, h=0.3)
    cfg["harness"]["push_directions"] = 2
    cfg["harness"]["push_ramp_w_s"] = 2.0
    cfg["harness"]["shake_ramp_s"] = 1.0
    cfg["harness"]["shake_max_g"] = 1.0
    out = push_test(w, cfg, blk.id)
    # a block of aspect b/h = 1/3 tips when the push at the centre of mass exceeds about (b/h) W
    assert 0.1 < out.value < 1.0
    s = shake_test(w, cfg)
    assert 0.0 < s.value <= 1.0


def test_failure_check_thresholds():
    w, cfg, blk = make_block_world()
    ref = w.state()
    assert failure_check(w, ref, [blk.id], 0.005, 2.0)[0] is False
    blk.x = blk.x + np.array([0.01, 0, 0])
    assert failure_check(w, ref, [blk.id], 0.005, 2.0)[0] is True
