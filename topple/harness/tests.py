"""Dynamic ground truth tests (Section 11.1).

Each test runs on a clone of the post release state at 1 kHz.

1. Tilt test: rotate gravity about horizontal axes at 0.5 degrees per second for 8 azimuths;
   the measured tilt margin M* is the minimum failure angle.
2. Shake test: horizontal ground acceleration a(t) = A(t) sin(2 pi f t) with f = 2 Hz and the
   amplitude ramped to 0.5 g; realised in the ground frame as an inertial force, so the
   effective gravity becomes g - a(t) e_x.
3. Push test: a ramped force at the top stone's centre of mass in 8 directions.
4. Fixed perturbation: a 3 degree tilt held for 2 s defines the binary outcome y.

Failure is any stone moving more than 5 mm or rotating more than 2 degrees from the
post release state (author's thresholds, sensitivity reported through the config).

Rotating gravity is the same experiment as tilting the ground, and before any stone has
moved the state of a resting tower does not depend on how fast the tilt was reached. The
tilt test therefore sweeps at a coarse rate to find the first motion, then restarts from the
resting state a few degrees below that angle at the specified 0.5 degrees per second.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.transforms import quat_conj, quat_mul, quat_to_rotvec


@dataclass
class TestOutcome:
    kind: str
    value: float  # critical tilt (deg), shake amplitude (g), push force over weight, or 0/1
    failed: bool
    per_direction: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)


def _dynamic_ids(world, ids=None):
    if ids is not None:
        return list(ids)
    return [b.id for b in world.bodies if not b.static and not b.kinematic]


def failure_check(world, ref_state, ids, disp_tol: float, rot_tol_deg: float) -> tuple[bool, float, float]:
    """Max displacement (m) and rotation (deg) of the listed bodies relative to ref_state."""
    max_d = 0.0
    max_r = 0.0
    for i in ids:
        b = world.bodies[i]
        d = float(np.linalg.norm(b.x - ref_state.x[i]))
        dq = quat_mul(b.q, quat_conj(ref_state.q[i]))
        r = float(np.degrees(np.linalg.norm(quat_to_rotvec(dq))))
        max_d = max(max_d, d)
        max_r = max(max_r, r)
    return (max_d > disp_tol or max_r > rot_tol_deg), max_d, max_r


def _rotated_gravity(g0: np.ndarray, alpha_deg: float, azimuth: float) -> np.ndarray:
    """Rotate gravity by alpha about the horizontal axis with the given azimuth."""
    axis = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
    a = np.radians(alpha_deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
    return R @ g0


def _hcfg(cfg: dict) -> dict:
    h = dict(cfg.get("harness", {}))
    h.setdefault("tilt_rate_deg_s", 0.5)
    h.setdefault("tilt_azimuths", 8)
    h.setdefault("tilt_max_deg", 60.0)
    h.setdefault("coarse_rate_deg_s", 10.0)
    h.setdefault("restart_margin_deg", 3.0)
    h.setdefault("shake_freq_hz", 2.0)
    h.setdefault("shake_max_g", 0.5)
    h.setdefault("shake_ramp_s", 10.0)
    h.setdefault("push_directions", 8)
    h.setdefault("push_ramp_w_s", 0.5)
    h.setdefault("push_max_w", 3.0)
    h.setdefault("fixed_tilt_deg", 3.0)
    h.setdefault("fixed_hold_s", 2.0)
    h.setdefault("fail_disp_m", 0.005)
    h.setdefault("fail_rot_deg", 2.0)
    h.setdefault("settle_s", 0.3)
    h.setdefault("check_every", 10)
    return h


def _tilt_one_azimuth(world, azimuth: float, h: dict, ids, rate: float, start_deg: float, max_deg: float, g0, ref) -> tuple[float, bool]:
    """Tilt from start_deg at ``rate`` until failure or max_deg. Returns (angle at failure or max, failed)."""
    w = world.clone()
    dt = w.dt
    w.gravity = _rotated_gravity(g0, start_deg, azimuth)
    # settle briefly at the start angle
    w.step(int(round(h["settle_s"] / dt)))
    failed, _, _ = failure_check(w, ref, ids, h["fail_disp_m"], h["fail_rot_deg"])
    if failed:
        return start_deg, True
    n_check = max(1, int(h["check_every"]))
    alpha = start_deg
    k = 0
    while alpha < max_deg:
        alpha = min(max_deg, alpha + rate * dt * n_check)
        w.gravity = _rotated_gravity(g0, alpha, azimuth)
        w.step(n_check)
        k += 1
        failed, _, _ = failure_check(w, ref, ids, h["fail_disp_m"], h["fail_rot_deg"])
        if failed:
            return alpha, True
    return max_deg, False


def tilt_test(world, cfg: dict, ids=None, azimuths=None) -> TestOutcome:
    h = _hcfg(cfg)
    ids = _dynamic_ids(world, ids)
    ref = world.state()
    g0 = np.array(world.gravity, dtype=float).copy()
    n_az = int(h["tilt_azimuths"])
    azimuths = np.linspace(0, 2 * np.pi, n_az, endpoint=False) if azimuths is None else np.asarray(azimuths)
    per = {}
    for az in azimuths:
        coarse, failed = _tilt_one_azimuth(world, az, h, ids, h["coarse_rate_deg_s"], 0.0, h["tilt_max_deg"], g0, ref)
        if failed:
            start = max(0.0, coarse - h["restart_margin_deg"])
            fine, failed2 = _tilt_one_azimuth(world, az, h, ids, h["tilt_rate_deg_s"], start, h["tilt_max_deg"], g0, ref)
            per[float(az)] = fine if failed2 else coarse
        else:
            per[float(az)] = h["tilt_max_deg"]
    vals = np.array(list(per.values()))
    m = float(vals.min())
    return TestOutcome("tilt", m, m < h["tilt_max_deg"], per_direction=per, details={"azimuth_min": float(list(per.keys())[int(vals.argmin())])})


def shake_test(world, cfg: dict, ids=None, direction=(1.0, 0.0, 0.0)) -> TestOutcome:
    """Rising sinusoidal horizontal excitation; returns the amplitude in g at first failure."""
    h = _hcfg(cfg)
    ids = _dynamic_ids(world, ids)
    w = world.clone()
    ref = w.state()
    g0 = np.array(w.gravity, dtype=float).copy()
    gmag = float(np.linalg.norm(g0))
    e = np.asarray(direction, dtype=float)
    e /= np.linalg.norm(e)
    f = h["shake_freq_hz"]
    T = h["shake_ramp_s"]
    dt = w.dt
    n = int(T / dt)
    n_check = max(1, int(h["check_every"]))
    amp_at_fail = h["shake_max_g"]
    failed = False
    t = 0.0
    k = 0
    while k < n:
        for _ in range(n_check):
            t += dt
            A = h["shake_max_g"] * min(1.0, t / T)
            a = A * gmag * np.sin(2 * np.pi * f * t)
            w.gravity = g0 - a * e
            w.step(1)
            k += 1
        failed, _, _ = failure_check(w, ref, ids, h["fail_disp_m"], h["fail_rot_deg"])
        if failed:
            amp_at_fail = h["shake_max_g"] * min(1.0, t / T)
            break
    return TestOutcome("shake", float(amp_at_fail), failed, details={"t_fail": t if failed else None})


def push_test(world, cfg: dict, top_id: int, ids=None) -> TestOutcome:
    """Ramped horizontal force at the top stone's centre of mass; returns min over directions of
    the force at failure divided by the top stone's weight."""
    h = _hcfg(cfg)
    ids = _dynamic_ids(world, ids)
    n_dir = int(h["push_directions"])
    top = world.bodies[top_id]
    W = top.mass * float(np.linalg.norm(world.gravity))
    per = {}
    for j in range(n_dir):
        ang = 2 * np.pi * j / n_dir
        e = np.array([np.cos(ang), np.sin(ang), 0.0])
        w = world.clone()
        ref = w.state()
        dt = w.dt
        n_check = max(1, int(h["check_every"]))
        s = 0.0
        failed = False
        t = 0.0
        while s < h["push_max_w"]:
            for _ in range(n_check):
                t += dt
                s = h["push_ramp_w_s"] * t
                w.external_force(top_id, s * W * e, w.bodies[top_id].x)
                w.step(1)
            failed, _, _ = failure_check(w, ref, ids, h["fail_disp_m"], h["fail_rot_deg"])
            if failed:
                break
        per[float(ang)] = s if failed else h["push_max_w"]
    vals = np.array(list(per.values()))
    m = float(vals.min())
    return TestOutcome("push", m, m < h["push_max_w"], per_direction=per)


def fixed_perturbation(world, cfg: dict, ids=None, azimuths=None) -> TestOutcome:
    """A fixed tilt held for a fixed time (binary outcome y = 1 if the tower survives all azimuths)."""
    h = _hcfg(cfg)
    ids = _dynamic_ids(world, ids)
    ref = world.state()
    g0 = np.array(world.gravity, dtype=float).copy()
    n_az = int(h["tilt_azimuths"])
    azimuths = np.linspace(0, 2 * np.pi, n_az, endpoint=False) if azimuths is None else np.asarray(azimuths)
    per = {}
    survived = True
    for az in azimuths:
        w = world.clone()
        w.gravity = _rotated_gravity(g0, h["fixed_tilt_deg"], az)
        n = int(h["fixed_hold_s"] / w.dt)
        n_check = max(1, int(h["check_every"]))
        failed = False
        k = 0
        while k < n:
            w.step(n_check)
            k += n_check
            failed, _, _ = failure_check(w, ref, ids, h["fail_disp_m"], h["fail_rot_deg"])
            if failed:
                break
        per[float(az)] = 0.0 if failed else 1.0
        survived &= not failed
        if failed:
            break
    return TestOutcome("fixed", 1.0 if survived else 0.0, not survived, per_direction=per)


def run_protocol(world, cfg: dict, top_id: int, ids=None, which=("tilt", "fixed", "shake", "push")) -> dict:
    """Run the selected tests on clones of ``world`` and return their outcomes keyed by name."""
    out = {}
    if "tilt" in which:
        out["tilt"] = tilt_test(world, cfg, ids)
    if "fixed" in which:
        out["fixed"] = fixed_perturbation(world, cfg, ids)
    if "shake" in which:
        out["shake"] = shake_test(world, cfg, ids)
    if "push" in which:
        out["push"] = push_test(world, cfg, top_id, ids)
    return out
