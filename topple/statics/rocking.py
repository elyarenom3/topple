"""Rocking-aware tilt margin (Margin 4, an addition to Section 7).

The tilt margin of Section 7.2 is the tilt at which the contact force program of the current
contact configuration becomes infeasible. The dynamic test of Section 11.1 declares failure when
any stone moves more than 5 mm or rotates more than 2 degrees, and stones on curved supports pivot
onto neighbouring hull vertices well before the program becomes infeasible. This margin predicts
the dynamic failure rule directly with a quasi static sweep:

for each azimuth, the tilt is increased in steps. At every step the whole configuration (stones
and the ground slab) is rotated into the gravity frame, each stone is settled bottom to top with the
same quasi static hull settling used for posterior samples (vertical seating and tipping about the
support polygon edge, ``statics.analysis.settle_hull_stone``), and the result is rotated back.
The margin is the first tilt at which

* a stone has rotated more than ``fail_rot_deg`` or moved more than ``fail_disp_m`` from its
  untilted pose (rigid fit of its hull vertices), or
* the contact force program with friction is infeasible on the settled configuration (sliding).

The last interval is bisected. The sweep carries the configuration from step to step, so a stone
that pivots onto a new support and holds there keeps accumulating rotation.
"""

from __future__ import annotations

import numpy as np

from ..core.mesh import box
from ..core.transforms import rotation_from_two_vectors
from .analysis import HullGeom, _rotate_hull, settle_hull_stone, tower_from_hulls
from .equilibrium import ContactLP
from .margins import rotated_gravity

_GROUND = None


def ground_slab() -> HullGeom:
    global _GROUND
    if _GROUND is None:
        _GROUND = HullGeom(box((4.0, 4.0, 0.2), center=(0.0, 0.0, -0.1)).vertices)
    return _GROUND


def _rigid_motion(V0: np.ndarray, V1: np.ndarray) -> tuple[float, float]:
    """Rotation angle (deg) and centroid displacement (m) of the rigid motion V0 -> V1 (Kabsch)."""
    c0, c1 = V0.mean(axis=0), V1.mean(axis=0)
    H = (V0 - c0).T @ (V1 - c1)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))
    return ang, float(np.linalg.norm(c1 - c0))


def _settle_under(hulls, coms, g: np.ndarray, band: float, step_deg: float):
    """Settle every stone (bottom to top) under gravity ``g``. Returns new hulls and centres of mass
    in the world frame, and whether every stone found a support."""
    down = np.array([0.0, 0.0, -1.0])
    R = rotation_from_two_vectors(g / np.linalg.norm(g), down)  # world -> gravity frame
    origin = np.zeros(3)
    ground = _rotate_hull(ground_slab(), R, origin)
    out_h, out_c, ok = [], [], True
    for h, c in zip(hulls, coms):
        hg = _rotate_hull(h, R, origin)
        cg = R @ c
        h2, c2, rot = settle_hull_stone(hg, cg, [ground] + out_h, band, max_tilt_deg=8.0, step_deg=step_deg)
        # the tipping loop can rock back and forth about two support edges, accumulating rotation
        # around an equilibrium; the stone has toppled only if its net rotation is large
        if rot >= 8.0 and _rigid_motion(hg.vertices, h2.vertices)[0] >= 6.0:
            ok = False
        out_h.append(h2)
        out_c.append(c2)
    Rt = R.T
    return [_rotate_hull(h, Rt, origin) for h in out_h], [Rt @ c for c in out_c], ok


def _state_fails(ref_hulls, hulls, fail_rot_deg, fail_disp_m) -> bool:
    for h0, h1 in zip(ref_hulls, hulls):
        ang, disp = _rigid_motion(h0.vertices, h1.vertices)
        if ang > fail_rot_deg or disp > fail_disp_m:
            return True
    return False


def _feasible(realised, hulls, coms, g, band, cone_sides) -> bool:
    rr = [dict(r, com=c) for r, c in zip(realised, coms)]
    tower = tower_from_hulls(rr, hulls, band, 8, g)
    if len(tower.contacts) == 0:
        return False
    return bool(ContactLP(tower, cone_sides).feasible(g)[0])


def rocking_tilt_margin(realised: list[dict], hulls: list[HullGeom], gravity=(0.0, 0.0, -9.81), cone_sides: int = 8, n_azimuth: int = 12, step_deg: float = 1.0, max_deg: float = 60.0, fail_rot_deg: float = 2.0, fail_disp_m: float = 0.005, band: float = 5e-4, bisect: int = 3, settle_step_deg: float = 0.25, support_band: float = 2e-5) -> tuple[float, float]:
    """Quasi static tilt at which any stone exceeds the dynamic failure rule. Returns (margin in
    degrees, azimuth of the minimum). ``realised`` and ``hulls`` describe the settled tower at zero
    tilt (as built by ``analysis.build_tower_hull``). The quasi static settling uses the vertices
    within ``support_band`` of a support as the support polygon (the dynamic engine pivots on the
    touching vertices); the force program uses the contact band ``band``."""
    g0 = np.asarray(gravity, dtype=float)
    coms0 = [np.asarray(r["com"], dtype=float) for r in realised]
    # reference: the configuration settled by the same quasi static model at zero tilt (the
    # engine's rest pose need not be at rest on the hulls with the tight support band)
    hulls, coms0, ok0 = _settle_under(list(hulls), coms0, g0, support_band, settle_step_deg)
    if not ok0 or not _feasible(realised, hulls, coms0, g0, band, cone_sides):
        return 0.0, 0.0
    best, best_az = float(max_deg), 0.0
    for az in np.linspace(0.0, 2.0 * np.pi, int(n_azimuth), endpoint=False):
        hulls_k, coms_k = list(hulls), list(coms0)
        a_ok = 0.0
        failed_at = None
        a = 0.0
        while a < best - 1e-9:
            a_next = min(a + step_deg, best)
            g = rotated_gravity(g0, az, a_next)
            h2, c2, ok = _settle_under(hulls_k, coms_k, g, support_band, settle_step_deg)
            if (not ok) or _state_fails(hulls, h2, fail_rot_deg, fail_disp_m) or not _feasible(realised, h2, c2, g, band, cone_sides):
                failed_at = a_next
                break
            hulls_k, coms_k, a_ok, a = h2, c2, a_next, a_next
        if failed_at is None:
            continue
        lo, hi = a_ok, failed_at
        for _ in range(int(bisect)):
            mid = 0.5 * (lo + hi)
            g = rotated_gravity(g0, az, mid)
            h2, c2, ok = _settle_under(hulls_k, coms_k, g, support_band, settle_step_deg)
            if (not ok) or _state_fails(hulls, h2, fail_rot_deg, fail_disp_m) or not _feasible(realised, h2, c2, g, band, cone_sides):
                hi = mid
            else:
                lo = mid
        if hi < best:
            best, best_az = hi, float(az)
    return best, best_az
