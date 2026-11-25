"""Simulation based placement scoring (condition E8, Section 11.3).

The estimated tower (posterior mean meshes at their observed poses) is built as a dynamic
world in the developer's engine and gravity is swept at a coarse rate for a few azimuths;
the score is the first angle at which any stone moves. This mirrors the physics simulator
scoring of Furrer et al. (2017) and Liu, Choi and Napp (2021) and carries no uncertainty.
Compute is matched to the static analysis by the sweep rate and the number of azimuths.
"""

from __future__ import annotations

import numpy as np

from ..core.config import get
from ..core.transforms import quat_conj, quat_mul, quat_to_rotvec


def _rotated_gravity(g0, alpha_deg, azimuth):
    axis = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
    a = np.radians(alpha_deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
    return R @ g0


def simulated_tilt_margin(realised: list[dict], cfg: dict, rate_deg_s: float = 10.0, n_azimuth: int = 4, max_deg: float = 60.0, disp_tol: float = 0.005, rot_tol_deg: float = 2.0) -> float:
    from ..physics.body import CollisionShape, RigidBody
    from ..physics.world import World

    if not realised:
        return 0.0
    cfg_w = dict(cfg["world"])
    world = World(cfg_w)
    ground = RigidBody.ground(friction=0.6)
    world.add_body(ground)
    ids = []
    for j, r in enumerate(realised):
        mesh = r["mesh"]
        density = r["mass"] / max(mesh.volume(), 1e-9)
        b = RigidBody.from_mesh(r.get("name", f"stone{j}"), mesh, density, friction=r["mu"], sdf_spacing=None)
        ids.append(world.add_body(b))
    world.step(int(0.2 / world.dt))
    ref = world.state()
    g0 = np.asarray(get(cfg, "world.gravity", (0, 0, -9.81)), dtype=float)
    best = max_deg
    for az in np.linspace(0, 2 * np.pi, n_azimuth, endpoint=False):
        w = world.clone()
        alpha = 0.0
        n_check = 10
        failed = False
        while alpha < best:
            alpha += rate_deg_s * w.dt * n_check
            w.gravity = _rotated_gravity(g0, alpha, az)
            w.step(n_check)
            for i in ids:
                b = w.bodies[i]
                d = np.linalg.norm(b.x - ref.x[i])
                r = np.degrees(np.linalg.norm(quat_to_rotvec(quat_mul(b.q, quat_conj(ref.q[i])))))
                if d > disp_tol or r > rot_tol_deg:
                    failed = True
                    break
            if failed:
                break
        if failed:
            best = min(best, alpha)
    return float(best)
