#!/usr/bin/env python3
"""Re-record the showcase tilt and collapse clips on a world rebuilt from the stored stones.

The world is rebuilt from the body frame meshes, masses and final poses in results/showcase (body
ids follow the subset map of the recorded clips). The true pair friction table is not stored with
the showcase, so every pair uses mu = 0.6 (the prior median); the clips serve the renders only.

python scripts/reclip_showcase.py --collapse-deg 25 --tilt-seconds 20
"""

from __future__ import annotations

import pickle

import numpy as np

from _common import RESULTS, config, parser

from topple.core.mesh import TriMesh
from topple.harness.tests import _rotated_gravity, failure_check
from topple.physics.body import CollisionShape, RigidBody
from topple.physics.recorder import StateRecorder
from topple.physics.world import World


def main(args):
    cfg = config(args)
    d = RESULTS / "showcase"
    with open(d / "showcase.pkl", "rb") as fh:
        data = pickle.load(fh)
    smap = data["subset_map"]
    order = sorted(smap, key=smap.get)
    w = World(cfg["world"])
    stones = {s["id"]: s for s in data["stones"]}
    for old in order:
        if old == data["ground_id"]:
            w.add_body(RigidBody.ground(friction=0.6))
            continue
        s = stones[old]
        m = TriMesh(s["V"], s["F"])
        mp = m.mass_properties(s["mass"] / m.volume())
        b = RigidBody(s["name"], s["mass"], mp["inertia_com"], CollisionShape.from_mesh(m, sdf_spacing=None), friction=0.6)
        b.q = data["final_state"]["q"][old].copy()
        b.x = data["final_state"]["x"][old].copy()
        w.add_body(b)
    w.damped_settle(seconds=0.4)
    ids = [smap[i] for i in data["tower_ids"]]
    ref = w.state()
    az = float(data["tilt"]["azimuth"])
    g0 = w.gravity.copy()
    h = cfg["harness"]

    def clip(seconds, angle_fn, every):
        ww = w.clone()
        rec = StateRecorder(ww, every=every)
        fail = None
        for k in range(int(seconds / ww.dt)):
            ww.gravity = _rotated_gravity(g0, angle_fn(k * ww.dt), az)
            ww.step(1)
            if fail is None and k % 10 == 0 and failure_check(ww, ref, ids, h["fail_disp_m"], h["fail_rot_deg"])[0]:
                fail = ww.time
        return rec, fail

    tilt, tf = clip(args.tilt_seconds, lambda t: 0.5 * t, 10)
    coll, cf = clip(3.0, lambda t: min(args.collapse_deg, 20.0 * t), 1)
    tilt.save(d / "tilt_record.npz")
    coll.save(d / "collapse_record.npz")
    data["tilt"].update({"fail_time": tf, "rate_deg_s": 0.5, "rebuilt_mu": 0.6})
    data["collapse"].update({"fail_time": cf, "angle": args.collapse_deg, "rebuilt_mu": 0.6})
    with open(d / "showcase.pkl", "wb") as fh:
        pickle.dump(data, fh)
    print(f"tilt clip {args.tilt_seconds} s, first failure at {None if tf is None else round((tf - tilt.times[0]) * 0.5, 2)} deg; collapse clip to {args.collapse_deg} deg, failure at {cf}")


if __name__ == "__main__":
    ap = parser("re-record showcase clips")
    ap.add_argument("--collapse-deg", type=float, default=25.0)
    ap.add_argument("--tilt-seconds", type=float, default=20.0)
    main(ap.parse_args())
