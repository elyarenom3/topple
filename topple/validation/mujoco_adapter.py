"""MuJoCo adapter (Todorov, Erez and Tassa 2012).

The scene is built programmatically with ``mujoco.MjSpec``: each dynamic body gets a free
joint, explicit mass and inertia at its frame origin (the centre of mass), and one convex
mesh geom per collision piece. Pair friction uses explicit contact pairs so that the pair
table of the developer engine is reproduced (MuJoCo otherwise takes the larger of the two
geom coefficients). Gravity is rotated through ``model.opt.gravity``.
"""

from __future__ import annotations

import copy
import tempfile

import numpy as np

from .common import BodyView, EngineWorld


class MuJoCoWorld(EngineWorld):
    def __init__(self, model, data, bodies, gravity, dt, body_ids, geom_of_body):
        super().__init__(bodies, gravity, dt)
        import mujoco

        self.mj = mujoco
        self.model = model
        self.data = data
        self.body_ids = body_ids  # our id -> mujoco body id (or -1 for static)
        self.geom_of_body = geom_of_body
        self._pending = {}
        self._sync_from_engine()

    @staticmethod
    def from_world(world, tmpdir=None):
        import mujoco

        spec = mujoco.MjSpec()
        spec.option.timestep = world.dt
        spec.option.gravity = np.asarray(world.gravity, dtype=float)
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        spec.option.noslip_iterations = 5
        # ground
        g = spec.worldbody.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_PLANE
        g.size = [5, 5, 0.1]
        g.name = "ground"
        g.friction = [0.6, 0.005, 0.0001]
        g.condim = 3
        views = []
        body_ids = {}
        geom_of_body = {}
        for b in world.bodies:
            if b.static:
                views.append(BodyView(b.id, b.name, b.mass, True, False, b.x.copy(), b.q.copy()))
                body_ids[b.id] = -1
                geom_of_body[b.id] = ["ground"]
                continue
            body = spec.worldbody.add_body()
            body.name = b.name
            body.pos = b.x
            body.quat = b.q
            body.mass = b.mass
            # body frame is the principal frame, so the inertia is diagonal
            body.inertia = np.diag(b.inertia_body)
            body.ipos = [0, 0, 0]
            body.iquat = [1, 0, 0, 0]
            if not b.kinematic:
                j = body.add_joint()
                j.type = mujoco.mjtJoint.mjJNT_FREE
                j.name = f"{b.name}_free"
            names = []
            for k, piece in enumerate(b.shape.pieces):
                mesh = spec.add_mesh()
                mesh.name = f"{b.name}_mesh{k}"
                mesh.uservert = piece.vertices.astype(np.float64).ravel()
                mesh.userface = piece.faces.astype(np.int32).ravel()
                geom = body.add_geom()
                geom.type = mujoco.mjtGeom.mjGEOM_MESH
                geom.meshname = mesh.name
                geom.name = f"{b.name}_geom{k}"
                geom.friction = [b.friction, 0.005, 0.0001]
                geom.condim = 3
                geom.margin = 0.0
                geom.mass = 0.0  # mass comes from the body inertial element
                names.append(geom.name)
            geom_of_body[b.id] = names
            views.append(BodyView(b.id, b.name, b.mass, False, b.kinematic, b.x.copy(), b.q.copy()))
        # explicit pair friction
        for i, a in enumerate(world.bodies):
            for bb in world.bodies[i + 1 :]:
                mu = world.pair_friction(a.id, bb.id)
                for ga in geom_of_body[a.id]:
                    for gb in geom_of_body[bb.id]:
                        p = spec.add_pair()
                        p.geomname1 = ga
                        p.geomname2 = gb
                        p.friction = [mu, mu, 0.005, 0.0001, 0.0001]
                        p.condim = 3
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        mj_ids = {}
        for b in world.bodies:
            mj_ids[b.id] = -1 if b.static else mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b.name)
        return MuJoCoWorld(model, data, views, world.gravity, world.dt, mj_ids, geom_of_body)

    def _sync_from_engine(self):
        for v in self.bodies:
            mid = self.body_ids[v.id]
            if mid < 0:
                continue
            v.x = self.data.xpos[mid].copy()
            v.q = self.data.xquat[mid].copy()

    def _apply_gravity(self):
        self.model.opt.gravity[:] = self._gravity

    def step(self, n: int = 1):
        for _ in range(n):
            self.data.xfrc_applied[:] = 0.0
            for bid, (f, p) in self._pending.items():
                mid = self.body_ids[bid]
                if mid >= 0:
                    self.data.xfrc_applied[mid, :3] = f
                    r = p - self.data.xpos[mid] if p is not None else np.zeros(3)
                    self.data.xfrc_applied[mid, 3:] = np.cross(r, f)
            self._pending = {}
            self.mj.mj_step(self.model, self.data)
            self.time += self.dt
        self._sync_from_engine()

    def external_force(self, body_id, f, p_world=None):
        self._pending[body_id] = (np.asarray(f, dtype=float), None if p_world is None else np.asarray(p_world, dtype=float))

    def clone(self):
        data2 = copy.copy(self.data)
        views = [BodyView(v.id, v.name, v.mass, v.static, v.kinematic, v.x.copy(), v.q.copy()) for v in self.bodies]
        w = MuJoCoWorld(self.model, data2, views, self._gravity, self.dt, self.body_ids, self.geom_of_body)
        w.time = self.time
        return w
