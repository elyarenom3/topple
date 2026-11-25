"""PyBullet adapter (Coumans and Bai 2016 to 2021).

Each dynamic body is a multi body with one convex mesh collision shape per piece (compound
when several), explicit mass and principal inertia, and the lateral friction set per body.
PyBullet combines pair friction as the product of the two coefficients, so each body gets
sqrt of its own coefficient and explicit pair values are not representable; the adapter
reports this in ``notes``. Clones use ``saveState`` and ``restoreState``.
"""

from __future__ import annotations

import numpy as np

from .common import BodyView, EngineWorld


class PyBulletWorld(EngineWorld):
    notes = "pair friction approximated as sqrt(mu_a) * sqrt(mu_b)"

    def __init__(self, client, bodies, gravity, dt, uid_of_body):
        super().__init__(bodies, gravity, dt)
        import pybullet as p

        self.p = p
        self.client = client
        self.uid_of_body = uid_of_body
        self._pending = {}
        self._sync_from_engine()

    @staticmethod
    def from_world(world, tmpdir=None, solver_iterations: int = 200, collision_margin: float = 5e-4, torsion_radius: float | None = None):
        """``collision_margin`` replaces Bullet's default convex margin (centimetres, which rounds
        small stones); spinning friction is set to mu times the contact patch radius
        ``torsion_radius`` (the developer's engine value by default), split between the two bodies
        because Bullet multiplies the per body coefficients."""
        import pybullet as p

        r_t = float(getattr(world, "torsion_radius", 0.002) if torsion_radius is None else torsion_radius)
        ground_mu = next((b.friction for b in world.bodies if b.static), 0.6)

        client = p.connect(p.DIRECT)
        p.resetSimulation(physicsClientId=client)
        p.setTimeStep(world.dt, physicsClientId=client)
        p.setGravity(*world.gravity, physicsClientId=client)
        p.setPhysicsEngineParameter(numSolverIterations=int(solver_iterations), physicsClientId=client)
        plane = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=client)
        ground_uid = p.createMultiBody(0, plane, physicsClientId=client)
        p.changeDynamics(ground_uid, -1, lateralFriction=float(np.sqrt(ground_mu)), spinningFriction=float(np.sqrt(ground_mu * r_t)), physicsClientId=client)
        views, uids = [], {}
        for b in world.bodies:
            if b.static:
                views.append(BodyView(b.id, b.name, b.mass, True, False, b.x.copy(), b.q.copy()))
                uids[b.id] = ground_uid
                continue
            shapes = []
            for piece in b.shape.pieces:
                shapes.append(p.createCollisionShape(p.GEOM_MESH, vertices=piece.vertices.tolist(), physicsClientId=client))
            if len(shapes) == 1:
                col = shapes[0]
            else:
                col = p.createCollisionShapeArray(
                    shapeTypes=[p.GEOM_MESH] * len(shapes),
                    vertices=[piece.vertices.tolist() for piece in b.shape.pieces],
                    physicsClientId=client,
                )
            q_xyzw = [b.q[1], b.q[2], b.q[3], b.q[0]]
            uid = p.createMultiBody(baseMass=0.0 if b.kinematic else b.mass, baseCollisionShapeIndex=col, basePosition=b.x.tolist(), baseOrientation=q_xyzw, physicsClientId=client)
            p.changeDynamics(uid, -1, localInertiaDiagonal=np.diag(b.inertia_body).tolist(), lateralFriction=float(np.sqrt(b.friction)), spinningFriction=float(np.sqrt(b.friction * r_t)), collisionMargin=float(collision_margin), restitution=0.0, physicsClientId=client)
            views.append(BodyView(b.id, b.name, b.mass, False, b.kinematic, b.x.copy(), b.q.copy()))
            uids[b.id] = uid
        return PyBulletWorld(client, views, world.gravity, world.dt, uids)

    def _sync_from_engine(self):
        for v in self.bodies:
            if v.static:
                continue
            pos, orn = self.p.getBasePositionAndOrientation(self.uid_of_body[v.id], physicsClientId=self.client)
            v.x = np.array(pos)
            v.q = np.array([orn[3], orn[0], orn[1], orn[2]])

    def _apply_gravity(self):
        self.p.setGravity(*self._gravity, physicsClientId=self.client)

    def step(self, n: int = 1):
        for _ in range(n):
            for bid, (f, pw) in self._pending.items():
                uid = self.uid_of_body[bid]
                pos = pw if pw is not None else self.p.getBasePositionAndOrientation(uid, physicsClientId=self.client)[0]
                self.p.applyExternalForce(uid, -1, f.tolist(), list(pos), self.p.WORLD_FRAME, physicsClientId=self.client)
            self._pending = {}
            self.p.stepSimulation(physicsClientId=self.client)
            self.time += self.dt
        self._sync_from_engine()

    def external_force(self, body_id, f, p_world=None):
        self._pending[body_id] = (np.asarray(f, dtype=float), None if p_world is None else np.asarray(p_world, dtype=float))

    def clone(self):
        """PyBullet state snapshots live inside one client; a clone restores the snapshot on use.

        The harness runs tests sequentially, so a clone is realised as "restore state now".
        """
        sid = self.p.saveState(physicsClientId=self.client)
        w = _RestoringPyBulletWorld(self, sid)
        return w


class _RestoringPyBulletWorld(PyBulletWorld):
    def __init__(self, parent: PyBulletWorld, sid: int):
        self.p = parent.p
        self.client = parent.client
        self.uid_of_body = parent.uid_of_body
        self.bodies = [BodyView(v.id, v.name, v.mass, v.static, v.kinematic, v.x.copy(), v.q.copy()) for v in parent.bodies]
        self._gravity = parent._gravity.copy()
        self.dt = parent.dt
        self.time = parent.time
        self._pending = {}
        self.p.restoreState(sid, physicsClientId=self.client)
        self._apply_gravity()
        self._sync_from_engine()
