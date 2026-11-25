"""Drake adapter (Tedrake and the Drake Development Team 2019).

A discrete ``MultibodyPlant`` at the developer engine's time step with the SAP contact
solver and point contact. Each dynamic body is a free body with its spatial inertia at the
frame origin and one ``Convex`` collision geometry per piece written to an OBJ file. Pair
friction is combined by Drake as 2 mu_a mu_b / (mu_a + mu_b); the adapter sets each body's
coefficient so that the stone to ground pairs match and reports the approximation in
``notes``. Gravity is rotated through the plant's gravity field.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from .common import BodyView, EngineWorld


class DrakeWorld(EngineWorld):
    notes = "pair friction combined by Drake's harmonic rule"

    def __init__(self, diagram, plant, simulator, context, bodies, gravity, dt, body_index):
        super().__init__(bodies, gravity, dt)
        self.diagram = diagram
        self.plant = plant
        self.simulator = simulator
        self.context = context
        self.body_index = body_index
        self._pending = {}
        self._sync_from_engine()

    @staticmethod
    def from_world(world, tmpdir=None, approx: str = "sap", penetration_allowance: float = 1e-4, stiction_tolerance: float = 1e-5, substeps: int = 5, contact: str = "hydroelastic", hydro_modulus: float = 5e7, hc_dissipation: float = 5.0, ground_hydroelastic: bool = False):
        """``approx`` is the discrete contact approximation (sap, lagged, similar). The plant runs at
        ``world.dt / substeps``; ``penetration_allowance`` sets the point contact stiffness and
        ``stiction_tolerance`` the slip speed below which friction is treated as sticking. ``contact``
        is "point" or "hydroelastic" (compliant stones on a rigid ground, with point contact as the
        fallback); hydroelastic patches resist spin, which point contacts cannot. With
        ``ground_hydroelastic`` false (default) the stone to ground pairs use point contact: the
        pressure field of a compliant stone moves the centre of pressure inside the tipping edge, so a
        block on the ground tips about 2 degrees early, while point contacts on a flat ground already
        give several contact points and so resist spin."""
        from pydrake.geometry import Convex, HalfSpace, ProximityProperties, AddContactMaterial
        from pydrake.math import RigidTransform, RotationMatrix
        from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, CoulombFriction, DiscreteContactApproximation
        from pydrake.multibody.tree import SpatialInertia, UnitInertia
        from pydrake.systems.analysis import Simulator
        from pydrake.systems.framework import DiagramBuilder
        from pydrake.common.eigen_geometry import Quaternion

        tmpdir = tmpdir or tempfile.mkdtemp()
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=world.dt / max(int(substeps), 1))
        approx_map = {"sap": DiscreteContactApproximation.kSap, "lagged": DiscreteContactApproximation.kLagged, "similar": DiscreteContactApproximation.kSimilar}
        plant.set_discrete_contact_approximation(approx_map[approx])
        from pydrake.geometry import AddCompliantHydroelasticProperties, AddRigidHydroelasticProperties
        from pydrake.multibody.plant import ContactModel

        if contact == "hydroelastic":
            plant.set_contact_model(ContactModel.kHydroelasticWithFallback)
        plant.set_penetration_allowance(float(penetration_allowance))
        plant.set_stiction_tolerance(float(stiction_tolerance))
        ground_mu = 0.6
        for b in world.bodies:
            if b.static:
                ground_mu = b.friction
        def props(mu, compliant: bool):
            """Proximity properties: friction, and for hydroelastic contact a compliant (stones) or
            rigid (ground) hydroelastic representation; contact patches give torsional resistance."""
            p_ = ProximityProperties()
            AddContactMaterial(dissipation=hc_dissipation, friction=CoulombFriction(mu, mu), properties=p_)
            if contact == "hydroelastic":
                if compliant:
                    AddCompliantHydroelasticProperties(resolution_hint=0.01, hydroelastic_modulus=hydro_modulus, properties=p_)
                else:
                    AddRigidHydroelasticProperties(properties=p_)
            return p_

        ground_props = props(ground_mu, False)
        if contact == "hydroelastic" and not ground_hydroelastic:
            ground_props = ProximityProperties()
            AddContactMaterial(dissipation=hc_dissipation, friction=CoulombFriction(ground_mu, ground_mu), properties=ground_props)
        plant.RegisterCollisionGeometry(plant.world_body(), RigidTransform(), HalfSpace(), "ground", ground_props)
        views, idx = [], {}
        for b in world.bodies:
            if b.static:
                views.append(BodyView(b.id, b.name, b.mass, True, False, b.x.copy(), b.q.copy()))
                continue
            I = b.inertia_body
            unit = UnitInertia(I[0, 0] / b.mass, I[1, 1] / b.mass, I[2, 2] / b.mass)
            body = plant.AddRigidBody(b.name, SpatialInertia(b.mass, np.zeros(3), unit))
            for k, piece in enumerate(b.shape.pieces):
                path = os.path.join(tmpdir, f"drake_{b.name}_{k}.obj")
                piece.save(path)
                plant.RegisterCollisionGeometry(body, RigidTransform(), Convex(path), f"{b.name}_c{k}", props(b.friction, True))
            views.append(BodyView(b.id, b.name, b.mass, False, b.kinematic, b.x.copy(), b.q.copy()))
            idx[b.id] = body.index()
        plant.mutable_gravity_field().set_gravity_vector(np.asarray(world.gravity, dtype=float))
        plant.Finalize()
        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        pc = plant.GetMyMutableContextFromRoot(context)
        for b in world.bodies:
            if b.static:
                continue
            body = plant.get_body(idx[b.id])
            q = Quaternion(b.q[0], b.q[1], b.q[2], b.q[3])
            plant.SetFreeBodyPose(pc, body, RigidTransform(RotationMatrix(q), b.x))
        simulator = Simulator(diagram, context)
        simulator.Initialize()
        return DrakeWorld(diagram, plant, simulator, context, views, world.gravity, world.dt, idx)

    def _sync_from_engine(self):
        pc = self.plant.GetMyContextFromRoot(self.context)
        for v in self.bodies:
            if v.static:
                continue
            X = self.plant.EvalBodyPoseInWorld(pc, self.plant.get_body(self.body_index[v.id]))
            v.x = np.array(X.translation())
            qq = X.rotation().ToQuaternion()
            v.q = np.array([qq.w(), qq.x(), qq.y(), qq.z()])

    def _apply_gravity(self):
        pc = self.plant.GetMyMutableContextFromRoot(self.context)
        self.plant.mutable_gravity_field().set_gravity_vector(self._gravity)
        # a parameter change requires the context to be refreshed
        self.plant.SetDefaultState(pc, pc.get_mutable_state()) if False else None

    def step(self, n: int = 1):
        from pydrake.multibody.math import SpatialForce
        from pydrake.multibody.plant import ExternallyAppliedSpatialForce

        pc = self.plant.GetMyMutableContextFromRoot(self.context)
        for _ in range(n):
            forces = []
            for bid, (f, p) in self._pending.items():
                eaf = ExternallyAppliedSpatialForce()
                eaf.body_index = self.body_index[bid]
                eaf.p_BoBq_B = np.zeros(3)
                eaf.F_Bq_W = SpatialForce(np.zeros(3), np.asarray(f, dtype=float))
                forces.append(eaf)
            self._pending = {}
            self.plant.get_applied_spatial_force_input_port().FixValue(pc, forces)
            self.time += self.dt
            self.simulator.AdvanceTo(self.time)
        self._sync_from_engine()

    def external_force(self, body_id, f, p_world=None):
        self._pending[body_id] = (np.asarray(f, dtype=float), None if p_world is None else np.asarray(p_world, dtype=float))

    def clone(self):
        from pydrake.systems.analysis import Simulator

        context2 = self.context.Clone()
        sim2 = Simulator(self.diagram, context2)
        sim2.Initialize()
        views = [BodyView(v.id, v.name, v.mass, v.static, v.kinematic, v.x.copy(), v.q.copy()) for v in self.bodies]
        w = DrakeWorld(self.diagram, self.plant, sim2, context2, views, self._gravity, self.dt, self.body_index)
        w.time = self.time
        w._apply_gravity()
        return w
