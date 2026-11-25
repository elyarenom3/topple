# Module interface contracts

Every module is written against these names. Section numbers refer to the technical
specification. Units are SI (metres, kilograms, seconds, radians) unless a name ends in
`_deg`. Arrays are float64 NumPy arrays. Quaternions are (w, x, y, z).

Shared foundations (written, see `topple/core`):

- `topple.core.transforms`: `SE3`, `quat_*`, `skew`, `tangent_basis`, `rotation_from_two_vectors`.
- `topple.core.mesh`: `TriMesh` (mass properties, validity, hull, curvature, subdivide, Taubin,
  sample_surface, signed_distance), `SDF`, primitives `box`, `icosphere`, `tetrahedron`, `cylinder`,
  `plane`, `concatenate`, and `convex_pieces`.
- `topple.core.raycast`: `BVH` with `intersect`, `occluded`, `count_hits`, `closest_points`; numba
  kernels `intersect_closest`, `intersect_any`, `ray_tri`, `point_tri_closest`.
- `topple.core.config`: `load_config(name, overrides)`, `config_hash`, `merge`, `get(cfg, "a.b.c")`.
- `topple.core.rng`: `make_rng(seed, tower, Stage, sub)`.
- `topple.physics.body`: `CollisionShape`, `RigidBody`, `Contact`, `WorldState`.

Rule: the estimator (`topple.perception`) and the planner (`topple.planning`) never import
`topple.harness` (which owns the hidden ground truth store). A test asserts this.

---------------------------------------------------------------------------------------------

## physics  (Section 6)

`topple/physics/world.py`

```python
class World:
    def __init__(self, cfg: dict | None = None, gravity=(0,0,-9.81), dt=1e-3, solver="pgs",
                 cone_sides=8, pgs_iterations=40, contact_margin=1.5e-3, max_manifold_points=8,
                 merging=False, baumgarte=0.2, slop=5e-4, restitution=0.0)
    bodies: list[RigidBody]            # body.id == index in this list
    gravity: np.ndarray                 # mutable: the tilt test rotates it
    time: float
    contacts: list[Contact]             # contacts of the last step, impulses filled in
    def add_body(self, body: RigidBody) -> int
    def remove_body(self, body_id: int) -> None
    def set_pair_friction(self, id_a: int, id_b: int, mu: float) -> None
    def pair_friction(self, id_a: int, id_b: int) -> float      # table, else sqrt(mu_a * mu_b)
    def step(self, n: int = 1) -> None                           # advances n * dt
    def state(self) -> WorldState
    def set_state(self, s: WorldState) -> None
    def clone(self) -> "World"                                   # shares shapes, copies state
    def kinetic_energy(self) -> float
    def body_by_name(self, name: str) -> RigidBody
    def set_kinematic_velocity(self, body_id, v, w) -> None      # prescribed twist, used by arm and shake table
    def external_force(self, body_id, f, p_world=None) -> None   # one step force (push test)
    recorder: StateRecorder | None                                # when set, every step appends state()
```

Semantics: `step` runs broad phase, narrow phase (`find_contacts`), the velocity level
solver (Stewart Trinkle / Anitescu Potra with a polyhedral cone of `cone_sides` sides, PGS by
default, `lemke` reference for small systems), implicit gyroscopic integration, and position
update. Kinematic bodies have zero inverse mass and keep their prescribed velocity. Static bodies
never move. `solver="xpbd"` selects the position based secondary solver with the same `World`
API. `world.contacts` after a step holds `Contact` objects with `fn, ft1, ft2` as impulses and the
cone residuals available through `world.last_residuals` (dict with `max_cone_violation`,
`max_complementarity`).

`topple/physics/collision.py`

```python
def find_contacts(world: World, margin: float | None = None) -> list[Contact]
    # broad phase: dynamic AABB tree over bodies (class AABBTree with insert/update/query_pairs)
    # narrow phase: convex piece pairs with GJK + EPA (functions gjk_distance, epa_penetration),
    # halfspace special case, contact manifold from the support patch within `margin`, reduced to
    # world.max_manifold_points by farthest point sampling, persistent feature ids for warm start.
    # Contacts are returned for pairs whose separation is below `margin` (depth = -separation).
def verify_contacts_sdf(world, contacts, tol) -> dict    # cross check with SDF queries (Section 6.4)
def gjk_distance(A_vertices, B_vertices) -> (dist, pa, pb)
def epa_penetration(A_vertices, B_vertices) -> (depth, normal, pa, pb)
```

`topple/physics/solver.py`: `solve_pgs(...)`, `solve_lemke(...)`, `build_contact_problem(...)`,
`cone_residuals(...)`. The LCP form of Section 6.3 is documented in the module docstring.

`topple/physics/xpbd.py`: `step_xpbd(world)` (Müller et al. 2020 rigid body XPBD).

`topple/physics/merging.py`: `MergeManager` (Coevoet et al. 2020): groups of bodies at rest with
a friction cone unmerge test, enabled by `World(merging=True)`.

`topple/physics/recorder.py`

```python
class StateRecorder:
    def __init__(self, world, every: int = 1)
    times: list[float]; states: list[WorldState]
    def append(self, state: WorldState)
    def pose_at(self, body_id: int, t: float) -> SE3    # linear position, slerp orientation
    def save(path) / load(path)
```

`topple/physics/arm.py` (Section 6.6, used by planning for IK and by the controller)

```python
class ArmModel:
    """7 revolute joints + 2 prismatic fingers, product of exponentials, reduced coordinates."""
    n_joints: int = 7
    q_min, q_max, dq_max, tau_max: np.ndarray (7,)
    finger_max: float                           # 0.08 m opening
    def __init__(self, name="panda7", base: SE3 = SE3())
    def fk(self, q) -> SE3                       # tool centre point (between the fingertips)
    def link_poses(self, q) -> list[SE3]         # 8 link frames (base to flange)
    def jacobian(self, q) -> np.ndarray          # (6, 7) spatial Jacobian of the TCP: rows [v; w]
    def inverse_dynamics(self, q, dq, ddq, gravity) -> tau     # recursive Newton Euler
    def forward_dynamics(self, q, dq, tau, gravity) -> ddq     # articulated body algorithm
    def mass_matrix(self, q) -> (7,7)
    def link_meshes(self) -> list[TriMesh]       # capsules/boxes in link frames for collision and render
    def finger_meshes(self, opening) -> (TriMesh, TriMesh)  # in TCP frame
class ArmInWorld:
    """Owns the kinematic link and finger bodies inside a World and drives them from joint state."""
    def __init__(self, world: World, model: ArmModel, finger_mu=1.0)
    q, dq: np.ndarray; opening: float
    def set_joint_state(self, q, dq, opening, d_opening=0.0)  # updates link body poses and twists
    def step_dynamics(self, tau, dt)                            # ABA integration with limits
    def wrist_wrench(self) -> np.ndarray (6,)                  # simulated force torque sensor from finger contacts
```

Tests `tests/test_physics.py` implement Section 12.1 items 1 to 8.

---------------------------------------------------------------------------------------------

## statics  (Section 7)

`topple/statics/tower.py`

```python
@dataclass
class StaticBody: id: int; mass: float; com: np.ndarray; is_ground: bool; name: str = ""
@dataclass
class StaticContact: body_a: int; body_b: int; point: np.ndarray; normal: np.ndarray; mu: float
class Tower:
    bodies: list[StaticBody]       # ground has mass 0 and is_ground True
    contacts: list[StaticContact]
    gravity: np.ndarray
    @staticmethod
    def from_world(world, margin=None, mu_override: dict | None = None) -> Tower   # uses find_contacts
    @staticmethod
    def from_estimates(bodies: list[dict], contacts: list[StaticContact]) -> Tower
    def upper_subassembly(self, cut_index) -> Tower   # Section 7.6
    def interfaces(self) -> list[list[StaticContact]]   # contacts grouped by body pair, bottom to top
```

`topple/statics/equilibrium.py`

```python
def contact_basis(contact, cone_sides) -> (B (6,3), C (N,3))        # Section 7.1, 6.2
def assemble(tower, gravity, cone_sides) -> (A, w_g, C_blocks, index)   # A f = -w_g, C f <= 0
def feasible(tower, gravity=None, cone_sides=8) -> (bool, forces | None)  # LP through scipy HiGHS
def min_norm_forces(tower, gravity=None, cone_sides=8) -> np.ndarray (n_contacts, 3)   # QP (OSQP, scipy fallback)
```

`topple/statics/margins.py`

```python
def tilt_margin(tower, cone_sides=8, n_azimuth=36, tol_deg=0.05, max_deg=89) -> (deg, azimuth_rad, per_azimuth)
def push_margin(tower, top_body, cone_sides=8, n_dir=8) -> (s_over_weight, direction)
def fc_margin(tower, cone_sides=8, force_budget_factor=2.0, length_scale=0.1) -> float
def support_polygon_margin(tower, body_id) -> (deg, d, h_c, polygon)      # Section 7.5
def subassembly_margins(tower, kind="tilt", **kw) -> list[float]            # per cut, bottom to top
def tower_margin(tower, kind="tilt", **kw) -> float                         # min over cuts
def analytic_block_tilt(mu, b, h) -> deg                                    # min(atan mu, atan b/h)
```

`topple/statics/sensitivity.py`: `margin_gradient(tower, wrt="com"|"mu"|"points", ...)` from the
dual solution (Section 7.7); `margin_jacobian_fd` finite difference check.

`topple/statics/survival.py`: `survival_probability(margins, tau_deg)`, `margin_summary(margins)`.

Tests `tests/test_statics.py`: block (analytic), block on wedge, three block arch, mass scaling
invariance, rotation about vertical invariance (Section 12.3).

---------------------------------------------------------------------------------------------

## stonegen  (Section 5)

```python
# topple/stonegen/families.py
def family_a_hull(rng, a, b, c, n_points=40, subdivisions=3, smooth_iters=10, fractal_amp=0.0) -> TriMesh
def family_b_ellipsoid(rng, a, b, c, amplitude, octaves=4, gain=0.5, subdivisions=3) -> TriMesh
def family_c_superquadric(rng, a, b, c, e1, e2, perturb=0.0, resolution=48) -> TriMesh
def family_d_voronoi(rng, block_extents, n_seeds, pick) -> TriMesh
# topple/stonegen/noise.py: perlin3(x, seed), fbm3(x, octaves, gain, lacunarity, seed), worley3
# topple/stonegen/erosion.py: erode(mesh, eta, steps, remesh_every=1) -> TriMesh   (Section 5.2)
# topple/stonegen/descriptors.py: descriptors(mesh) -> dict(elongation, flatness, convexity, sphericity, a, b, c)
# topple/stonegen/generator.py
@dataclass
class StoneSpec: name, mesh (body frame, principal axes), density, volume, mass, com (zeros), inertia (3,3),
                 stone_class, family, descriptors, material (render preset name), seed
class StoneGenerator:
    def __init__(self, cfg: dict)
    def sample(self, rng, stone_class: str | None = None) -> StoneSpec      # rejection sampling on class boxes
    def sample_pile(self, rng, n) -> list[StoneSpec]
def stone_to_body(spec, friction, sdf_spacing) -> RigidBody
```

Each mesh passes `TriMesh.is_valid()`; `StoneSpec.mesh` is in the principal frame.

---------------------------------------------------------------------------------------------

## perception  (Section 8)

```python
# topple/perception/camera.py
class PinholeCamera: K (3,3), width, height; from_fov(w, h, fov_deg); project(P_cam); backproject(u, v, z); rays(T_world_cam) -> (origins, dirs)
# topple/perception/depth_sensor.py
class DepthSensor:
    def __init__(self, camera: PinholeCamera, cfg_sensor: dict, rng)
    def render_depth(self, meshes_world: list[TriMesh], T_world_cam: SE3, mask_ids=None) -> DepthImage
        # one primary ray per pixel against exact meshes (core.raycast), then noise model
@dataclass
class DepthImage: depth (H,W), valid (H,W) bool, sigma (H,W), T_world_cam: SE3, K, normals (H,W,3), incidence (H,W), object_id (H,W)
# topple/perception/noise.py: structured_light_shadow, quantise_disparity, khoshelham_sigma(z), nguyen_sigma(z, theta), handa_perturb
# topple/perception/tsdf.py
class TSDFVolume: __init__(origin, spacing, shape, truncation); integrate(depth_image, sigma, T_world_cam, K); D, W arrays; extract_mesh() -> TriMesh (marching cubes, own implementation, cross checked with skimage)
# topple/perception/marching_cubes.py: marching_cubes(values, origin, spacing, iso=0.0) -> TriMesh
# topple/perception/icp.py: icp_point_to_plane(src_pts, dst_pts, dst_normals, T0, iters) -> SE3
# topple/perception/posterior.py
class ShapePosterior:  # Section 8.5
    def __init__(self, tsdf, completion="hull"|"mirror"|"gp", correlation_length=5e-3, rng)
    mean_mesh: TriMesh
    def sample(self, n) -> list[TriMesh]
# topple/perception/mass.py: mass_properties(mesh, density) (wraps TriMesh), density_prior(rng, lo, hi), random_density_field_com(mesh, rng)
# topple/perception/friction.py: FrictionPrior(median, log_sigma).sample(rng, n); GridPosterior.update(slip_angle_obs, noise)
# topple/perception/estimator.py
@dataclass
class StoneEstimate: mean_mesh, samples: list[StoneSample], density, mu_samples, meta
@dataclass
class StoneSample: mesh, mass, com, inertia, mu, pose_offset: SE3 (pose uncertainty)
class Estimator:
    def __init__(self, cfg, rng)
    def estimate(self, depth_images: list[DepthImage], n_samples) -> StoneEstimate
# topple/perception/propagate.py
def monte_carlo(fn, samples) -> np.ndarray
def unscented(fn, mean, cov, alpha=1e-3, beta=2, kappa=0) -> (mean, var, sigma_points, values)
def first_order(fn, grad_fn, mean, cov) -> (mean, var)
# topple/perception/viewpoints.py: hemisphere_views(center, radius, n, rng) -> list[SE3]; in_hand_views(...)
```

Tests `tests/test_perception.py`: noise free depth recovers meshes below voxel size; volume and COM
error fall with view count; simulation based calibration of the COM posterior (Section 12.4).

---------------------------------------------------------------------------------------------

## planning  (Section 9)

```python
# topple/planning/resting.py: resting_orientations(body_mesh, n_drops, rng, world_factory) -> list[(quat, prob)]
# topple/planning/candidates.py: sample_candidates(top_surface_points, stone_spec, n, rng) -> list[SE3]; settle_candidate(world, body, pose, steps) -> SE3
# topple/planning/objective.py: objective(margin_samples, dh, kappa, lambda_h) -> float; chance_ok(p_hat, delta)
# topple/planning/optimise.py: cma_es(f, x0, sigma0, iters, rng), nelder_mead(f, x0, ...), gradient_ascent(f, grad, x0, ...)
# topple/planning/grasp.py: antipodal_grasps(mesh, mu_g, n_pairs, rng) -> list[Grasp]; ferrari_canny(wrenches, L); rank_grasps(grasps, com, target_pose, clearance_fn)
# topple/planning/kinematics.py: ik_dls(arm, target: SE3, q0, iters, lam, joint_limit_weight) -> (q, err); ik_sdls(...)
# topple/planning/rrt.py: class RRTConnect(sample_fn, collision_fn, q_min, q_max, step); plan(q0, q1) -> path; class RRTStar(...); shortcut(path, collision_fn, rng); time_parameterise(path, dq_max) -> (times, q(t))
# topple/planning/controller.py
class PlacementController:   # Section 9.6
    def __init__(self, world, arm_in_world, cfg_arm)
    def execute(self, trajectory, stone_body, target_pose, recorder=None) -> PlacementResult(contact_detected, released_pose, wrist_log)
# topple/planning/planner.py
class Planner:
    def __init__(self, cfg, rng)
    def choose(self, tower_estimate, pile_estimates, margin_fn) -> Decision(stone_index, pose, grasp, margin_samples, p_hat, J, candidates)
```

Planning uses `topple.statics` for margins and `topple.physics` for settling; it never imports
`topple.harness`.

---------------------------------------------------------------------------------------------

## render  (Section 10)

```python
# topple/render/scene.py
class Material: fields of the Burley principled convention (base_color, roughness, metallic, specular,
    anisotropy, wet (coat), flakes, subsurface (mfp), emission, texture: callable | None); presets in render/materials.py
class Instance: mesh (TriMesh, body frame), material, T0: SE3, T1: SE3 | None (shutter close), object_id, overlay: bool
class Scene: add(instance); set_ground(kind); lights: list; sky; medium; def build() -> accelerated scene
class Camera: position, look_at, up, focal_mm, f_number, focus_distance, sensor=(36,24), width, height,
    shutter=(0,1/48), aperture_blades=9, ortho=False; def dof_limits() -> (H, D_near, D_far)
# topple/render/lights.py: AreaLight, SunLight(angular_diameter_deg=0.53), HosekWilkieSky(elevation, turbidity), HDREnvironment(path)
# topple/render/presets.py: lighting_preset(name) -> list[lights]; camera_preset(name, scene_bounds); material_preset(name, rng)
# topple/render/integrator.py: render(scene, camera, spp, max_depth, seed, passes=("beauty","albedo","normal","depth","object_id","overlay")) -> dict[str, np.ndarray]
# topple/render/overlays.py: arrow, cone, tube, sphere, ellipsoid, polygon_outline, bar, heatmap_plane, dial, polyline -> TriMesh + colour
# topple/render/color.py: aces_rrt_odt, bradford_adapt, exposure_from_grey_card, srgb_encode, colormap("batlow"|"vik"|"roma")
# topple/render/io.py: write_exr(path, passes: dict), read_exr(path), write_png(path, rgb)
# topple/render/denoise.py: denoise(beauty, albedo, normal) (OIDN if importable, else guided filter)
# topple/render/flip.py: flip_error(ref, test) -> (mean, map)
# topple/render/video.py: encode(frames_dir, out, fps=24), camera_path_min_jerk(keyframes, times)
# topple/render/sampling.py: sobol_owen(n, dim, seed), blue_noise_mask(size), polygon_aperture_sample(u, blades)
```

Overlays are triangle meshes with unlit emissive materials, rendered in a second pass with the same
camera and composited alpha over the beauty pass (Section 10.6).

---------------------------------------------------------------------------------------------

## harness, eval, validation  (Sections 11, 12)

```python
# topple/harness/truth.py: GroundTruthStore (hidden): stones, true mu table, densities; only harness and eval read it
# topple/harness/tests.py: tilt_test, shake_test, push_test, fixed_perturbation(world, cfg) -> results with M* and y
# topple/harness/guard.py: assert_no_truth_imports()
# topple/eval/metrics.py: brier, murphy_decomposition, ece, mce, reliability_diagram, coverage, pit, spearman
# topple/eval/sobol.py: saltelli_design, first_order_indices
# topple/eval/report.py: results_table, bootstrap_ci
# topple/validation/{mujoco_adapter,pybullet_adapter,drake_adapter}.py: export_world(world) and run the same tilt test
```
