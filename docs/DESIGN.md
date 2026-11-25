# Design decisions and deviations

This file records every place where the implementation adds to or departs from the technical
specification, and why. Section numbers refer to the specification.

## Language and libraries (Section 13.1)

The physics engine and the path tracer are Python with numba kernels for the hot loops (ray
casting, GJK and EPA, contact manifolds, projected Gauss Seidel, XPBD, the path tracing loop).
The specification lists this as the faster first option. Linear programs use HiGHS (through
scipy, and through `highspy` for the repeated tilt margin solves), quadratic programs use OSQP.
JAX is not used: margin derivatives come from the dual solution of the linear program
(Section 7.7) and the first order propagation uses finite differences on the reduced vector.

## Physics (Section 6)

- **Torsional patch friction.** A point contact cannot resist spin about its normal, so a stone
  resting on one or two loaded points kept any small yaw impulse and crept at about 2 degrees
  per second, enough to trip the 2 degree failure rule with no perturbation at all. Each contact
  now carries a torsional row bounded by mu times r_t times the normal impulse (a contact patch
  of radius r_t = 2 mm, `world.torsion_radius`, author's choice), solved in the same Gauss Seidel
  sweep. Setting it to 0 recovers point contacts. A test checks the spin deceleration of a
  sphere against mu r_t m g / I.
- **Settling damping.** Rigid rolling contacts do not dissipate rocking energy, so a released
  stone on a curved support could rock for seconds. The settling procedures (pile drop, candidate
  settling in the planner, post release settling) step with a velocity damping of 60 per second,
  then step without damping until every stone moves less than 0.1 mm and 0.05 degrees in each of
  four consecutive windows of 50 steps (up to four damped and undamped cycles), and zero the
  velocities. No damping is applied during any dynamic test.
- **Oracle lowering.** A kinematic stone touching the static ground produces no impulse (both
  have zero inverse mass), so the oracle placement finds the first contact by conservative
  advancement on the separation distance, the quasi static limit of the 1 cm/s descent of
  Section 9.6. The arm placement keeps force sensing through the wrist.
- **PGS iterations.** The default is 100 iterations with warm starting and a convergence test,
  which keeps the ten cube stack drift at 0.05 mm (limit 0.1 mm) at no cost once at rest.
- **Touching hulls.** GJK reports a distance below 0.1 micrometre for every resting contact, and
  EPA started from such a flat simplex can return a large depth with a wrong witness point (one
  case gave 1.3 mm, and the position correction launched the stone). The query now moves the
  second piece apart along the centroid direction by 0.1 mm (then 3 mm), runs GJK on the
  separated pair, takes the normal from the closest simplex triangle, and subtracts the shift
  along that normal; EPA is used only for deeper overlaps.
- **Manifold points.** The contact manifold keeps the deepest point and then fills its slots with
  points within 0.1 mm of the deepest before speculative points further away (farthest point
  sampling within each tier). A purely spatial selection had dropped the three vertices of the
  facet a stone rested on, so the stone stood on one point and rocked. Near ties keep a stable
  order so that persistent contacts keep their warm start.
- **Settling wobble.** During the damped part of a settle the gravity direction wobbles on a cone
  of 0.5 degrees at 2 turns per second (`World.damped_settle`, author's choice). A stone that
  damping had left balanced on a mesh edge tips onto a stable facet before the tests start;
  without it, a 1 degree tilt tipped such a stone by more than 2 degrees.
- These three changes came after the first round of experiments, which is archived in
  `results/archive_v1/`; every reported experiment was run again with them.

## Static analysis (Section 7)

- **Geometry.** The engine collides convex pieces, which for every generated stone (convexity
  0.99 to 1.0) is the convex hull. The static analysis uses the same hull geometry, with
  closed form plane queries for seating and contact detection (170 times faster than mesh
  queries). `statics.geometry: mesh` restores the surface mesh.
- **Contact band.** Contacts are hull vertices within 0.5 mm of the support after seating
  (`statics.contact_band`, author's choice). The measured critical tilt depends on this choice
  because a stone can rock onto vertices just outside the band before the 2 degree rule trips;
  the E1 records allow a sensitivity analysis.
- **Sample settling.** A posterior sample perturbs shape and pose, so at the planned pose it can
  float, sink or rest off balance. Each sample stone is seated vertically and then tipped about
  its support polygon edge in 0.5 degree steps until its centre of mass is over the support, before
  the margin is computed. Without this step single stones on flat ground received margins near
  zero because their sampled undersides rested on arbitrary triangles.

- **Rocking aware margin (Margin 4, an addition).** The tilt margin is the tilt at which the force
  program becomes infeasible, while the dynamic test fails a stone that rotates 2 degrees, which a
  stone on curved facets does by pivoting onto the next facet well before infeasibility.
  `statics/rocking.py` steps the tilt (1 degree, then bisection) for each azimuth, rotates the
  hulls into the gravity frame, settles each stone quasi statically (support polygon from the
  vertices within 0.02 mm of a support), and reports the first tilt at which a stone has moved
  2 degrees or 5 mm from the model's own settled state at zero tilt, or the force program is
  infeasible. It is evaluated next to the tilt margin in E1 and E2 (kind `rocking`).

## Perception (Section 8)

- **Mean mesh smoothing.** Ten Taubin iterations on the posterior mean mesh remove millimetre
  spikes of the fused surface that otherwise inflated the hull used for contact (convexity rose
  from 0.76 to 0.93 to 0.97 on the quick profile).
- **Pose uncertainty.** The fused model averages independent per view pose errors, so the sensor
  part of the pose posterior scales with one over the square root of the number of views.
- **Poisson completion.** `tsdf.completion: poisson` fits a screened Poisson surface (Kazhdan and
  Hoppe 2013) to the observed oriented points on the TSDF grid (finite differences, conjugate
  gradients) and uses the signed distance to the extracted surface as the prior of the unobserved
  voxels. The default stays `hull`, as in the reported experiments.
- **Sobol groups (E3).** A shape mode belongs to the underside group when its energy is more
  concentrated on the vertices facing down at the placed orientation (normal z below -0.5) than a
  uniform displacement; the first round grouped by observed voxels, which left the group empty.

## Planning (Section 9)

- Candidates are settled dynamically on the estimated geometry (Section 9.1) in a support world
  where the placed stones are static. A candidate is stable when it comes to rest on the current
  top stone only (a tower is one column).
- Candidate centres are drawn over the inner half and the upper quarter of the top stone, where
  its surface is close to horizontal.
- The height reward is lambda_h = 200 degrees of margin per metre (author's choice; the
  specification leaves the unit open).
- Candidate screening uses 8 Monte Carlo samples and 12 azimuths with a normal approximation of
  the survival probability; the chosen pose is evaluated again with the full method.

## Experiment protocol (Section 11)

- **Probe placements.** With the chance constraint p_hat >= 0.95, every placement that builds the
  tower has a high predicted probability, which leaves the reliability diagram empty below 0.95.
  At every step two more candidates, drawn from different bins of predicted probability, are
  placed and tested on clones of the true world (ground plus tower). They do not change the
  tower and are flagged `probe` in the records; tower height and collapse rate use the building
  placements only.
- **Stopping.** The tower is complete when no candidate meets the chance constraint; the best
  candidate is still predicted and tested as a probe.
- **Shared placements.** E1 and E7 evaluate all margins and cone discretisations on the same
  exact input placements; E2 and E8 evaluate all propagation methods and the simulation score on
  the same perceived placements. Each placement is tested once.
- **Tests on the tower alone.** The dynamic tests run on a clone holding the ground and the
  tower, without the remaining pile stones.
- **Tilt sweep.** Rotating gravity at 0.5 degrees per second over 8 azimuths to 60 degrees is
  up to 16 simulated minutes per placement. Before the first motion the state of a resting
  tower does not depend on the rate, so each azimuth first sweeps at 10 degrees per second, then
  restarts 3 degrees below the first failure at the specified 0.5 degrees per second.
- **Scale.** The reported experiments use `configs/study.yaml` (24 towers per condition,
  320 x 240 depth images, 64 Monte Carlo samples, 6 test azimuths). The default config is the
  full specification scale.
- **Placement.** The calibration runs E1, E2, E5 and E6 use the oracle placement (Section 13.4,
  M7). E4 places every stone with the arm (grasp selection, IK, RRT-Connect, admittance descent,
  force contact detection, compliant release), the full autonomous loop; a placement that the
  arm cannot complete falls back to the oracle and is flagged `arm_fallback`. Choices made for
  the arm:
  - The release is a compliant handover: at contact the stone becomes dynamic and is held to the
    tool frame by a force limited spring damper (30 N, 1 N m) whose stiffness ramps to zero while
    the fingers open, so the support takes the load gradually.
  - The descent is an admittance realisation of the Cartesian impedance law on a position
    controlled arm: the setpoint moves by the filtered wrist force over the stiffness, limited to
    5 mm and 2 cm/s.
  - The closed fingers hold the stone as an attachment to the tool frame from grasp closure until
    the fingers open past the grasp width; friction carrying between kinematic fingers was not
    reliable (the stone slipped while lifted).
  - The stone can shift while the fingers close, so the place and approach poses are planned
    again from the held offset (the in hand pose).
  - The wrist force sensor reads the reaction on the held stone through a compliance of 2e4 N/m
    on the penetration depth, because a kinematic held stone exchanges no impulse with a static
    support.
  - Without regrasping, a stone keeps its pile orientation up to a yaw, so a grasp axis that is
    horizontal at the pick stays horizontal at the place (`planner.keep_pile_orientation`).
  - The demonstration uses a pile laid out on arcs within the arm's reach
    (`stones.pile_layout: arc`) and a 12 cm finger opening (`arm.finger_max`, the Panda hand
    opens 8 cm); the calibration runs keep the grid layout. These two config keys were added after
    the study runs; their defaults reproduce the earlier behaviour.

## Rendering (Section 10)

- The sky is the Hošek and Wilkie (2012) model with the authors' CIE XYZ coefficient tables (BSD
  licence, from the reference implementation as redistributed in Blender 2.93), evaluated with
  the reference interpolation. Preetham et al. (1999) remains an option. The sun disc radiance
  uses Preetham's transmittance for both.
- The spectral mode traces three stratified wavelengths per sample in the three channels of the
  kernel (hero wavelength sampling uses four). RGB values are upsampled with a smooth three band
  basis that sums to one, so white reflectance is 1 at every wavelength; colour matching
  functions are the fit of Wyman, Sloan and Shirley (2013) and the white is a 6504 K blackbody.
  A dispersive dielectric (Cauchy) keeps only the hero wavelength after a refraction.
- Water is a smooth dielectric (refraction, Fresnel, absorption, radiance scaled by (n_i/n_t)^2
  across the interface) whose normal carries the ripple bump. Manifold next event estimation
  connects points under water to the sun through one refraction: a Newton solve on the specular
  constraint, the beam compression from finite differences, and the sun disc dropped from paths
  that the estimator already counts. One solution is found per point, so very strong ripples
  lose some caustic energy (half at three times the river preset's ripple strength).
- The macro still H2 uses a traced lens system: the double Gauss 50 mm F/2 of Kolb, Mitchell and
  Hanrahan (1995), focused by bisection on the paraxial object distance, with exit pupil bounds
  per film radius for sampling. Its vignetting and field curvature come from the prescription.
- Subsurface scattering is an isotropic random walk; sparkle uses hashed micro flakes.
- Intel Open Image Denoise is used when the `oidn` Python package is installed; otherwise a joint
  bilateral filter on the albedo and normal buffers.
- V4 has modal impact sound (`render/sound.py`): mass spring surface modes per stone, scaled so the
  lowest mode matches the fundamental of a granite sphere of the same volume, excited by the
  impulses read from the recorded clip.
- The committed stills and videos use reduced resolution and sample counts; the scripts take the
  specification settings as arguments.

## Validation (Section 12.2)

MuJoCo 4.0.1, Drake and PyBullet run the cross engine tests. PyBullet runs in a separate
Python 3.11 environment (`.conda311`, conda forge build), since it has no wheel for Python 3.13;
`fig_engine_comparison.py --engines pybullet --merge` adds its results to the same towers.
Engine settings: Drake uses SAP with compliant hydroelastic contact between stones (patches
resist spin) and point contact with the ground (a compliant pressure field moves the centre of
pressure inside the tipping edge, so a block on the ground tipped about 2 degrees early);
PyBullet uses 200 solver iterations, a 0.5 mm convex margin (its default margin rounds small
stones) and spinning friction mu r_t. A test checks that a settled three stone tower holds for
2 s at zero tilt in every engine, and F4 uses only towers that hold in the developer's engine.

## Items not implemented

- The force torque placement correction of arXiv:2404.17668 (Section 9.6, optional).
- The extensions of Section 14, apart from the arm placement demonstration and the friction slip
  measurement posterior, which are part of the core modules.
