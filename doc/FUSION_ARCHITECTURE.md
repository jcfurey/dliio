# Fusion architecture — dliio + robot_localization EKF

## September 8: downstream 3D tilt experiment

`launch/dlio_ekf.launch.py` adds an optional filter beside an existing DLIO
session. It uses `cfg/robot_localization_tilt.yaml`, requires the declared
`robot_localization` runtime dependency, and does not require wheel odometry
or an Exyn package. The older `dlio_rl.launch.py` below is a separate
wheel/track example and is not the configuration for the September tunnel bag.

```bash
ros2 launch direct_lidar_inertial_odometry dlio_ekf.launch.py \
  use_sim_time:=true body_frame:=base_link \
  odom_topic:=dlio/odom_node/odom gravity_topic:=dliio/auxiliary_gravity
```

Use the actual body frame (`vehicle` for the Exyn replay). The gravity input is
`geometry_msgs/Vector3Stamped`: **world +Z expressed in that body frame**, with
its acquisition stamp. Supply a reviewed sensor mounting transform upstream;
the adapter rejects a different frame instead of guessing an extrinsic. It
converts this direction to `sensor_msgs/Imu` roll/pitch with arbitrary zero yaw.
Only roll/pitch are enabled in the EKF. Gyro and acceleration blocks are marked
unavailable, and are not fused. Near-vertical pitch is rejected because this
experiment uses the Euler-angle robot_localization state.

The filter integrates differential DLIO XYZ and retains absolute DLIO yaw.
All three translation axes remain measured; `two_d_mode` stays false and no
zero-height or zero-vertical-velocity constraint is introduced. DLIO roll/pitch,
observer twist and accelerations are excluded. Gravity tilt stays absolute
(`imu0_differential: false`, `imu0_relative: false`). This can reduce the
component of vertical drift caused by integrating motion with a wrong attitude;
it cannot observe translation that the scan matcher has already lost.

`dlio_ekf_inputs.py` preserves acquisition stamps and the input pose. It limits
odometry to 5 Hz and tilt to 20 Hz, rejects duplicate/old stamps and invalid
poses, and requires a restart on bag rewind. The EKF enables 2 seconds of
lagged-measurement history for the slower DLIO output. The installed RL 3.10.0
node waits for nonzero `/clock` before creating its publishers: a replay must
publish a held clock, wait for subscribers, then release measurements. The
workspace experiment runner exercises this sequence.

Output is `odometry/tilt_filtered`, configurable with `filtered_topic:=`.
**This launch publishes no TF.** DLIO's existing `odom → body` and the graph's
`map → odom` retain their existing meaning. The comparison output does not
feed the DLIO observer, alter archived poses, deform the map or change camera
draping. Replacing a TF without correcting the historical map would mix
incompatible trajectories. If this experiment helps, frontend gravity/bias
work and validated graph constraints are the routes to consistent geometry.

Noise is explicit and still experimental. The default `covariance_policy:=assumed`
uses diagonal pose sigmas of 0.3 m / 0.1 rad and a tilt sigma of 5 degrees; all
are launch arguments (`assumed_position_sigma`, `assumed_angle_sigma`,
`assumed_tilt_sigma`, with angular values in radians). They are sensitivity
assumptions, not calibrated errors. `covariance_policy:=published` preserves
all 36 pose-covariance entries and rejects nonfinite, nonsymmetric or
non-positive-definite input. Passing that numerical check does not establish
calibration. DLIO differential XYZ and absolute yaw still share information;
the EKF does not model their cross-correlation between separate updates.
The BNO also shares information with DLIO when frontend auxiliary-gravity
aiding is enabled. Neither decimation nor a full output matrix removes these
dependencies or makes the posterior an honest full uncertainty estimate.

Validation includes an actual ROS EKF test with a known 10-degree attitude
error and true 0.05 m/s vertical motion: the filtered trajectory must preserve
the real slope, correct the attitude-induced component, retain finite positive
covariance and publish no TF. Captured-run comparisons, limitations and timing
receipts are documented by the integrating workspace; they are not portable
ground-truth fixtures in this package.

The parameter choices follow the primary
[robot_localization configuration guide](https://github.com/cra-ros-pkg/robot_localization/blob/rolling-devel/doc/configuring_robot_localization.rst)
and its [differential pose implementation](https://github.com/cra-ros-pkg/robot_localization/blob/rolling-devel/src/ros_filter.cpp).
Tests run against the locally installed version, not an assumption that a
moving upstream branch has identical behavior.

## July 9 wheel/track fusion design

Response to the operational tunnel-slosh problem (the corkscrew/figure-eight
oscillation along the tunnel axis) and the 2026-07-08 X-ICP runaway. Based on
the survey of degradation-robust systems: the SubT-winning approach is
**redundant odometry sources fused with health gating**, not a perfect single
estimator (CompSLAM, arXiv:2505.06483; SuperLoc, arXiv:2412.02901). The slosh
mode itself is named in the literature — *degradation*: "small, back-and-forth
oscillations of the odometry," as distinct from unidirectional drift — and
mainstream LIO (FAST-LIO2) diverges outright in the same self-similar sections,
so swapping packages is not the fix.

## The three-layer design

**Layer 1 — keep the estimator from eating its own map** (this repo):
- `odom/keyframe/degenGate`: veto keyframe creation while the gate holds
  degenerate axes. A sloshing pose that keyframes plants duplicated wall
  structure in the submap ("map-lock" aliasing), which feeds the oscillation
  back — the self-contamination loop. Field-converged practice (DLIOM's
  accurate-state keyframe selection, arXiv:2305.01843; DAMM-LOAM; DALI-SLAM).
- `odom/xicp/partialBudget{Trans,Rot}` (`xicpBudgetedAdmit`): the ternary
  gate's partial band previously admitted `keep·comp` along marginal axes with
  **no cap** — fail-open when the marginal-band signal is dishonest (the
  2026-07-08 runaway: the aliased photometric drive both lifts the weak axis
  into the band and supplies the pull). Budgeting the per-scan cumulative
  admission (mirroring the visual-rescue budget) makes X-ICP fail-BOUNDED,
  matching Tuna et al.'s controlled partial update.

**Layer 2 — heuristic per-axis output**:
- The governor's rank-1 covariance inflation (`covPosVar`/`covRotVar`) marks
  exactly the held axes untrusted in the published `/odom` covariance — this is
  a heuristic signal a downstream fusion layer can use to de-weight that axis.
  It is not a calibrated covariance of the full estimator state or its errors.
- `odom/publishTf: false`: the EKF owns `odom -> base_link`; dliio contributes
  odometry *messages* only.

**Layer 3 — the fusion EKF** (`cfg/robot_localization_ekf.yaml`,
`launch/dlio_rl.launch.py`; requires `ros-<distro>-robot-localization`):
- dliio pose fused **differentially** (deltas, covariance-weighted): slosh
  increments along the inflated axis are largely ignored; the five observable
  axes pass at full weight. A Mahalanobis gate rejects residual map-lock jumps.
- **Rig kinematic odometry** (`rig_odom_topic`) supplies forward velocity — the
  along-tunnel information LiDAR cannot observe. The 07052026 rig's
  `base_link_correction` TF chain indicates a nav stack already runs on the
  platform; its odometry is the cheapest second source. (Extreme-tunnel systems
  end up here: CM-LIUW adds UWB+wheel in coal mines.)
- IMU angular velocity only (dliio already integrates the IMU tightly; r_l has
  no IMU bias states, so re-fusing orientation/accel adds drift).
- dliio **twist is deliberately not fused** — during slosh the velocity state
  is the oscillation's flywheel.

## Honest limits (read before relying on it)

1. **GIGO / downstream-only**: the EKF smooths the *output*; it has no feedback
   into the scan matcher. Layer 1 is what keeps dliio internally sane —
   without it the submap still corrupts and dliio can still diverge internally
   even while `/odometry/filtered` looks smooth.
2. **No second source, less value**: with `rig_odom_topic` absent, the tunnel
   axis rides the EKF motion model between dliio updates — smoother than raw
   slosh, but it will drift. The architecture's value scales with the quality
   of the kinematic source.
3. **SuperLoc's critique applies**: injecting the prior *into* the alignment
   before failure is stronger than blending after it. The merged
   `directionSeparateTerm` machinery is the hook for that upgrade — feeding the
   rig odometry prior along held axes inside the solve is the follow-on if the
   EKF layer proves insufficient (see doc/BAKEOFF_SUPERLOC.md).

## Validation plan

On the 06042026 substrate (n ≥ 24): combo + Layer-1 fixes vs combo alone —
watch DIV rate, crash rate, and **slosh amplitude/period** (the new operational
metric; log along-axis position + yaw against gate state). On the 07052026 bag:
the launch's isolation matrix, with `photometricWeight: 0` per the runaway
analysis. The keyframe gate's freeze-test prediction: slosh collapses toward
slow drift when the gate is on.
