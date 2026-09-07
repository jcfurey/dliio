# Measurement-time observer and conditional covariance

This is an opt-in experiment. `odom/observer/measurementTimeUpdates` defaults
to `false`. The existing configured covariance behavior remains the default.
The workspace [experiment report](../../../docs/dliio-timed-observer-experiments.md)
records the actual long-bag runs, parameters, artifacts, and limitations.

## What changes

The geometric observer retains its configured gains. A LiDAR update is
applied at its scan acquisition midpoint, starting from the preceding scan
checkpoint. Buffered **raw, body-frame IMU samples** are then replayed to the
latest state timestamp with the corrected biases. Deskew also subtracts the
current scan checkpoint's biases from raw samples instead of reusing cached
values corrected with an older bias estimate.

The high-rate mean uses the existing normalized first-order quaternion step
and constant-acceleration position/velocity step. The scan update uses the
existing position, velocity, attitude, accelerometer-bias, and gyro-bias gains
at the matched time. It is a discrete geometric observer, not an EKF: the
covariance never chooses a gain. Changing the declared noise leaves the
mean unchanged. Registration, map observations, and the next deskew pose
still use the registered pose, as in the existing architecture; published
odometry is the observer state.

The legacy observer also has a separate timing fix. Its stored previous
corrected state is at a publication timestamp, which can be at or after the
next scan midpoint. Equal timestamps now use that stored state directly;
earlier queries integrate backward through the available IMU window.
Under time reversal, velocity and angular velocity change sign, while
acceleration and gravity do not. This reuses the forward deskew kernel.
The change removes a fallback that compared a scan pose with a later state.
Startup without an available corrected checkpoint retains its old fallback
and reports `Observer Time Aligned = 0`.

## Coordinates and propagation

The full 15-dimensional local error is ordered as:

```text
[position_world, attitude_body, velocity_world, accel_bias_body, gyro_bias_body]
```

Each entry is a three-vector. Orientation uses a right perturbation
`q_true = q_est * Exp(theta_body)`, with the quaternion mapping body to world.
The implementation differentiates its actual discrete mean operations using
central manifold differences with a step of `1e-5`, instead of substituting
a Kalman update or a Hessian inverse. Quaternion perturbation and derivative
conventions follow the local/global distinction in
[Solà, §§4.4 and 5](https://arxiv.org/html/1711.02508v1).

Let `F` be the state transition Jacobian and `D` the sensitivity to the
measured IMU input. A sample has noise covariance
`Q = diag(accel_density² / sample_period, gyro_density² / sample_period)`.
If a scan divides an IMU interval, both parts use the **same noisy sample**.
The snapshot therefore retains `C = Cov(state_error, sample_noise)`, along
with the input stamp and `Q`. For that sample:

```text
P_next = F P Fᵀ + D Q Dᵀ - F C Dᵀ - D Cᵀ Fᵀ + Q_bias_walk
C_next = F C - D Q
```

`C` resets to zero for a new raw input. Bias random-walk variance is added
to the corresponding blocks in proportion to the integration duration.
The sample variance uses its original interval, even when propagation stops
partway through it. Treating each fraction as independent would underestimate
the uncertainty: a unit-noise one-second sample split at 0.4 s must still
produce velocity variance 1, rather than `0.4² + 0.6² = 0.52`.

For the geometric pose correction, `A` differentiates the correction with
respect to state and `Z` with respect to an assumed external pose error:

```text
P_corrected = A P Aᵀ + Z R_pose Zᵀ
C_corrected = A C
```

The update expresses errors in the tangent at the resulting orientation,
including quaternion normalization and bias/position/velocity coupling.
Retained raw inputs are replayed from this corrected checkpoint. Replaying
a delayed measurement yields the same final state and covariance as applying
it promptly, provided the measurement sequence and raw samples are identical.

## What this covariance means

`R_pose` is a **declared independent external-pose noise assumption**. DLIO's
actual GICP pose reuses its own map, deskew, and inertial prior. Those errors
are correlated with the observer and with preceding registrations. These
correlations are **not modeled here**. Incorrect correspondences, curved
tunnel ambiguity, systematic sensor/mounting errors, and accumulated map
drift are also outside this local Gaussian model.

With auxiliary gravity enabled, `R_pose` describes the *adjusted* pose as
an assumed external measurement. It is not computed from BNO covariance,
the gravity blend Jacobian, or the GICP residuals. The BNO attitude's temporal
correlation and shared information with its gyro are not estimated.

Consequently, this is a propagated **conditional covariance of the observer**,
not calibrated global trajectory/map uncertainty and not a justification
for treating DLIO and its primary IMU as independent downstream EKF inputs.
The small residual of an incorrect, self-consistent map can still coincide
with a small conditional covariance. Geometry diagnostics and sensor
disagreement remain separate integrity evidence. The ICP covariance
[rematching analysis](https://arxiv.org/abs/1410.7632) gives another reason
not to interpret an arbitrary GICP normal-matrix inverse as global uncertainty.

The [hierarchical observer paper](https://arxiv.org/html/2303.02777v1)
motivates the orientation-to-translation/bias coupling. Its analysis assumes
upstream pose measurements and does not establish this noise model or certify
an incorrect tunnel registration. The sampled implementation and its
conditional covariance are local engineering choices tested here.

### Declared noise and initial uncertainty

All values below are experiment assumptions, not calibration from this bag:

| Parameter | Default | Units |
|---|---:|---|
| `odom/observer/accelNoiseDensity` | 0.03 | m/s²/√Hz |
| `odom/observer/gyroNoiseDensity` | 0.002 | rad/s/√Hz |
| `odom/observer/accelBiasWalk` | 0.0005 | m/s²/√s |
| `odom/observer/gyroBiasWalk` | 0.00005 | rad/s/√s |
| `odom/observer/poseNoise` | `[.09,.09,.09,.01,.01,.01]` | world position m², world rotation rad² |

Initialization uses this pose covariance, converts the orientation block
to the body tangent, and assumes independent per-axis variances of 0.25 for
world velocity, 0.04 for accelerometer bias, and 0.0001 for gyro bias.
The first registered pose initializes the checkpoint. Samples buffered after
the currently propagated state timestamp are left for their owning IMU
callback, avoiding double propagation during concurrent initialization.
Before initialization, the IMU callback advances its timestamp only after
the raw sample is buffered. After initialization it propagates the model
even while the first scan callback is still finishing. Thus the startup
flag cannot advance the output timestamp without advancing the model.

Settings and geometric gains require a restart in this mode. Attenuated
observer gains, velocity damping, slosh response, and extra held-axis
covariance inflation are currently rejected because this covariance model
does not include those operations. Raw replay is bounded to 8,192 samples;
pose corrections must advance the checkpoint by no more than one second.
The current implementation does not recover automatically from a longer
checkpoint gap: subsequent pose updates remain rejected, and a restart is
required. This restriction is separate from a single scan's computation time.

## ROS output and diagnostics

Pose covariance uses world position and a fixed/world-axis orientation
tangent, consistent with the declared ROS convention. The attitude projection
is `theta_world = R(q) theta_body`, including its position cross terms.
Twist is expressed in `child_frame_id`: body velocity is `Rᵀ v`, with
derivatives `[Rᵀv]×` with respect to body attitude and `Rᵀ` with respect to
world velocity. Angular velocity is measured gyro minus estimated gyro bias;
its covariance includes direct measurement noise and its correlation with
the propagated state. Both 6×6 ROS arrays retain every cross term, in
row-major order. See [REP-103](https://github.com/ros-infrastructure/rep/blob/master/rep-0103.rst).

State, covariance, and timestamp are copied under the same observer mutex.
Odometry, pose, and their TF are withheld until initialization, and after
the model becomes invalid. Invalid raw input, replay overflow, nonfinite or
non-positive covariance, and bias clipping invalidate the model until a
restart. Clipping must not manufacture apparent confidence from a zero
derivative. Rejected pose updates are counted; prediction continues without
those updates. Scan/map publications remain separate from observer output.

New diagnostics include:

- `Odometry Covariance Model`: `conditional geometric observer; independent external pose noise assumed`.
- `Observer Covariance Propagated` and `Observer Covariance Valid`: model
  initialization/numerical validity, **not statistical calibration**.
- `Observer Rejected Raw IMU` and `Observer Rejected Pose Update`.
- `Observer Joint Covariance Row Major [p_world,theta_body,v_world,ba_body,bg_body]`:
  all 225 entries, paired with `Observer State Stamp (s)`.
- Existing geometry, convergence, biases, raw innovations, and IMU delivery
  metrics remain available independently.

## Optional BNO gravity support

`odom/auxGravity/enabled` is independently opt-in. The input is a
`geometry_msgs/Vector3Stamped` unit **world-up direction expressed in the
configured body frame**, on `/dliio/auxiliary_gravity` by default. DLIO checks
the exact frame name, finite unit direction, and increasing timestamps.
Only received samples bracketing the scan timestamp can be interpolated;
there is no extrapolation or waiting for future samples.

The minimal left rotation moves the measured world-up direction toward +Z.
Correction fraction is `1-exp(-gain*scan_dt)`, with a separate rate cap.
Defaults are gain 1/s, maximum correction rate 5°/s, maximum disagreement
30°, and maximum bracketing gap 50 ms. A gain of zero monitors without
changing a pose. Invalid time intervals and large disagreements are rejected.
The correction changes registration orientation before observer/map updates
and before it seeds subsequent deskew. It does not directly change position.

This deterministic tilt blend is not a Kalman update. No BNO absolute heading
is fused. Rotating the world about gravity commutes with the correction;
this does not promise an unchanged Euler yaw at arbitrary tilt. Diagnostics
retain the raw GICP quaternion, gravity disagreement, applied angle, bracket
stamps, and accepted/rejected counts.

## Verification

Tests cover constant acceleration, backward rotation/translation integration,
the equal/future observer checkpoint regression, delayed versus timely
fractional scan updates, split-sample IMU noise correlation, full ROS pose
and twist projections, invalid inputs, covariance invalidation at clipping,
and concurrent initialization's buffered future sample. A seeded 800-particle
Monte Carlo experiment checks the full joint covariance and average NEES
under the declared independent-noise model. Gravity tests cover brackets,
unit/finite input rejection, shortest tilt correction, rate limiting, and
invariance to a changed navigation heading. Real-bag comparisons have no
independent trajectory ground truth.
