# Testing & sanitizers

## Unit & integration tests (49, run in CI)

`colcon test` -> 7 gtest suites:
- `test_nano_gicp` — GICP alignment, block-wise degeneracy gate, small-cloud
  covariance guard, PLANE vs MIN_EIG regularization (synthetic plane/corner).
- `test_imu_integration` — continuous-time deskew kernel vs closed-form
  constant-accel / constant-omega / gravity-cancel trajectories.
- `test_intensity` — the radiometric `correctIntensity` kernel: range-only at
  normal incidence, incidence (1/cos) brightening, cosMin grazing-angle floor,
  255 saturation, bad-input guards.
- `test_visual_residual` — camera frame-to-frame, camera frame-to-map, and
  COIN-LIO LiDAR-image residuals: residual->0 at truth, analytic-vs-finite-
  difference Jacobian (sign guards), Gauss-Newton descent, occlusion-gate
  rejection, elevation-LUT projection.
- `test_sensor_detection` — the field-name -> `SensorType` mapping, including the
  Hesai/Livox split by `timestamp` magnitude and the no-points / boundary cases.
- `test_node_concurrency` — the live-node TSan harness (see below); under a normal
  build it is a functional smoke test that the node processes scans and shuts
  down cleanly (worker-thread joins).
- `test_pipeline` — end-to-end: drives a real `OdomNode` through a
  MultiThreadedExecutor with synthetic-but-consistent LiDAR (a static, offset box
  -> full 3D observability) + IMU and asserts the *estimate*: stationary
  no-drift, constant-velocity tracking, `imu/normalized` g-unit handling,
  `extrinsics/source: tf` gating, live-param range rejection, and a
  generously-bounded hot-path compute-time smoke (read off `/diagnostics`).
  Tolerances are divergence/regression guards on synthetic data, not accuracy
  specs (see the PR notes / test comments).

## ASan + UBSan (reproducible)

```
colcon build --build-base build_asan --install-base install_asan \
  --cmake-args -DDLIIO_SANITIZE=address -DCMAKE_BUILD_TYPE=RelWithDebInfo
ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 UBSAN_OPTIONS=print_stacktrace=1 \
  ./build_asan/direct_lidar_inertial_odometry/test_nano_gicp
#   (repeat for the other suites, e.g. test_imu_integration, test_intensity,
#    test_visual_residual, test_sensor_detection, test_pipeline,
#    test_node_concurrency)
```
`detect_leaks=0` suppresses one-shot leaks in PCL/OpenCV/rclcpp statics; the
heap-overflow / use-after-free / UB checks are what matter for the hot-path math.
**Status: clean on all 49 tests** (the hard CI gate).

## TSan — clean (via the live-node harness)

`test_node_concurrency` constructs a real `OdomNode` with every optional path
enabled and pumps synthetic IMU + organized Ouster-like scans + camera frames
through a MultiThreadedExecutor, so the IMU / scan / image callbacks, the 100 Hz
pose timer, the background submap thread, and the dashboard thread all run
concurrently. This is the configuration that actually reaches the cross-thread
state, so TSan can see it.

```
colcon build --build-base build_tsan --install-base install_tsan \
  --cmake-args -DDLIIO_SANITIZE=thread -DCMAKE_BUILD_TYPE=RelWithDebInfo
OMP_NUM_THREADS=1 \
TSAN_OPTIONS="suppressions=$(pwd)/src/direct_lidar_inertial_odometry/test/tsan.supp" \
  ./build_tsan/direct_lidar_inertial_odometry/test_node_concurrency
```
`OMP_NUM_THREADS=1` silences the (uninstrumented) OpenMP runtime; `test/tsan.supp`
covers the most common callback-group-serialized accesses and one third-party
(OpenNI2) library issue.

**What this is and isn't.** The harness is a *triage tool*, not a hard gate. It
found the 5 genuine cross-thread races below, which were fixed. As a permanent
"zero reports" gate it is unreliable: rclcpp's executor is not TSan-instrumented,
so a MutuallyExclusive callback group's serialized-but-thread-migrated accesses
look like races to TSan, and they cannot be suppressed by function name without
also masking *real* races in those same functions (TSan suppresses a race if
EITHER access stack matches a suppression). The set of false positives that
surfaces also varies run to run (the harness is concurrent). It therefore runs
in CI as an **informational, continue-on-error job**; ASan/UBSan is the hard gate.

**Reading the output:** a GENUINE race has its two access stacks in DIFFERENT
callback-group domains -- e.g. the pose timer vs the IMU callback, the dashboard
thread vs the IMU callback, or the background submap thread vs the main scan
callback. Two stacks within the SAME MutuallyExclusive group (both in
callbackImu/transformImu, both in callbackPointCloud, both in publishPose) are
the executor-migration false positives.

### Races this harness found and fixed (not suppressed)
- `publishPose` / `debug()` read `state`+`imu_stamp` cross-group -> snapshot under `geo.mtx`.
- `imu_rates` written by the IMU callback, read by `publishDiagnostics` every scan -> `mtx_imu`.
- IMU-buffer iteration on the scan thread vs `push_front` -> range copied under `mtx_imu`.
- `geo.first_opt_done` cross-group bool -> `std::atomic<bool>`.
- background submap concat reads `keyframe_normals` / `keyframe_visual_refs` vs main `push_back` -> copy shared_ptrs under `keyframes_mutex`.

What `test/tsan.supp` covers (NOT bugs): accesses serialized by rclcpp
MutuallyExclusive callback groups (the executor's happens-before is invisible to
TSan because rclcpp/rmw are uninstrumented), verified single-group by inspection.

## TSan caveats (general)

The unit tests are **single-threaded** drivers (they call the registration/
integration functions directly), so TSan on them exercises only the OpenMP
reduction loops, NOT the node's callback-group concurrency where the real races
live (IMU vs scan vs background-submap vs dashboard threads). Two consequences:

1. **OpenMP false positives.** Stock libgomp is not TSan-instrumented; build with
   `-DDLIIO_SANITIZE=thread` AND run with `OMP_NUM_THREADS=1` to silence the
   runtime and see only application races.
2. **To actually observe the fixed races** (imu_rates writer/reader,
   keyframe-vector concat vs push_back, dashboard-thread stats) TSan must drive a
   live `OdomNode` with concurrent synthetic IMU + organized-PointCloud2 + Image
   publishers under a MultiThreadedExecutor. That harness now exists
   (`test_node_concurrency`) and is the only configuration that reaches those
   code paths.

## Open testing roadmap (synthetic data)

- ~~**Concurrent node harness (TSan target).**~~ **Done** (`test_node_concurrency`):
  a real `OdomNode` is driven with synthetic organized Ouster-like scans + IMU +
  camera frames through a MultiThreadedExecutor; it found and validated the fixes
  for the races listed above and exercises live param-set mid-run.
- ~~**End-to-end accuracy / synthetic-scan generator.**~~ **Partly done**
  (`test_pipeline`): a static offset-box world gives full 3D observability and the
  estimate is checked (stationary no-drift, constant-velocity tracking, plus the
  `imu/normalized` / tf-extrinsics / live-param / perf-smoke cases). Still open to
  reuse the generator for `buildLidarIntensityImage` self-cal accuracy (recover the
  known az/el model + non-uniform LUT) and an occlusion-gate end-to-end (place an
  occluder, assert far points are culled).
- **Property tests:** random valid covariances -> Mahalanobis PSD; random poses ->
  residual Jacobian finite-difference (extend the existing per-term checks).
- **Bag-replay regression** — the real accuracy guard (synthetic tolerances above
  are divergence guards only); needs a representative tunnel bag (see REVIEW §V.1).
