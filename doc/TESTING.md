# Testing & sanitizers

## Unit tests (25, run in CI)

`colcon test` -> 3 gtest suites:
- `test_nano_gicp` — GICP alignment, block-wise degeneracy gate, small-cloud
  covariance guard, PLANE vs MIN_EIG regularization (synthetic plane/corner).
- `test_imu_integration` — continuous-time deskew kernel vs closed-form
  constant-accel / constant-omega / gravity-cancel trajectories.
- `test_visual_residual` — camera frame-to-frame, camera frame-to-map, and
  COIN-LIO LiDAR-image residuals: residual->0 at truth, analytic-vs-finite-
  difference Jacobian (sign guards), Gauss-Newton descent, occlusion-gate
  rejection, elevation-LUT projection.

## ASan + UBSan (reproducible)

```
colcon build --build-base build_asan --install-base install_asan \
  --cmake-args -DDLIIO_SANITIZE=address -DCMAKE_BUILD_TYPE=RelWithDebInfo
ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 UBSAN_OPTIONS=print_stacktrace=1 \
  ./build_asan/direct_lidar_inertial_odometry/test_nano_gicp
#   (repeat for test_imu_integration, test_visual_residual)
```
`detect_leaks=0` suppresses one-shot leaks in PCL/OpenCV/rclcpp statics; the
heap-overflow / use-after-free / UB checks are what matter for the hot-path math.
**Status: clean on all 25 tests** (2026-06-13).

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
suppresses callback-group-serialized accesses TSan can't reason about and one
third-party (OpenNI2) library issue. **Status: clean (0 warnings)** after the
2026-06-13 race fixes; the harness building this set of fixes is what found them.

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
   publishers under a MultiThreadedExecutor. That integration harness is the
   open next step (see below) — it is the only configuration that reaches those
   code paths.

## Open testing roadmap (synthetic data)

- **Concurrent node harness (TSan target).** Construct `OdomNode`, pump synthetic
  organized Ouster-like scans (range+reflectivity grid with known beam model) +
  IMU at rate + camera frames, spin a MultiThreadedExecutor a few seconds under
  `-DDLIIO_SANITIZE=thread`, OMP_NUM_THREADS=1. Exercises every cross-thread
  member this sweep touched.
- **Synthetic organized-scan generator** reused for: `buildLidarIntensityImage`
  self-cal accuracy (recover known az/el model + non-uniform LUT), occlusion-gate
  end-to-end (place an occluder, assert far points are culled), deskew of a known
  swept trajectory.
- **Property tests:** random valid covariances -> Mahalanobis PSD; random poses ->
  residual Jacobian finite-difference (extend the existing per-term checks).
