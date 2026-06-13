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

## TSan — scope and how to make it useful

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
