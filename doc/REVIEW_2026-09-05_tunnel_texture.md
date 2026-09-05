# 0705 tunnel texture and timing review

Base: `fda3109` on `cam-dev`. Changes are isolated on `fix/ouster-field-sweep`.
Dataset: `07052026_4_an`, OS-1-64-U13, 1024 × 64 at 10 Hz,
`RNG19_RFL8_SIG16_NIR16`, recorded 2026-07-05.

## Findings and fixes

The recorded signal and reflectivity images clearly contain graffiti. A captured
full-resolution scan had 43,495 valid returns; the existing 1 m crop retained
29,522. The inferred spherical projection had absolute column errors below
0.355 pixels and row errors below 0.212 pixels at the 99th percentile in that
sample. These figures describe a sample, not a calibration guarantee.

The following logic defects prevented reliable use of that texture:

1. The degeneracy gate omitted the LiDAR flow term when deciding whether
   appearance could rescue a weak axis. It also classified a Hessian containing
   the 3D photometric term as geometric. The gate now receives a separate
   geometric Hessian and evaluates appearance information before damping.
2. Step acceptance ignored all image costs, including flow. It could reject a
   useful texture correction, return an untested final step, and consume rescue
   budget on a rejected step. Acceptance now includes the image residual costs,
   checks the final proposal, and restores budgets after rejection. The fixed
   Huber cost matches the corresponding robust gradient. Experimental adaptive
   kernels and directional preconditioners remain optional; their state-dependent
   weighting is not a guarantee of descent on a fixed statistical objective.
3. The 3D intensity fit required full spatial rank, rejecting paint on planar
   walls. A minimum-norm fit now estimates the supported tangential gradient and
   rejects line-like support. Its residual evaluates the same local interpolated
   intensity model as its Jacobian, allowing sub-voxel steps to be evaluated.
4. Zero-mean flow patches omitted the derivative of the moving patch mean. A
   uniform ramp therefore claimed motion information after its informative
   component had been removed. Both residual and gradient now subtract the mean.
5. Flow did not check previous-frame range consistency, and missing returns could
   become apparent texture edges. Previous/current range images now validate
   visibility and the entire interpolation/gradient footprint. The map term also
   requires valid support. Ceiling division enforces the 4,000-point iteration cap.
6. Cropping early returns could shift Ouster/Velodyne relative timestamps against
   the IMU. Relative times now keep the message-header origin; the sampled 0705
   cloud's first retained return was 5.46 ms after that origin.
7. The observer compared a LiDAR midpoint pose with an IMU state already at the
   latest sample. It now reconstructs the observer prediction at the measurement
   time from its preceding corrected state and buffered IMU data. Orientation
   corrections are transported to the current state. State/timestamp publication
   and bias snapshots share the observer mutex. Diagnostics distinguish successful
   time alignment and geometric weakness from axes held after texture fusion.

## Measurement roles

| Input | Use in the candidate |
|---|---|
| XYZ and calibrated beam geometry from ouster-ros | Geometric registration and image projection |
| Per-return `t` | Deskewing, with the correct header time origin |
| Organized beam rows | Elevation lookup for image projection |
| Range derived from XYZ | Reject occluded surfaces and missing-return texture |
| Raw signal (`intensity`/`signal`) | Full-resolution inter-frame texture |
| Calibrated reflectivity | Spatial photometric map gradients; separate stored measurement |
| Ambient/near-IR | Supported image alternative; excluded from this profile after the initial ablation |

Signal and reflectivity derive from the same return and are not independent
sensors. The candidate combines their spatial/inter-frame roles with explicit
weights, without interpreting the pair as independent statistical confidence.
Ouster's [sensor data documentation](https://static.ouster.dev/sensor-docs/image_route1/image_route3/sensor_data/sensor-data.html)
describes these fields and the calibrated beam geometry. The image-based use of
LiDAR appearance follows the broad approach demonstrated by
[COIN-LIO](https://github.com/ethz-asl/COIN-LIO); this implementation is not COIN-LIO.

## Validation and reproducibility

The workspace helpers `scripts/dliio_ouster_cache.py`,
`scripts/dliio_compile_replay.py`, `scripts/dliio_nrun_cache.py`, and
`scripts/dliio_analyze_cache_sweep.py` retain the inputs, executable/library hashes,
parameters, per-scan poses, costs, and run outcomes under
`results/dliio_ouster_sweep_2026-09-05/`.

The cache contains 5,511 complete scans and 69,062 IMU samples. Fifteen partial
frames are recorded as excluded. A decoded frame matched a ROS-driver capture
exactly in signal, reflectivity, ambient, range, ring, and point timestamps;
maximum XYZ disagreement was 3.82 micrometers. Both estimator versions receive
the same complete scans and IMU order. The cache uses the common sensor clock,
waits for submap work at each scan, and isolates the estimator from DDS loss and
executor scheduling. It does not test the driver's ROS receive-time timestamp
mode or prove behavior under transport loss.

The regression suite includes known-motion paint fixed to the world in an
infinite square tunnel, forward/reverse/stationary cases with a biased prior,
textureless controls, planar gradients, cost/Jacobian agreement, invalid image
support, timestamp preservation, and observer latency with real innovations.
The existing registration, IMU, pipeline, map/export, and concurrency tests also
remain part of the validation. Memcheck covers the changed image sampling and
registration paths.

Five alternating repetitions per version completed all 5,511 scans. Each used
four pinned CPUs, OpenMP 4, and the same cached inputs. The baseline is `fda3109`
with the previous RViz parameters; the candidate combines the fixes with
`cfg/examples/ouster_07052026_texture.yaml`. This compares the complete solution,
not an isolated estimate of each code change's contribution.

| Metric | Baseline median (min–max) | Candidate median (min–max) |
|---|---:|---:|
| Complete runs | 5/5 | 5/5 |
| Short-period axial travel, m | 25.50 (25.50–25.50) | 2.52 (2.37–2.81) |
| Total axial travel, m | 167.84 (167.84–167.84) | 113.06 (112.56–113.98) |
| Axial extent, m | 32.09 (32.09–32.09) | 33.12 (33.01–33.32) |
| Per-run callback p50, ms | 6.24 (6.16–6.32) | 7.64 (7.45–7.76) |
| Per-run callback p95, ms | 10.89 (10.57–11.05) | 15.88 (15.38–16.68) |
| Peak resident memory, MiB | 265.54 (265.19–272.57) | 266.33 (260.22–278.22) |
| Final-segment axial travel, m | 2.43 (2.43–2.43) | 0.16 (0.13–0.29) |

Short-period travel is total variation of world X minus total variation after
Gaussian smoothing with sigma 0.5 s. It fell 90.1%, but includes any real brief
motion too. No pose alignment or scale fitting was used. The final segment is
recording time 530 s onward; its candidate trajectory is smoother but still
drifts by 0.11–0.29 m across repetitions. Runtime excludes initialization before
5 s and measures the point-cloud callback, excluding separately timed submap
completion and ROS transport/RViz. These are estimator timings, not end-to-end
latency guarantees.

All 248 GoogleTests passed across 17 CTest suites. Memcheck reported zero errors
for the changed image sampling and registration cases (leak checking was not
enabled). The workspace report is
`results/REVIEW_2026-09-05_dliio_tunnel_texture.md`; per-run metrics, configuration
hashes, logs, and the all-repetition plot are in
`results/dliio_ouster_sweep_2026-09-05/`.

## Limits

The bag has no independent pose ground truth. Short-period axial travel and
repeatability quantify behavior, not absolute accuracy. A visually smoother
trajectory alone cannot establish correct distance. Preliminary image matching
between revisits was inconclusive after excluding sensor/body returns and is
not used as a ground-truth surrogate.

The old low-weight 3D-only configuration diverged on the changed estimator in
control runs, including after the timing fix. Keeping the old parameters is
therefore a known regression on this recording; use the tested full-resolution
flow profile with this change. This review does not establish that every
pre-existing photometric configuration remains stable. Near-IR was less useful
in the initial segment and is not blindly added to the objective. Raw images still use a
spherical model inferred from points and a single scan pose; exact beam-offset
projection and per-pixel motion compensation remain possible improvements.

Observer time alignment requires the preceding corrected state to be older than
the new scan midpoint and buffered IMU measurements to cover that interval.
Initialization or severe backlog can fall back to the original comparison;
`Observer Time Aligned` and `Observer IMU Lead (ms)` expose this limitation.
