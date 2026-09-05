# Development-branch review: Ouster channels

Base: `dev` at `a8fb6ed`. Scope: sensor field intake, channel preprocessing,
deskew/voxel preservation, and photometric/image residual inputs.

| Priority | Finding | Fix |
|---|---|---|
| P1 | Raw intensity correction saturated at 255, including configurations declaring a 16-bit photometric scale. | Preserve raw signal and store unclipped radiometric correction separately. |
| P1 | Image channel selection reused the reflectivity slot; enabling 3D reflectivity overrode image selection, while image denoising altered the 3D signal. | Give image samples their own field and use it in map and flow references. |
| P1 | Native `signal`/`near_ir` names were not resolved; typed scalar iterators walked through row padding. | Decode checked scalar fields with aliases, datatype conversion, row strides, and byte order. |
| P1 | Image calibration unwrapped clockwise azimuth into roughly `[0, -2π]`, but map/flow/reference projection used `atan2` in `[-π, π]`, rejecting half the scan. | Resolve the angle onto the calibrated image interval in all three consumers; test full-scan participation across the seam. |
| P2 | Geometry-only voxelization discarded reflectivity; flow-only skipped custom channel transfer, selection, and denoising. | Preserve raw and derived fields in every mode and share preprocessing between image consumers. |
| P2 | Live enabling or scale changes left gradients absent/stale, including shared submap targets. | Invalidate and rebuild gradients using the receiving registration's channel, scale, and neighborhood settings. |
| P2 | OpenMP gradient workers wrote bit-packed `vector<bool>` entries; live settings could race the background gradient build. | Use one byte per validity flag and serialize background target preparation against live setting changes. |

All six initial wire-format regressions failed against the development base
before the fixes. The pre-existing 16 CTest suites passed on that base.

The first `/ouster/points` message in the local `06042026_` bag is an organized
64×1024 original-format cloud, with 48-byte points. Raw channel min/median/max:

| Channel | Min | Median | Max |
|---|---:|---:|---:|
| intensity | 1 | 1006 | 4441 |
| reflectivity | 0 | 10 | 104 |
| ambient | 338 | 1396 | 23346 |

These values confirm that an 8-bit signal clamp destroys useful dynamic range
on the available data. They do not establish an improvement in odometry accuracy.
Usage and remaining sensor/model assumptions are documented in
[OUSTER_CHANNELS.md](OUSTER_CHANNELS.md).

A recorded-scan check passed both channel modes through actual node intake,
deskew, voxel averaging, and stationary self-registration. It verified 43,539
valid raw points and ambient-image samples against the message, producing
19,577 voxels. Reflectivity contributed 18,523 photometric residuals; corrected
intensity contributed 18,777. Both self-registration residual RMS values were
zero. This checks channel transport and engagement, not motion accuracy.

Final validation: Release build on ROS 2 Lyrical with Cyclone DDS and
`OMP_NUM_THREADS=4`; **238 tests in 17 CTest suites passed**, with no failures,
errors, or disabled tests. This includes the live-node concurrency and trajectory
pipeline suites, azimuth-seam regression, and PCL grid-overflow fallback.
Modified YAML parses, the sanitizer runner passes `bash -n`, and
`git diff --check` is clean. The CI sanitizer subset now includes intensity and
wire-format channel tests; sanitizers were not run locally for this review.
