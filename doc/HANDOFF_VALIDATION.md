# Handoff validation — 2026-09-05

The `cam-dev` handoff was built and exercised in isolated ROS installations.
Only this repository was mounted for the dliio builds. Ouster was built
separately from the revision in `ouster.repos`; no parent-workspace estimator,
bridge, TF, or camera nodes were available to the replay containers.

| ROS | Ubuntu | GCC | CMake | Test cases | Paired scans in 75 s | Avg / max callback ms |
|---|---|---|---|---|---|---|
| Humble | 22.04.5 | 11.4.0 | 3.22.1 | 271 / pass | 750 | 9.21 / 47.85 |
| Jazzy | 24.04.4 | 13.3.0 | 3.28.3 | 271 / pass | 750 | 9.26 / 39.52 |
| Kilted | 24.04.4 | 13.3.0 | 3.28.3 | 271 / pass | 750 | 9.26 / 59.12 |
| Lyrical | 26.04 | 15.2.0 | 4.2.3 | 271 / pass | 750 | 8.99 / 31.27 |

Each version passed:

- 259 C++ cases across 18 suites and 12 Python launch/real-bag metadata cases.
  Colcon's full aggregate displays 290 because it also counts 19 CTest runners.
- Installed component loading, latched map delivery, signal/reflectivity
  preservation, and SavePCD generated type support and input validation.
- Compilation and execution of the enabled raw Livox adapter, using the
  repository's message-only fixture; correct converted fields/timestamps and
  rejection of a malformed point-count/array-length pair. No Livox node ran.
- An independent C++ project finding the installed package, compiling its
  generated service headers, linking the shared type-support library, and
  running successfully.
- A headless `07052026_4_an` packet replay at **rate 1.0**, sampled for 75 seconds
  after startup. All 750 scan clouds paired exactly with their registered poses;
  all 10 sampled keyframes paired exactly with their poses. All versions reported
  **zero estimated dropped scans and zero computation overruns**.

The replay graphs contained OusterCloud, dliio odometry and preview map, the
component container, bag player, and temporary launch/probe support nodes.
Registered scans, keyframes, and maps retained `intensity`, `reflectivity`,
`intensity_corrected`, and `lidar_intensity`, in `odom`, without invalid processed
point-time aliases. Reflectivity registration and the raw-intensity image-flow
term were active.

Humble used an existing Ubuntu 22.04 ROS Humble dependency base, with declared
rosdep dependencies installed; Jazzy/Kilted/Lyrical used their official ROS
ros-base images with rosdep. Builds were Release, compiler parallelism 2, with
Cyclone DDS, four OpenMP threads and one OpenBLAS thread. Replay domains were
isolated and each process was limited to four CPU cores on the test machine.
CI uses the official ROS base image for every distribution.

## Compatibility fixes verified

Humble uses the older cv_bridge header and service QoS API. Its intra-process
manager rejects transient-local endpoints: dliio's latched map publisher now
uses DDS, and the launch selects DDS for OusterCloud on Humble while keeping it
in the same process. dliio keyframes retain intra-process delivery. Jazzy's bag
writer requires a topic ID in test fixtures. The packet-player flags are common
to all four versions. The Livox option defaults to ON; optional dependency
exports follow whether the adapter was actually built.

The host Lyrical installation also passed the complete suite, installed Livox
check, and C++ consumer check. The repository's 0705 RViz launch is the same
canonical workflow used by these packet checks.

## Limits of this evidence

These are compatibility and short full-speed runtime checks on x86-64. They are
not four complete-bag ground-truth accuracy studies or Livox hardware/SDK tests.
Raw Ouster clouds were delivered to dliio within the component process; the
external diagnostic probe did not receive the large raw cloud topic, although
it received the processed scans and maps. The driver occasionally produced
repeated IMU timestamps in receive-time mode; dliio rejected those correctly.

The tunnel can still report a weak geometric direction and hold its prior where
texture is insufficient. Slow drift and dedicated mapping/loop closure remain
separate work; this pass does not claim to eliminate them. Callback measurements
are sampled diagnostics, not a controlled performance comparison between ROS
versions. Full-speed delivery on other hardware needs its own replay check.
