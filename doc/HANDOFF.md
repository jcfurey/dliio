# Standalone handoff

This repository supplies the estimator, current map preview, launch files,
profiles, RViz configuration, and replay/verification tools. The Ouster workflow
starts only dliio components and, for packet input, the Ouster cloud component.
An optional ROS bag player supplies recorded data and `/clock`; RViz is optional.
No RESPLE, SuperMap, Livox driver node, robot_localization, camera, or external
static-transform node is needed. Standard ROS libraries and message packages
are build/runtime dependencies, installed with rosdep.

## Clean source build

Use a shell with only the selected ROS distribution sourced. The supported
platforms are Ubuntu 22.04 / Humble, Ubuntu 24.04 / Jazzy and Kilted, and Ubuntu
26.04 / Lyrical. Do not mix ROS distributions or reuse a build/install tree
between them. Docker equivalents are in [docker/README.md](../docker/README.md).

```bash
mkdir -p dliio_ws/src
cd dliio_ws
git clone --branch cam-dev https://github.com/jcfurey/dliio.git src/dliio
source /opt/ros/jazzy/setup.bash  # substitute humble, kilted, or lyrical
sudo apt-get update
sudo apt-get install -y python3-rosdep python3-colcon-common-extensions python3-vcstool
# Run sudo rosdep init once if this machine has never used rosdep.
rosdep update --rosdistro "$ROS_DISTRO"
rosdep install --from-paths src/dliio --ignore-src -r -y --rosdistro "$ROS_DISTRO"
export CMAKE_BUILD_PARALLEL_LEVEL=2 MAKEFLAGS=-j2
colcon build --packages-select direct_lidar_inertial_odometry \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DDLIIO_ENABLE_LIVOX=ON
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

A source archive can be extracted as `src/dliio` instead of cloning. Nothing
needs the parent test workspace or its submodules. Runtime assets are installed
under `share/direct_lidar_inertial_odometry`; source paths are not used by launch.

`DLIIO_ENABLE_LIVOX` defaults to **ON**. When the `livox_ros_driver2` message
package is in the sourced underlay, dliio compiles its raw `CustomMsg` adapter.
When absent, CMake reports that the adapter is unavailable; Ouster and Livox
PointCloud2 inputs still work. Build/source your Livox driver first and rebuild
dliio to enable raw Livox input. No Livox node needs to run for Ouster operation.
When adding that dependency to an existing build, pass `--cmake-force-configure`
to colcon so CMake reruns dependency detection. The existing Livox
input/remapping and `imu/normalized` settings are retained.

## Ouster driver

If a driver already publishes organized `/ouster/points` and `/ouster/imu`,
dliio can consume those directly. Preserve point time, signal, reflectivity,
ring, range, and ambient/near-IR fields. Reduced XYZ-only clouds discard the
texture and deskew cues used by the tunnel profile.

For the packet workflow, install the revision pinned in `ouster.repos`:

```bash
# From dliio_ws, after sourcing the selected ROS distribution:
vcs import src < src/dliio/ouster.repos
git -C src/ouster-ros submodule update --init --recursive
rosdep install --from-paths src/ouster-ros --ignore-src -y --rosdistro "$ROS_DISTRO"
colcon build --packages-up-to ouster_ros \
  --executor sequential --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

The pin is `jcfurey/ouster-ros` at
`338fa84a9d988eaae762c79bdd7bacbae497280a` (including its pinned SDK). It provides
the reliable packet subscriptions used by the supplied replay QoS configuration.
No external bridge node or Python Ouster SDK is needed.

Humble's intra-process manager rejects the driver's latched metadata
subscription, so the launch uses DDS delivery for that component on Humble.
It remains in the same process as dliio. dliio's keyframes still use intra-process
delivery; its latched preview-map publisher uses DDS on all four distributions.

## Live points and recorded data

For a live Ouster driver, copy `cfg/dlio.yaml`, fill in your sensor extrinsics,
and keep LiDAR and IMU timestamps synchronized. dliio publishes its own static
base-to-sensor transforms from that YAML:

```bash
ros2 launch direct_lidar_inertial_odometry dlio_ouster.launch.py \
  mode:=points robot_config:=/absolute/path/my_robot.yaml rviz:=true
```

The default topics are `/ouster/points` and `/ouster/imu`. Override them with
`pointcloud_topic:=... imu_topic:=...`. For a bag containing those decoded
topics, add `bag:=/absolute/path/bag`; the launch starts a player with simulated
time automatically. `params_file:=...` replaces algorithm defaults and
`robot_config:=...` is applied after the selected profile.

For the 0705 packet bag, use the installed profile and RViz configuration:

```bash
ros2 launch direct_lidar_inertial_odometry dlio_ouster.launch.py \
  mode:=packets profile:=0705 bag:=/absolute/path/07052026_4_an \
  rviz:=true rate:=1.0 run_dir:=/absolute/path/dliio_run
```

The launch reads `/ouster/metadata` directly from the bag, validates and writes
`ouster-metadata.json` into `run_dir`, loads Ouster and dliio into one process,
then starts playback at **1×**. It applies the recorded sensor extrinsics,
reflectivity registration, and raw-signal image-flow profile. This calibration
is specific to the 0705 sensor mounting; use your own YAML for other hardware.

`rviz:=false` runs headless. `map:=false` omits the accumulated-map preview.
The map is not a dedicated mapping backend and has no loop closure. See the
[mapping contract](MAPPING_INTERFACE.md) before consuming registered clouds.

For an external player, supply `mode:=packets metadata:=/path/sensor.json`,
omit `bag`, and play packets with `--clock` and the installed
`cfg/ouster_packet_replay_qos.yaml`. Packet mode defaults to simulated time.
Stop/restart dliio and its map before seeking backward or repeating a bag.

The self-contained launch selects YAML extrinsics and disables optional camera
residuals. Other legacy launch files retain their existing integration options.

## Verification

```bash
colcon test --packages-select direct_lidar_inertial_odometry
colcon test-result --verbose
# In an otherwise unused ROS domain, test the installed package without hardware:
ROS_DOMAIN_ID=147 ros2 run direct_lidar_inertial_odometry verify_install.py
# During a 0705 replay, in the same domain as the replay:
ros2 run direct_lidar_inertial_odometry inspect_replay.py \
  --seconds 75 --output dliio_run/probe.json
# Or start, check, and stop a complete headless 1x packet replay automatically:
ROS_DOMAIN_ID=147 ros2 run direct_lidar_inertial_odometry verify_packet_replay.py \
  /absolute/path/07052026_4_an --seconds 75
```

The installed smoke check loads the components through the real launch file,
checks latched map delivery and preserved signal/reflectivity, and calls the
generated SavePCD service. The replay probe records scan/keyframe pose pairing,
fields, TF, and processing diagnostics. CLI support nodes and these temporary
test probes may appear in the graph; they are not estimator dependencies.

Offline cache decoding, C++ replay, and repeated comparison tools are in
`scripts/` in this repository and installed under its share directory. Use
their `--help` options. They are diagnostic tools specific to the documented
0705 packet format, not a general replacement for the Ouster driver. Repeated
accuracy comparisons need at least five runs; a compatibility smoke test alone
does not establish ground-truth trajectory accuracy.

CI compiles and tests clean Humble, Jazzy, Kilted, and Lyrical images. It also
checks the installed launch and service, including Humble's latched-publisher
composition behavior, and compiles an independent C++ consumer of the installed
service. Local validation evidence is recorded in
[HANDOFF_VALIDATION.md](HANDOFF_VALIDATION.md).

The native matrix additionally builds the raw Livox adapter against the
interface-only fixture in `test/livox_interfaces`, checks converted fields and
timestamps, and verifies rejection of inconsistent point counts. This covers
the adapter without requiring a Livox node; it is not hardware/SDK validation.
