# Integrating DLIO into another robot or workspace

The ROS package is `direct_lidar_inertial_odometry`; the repository is `dliio`.
Its input boundary is synchronized `sensor_msgs/msg/PointCloud2` and
`sensor_msgs/msg/Imu`. Your driver or bag player supplies those messages. The
generic launch needs no Exyn packages, Ouster driver, vendor decoder, camera,
external estimator, or files from the workspace that developed this fork.

## Build just this repository

Start in a shell with only your ROS distribution sourced:

```bash
source /opt/ros/lyrical/setup.bash  # use the distribution installed on your robot
mkdir -p dliio_ws/src
cd dliio_ws
git clone --branch cam-dev https://github.com/jcfurey/dliio.git src/dliio
rosdep install --from-paths src --ignore-src -r -y --rosdistro "$ROS_DISTRO"
CMAKE_BUILD_PARALLEL_LEVEL=2 colcon build --base-paths src \
  --packages-select direct_lidar_inertial_odometry
source install/setup.bash
```

Install and initialize `rosdep` and `colcon` first on a new machine; the
[handoff guide](HANDOFF.md#clean-source-build) gives those commands. CMake
requires GTSAM >= 4.2, pybind11, Python development headers, PCL, Eigen, OpenCV
and OpenMP; the manifest also declares the ROS and Python runtime dependencies,
including SciPy. The pose graph extension is part of the normal build. No
Python GTSAM wheel, parent overlay, or manual `PYTHONPATH` is required.

CI targets Humble, Jazzy, Kilted and Lyrical. The isolated integration check for
this change was exercised locally on Lyrical; the other distributions need
their CI runs. Use a new build/install tree for each distribution.

The optional raw Livox adapter is detected at configure time. For
`livox_ros_driver2/CustomMsg`, build and source that message package first,
then configure DLIO with `DLIIO_ENABLE_LIVOX=ON` (the default). PointCloud2 input
works without it; `-DDLIIO_ENABLE_LIVOX=OFF` explicitly excludes the adapter.
The pinned `ouster.repos` is only for the optional Ouster packet workflow.

For a reproducible deployment, record a DLIO commit in your workspace's chosen
dependency mechanism, such as a Git submodule or a pinned `.repos` entry. The
development branch in the example is a starting point, not a version pin.

## Configure your robot

Copy the installed template into your own bringup package:

```bash
cp "$(ros2 pkg prefix --share direct_lidar_inertial_odometry)/cfg/examples/robot_template.yaml" \
  /path/to/my_robot_bringup/config/dlio_robot.yaml
```

Fill in the measured sensor transforms and frame names. The template's identity
matrices are placeholders. `cfg/dlio.yaml` remains the historical Ouster example
used when `robot_config` is omitted; it includes a sensor-specific IMU offset
and camera intrinsics. Supply your own file for another robot.

`extrinsics/baselink2lidar` and `baselink2imu` store the **sensor pose in the
base frame**: `p_base = R * p_sensor + t`, with a row-major 3×3 rotation and
translation in metres. With `extrinsics/source: yaml`, DLIO also broadcasts
these static transforms. With `tf`, it consumes your existing base-to-sensor TF
and waits at startup; the current implementation falls back to the supplied
YAML after roughly ten seconds if lookup fails. The optional camera's
`cam2lidar` similarly maps camera coordinates into LiDAR coordinates.

Inputs must use a shared acquisition clock. IMU angular velocity is in rad/s;
acceleration is in m/s² and includes gravity. `imu/normalized: true` is only for
devices reporting acceleration in g. Remain stationary during the configured
IMU calibration interval, or provide calibrated intrinsics and disable that
startup procedure. A moving sensor head must be corrected by your input adapter
into a consistent physical frame before registration.

Preserve XYZ and per-point time: Ouster `t` is relative nanoseconds, Velodyne
`time` is relative seconds, and the `timestamp` sensor paths are described in
the [sensor notes](../README.md#sensor-setup). Preserve raw intensity separately
from calibrated reflectivity. A missing timing field disables deskew; a field
named intensity does not by itself establish a calibrated reflectivity scale.
See [channel semantics](OUSTER_CHANNELS.md) for photometric configuration.

## Launch against your existing sensor topics

Odometry without a mapping backend:

```bash
ros2 launch direct_lidar_inertial_odometry dlio.launch.py \
  robot_config:=/path/to/my_robot_bringup/config/dlio_robot.yaml \
  pointcloud_topic:=/robot/lidar/points imu_topic:=/robot/imu/data \
  mapper:=none
```

To retain a persistent map, use `mapper:=persistent` and an explicit archive
directory. Each launch creates a new session database in that directory:

```bash
ros2 launch direct_lidar_inertial_odometry dlio.launch.py \
  robot_config:=/path/to/my_robot_bringup/config/dlio_robot.yaml \
  pointcloud_topic:=/robot/lidar/points imu_topic:=/robot/imu/data \
  mapper:=persistent archive_directory:=/path/to/run/maps \
  composed:=true
```

`mapper:=preview` retains the original accumulated cloud and `save_pcd` service
and remains the default for compatibility. `composed:=true` loads the C++ nodes
in one container; the Python persistent mapper runs in its own process.
`dlio_composed.launch.py` is a compatibility wrapper for the same generic launch.

Parameter precedence is: mapping defaults (persistent mapper only),
`params_file`, `robot_config`, then launch overrides. Each optional configuration
argument replaces its corresponding installed default file. Robot frames and
calibration therefore win over frame defaults in the algorithm file. Launch
overrides control simulated time and the matching atomic observation transport
for persistent mapping. The generic launch honors your extrinsics source and
optional aiding settings; it does not select a recording-specific tuning profile.

For an existing bag, set `use_sim_time:=true`, wait for both sensor subscriptions,
and start `ros2 bag play /path/to/bag --clock` in the same ROS domain. The generic
launch does not start a player or sensor driver. Restart the estimator and
mapper before seeking backwards or repeating a capture. Input QoS is
best-effort sensor data; completion of playback alone does not prove every
scan or mapping observation was processed.

Add `rviz:=true` to use the installed viewer. With custom frame names, pass
`rviz_frame:=robot1/odom` or `robot1/map` as appropriate; `rviz_config` selects
your own view. RViz is optional for every mapping mode.

## Namespaces, outputs and TF ownership

`namespace:=robot1` scopes the nodes and their relative input/output topics,
diagnostics and mapping services. An absolute input such as `/shared/imu`
remains absolute. Frame IDs are message data and are **not** automatically
prefixed: configure `robot1/odom`, `robot1/base_link`, `robot1/lidar`,
`robot1/imu` and `robot1/map` in that robot's YAML. Give simultaneous persistent
instances distinct storage directories.

The root-namespace defaults remain:

| Interface | Topic or service |
| --- | --- |
| Odometry and scan-time pose | `/dlio/odom_node/odom`, `/dlio/odom_node/scan_pose` |
| Deskewed cloud | `/dlio/odom_node/pointcloud/deskewed` |
| Atomic cloud, pose and registration quality | `/dlio/odom_node/mapping_observation` |
| Preview or persistent map | `/dlio/map_node/map` |
| Frontend and mapper diagnostics | `/diagnostics`, `/dlio/mapping/diagnostics` |
| Persistent archive operations | `/dlio/mapping/save_map`, `load_map`, `export_pcd` |
| Graph and pose revisions | `/dlio/mapping/update_pose_graph`, `apply_pose_revision`, `restore_pose_revision` |

These names gain `/robot1` when that namespace is selected. The global `/tf`
and `/tf_static` transport remains shared; distinct configured frame IDs keep
the robots' trees separate. Root-namespace callers keep their existing names.

DLIO's `odom/publishTf` controls its odom-to-base transform;
`mapping/publish_tf` controls the persistent mapper's map-to-odom transform.
Assign one publisher to each edge. An external EKF can consume the odometry
while owning the odom-to-base edge, but it does not automatically replace the
mapper's archived DLIO poses. Account for that distinction when composing frames
or applying revisions. Configured diagonal covariance and the timed observer's
conditional covariance are not calibrated global uncertainty or independent
pose-graph edge noise; see [TIMED_OBSERVER.md](TIMED_OBSERVER.md).

## Include DLIO in your package

Declare `<exec_depend>direct_lidar_inertial_odometry</exec_depend>` in a bringup
package. Add the usual `launch`, `launch_ros` and `ament_index_python`
dependencies if your launch imports them. Resolve both packages through the
ament index instead of referencing a source checkout:

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    return LaunchDescription([IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('direct_lidar_inertial_odometry'), 'launch', 'dlio.launch.py'])),
        launch_arguments={
            'namespace': 'robot1',
            'robot_config': PathJoinSubstitution([
                FindPackageShare('my_robot_bringup'), 'config', 'dlio_robot.yaml']),
            'pointcloud_topic': 'lidar/points',
            'imu_topic': 'imu/data',
            'mapper': 'none',
        }.items())])
```

For direct component loading, use `dlio::OdomNode` and optionally
`dlio::MapNode`; their remapping contract is in the generic launch. The latter
is the preview mapper. The persistent mapper executable is
`dlio_mapping_node.py` and its Python API is installed as `dliio_mapping` in
the normal Python package directory. Generated messages and services import
from `direct_lidar_inertial_odometry.msg` and `.srv`. Consumers need no
`sys.path` edits or copied source files. For C++ interface consumers, see the
small [installed consumer](../test/installed_consumer/CMakeLists.txt).
That example is installed with the package and can be built without a source
checkout:

```bash
cmake -S "$(ros2 pkg prefix --share direct_lidar_inertial_odometry)/test/installed_consumer" \
  -B /path/to/consumer_build
cmake --build /path/to/consumer_build
/path/to/consumer_build/consumer
```

## Verify the installed integration

In an otherwise unused ROS domain, from any working directory:

```bash
ROS_DOMAIN_ID=147 ros2 run direct_lidar_inertial_odometry verify_integration.py \
  --output /path/to/validation/integration.json
```

This loads one standalone frontend and one composed frontend in separate
namespaces, validates custom frame precedence and topic wiring, feeds synthetic
atomic observations to their independent mappers, saves both archives, checks
service isolation, and checks shutdown. It requires no sensor hardware or bag.
It tests installed Python/native imports too. It does not measure LIO accuracy.
CI and the Docker test image run it alongside the existing component, mapping,
unit and independent C++ consumer checks.

The [persistent mapping guide](MAPPING_NODE.md), [wire contract](MAPPING_INTERFACE.md),
[pose revisions](POSE_REVISIONS.md) and [graph API](POSE_GRAPH.md) document the
backend. Validated loop injection exists; automatic place retrieval and loop
registration remain external. The [Ouster handoff](HANDOFF.md) adds optional
packet decoding and dataset profiles on top of this generic integration.
