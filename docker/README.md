# Building and running dliio in Docker

Run these commands from this repository's root. No parent test workspace is
needed. The same Dockerfile supports `humble`, `jazzy`, `kilted`, and `lyrical`.
Standard dependencies, ROS bag tools, Cyclone DDS, and RViz are installed through
`package.xml`. `DLIIO_ENABLE_LIVOX` remains ON; its raw adapter is compiled when
Livox message definitions are available (see [handoff](../doc/HANDOFF.md)).

```bash
# Build and run the complete tests plus installed-component smoke check.
docker build -f docker/Dockerfile --build-arg ROS_DISTRO=jazzy --target test .
# Build a runtime image; substitute any supported ROS distribution.
docker build -f docker/Dockerfile --build-arg ROS_DISTRO=jazzy -t dliio:jazzy .
```

The default launch consumes an existing Ouster driver's points/IMU. For a bag of
decoded points, use the built-in player:

```bash
mkdir -p dliio_run
docker run --rm --net=host --ipc=host \
  -v "$PWD/bags:/bags:ro" -v "$PWD/dliio_run:/out" dliio:jazzy \
  ros2 launch direct_lidar_inertial_odometry dlio_ouster.launch.py \
    mode:=points bag:=/bags/my_run rate:=1.0 run_dir:=/out
```

For **packet bags**, build the optional image containing the pinned Ouster
component. This fetches only the Ouster driver and its SDK, then builds them:

```bash
docker build -f docker/Dockerfile.ouster \
  --build-arg BASE_IMAGE=dliio:jazzy -t dliio:jazzy-ouster .
docker run --rm --net=host --ipc=host \
  -v "$PWD/bags:/bags:ro" -v "$PWD/dliio_run:/out" dliio:jazzy-ouster \
  ros2 launch direct_lidar_inertial_odometry dlio_ouster.launch.py \
    mode:=packets profile:=0705 bag:=/bags/07052026_4_an rate:=1.0 run_dir:=/out
```

The dataset is supplied separately. This runs headless by default. For RViz,
use the host's RViz with `launch/dlio_0705_texture.rviz` and the same ROS domain,
or pass your desktop's display access into the container and add `rviz:=true`.
Do not assume a headless Docker build has a display server.

`docker/docker-compose.yml` retains convenient `dlio`, `bag`, `test`, `asan`,
and `tsan` services. Its default middleware is Cyclone DDS. The compose bag
service expects decoded point/IMU topics; use the packet command above for
raw Ouster packets.

| Build argument | Default | Meaning |
|---|---|---|
| `ROS_DISTRO` | `jazzy` | `humble`, `jazzy`, `kilted`, or `lyrical` |
| `BASE_IMAGE` | `ros:$ROS_DISTRO-ros-base` | Override the base; must match the selected distribution |
| `ROS_IMAGE_FLAVOR` | `ros-base` | Alternate official ROS image flavor |
| `BUILD_JOBS` | `2` | Limit CMake/Make compiler concurrency |
| `CMAKE_BUILD_TYPE` | `Release` | Build mode |
| `BUILD_TESTING` | `ON` | Compile the test suites |
| `DLIIO_SANITIZE` | empty | `address`, `thread`, or `undefined` |
| `SKIP_ROSDEP` | `0` | Skip dependency installation only if the base already contains every declared dependency |

`deps`, `build`, `test`, `sanitize`, and `runtime` are separate targets. The
runtime image is the default target and sources ROS and the installed package.
An existing `perception` image alone does **not** guarantee that every runtime
or test dependency is installed; prepare a complete dependency image before
using `SKIP_ROSDEP=1` offline. The optional Ouster image also needs network
access to fetch its pinned source, unless its build layers are cached.

```bash
docker build -f docker/Dockerfile --target sanitize \
  --build-arg DLIIO_SANITIZE=address \
  --build-arg CMAKE_BUILD_TYPE=RelWithDebInfo .
```

CI runs the native package checks and Docker test target on all four ROS
versions. The native matrix additionally compiles and exercises the raw Livox
adapter using a message-only test fixture; no Livox driver node is started.
