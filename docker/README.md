# Building and running dliio in Docker

Everything here reproduces `.github/workflows/ci.yml` in a container so the
ROS 2 side (`odom.cc`, `map.cc`, the launch files, the full `colcon test`
suite) can be compiled and run on a machine that has no ROS 2 install. Run all
commands from the **repository root**.

| File | Purpose |
|---|---|
| `Dockerfile` | multi-stage: `deps` -> `build` -> `test` / `sanitize` / `runtime` |
| `docker-compose.yml` | named services for build, test, sanitizers, and bag replay |
| `entrypoint.sh` | sources `/opt/ros/$ROS_DISTRO` + `/ws/install` and `exec`s the command |
| `run-sanitized-tests.sh` | the CI ASan/UBSan and TSan test subsets |
| `../.dockerignore` | keeps `.git`, host build trees and bags out of the context |

## Quick start

```sh
# build the package and run every gtest (fails the build on a failing test)
docker build -f docker/Dockerfile --target test .

# slim runtime image (deps + install/ only)
docker build -f docker/Dockerfile -t dliio .

# ASan+UBSan on the hot-path suites, exactly like the CI hard gate
docker build -f docker/Dockerfile --target sanitize \
  --build-arg DLIIO_SANITIZE=address --build-arg CMAKE_BUILD_TYPE=RelWithDebInfo .
```

Or with compose (same stages, memorable names):

```sh
docker compose -f docker/docker-compose.yml build test   # build + colcon test
docker compose -f docker/docker-compose.yml build asan   # ASan/UBSan subset
docker compose -f docker/docker-compose.yml build tsan   # TSan harness (informational)
docker compose -f docker/docker-compose.yml build dlio   # runtime image
```

## Running odometry from a `ros2 bag`

DLIO needs the cloud **and** the IMU topic, and `use_sim_time:=true` with
`ros2 bag play --clock` (see the top-level README). The `dlio` service uses
host networking + host IPC so it joins the host's ROS 2 graph directly:

```sh
mkdir -p bags out && cp -r /path/to/my_run bags/      # a rosbag2 directory
POINTCLOUD_TOPIC=/ouster/points IMU_TOPIC=/ouster/imu BAG=my_run \
  docker compose -f docker/docker-compose.yml --profile bag up dlio bag
```

* `cfg/` is bind-mounted read-only at `/cfg`, so `ROBOT_CONFIG=/cfg/examples/...`
  and edits to `cfg/*.yaml` take effect on the next `up`, no rebuild.
* `out/` is mounted at `/out` for the map export:
  `ros2 service call /save_pcd direct_lidar_inertial_odometry/srv/SavePCD "{leaf_size: 0.2, save_path: /out}"`
* `BAG_RATE=0.5` slows playback when the best-effort cloud subscription drops scans.
* Plain `docker run` works too:
  `docker run --rm --net=host --ipc=host -v $PWD/cfg:/cfg:ro dliio ros2 launch direct_lidar_inertial_odometry dlio.launch.py use_sim_time:=true pointcloud_topic:=/ouster/points imu_topic:=/ouster/imu`

RViz is not in the `ros-base` image. Either run `rviz2` on the host with the
same `ROS_DOMAIN_ID` (the recommended way -- `launch/dlio.rviz` is the config),
or build with `--build-arg ROS_IMAGE_FLAVOR=desktop` and pass `DISPLAY` /
`/tmp/.X11-unix` into the container.

## Build arguments

| Arg | Default | Notes |
|---|---|---|
| `ROS_DISTRO` | `jazzy` | image tag and `setup.bash`; `kilted`, `lyrical`, ... work if the base image exists |
| `ROS_IMAGE_FLAVOR` | `ros-base` | `perception` pre-installs every `package.xml` dependency |
| `BASE_IMAGE` | `ros:$ROS_DISTRO-$ROS_IMAGE_FLAVOR` | full override |
| `SKIP_ROSDEP` | `0` | `1` skips `apt-get`/`rosdep` in the `deps` stage |
| `CMAKE_BUILD_TYPE` | `Release` | `RelWithDebInfo` for sanitizers |
| `DLIIO_SANITIZE` | *(empty)* | `address`, `thread`, `undefined` (see `doc/TESTING.md`); empty makes the `sanitize` stage a no-op |
| `BUILD_TESTING` | `ON` | `OFF` to skip compiling the gtests |

`runtime` is the last stage, so it is what a plain `docker build .` produces.
Use BuildKit (the default since Docker 23): it builds only the stages the
chosen `--target` depends on. The deprecated classic builder
(`DOCKER_BUILDKIT=0`) instead builds every stage *preceding* the target, so
asking it for `runtime` also runs the test stage.

### Air-gapped / proxied hosts

`rosdep install` needs `packages.ros.org`. When that is unreachable, use the
`perception` image, which already ships `pcl_ros`, `pcl_conversions`,
`cv_bridge`, `image_transport` and `libopencv-dev`, and skip rosdep:

```sh
docker build -f docker/Dockerfile --target test \
  --build-arg ROS_IMAGE_FLAVOR=perception --build-arg SKIP_ROSDEP=1 .
```

Nothing in the `build`/`test` stages touches the network, so this builds and
tests fully offline once the base image is present.

## CI

`.github/workflows/docker.yml` builds the `test` target (the whole suite inside
the image) whenever `docker/**`, `.dockerignore`, `CMakeLists.txt` or
`package.xml` change, and on manual dispatch. The native jobs in `ci.yml`
remain the primary gate.
