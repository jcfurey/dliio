# syntax=docker/dockerfile:1.7
# =============================================================================
# direct_lidar_inertial_odometry (dliio) -- ROS 2 build / test / runtime image
#
# Mirrors .github/workflows/ci.yml: ros:<distro>-ros-base + rosdep (deps from
# package.xml) + colcon build/test. Build from the REPOSITORY ROOT:
#
#   docker build -f docker/Dockerfile --target test    .      # build + colcon test
#   docker build -f docker/Dockerfile --target runtime .      # slim runtime image
#   docker build -f docker/Dockerfile --target sanitize \
#       --build-arg DLIIO_SANITIZE=address \
#       --build-arg CMAKE_BUILD_TYPE=RelWithDebInfo .         # ASan+UBSan subset
#
# Build args
#   ROS_DISTRO        jazzy (default; what CI runs) | kilted | lyrical | ...
#   ROS_IMAGE_FLAVOR  ros-base (default, CI parity) | perception
#                     'perception' pre-installs every dependency in package.xml
#                     (pcl_ros, pcl_conversions, cv_bridge, image_transport,
#                     libopencv-dev), so with SKIP_ROSDEP=1 the build needs no
#                     access to packages.ros.org at all (air-gapped / proxied hosts).
#   BASE_IMAGE        full override of the base image (defaults to
#                     ros:${ROS_DISTRO}-${ROS_IMAGE_FLAVOR})
#   SKIP_ROSDEP       0 (default) | 1 -- skip apt-get/rosdep in the deps stage
#   CMAKE_BUILD_TYPE  Release (default) | RelWithDebInfo | Debug
#   DLIIO_SANITIZE    "" (default) | address (ASan+UBSan) | thread (TSan) | undefined
#   BUILD_TESTING     ON (default) | OFF
#
# Stages (in file order)
#   deps     base image + package.xml dependencies (cached across source edits)
#   build    colcon build of the package into /ws/install
#   test     colcon test + test-result (fails the build on a failing test)
#   sanitize runs the CI sanitizer test subset (no-op unless DLIIO_SANITIZE is set)
#   runtime  deps + /ws/install only, entrypoint sources ROS + the workspace;
#            last stage, so it is what a plain `docker build .` produces
# =============================================================================
ARG ROS_DISTRO=jazzy
ARG ROS_IMAGE_FLAVOR=ros-base
ARG BASE_IMAGE=ros:${ROS_DISTRO}-${ROS_IMAGE_FLAVOR}

# ---- deps -------------------------------------------------------------------
FROM ${BASE_IMAGE} AS deps
ARG ROS_DISTRO
ARG SKIP_ROSDEP=0
ENV ROS_DISTRO=${ROS_DISTRO} \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /ws

# Only package.xml first, so the slow dependency layer survives source edits.
COPY package.xml src/direct_lidar_inertial_odometry/package.xml
RUN if [ "${SKIP_ROSDEP}" = "1" ]; then \
      echo "SKIP_ROSDEP=1: trusting the base image to provide every dependency in package.xml"; \
    else \
      apt-get update \
      && rosdep update --rosdistro "${ROS_DISTRO}" \
      && rosdep install --from-paths src --ignore-src -y --rosdistro "${ROS_DISTRO}" \
      && rm -rf /var/lib/apt/lists/*; \
    fi

# ---- build ------------------------------------------------------------------
FROM deps AS build
ARG CMAKE_BUILD_TYPE=Release
ARG DLIIO_SANITIZE=
ARG BUILD_TESTING=ON
# Exported so the sanitize stage / run-sanitized-tests.sh know how the tree was built.
ENV DLIIO_SANITIZE=${DLIIO_SANITIZE} \
    CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}

# The whole package (see .dockerignore for what is left out of the context).
COPY . src/direct_lidar_inertial_odometry

RUN source "/opt/ros/${ROS_DISTRO}/setup.bash" \
    && colcon build --event-handlers console_cohesion+ \
         --cmake-args \
           -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
           -DBUILD_TESTING="${BUILD_TESTING}" \
           -DDLIIO_SANITIZE="${DLIIO_SANITIZE}"

# ---- test -------------------------------------------------------------------
# `docker build --target test` is the gate: a failing gtest fails the image build.
FROM build AS test
RUN source "/opt/ros/${ROS_DISTRO}/setup.bash" \
    && colcon test --event-handlers console_cohesion+ \
    && colcon test-result --verbose

# ---- sanitize ---------------------------------------------------------------
# Same subset and options as the asan-ubsan / tsan-informational CI jobs.
# Needs --build-arg DLIIO_SANITIZE=address|thread; without it the stage is a
# no-op rather than an error, so the classic builder (DOCKER_BUILDKIT=0), which
# builds every stage preceding --target, can walk through it to `runtime`.
FROM build AS sanitize
RUN if [ -z "${DLIIO_SANITIZE}" ]; then \
      echo "sanitize: DLIIO_SANITIZE not set, nothing to run"; \
      echo "          re-run with --build-arg DLIIO_SANITIZE=address|thread"; \
    else \
      src/direct_lidar_inertial_odometry/docker/run-sanitized-tests.sh; \
    fi

# ---- runtime ----------------------------------------------------------------
# Last on purpose: the final stage is what a plain `docker build .` produces.
FROM deps AS runtime
COPY --from=build /ws/install /ws/install
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
# Override topics/params at run time, e.g.
#   docker run --rm --net=host --ipc=host dliio \
#     ros2 launch direct_lidar_inertial_odometry dlio.launch.py \
#       use_sim_time:=true pointcloud_topic:=/ouster/points imu_topic:=/ouster/imu
CMD ["ros2", "launch", "direct_lidar_inertial_odometry", "dlio.launch.py"]
