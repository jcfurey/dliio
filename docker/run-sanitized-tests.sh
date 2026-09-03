#!/usr/bin/env bash
# Run the sanitizer test subset exactly as .github/workflows/ci.yml does.
#
#   DLIIO_SANITIZE=address  -> ASan+UBSan on the hot-path suites (hard gate in CI)
#   DLIIO_SANITIZE=thread   -> TSan on the live-node concurrency harness
#                              (informational in CI; see doc/TESTING.md)
#
# Expects a colcon workspace at /ws built with -DDLIIO_SANITIZE=<same value>
# (the `build` stage of docker/Dockerfile exports DLIIO_SANITIZE for this).
set -euo pipefail

WS="${WS:-/ws}"
PKG_DIR="${WS}/src/direct_lidar_inertial_odometry"
BUILD_DIR="${WS}/build/direct_lidar_inertial_odometry"

# shellcheck disable=SC1090
source "/opt/ros/${ROS_DISTRO}/setup.bash"
# shellcheck disable=SC1091
source "${WS}/install/setup.bash"

case "${DLIIO_SANITIZE:-}" in
  address|undefined)
    export ASAN_OPTIONS=detect_leaks=0:halt_on_error=1:abort_on_error=1
    export UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1
    rc=0
    for t in test_nano_gicp test_imu_integration test_visual_residual test_node_concurrency; do
      echo "=== ${t} (${DLIIO_SANITIZE}) ==="
      "${BUILD_DIR}/${t}" || rc=1
    done
    exit "${rc}"
    ;;
  thread)
    # Stock libgomp is not TSan-instrumented: single-thread OpenMP so only the
    # node's own callback-group concurrency is observed.
    export OMP_NUM_THREADS=1
    export TSAN_OPTIONS="halt_on_error=0:report_thread_leaks=0:suppressions=${PKG_DIR}/test/tsan.supp"
    echo "=== test_node_concurrency (thread) ==="
    "${BUILD_DIR}/test_node_concurrency"
    ;;
  "")
    echo "DLIIO_SANITIZE is empty: build with --build-arg DLIIO_SANITIZE=address|thread first" >&2
    exit 2
    ;;
  *)
    echo "Unknown DLIIO_SANITIZE='${DLIIO_SANITIZE}' (expected address|thread|undefined)" >&2
    exit 2
    ;;
esac
