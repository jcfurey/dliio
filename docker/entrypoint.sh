#!/usr/bin/env bash
# Container entrypoint: source the ROS 2 underlay and (when present) the
# dliio workspace overlay, then exec the requested command.
set -e

if [ -z "${ROS_DISTRO:-}" ]; then
  echo "entrypoint: ROS_DISTRO is not set" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "/opt/ros/${ROS_DISTRO}/setup.bash"

if [ -f /ws/install/setup.bash ]; then
  # shellcheck disable=SC1091
  source /ws/install/setup.bash
fi

exec "$@"
