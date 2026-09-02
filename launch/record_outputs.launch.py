#
#   Copyright (c)
#
#   The Verifiable & Control-Theoretic Robotics (VECTR) Lab
#   University of California, Los Angeles
#
#   Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez
#   Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu
#

# Record ONLY the lightweight estimator outputs during a scored/timed run, so
# visualization (RViz/Foxglove) can be done afterward from this small bag with
# no contention on the live estimator. Deliberately excludes the per-scan dense
# deskewed cloud and the full map -- those are the heavy topics that starve the
# node under co-scheduling (see doc/TUNNEL_FINDINGS.md).
#
# Usage (run alongside the headless estimator + bag playback):
#   ros2 launch direct_lidar_inertial_odometry record_outputs.launch.py \
#     out:=/path/to/run1_outputs
# Then visualize later:  ros2 bag play /path/to/run1_outputs --clock  (+ RViz/Foxglove)

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    declare_out = DeclareLaunchArgument(
        'out', default_value='dliio_outputs',
        description='Output bag directory'
    )

    topics = [
        '/dlio/odom_node/odom',
        '/dlio/odom_node/pose',
        '/dlio/odom_node/path',
        '/dlio/odom_node/keyframes',
        '/diagnostics',
        '/tf',
        '/tf_static',
    ]

    record = ExecuteProcess(
        cmd=['ros2', 'bag', 'record', '-o', LaunchConfiguration('out')] + topics,
        output='screen',
    )

    return LaunchDescription([declare_out, record])
