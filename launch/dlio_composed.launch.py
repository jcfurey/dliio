#
#   Copyright (c)
#
#   The Verifiable & Control-Theoretic Robotics (VECTR) Lab
#   University of California, Los Angeles
#
#   Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez
#   Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu
#

"""Compatibility entry point for the generic launch with C++ composition enabled."""
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    package = Path(get_package_share_directory('direct_lidar_inertial_odometry'))
    return LaunchDescription([IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(package / 'launch/dlio.launch.py')),
        launch_arguments={'composed': 'true'}.items())])
