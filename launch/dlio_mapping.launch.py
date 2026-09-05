"""The standalone Ouster workflow with the persistent mapper enabled."""
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    package = Path(get_package_share_directory('direct_lidar_inertial_odometry'))
    return LaunchDescription([IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(package / 'launch/dlio_ouster.launch.py')),
        launch_arguments={'mapper': 'persistent'}.items())])
