"""Open a saved mapping archive without an odometry node or sensor driver."""
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def setup(context):
    package = Path(get_package_share_directory('direct_lidar_inertial_odometry'))
    path = Path(LaunchConfiguration('load_map').perform(context)).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f'Archive does not exist: {path}')
    config = LaunchConfiguration('mapping_config').perform(context)
    if not Path(config).is_file():
        raise ValueError(f'Mapping configuration does not exist: {config}')
    actions = [Node(package='direct_lidar_inertial_odometry', executable='dlio_mapping_node.py',
        name='dlio_mapping_node', parameters=[config, {'mapping/load_path': str(path), 'use_sim_time': False,
            'frames/map': LaunchConfiguration('map_frame').perform(context),
            'frames/odom': LaunchConfiguration('odom_frame').perform(context)}],
        remappings=[('map', '/dlio/map_node/map')], output='screen')]
    rviz = LaunchConfiguration('rviz').perform(context).lower()
    if rviz not in ('true', 'false'):
        raise ValueError('rviz must be true or false')
    if rviz == 'true':
        actions.append(Node(package='rviz2', executable='rviz2', name='dlio_map_rviz',
            arguments=['-d', str(package / 'launch/dlio_mapping.rviz')], output='screen'))
    return actions


def generate_launch_description():
    package = Path(get_package_share_directory('direct_lidar_inertial_odometry'))
    return LaunchDescription([
        DeclareLaunchArgument('load_map', description='Absolute path to a sealed .dliomap snapshot'),
        DeclareLaunchArgument('mapping_config', default_value=str(package / 'cfg/mapping.yaml')),
        DeclareLaunchArgument('map_frame', default_value='map'),
        DeclareLaunchArgument('odom_frame', default_value='odom'),
        DeclareLaunchArgument('rviz', default_value='true'),
        OpaqueFunction(function=setup)])
