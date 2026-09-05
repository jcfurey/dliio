#
#   Copyright (c)
#
#   The Verifiable & Control-Theoretic Robotics (VECTR) Lab
#   University of California, Los Angeles
#
#   Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez
#   Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu
#

# Composed variant: both nodes in one multithreaded component container with
# intra-process communication, so keyframe clouds pass between the odometry
# and map nodes without RMW serialization. Functionally equivalent to
# dlio.launch.py otherwise.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    current_pkg = FindPackageShare('direct_lidar_inertial_odometry')

    declare_pointcloud_topic_arg = DeclareLaunchArgument(
        'pointcloud_topic', default_value='points_raw',
        description='Pointcloud topic name'
    )
    declare_imu_topic_arg = DeclareLaunchArgument(
        'imu_topic', default_value='imu_raw',
        description='IMU topic name'
    )
    declare_camera_topic_arg = DeclareLaunchArgument(
        'camera_topic', default_value='image_raw',
        description='Camera image topic (only used when odom/visual/enabled)'
    )
    declare_use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use simulation (/clock) time.'
    )
    declare_robot_config_arg = DeclareLaunchArgument(
        'robot_config',
        default_value=PathJoinSubstitution([current_pkg, 'cfg', 'dlio.yaml']),
        description='Per-robot configuration (extrinsics, IMU intrinsics).'
    )
    declare_params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([current_pkg, 'cfg', 'params.yaml']),
        description='Algorithm parameters.'
    )

    use_sim_time = ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)

    parameters = [
        LaunchConfiguration('robot_config'),
        LaunchConfiguration('params_file'),
        {
            'use_sim_time': use_sim_time,
            # the ANSI dashboard garbles container-multiplexed logs
            'odom/debug/dashboard': False,
        },
    ]

    container = ComposableNodeContainer(
        name='dlio_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        output='screen',
        composable_node_descriptions=[
            ComposableNode(
                package='direct_lidar_inertial_odometry',
                plugin='dlio::OdomNode',
                name='dlio_odom_node',
                parameters=parameters,
                remappings=[
                    ('pointcloud', LaunchConfiguration('pointcloud_topic')),
                    ('imu', LaunchConfiguration('imu_topic')),
                    ('camera', LaunchConfiguration('camera_topic')),
                    ('odom', 'dlio/odom_node/odom'),
                    ('pose', 'dlio/odom_node/pose'),
                    ('scan_pose', 'dlio/odom_node/scan_pose'),
                    ('kf_pose_stamped', 'dlio/odom_node/keyframe_pose'),
                    ('path', 'dlio/odom_node/path'),
                    ('kf_pose', 'dlio/odom_node/keyframes'),
                    ('kf_cloud', 'dlio/odom_node/pointcloud/keyframe'),
                    ('deskewed', 'dlio/odom_node/pointcloud/deskewed'),
                ],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
            ComposableNode(
                package='direct_lidar_inertial_odometry',
                plugin='dlio::MapNode',
                name='dlio_map_node',
                parameters=parameters,
                remappings=[
                    ('keyframes', 'dlio/odom_node/pointcloud/keyframe'),
                    ('map', 'dlio/map_node/map'),
                ],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
        ],
    )

    return LaunchDescription([
        declare_pointcloud_topic_arg,
        declare_imu_topic_arg,
        declare_camera_topic_arg,
        declare_use_sim_time_arg,
        declare_robot_config_arg,
        declare_params_file_arg,
        container,
    ])
