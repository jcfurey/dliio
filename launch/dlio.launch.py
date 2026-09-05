#
#   Copyright (c)
#
#   The Verifiable & Control-Theoretic Robotics (VECTR) Lab
#   University of California, Los Angeles
#
#   Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez
#   Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu
#

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    current_pkg = FindPackageShare('direct_lidar_inertial_odometry')

    # Arguments
    declare_rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='false',
        description='Launch RViz'
    )
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
        description='Use simulation (/clock) time. Set true under Gazebo or '
                    'rosbag play --clock.'
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

    # Parameters: robot config first, algorithm params second, launch-level
    # overrides last (later entries win).
    parameters = [
        LaunchConfiguration('robot_config'),
        LaunchConfiguration('params_file'),
        {'use_sim_time': use_sim_time},
    ]

    # DLIO Odometry Node
    dlio_odom_node = Node(
        package='direct_lidar_inertial_odometry',
        executable='dlio_odom_node',
        output='screen',
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
    )

    # DLIO Mapping Node
    dlio_map_node = Node(
        package='direct_lidar_inertial_odometry',
        executable='dlio_map_node',
        output='screen',
        parameters=parameters,
        remappings=[
            ('keyframes', 'dlio/odom_node/pointcloud/keyframe'),
            ('map', 'dlio/map_node/map'),
        ],
    )

    # RViz
    rviz_config_path = PathJoinSubstitution([current_pkg, 'launch', 'dlio.rviz'])
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='dlio_rviz',
        arguments=['-d', rviz_config_path],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
        condition=IfCondition(LaunchConfiguration('rviz'))
    )

    return LaunchDescription([
        declare_rviz_arg,
        declare_pointcloud_topic_arg,
        declare_imu_topic_arg,
        declare_camera_topic_arg,
        declare_use_sim_time_arg,
        declare_robot_config_arg,
        declare_params_file_arg,
        dlio_odom_node,
        dlio_map_node,
        rviz_node
    ])
