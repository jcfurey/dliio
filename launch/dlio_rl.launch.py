# dliio + robot_localization EKF fusion launch (doc/FUSION_ARCHITECTURE.md).
#
# Runs dlio with the tunnel + X-ICP + fusion overlays layered (governor cov
# inflation ON, odom/publishTf OFF, keyframe degen gate ON) and an
# ekf_filter_node (robot_localization) that fuses:
#   - dliio pose (differential, covariance-weighted -- the held tunnel axis
#     arrives inflated and is de-weighted),
#   - a rig kinematic odometry twist (rig_odom_topic -- wheel/track/nav-stack;
#     the along-tunnel information LiDAR cannot observe),
#   - IMU angular velocity.
# The EKF owns odom->base_link and publishes the fused state on /odometry/filtered.
#
# Requires ros-<distro>-robot-localization (runtime only; not a build dep of
# this package). If the rig has no second odometry source, the EKF still
# smooths dliio via the covariance-weighted differential fusion, but the
# along-tunnel axis then rests on the (drifting) motion model -- see the doc.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    current_pkg = FindPackageShare('direct_lidar_inertial_odometry')

    args = [
        DeclareLaunchArgument('pointcloud_topic', default_value='pointcloud',
                              description='LiDAR PointCloud2 topic.'),
        DeclareLaunchArgument('imu_topic', default_value='imu',
                              description='IMU topic.'),
        DeclareLaunchArgument('rig_odom_topic', default_value='rig/odom',
                              description='Rig kinematic odometry (wheel/track) topic for the EKF.'),
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='Bag replay -> true.'),
        DeclareLaunchArgument('robot_config',
                              default_value=PathJoinSubstitution([current_pkg, 'cfg', 'dlio.yaml']),
                              description='Per-robot config (extrinsics etc.).'),
        DeclareLaunchArgument('params_file',
                              default_value=PathJoinSubstitution([current_pkg, 'cfg', 'params.yaml']),
                              description='Algorithm parameters.'),
        DeclareLaunchArgument('ekf_params_file',
                              default_value=PathJoinSubstitution([current_pkg, 'cfg', 'robot_localization_ekf.yaml']),
                              description='robot_localization EKF parameters.'),
    ]

    use_sim_time = ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)

    # dliio parameters: robot config, algorithm params, then the tunnel/X-ICP/
    # fusion overlays in order (later entries win).
    dlio_parameters = [
        LaunchConfiguration('robot_config'),
        LaunchConfiguration('params_file'),
        PathJoinSubstitution([current_pkg, 'cfg', 'examples', 'ouster_tunnel.yaml']),
        PathJoinSubstitution([current_pkg, 'cfg', 'examples', 'ouster_tunnel_xicp.yaml']),
        PathJoinSubstitution([current_pkg, 'cfg', 'examples', 'ouster_tunnel_rl.yaml']),
        {'use_sim_time': use_sim_time},
    ]

    dlio_odom_node = Node(
        package='direct_lidar_inertial_odometry',
        executable='dlio_odom_node',
        output='screen',
        parameters=dlio_parameters,
        remappings=[
            ('pointcloud', LaunchConfiguration('pointcloud_topic')),
            ('imu', LaunchConfiguration('imu_topic')),
            ('odom', 'dlio/odom_node/odom'),
            ('pose', 'dlio/odom_node/pose'),
            ('path', 'dlio/odom_node/path'),
            ('kf_pose', 'dlio/odom_node/keyframes'),
            ('kf_cloud', 'dlio/odom_node/pointcloud/keyframe'),
            ('deskewed', 'dlio/odom_node/pointcloud/deskewed'),
        ],
    )

    dlio_map_node = Node(
        package='direct_lidar_inertial_odometry',
        executable='dlio_map_node',
        output='screen',
        parameters=dlio_parameters,
        remappings=[
            ('keyframes', 'dlio/odom_node/pointcloud/keyframe'),
            ('map', 'dlio/map_node/map'),
        ],
    )

    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[LaunchConfiguration('ekf_params_file'),
                    {'use_sim_time': use_sim_time}],
        remappings=[
            ('dlio/odom', 'dlio/odom_node/odom'),
            ('rig/odom', LaunchConfiguration('rig_odom_topic')),
            ('imu/data', LaunchConfiguration('imu_topic')),
            ('odometry/filtered', 'odometry/filtered'),
        ],
    )

    return LaunchDescription(args + [dlio_odom_node, dlio_map_node, ekf_node])
