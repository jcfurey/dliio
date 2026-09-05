import launch
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource


# Isolation variant of dlio_07052026.launch.py: same raw-packet decode +
# extrinsics/frame-rename overlay, but WITHOUT ouster_tunnel.yaml (gate/
# regularization/photometric-weight/maxCorr clamps) or any of the tunnel_*
# overlays that build on top of it (visual/lidar_image/xicp — their headers
# all say "ON TOP of ouster_tunnel.yaml", so loading them without the tunnel
# base isn't a meaningful config; dropped here rather than kept as dead
# args).
#
# Exists to answer one question live: on 07052026_4_an, dlio_07052026.launch.py
# (tunnel-mode base) catastrophically diverged (2026-07-09, RViz smoke test,
# final pose ~[33M,-5.9M,-30M] m) and tunnel+lidar_image crashed
# (std::out_of_range, exit -6). Neither config is 06042026-validated for THIS
# bag — ouster_tunnel.yaml's gate/regularization/photometricWeight/maxCorr
# numbers were tuned on 06042026 and never checked against this dataset's own
# geometry/reflectivity. This file isolates whether that overlay is actively
# HARMFUL here vs. plain dlio.yaml defaults (still with the correct
# 07052026_extrinsics — that overlay is load-bearing, not a tunnel-mode
# artifact: without it the 180deg os_lidar<->os_imu yaw goes unapplied and
# base_link collides with the bag's own rig TF tree).
#
# Same conventions as the parent file: no bag_file arg (play the bag
# yourself), --clock required for the scoped-sim-time decode nodes, dlio_*
# nodes stay wall-clock (use_sim_time hardcoded false).
def generate_launch_description():
    dliio_pkg = FindPackageShare('direct_lidar_inertial_odometry')

    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='false', description='Launch RViz (dlio.rviz).')
    timestamp_mode_arg = DeclareLaunchArgument(
        'timestamp_mode', default_value='TIME_FROM_ROS_TIME',
        description='ouster_ros os_cloud timestamp_mode — see '
                    'dlio_07052026.launch.py header for the full rationale.')
    pointcloud_topic_arg = DeclareLaunchArgument(
        'pointcloud_topic', default_value='/ouster/points')
    imu_topic_arg = DeclareLaunchArgument(
        'imu_topic', default_value='/ouster/imu')

    ouster_replay_launch = PathJoinSubstitution(
        [FindPackageShare('ouster_ros'), 'launch', 'replay.launch.xml'])
    ouster_decode = GroupAction(
        scoped=True,
        actions=[
            IncludeLaunchDescription(
                XMLLaunchDescriptionSource(ouster_replay_launch),
                launch_arguments={
                    'viz': 'false',
                    'pub_static_tf': 'false',
                    'bag_file': '',
                    'timestamp_mode': LaunchConfiguration('timestamp_mode'),
                }.items()),
        ])

    cfg_examples = PathJoinSubstitution([dliio_pkg, 'cfg', 'examples'])
    parameters = [
        PathJoinSubstitution([dliio_pkg, 'cfg', 'dlio.yaml']),
        # Extrinsics + dliio_base_link frame rename only — no tunnel-mode
        # overlay. See header for why this one file still applies.
        PathJoinSubstitution([cfg_examples, 'ouster_07052026_extrinsics.yaml']),
        {
            'odom/debug/dashboard': False,
            'use_sim_time': False,
        },
    ]

    dlio_odom_node = Node(
        package='direct_lidar_inertial_odometry',
        executable='dlio_odom_node',
        output='screen',
        parameters=parameters,
        remappings=[
            ('pointcloud', LaunchConfiguration('pointcloud_topic')),
            ('imu', LaunchConfiguration('imu_topic')),
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
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='dlio_rviz',
        arguments=['-d', PathJoinSubstitution([dliio_pkg, 'launch', 'dlio.rviz'])],
        output='screen',
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    return launch.LaunchDescription([
        rviz_arg,
        timestamp_mode_arg,
        pointcloud_topic_arg,
        imu_topic_arg,
        ouster_decode,
        dlio_odom_node,
        dlio_map_node,
        rviz_node,
    ])
