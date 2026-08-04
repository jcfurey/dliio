import launch
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource


# Launch for the 07052026_4_an dataset (raw-packet Ouster OS-1-64 + IMU +
# Lucid camera). Unlike the stock dlio.launch.py (which assumes an
# already-decoded bag, e.g. one topic remap away), this bag holds RAW
# record-mode packets (/ouster/{lidar,imu}_packets + /ouster/metadata) — this
# launch brings up ouster_ros's os_cloud/os_image composable nodes
# (src/ouster-ros in the top-level repo, resple_07052026.launch.py's sibling
# for RESPLE) to decode them into /ouster/points + /ouster/imu first.
#
# Doesn't IncludeLaunchDescription dlio.launch.py: that file hardcodes a
# single params_file, but this dataset needs several YAML overlays layered in
# order (extrinsics always; visual/lidar_image optionally) — see the
# `parameters` list below. So this defines its own dlio_odom_node/
# dlio_map_node/rviz2 Nodes, matching dlio.launch.py's structure exactly
# (upstream launch files aren't modified in place, per convention — a new
# sibling file instead, same as RESPLE's dataset launches).
#
# TUNNEL MODE (found live 2026-07-08): this bag's environment shows the same
# along-axis map-lock oscillation/corkscrew as the 06042026 tunnel — so
# ouster_tunnel.yaml's gate/regularization/photometric-weight base is loaded
# unconditionally here (not gated behind an arg; there's no evidence yet this
# dataset needs anything else). use_xicp:=true layers
# ouster_tunnel_xicp.yaml's n=24-validated ternary gate refinement on top —
# the only lever in the whole 06042026 investigation confirmed to actually
# PREVENT the collapse (vs. bounding it after the fact, or the lidar_image/
# camera terms which don't reach this axis at all). Default false pending a
# from-scratch validation on THIS dataset (n=24 was 06042026-specific).
#
# TIMESTAMP-DOMAIN GOTCHA (found live 2026-07-08): this sensor's metadata says
# timestamp_mode=TIME_FROM_INTERNAL_OSC (its own free-running oscillator,
# unrelated to wall time) — fine for LiDAR+IMU-only math (self-consistent
# within one clock domain; RESPLE's resple_07052026.launch.py works
# unmodified with it), but it silently starves odom/visual's camera<->scan
# time matching: LiDAR points came in stamped ~4600s (sensor uptime) while
# the Lucid camera stamps in real epoch time (~1.78e9s) — a ~1.78-billion-
# second gap against the 0.05s (odom/visual/maxTimeDiff) matching window, so
# "Visual Match dt" was permanently -1 (no candidate ever found) and Visual
# Points stayed 0 regardless of weight/denseSource tuning. Forcing
# TIME_FROM_ROS_TIME on the decode (default below) fixes that — LiDAR/IMU
# then stamp at ROS receive-time, same domain as the camera. Trade-off:
# doing so occasionally logs "IMU data does not cover the scan period" (LiDAR
# and IMU packets no longer share one hardware clock, so their relative
# offset is only as tight as decode-time scheduling) — a real but so-far
# minor cost; set timestamp_mode:=TIME_FROM_INTERNAL_OSC to go back to
# tight internal sync if not using use_visual.
def generate_launch_description():
    dliio_pkg = FindPackageShare('direct_lidar_inertial_odometry')

    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='false', description='Launch RViz (dlio.rviz).')
    use_visual_arg = DeclareLaunchArgument(
        'use_visual', default_value='false',
        description='Enable the direct camera photometric term + frame-to-map '
                    'anchor (odom/visual/*, cfg/examples/ouster_tunnel_visual.yaml '
                    '+ ouster_07052026_visual.yaml). Needs the Lucid camera, '
                    'played from the bag on camera_topic.')
    use_lidar_image_arg = DeclareLaunchArgument(
        'use_lidar_image', default_value='false',
        description='Enable the COIN-LIO-style LiDAR reflectivity/ambient '
                    'frame-to-map anchor (odom/lidar_image/*, cfg/examples/'
                    'ouster_tunnel_lidarimg.yaml). Derived from /ouster/points '
                    'per-point fields directly — no os_image / camera needed.')
    use_xicp_arg = DeclareLaunchArgument(
        'use_xicp', default_value='false',
        description='Enable the X-ICP ternary localizability gate '
                    '(odom/xicp/*, cfg/examples/ouster_tunnel_xicp.yaml) — '
                    'n=24-validated on 06042026 to prevent (not just bound) '
                    'the tunnel along-axis collapse.')
    timestamp_mode_arg = DeclareLaunchArgument(
        'timestamp_mode', default_value='TIME_FROM_ROS_TIME',
        description='ouster_ros os_cloud timestamp_mode. TIME_FROM_ROS_TIME '
                    '(default) puts LiDAR/IMU stamps in the same clock domain '
                    'as the camera (required for use_visual:=true). '
                    'TIME_FROM_INTERNAL_OSC keeps this sensor\'s native mode '
                    '(tighter internal LiDAR<->IMU sync; only use if '
                    'use_visual:=false).')
    pointcloud_topic_arg = DeclareLaunchArgument(
        'pointcloud_topic', default_value='/ouster/points')
    imu_topic_arg = DeclareLaunchArgument(
        'imu_topic', default_value='/ouster/imu')
    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic', default_value='/lucid_camera_1/image_raw')

    # Decode /ouster/{lidar,imu}_packets + /ouster/metadata into
    # /ouster/points + /ouster/imu. No bag_file arg — play the bag in a
    # second terminal, same convention as every dataset launch here.
    # viz:=false / pub_static_tf:=false: see resple_07052026.launch.py's
    # identical rationale (this dataset's own launch, not modified — the
    # comment there covers why in full). Scoped group: replay.launch.xml's
    # unscoped <set_parameter use_sim_time=true/> must not leak onto
    # dlio_odom_node/dlio_map_node (both run wall-clock here, matching every
    # other dliio script's default USE_SIM=false).
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

    # Layered parameter files (later wins). The extrinsics overlay (frame
    # rename + this rig's precise os_lidar<->os_imu extrinsic) always
    # applies. The visual/lidar_image overlay files are always loaded (cheap:
    # just declares weights/intrinsics) but their *_enabled flags are
    # overridden by the final dict below, gated on the launch args — so
    # use_visual/use_lidar_image toggle the actual behavior without needing
    # a second copy of these files.
    cfg_examples = PathJoinSubstitution([dliio_pkg, 'cfg', 'examples'])
    parameters = [
        PathJoinSubstitution([dliio_pkg, 'cfg', 'dlio.yaml']),
        # Tunnel-mode base (gate/regularization/photometric weight) — see the
        # header comment above. Loaded before our own extrinsics overlay so
        # its coarse datasheet IMU extrinsic gets overridden, not the other
        # way around.
        PathJoinSubstitution([cfg_examples, 'ouster_tunnel.yaml']),
        PathJoinSubstitution([cfg_examples, 'ouster_07052026_extrinsics.yaml']),
        PathJoinSubstitution([cfg_examples, 'ouster_tunnel_visual.yaml']),
        PathJoinSubstitution([cfg_examples, 'ouster_07052026_visual.yaml']),
        PathJoinSubstitution([cfg_examples, 'ouster_tunnel_lidarimg.yaml']),
        PathJoinSubstitution([cfg_examples, 'ouster_tunnel_xicp.yaml']),
        {
            'odom/visual/enabled': ParameterValue(LaunchConfiguration('use_visual'), value_type=bool),
            'odom/visual/map/enabled': ParameterValue(LaunchConfiguration('use_visual'), value_type=bool),
            'odom/lidar_image/enabled': ParameterValue(LaunchConfiguration('use_lidar_image'), value_type=bool),
            'odom/xicp/ternaryEnabled': ParameterValue(LaunchConfiguration('use_xicp'), value_type=bool),
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
            ('camera', LaunchConfiguration('camera_topic')),
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
        use_visual_arg,
        use_lidar_image_arg,
        use_xicp_arg,
        timestamp_mode_arg,
        pointcloud_topic_arg,
        imu_topic_arg,
        camera_topic_arg,
        ouster_decode,
        dlio_odom_node,
        dlio_map_node,
        rviz_node,
    ])
