"""Standalone Ouster + dliio workflow for Humble, Jazzy, Kilted, and Lyrical."""
import json
import math
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, SetEnvironmentVariable, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def boolean(value):
    if value.lower() not in ('true', 'false'):
        raise ValueError(f'Expected true or false, got {value!r}')
    return value.lower() == 'true'


def extract_metadata(bag, destination):
    # Read only the recorded metadata, through the distro's own storage plugin.
    # No helper ROS node, workspace source path, or sensor SDK is needed.
    from rclpy.serialization import deserialize_message
    from rosbag2_py import ConverterOptions, SequentialReader, StorageFilter, StorageOptions
    from std_msgs.msg import String
    reader = SequentialReader()
    reader.open(StorageOptions(uri=str(bag), storage_id=''), ConverterOptions('', ''))
    topics = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    if topics.get('/ouster/metadata') != 'std_msgs/msg/String':
        raise ValueError('Bag needs /ouster/metadata (std_msgs/msg/String), or metadata:=/path/sensor.json')
    reader.set_filter(StorageFilter(topics=['/ouster/metadata']))
    if not reader.has_next():
        raise ValueError('The bag contains no Ouster metadata message; supply metadata:=/path/sensor.json')
    _, serialized, _ = reader.read_next()
    metadata = deserialize_message(serialized, String).data
    json.loads(metadata)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(metadata)
    return destination


def build_actions(args, package):
    mode, profile = args['mode'], args['profile']
    if mode not in ('points', 'packets'):
        raise ValueError('mode must be points (existing Ouster driver) or packets (recorded Ouster packets)')
    if profile not in ('generic', '0705'):
        raise ValueError('profile must be generic or 0705')
    if args['mapper'] not in ('preview', 'persistent'):
        raise ValueError('mapper must be preview or persistent')
    mapping_config = Path(args['mapping_config']).expanduser() if args['mapping_config'] else package / 'cfg/mapping.yaml'
    if args['mapper'] == 'persistent' and not mapping_config.is_file():
        raise ValueError(f'Mapping configuration does not exist: {mapping_config}')
    rate = float(args['rate'])
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError('rate must be finite and positive')
    bag = Path(args['bag']).expanduser().resolve() if args['bag'] else None
    if bag and not bag.exists():
        raise ValueError(f'Bag does not exist: {bag}')
    sim = (bag is not None or mode == 'packets') if args['use_sim_time'] == 'auto' else boolean(args['use_sim_time'])
    if bag and not sim:
        raise ValueError('Bag replay requires use_sim_time:=true (or auto)')
    params_file = Path(args['params_file']).expanduser() if args['params_file'] else package / 'cfg/params.yaml'
    params = [str(params_file)]
    if profile == '0705':
        params += [str(package / 'cfg/examples' / name) for name in (
            'ouster_tunnel.yaml', 'ouster_07052026_extrinsics.yaml', 'ouster_07052026_texture.yaml')]
    else:
        params.insert(0, str(package / 'cfg/dlio.yaml'))
    if args['robot_config']:
        params.append(str(Path(args['robot_config']).expanduser()))
    for filename in params:
        if not Path(filename).is_file():
            raise ValueError(f'Configuration file does not exist: {filename}')
    # This handoff owns its static extrinsics and requires no camera/fusion node.
    params.append({'use_sim_time': sim, 'extrinsics/source': 'yaml',
                   'odom/visual/enabled': False, 'odom/visual/map/enabled': False,
                   'odom/debug/dashboard': False})
    intra = [{'use_intra_process_comms': True}]
    components = []
    if mode == 'packets':
        metadata = Path(args['metadata']).expanduser().resolve() if args['metadata'] else None
        if metadata is None:
            if bag is None:
                raise ValueError('Packet mode requires bag:=... or metadata:=... for an external bag player')
            metadata = extract_metadata(bag, Path(args['run_dir']).expanduser().resolve() / 'ouster-metadata.json')
        if not metadata.is_file():
            raise ValueError(f'Metadata file does not exist: {metadata}')
        json.loads(metadata.read_text())
        get_package_share_directory('ouster_ros')  # fail before spawning any nodes
        components.append(ComposableNode(
            package='ouster_ros', plugin='ouster_ros::OusterCloud',
            namespace='ouster', name='os_cloud', parameters=[{
                'use_sim_time': sim, 'metadata': str(metadata),
                'timestamp_mode': 'TIME_FROM_ROS_TIME', 'proc_mask': 'PCL|IMU',
                'point_type': 'original', 'organized': True, 'destagger': True,
                'pub_static_tf': False, 'use_system_default_qos': True,
                'lidar_packet_reliable': True,
                'qos_overrides./ouster/lidar_packets.subscription.depth': 4096,
                'qos_overrides./ouster/imu_packets.subscription.depth': 1000,
            # Humble rejects Ouster's transient-local metadata subscription
            # when intra-process delivery is enabled. Keep the driver in the
            # same container and let its points/IMU use the DDS local path.
            }], extra_arguments=[{'use_intra_process_comms': os.environ.get('ROS_DISTRO') != 'humble'}]))
    pointcloud = args['pointcloud_topic']
    imu = args['imu_topic']
    if mode == 'packets' and (pointcloud != '/ouster/points' or imu != '/ouster/imu'):
        raise ValueError('Packet mode publishes /ouster/points and /ouster/imu; topic overrides apply to points mode')
    components.append(ComposableNode(
        package='direct_lidar_inertial_odometry', plugin='dlio::OdomNode', name='dlio_odom_node',
        parameters=params, remappings=[
            ('pointcloud', pointcloud), ('imu', imu),
            ('odom', '/dlio/odom_node/odom'), ('pose', '/dlio/odom_node/pose'),
            ('scan_pose', '/dlio/odom_node/scan_pose'), ('path', '/dlio/odom_node/path'),
            ('kf_pose', '/dlio/odom_node/keyframes'), ('kf_pose_stamped', '/dlio/odom_node/keyframe_pose'),
            ('kf_cloud', '/dlio/odom_node/pointcloud/keyframe'),
            ('deskewed', '/dlio/odom_node/pointcloud/deskewed')], extra_arguments=intra))
    if boolean(args['map']) and args['mapper'] == 'preview':
        components.append(ComposableNode(
            package='direct_lidar_inertial_odometry', plugin='dlio::MapNode', name='dlio_map_node',
            parameters=params, remappings=[('keyframes', '/dlio/odom_node/pointcloud/keyframe'),
                                          ('map', '/dlio/map_node/map')], extra_arguments=intra))
    actions = [ComposableNodeContainer(
        name='dlio_ouster_container', namespace='', package='rclcpp_components',
        executable='component_container_mt', parameters=[{'thread_num': 6}],
        composable_node_descriptions=components, output='screen')]
    if boolean(args['map']) and args['mapper'] == 'persistent':
        directory = (Path(args['archive_directory']).expanduser() if args['archive_directory'] else
                     Path(args['run_dir']).expanduser() / 'mapping')
        actions.append(Node(package='direct_lidar_inertial_odometry', executable='dlio_mapping_node.py',
            name='dlio_mapping_node', parameters=[*params, str(mapping_config),
                {'mapping/storage_directory': str(directory.resolve()), 'mapping/load_path': ''}],
            remappings=[('keyframes', '/dlio/odom_node/pointcloud/keyframe'),
                        ('keyframe_pose', '/dlio/odom_node/keyframe_pose'), ('map', '/dlio/map_node/map')],
            output='screen'))
    if boolean(args['rviz']):
        actions.append(Node(package='rviz2', executable='rviz2', name='dlio_rviz',
            arguments=['-d', str(package / 'launch/dlio_0705_texture.rviz')],
            parameters=[{'use_sim_time': sim}], output='screen'))
    if bag:
        topics = ['/ouster/metadata', '/ouster/lidar_packets', '/ouster/imu_packets'] if mode == 'packets' else [pointcloud, imu]
        cmd = ['ros2', 'bag', 'play', str(bag), '--rate', str(rate), '--clock', '1000',
               '--topics', *topics, '--read-ahead-queue-size', '10000', '--disable-keyboard-controls']
        if mode == 'packets':
            cmd += ['--qos-profile-overrides-path', str(package / 'cfg/ouster_packet_replay_qos.yaml')]
        # Use only flags present in both Humble and Jazzy.
        actions.append(TimerAction(period=8., actions=[ExecuteProcess(cmd=cmd, output='screen')]))
    return actions


DEFAULTS = {
    'mode': 'points', 'profile': 'generic', 'bag': '', 'metadata': '', 'rate': '1.0',
    'rviz': 'false', 'map': 'true', 'use_sim_time': 'auto', 'robot_config': '', 'params_file': '',
    'mapper': 'preview', 'mapping_config': '', 'archive_directory': '',
    'pointcloud_topic': '/ouster/points', 'imu_topic': '/ouster/imu',
    'run_dir': str(Path.cwd() / 'dliio_run'),
}


def setup(context):
    args = {name: LaunchConfiguration(name).perform(context) for name in DEFAULTS}
    return build_actions(args, Path(get_package_share_directory('direct_lidar_inertial_odometry')))


def generate_launch_description():
    return LaunchDescription([
        *[DeclareLaunchArgument(name, default_value=value) for name, value in DEFAULTS.items()],
        SetEnvironmentVariable('OMP_NUM_THREADS', os.environ.get('OMP_NUM_THREADS', '4')),
        SetEnvironmentVariable('OPENBLAS_NUM_THREADS', os.environ.get('OPENBLAS_NUM_THREADS', '1')),
        OpaqueFunction(function=setup),
    ])
