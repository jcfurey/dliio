#
#   Copyright (c)
#
#   The Verifiable & Control-Theoretic Robotics (VECTR) Lab
#   University of California, Los Angeles
#
#   Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez
#   Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu
#

"""Sensor-independent DLIO bringup for existing PointCloud2 and IMU publishers."""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


PACKAGE = 'direct_lidar_inertial_odometry'
DEFAULTS = {
    'namespace': '', 'pointcloud_topic': 'points_raw', 'imu_topic': 'imu_raw',
    'camera_topic': 'image_raw', 'robot_config': '', 'params_file': '',
    'use_sim_time': 'false', 'composed': 'false', 'mapper': 'preview',
    'mapping_config': '', 'archive_directory': '', 'loop_closure_config': '',
    'rviz': 'false', 'rviz_config': '', 'rviz_frame': '',
}
DESCRIPTIONS = {
    'namespace': 'ROS namespace for all DLIO nodes, topics and services; TF frame IDs come from robot_config.',
    'robot_config': 'Robot calibration and frames, applied after algorithm defaults. Empty retains cfg/dlio.yaml.',
    'params_file': 'Algorithm configuration; empty uses cfg/params.yaml.',
    'mapper': 'none: odometry only; preview: accumulated map; persistent: archive and pose graph services.',
    'archive_directory': 'Required with mapper:=persistent. Directory for a new recording database.',
    'mapping_config': 'Persistent mapper limits and publication settings; empty uses cfg/mapping.yaml.',
    'loop_closure_config': 'Optional automatic loop JSON with explicit assumed noise and geometry limits; requires persistent mapping.',
    'composed': 'Load the C++ nodes in one container. The persistent Python mapper stays in its own process.',
    'use_sim_time': 'Use /clock from simulation or an external ros2 bag play --clock.',
    'rviz_frame': 'RViz fixed frame; empty uses map for persistent mapping, otherwise odom.',
}
OUTPUTS = {
    'odom': 'dlio/odom_node/odom', 'pose': 'dlio/odom_node/pose',
    'scan_pose': 'dlio/odom_node/scan_pose', 'kf_pose_stamped': 'dlio/odom_node/keyframe_pose',
    'path': 'dlio/odom_node/path', 'kf_pose': 'dlio/odom_node/keyframes',
    'kf_cloud': 'dlio/odom_node/pointcloud/keyframe',
    'deskewed': 'dlio/odom_node/pointcloud/deskewed',
    'mapping_cloud': 'dlio/odom_node/pointcloud/mapping',
    'mapping_pose': 'dlio/odom_node/mapping_pose',
    'mapping_observation': 'dlio/odom_node/mapping_observation',
}


def boolean(value):
    if value.lower() not in ('true', 'false'):
        raise ValueError('Boolean launch arguments must be true or false')
    return value.lower() == 'true'


def build_actions(args, package):
    mapper = args['mapper']
    if mapper not in ('none', 'preview', 'persistent'):
        raise ValueError('mapper must be none, preview, or persistent')
    sim, composed, rviz = (boolean(args[name]) for name in ('use_sim_time', 'composed', 'rviz'))
    persistent = mapper == 'persistent'
    if persistent and not args['archive_directory'].strip():
        raise ValueError('mapper:=persistent requires archive_directory:=/path/to/a/recording')
    if args['loop_closure_config'] and not persistent:
        raise ValueError('loop_closure_config requires mapper:=persistent')

    def configuration(argument, default):
        path = Path(args[argument]).expanduser() if args[argument] else package / default
        if not path.is_file():
            raise ValueError(f'{argument}: configuration file does not exist: {path}')
        return str(path.resolve())

    # Algorithm defaults contain frame defaults too. Apply robot calibration
    # last so names and extrinsics supplied by the integrating robot take effect.
    parameters = [configuration('params_file', 'cfg/params.yaml'),
                  configuration('robot_config', 'cfg/dlio.yaml')]
    overrides = {'use_sim_time': sim}
    if persistent:
        overrides.update({'map/observation/enabled': True, 'map/keyframe/filtered': False})
    if composed:
        overrides['odom/debug/dashboard'] = False
    parameters.append(overrides)
    namespace = args['namespace']
    odom_remaps = [('pointcloud', args['pointcloud_topic']), ('imu', args['imu_topic']),
                   ('camera', args['camera_topic']), *OUTPUTS.items()]
    map_remaps = [('keyframes', OUTPUTS['kf_cloud']), ('map', 'dlio/map_node/map')]
    if composed:
        components = [ComposableNode(package=PACKAGE, plugin='dlio::OdomNode',
            namespace=namespace, name='dlio_odom_node', parameters=parameters,
            remappings=odom_remaps, extra_arguments=[{'use_intra_process_comms': True}])]
        if mapper == 'preview':
            components.append(ComposableNode(package=PACKAGE, plugin='dlio::MapNode',
                namespace=namespace, name='dlio_map_node', parameters=parameters,
                remappings=map_remaps, extra_arguments=[{'use_intra_process_comms': True}]))
        actions = [ComposableNodeContainer(package='rclcpp_components', executable='component_container_mt',
            namespace=namespace, name='dlio_container', output='screen',
            composable_node_descriptions=components)]
    else:
        actions = [Node(package=PACKAGE, executable='dlio_odom_node', namespace=namespace,
                        parameters=parameters, remappings=odom_remaps, output='screen')]
        if mapper == 'preview':
            actions.append(Node(package=PACKAGE, executable='dlio_map_node', namespace=namespace,
                                parameters=parameters, remappings=map_remaps, output='screen'))
    if persistent:
        mapping_config = configuration('mapping_config', 'cfg/mapping.yaml')
        loop_config = configuration('loop_closure_config', '') if args['loop_closure_config'] else ''
        actions.append(Node(package=PACKAGE, executable='dlio_mapping_node.py', namespace=namespace,
            sigterm_timeout='60.0', sigkill_timeout='30.0',
            parameters=[mapping_config, *parameters, {
                'mapping/storage_directory': str(Path(args['archive_directory']).expanduser().resolve()),
                'mapping/load_path': '', 'mapping/input_source': 'observations',
                'mapping/transport': 'observation'},
                *([{'mapping/worker_queue': 64}] if loop_config else [])],
            remappings=[('observation', OUTPUTS['mapping_observation']), ('map', 'dlio/map_node/map')],
            output='screen'))
        if loop_config:
            actions.append(Node(package=PACKAGE, executable='dlio_loop_closure_node.py', namespace=namespace,
                sigterm_timeout='60.0',
                parameters=[{'use_sim_time': sim, 'loop_closure/configuration_file': loop_config,
                    'loop_closure/output_directory': str(Path(args['archive_directory']).expanduser().resolve()/'loops')}],
                output='screen'))
    if rviz:
        config = configuration('rviz_config', 'launch/dlio_mapping.rviz' if persistent else 'launch/dlio.rviz')
        topics = set(OUTPUTS.values()) | {'dlio/map_node/map', 'dlio/mapping/path'}
        actions.append(Node(package='rviz2', executable='rviz2', name='dlio_rviz', namespace=namespace,
            arguments=['-d', config, '-f', args['rviz_frame'] or ('map' if persistent else 'odom')],
            parameters=[{'use_sim_time': sim}], remappings=[('/'+topic, topic) for topic in sorted(topics)],
            output='screen'))
    return actions


def setup(context):
    args = {name: LaunchConfiguration(name).perform(context) for name in DEFAULTS}
    return build_actions(args, Path(get_package_share_directory(PACKAGE)))


def generate_launch_description():
    return LaunchDescription([
        *[DeclareLaunchArgument(name, default_value=value, description=DESCRIPTIONS.get(name, name))
          for name, value in DEFAULTS.items()],
        OpaqueFunction(function=setup),
    ])
