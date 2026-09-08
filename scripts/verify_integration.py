#!/usr/bin/env python3
"""Check installed, namespaced DLIO integration with synthetic mapping input.

Loads standalone and composed frontends, exercises their ROS wiring and two
independent persistent mappers. This checks integration, not trajectory accuracy.
"""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

import numpy as np
import rclpy
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from nav_msgs.msg import Odometry
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import PointCloud2
from tf2_msgs.msg import TFMessage
from direct_lidar_inertial_odometry.msg import MappingObservation
from direct_lidar_inertial_odometry.srv import MapArchive
import dliio_mapping
import _dliio_pose_graph
from dliio_mapping.input import decode_cloud
from dliio_mapping.node import cloud_message


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
            raise RuntimeError('Integration launch did not shut down cleanly')
    if process.returncode != 0:
        raise RuntimeError(f'Integration launch exited with code {process.returncode}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path.cwd()/'dliio_run/integration.json')
    args = parser.parse_args()
    output = args.output.resolve()
    run = output.parent/('integration-' + uuid.uuid4().hex[:8])
    run.mkdir(parents=True)
    result = {'passed': False, 'artifacts': str(run),
              'python_mapping': dliio_mapping.__file__, 'pose_graph': _dliio_pose_graph.__file__}
    output.write_text(json.dumps(result, indent=2)+'\n')
    prefix = Path(get_package_prefix('direct_lidar_inertial_odometry'))
    # Symlink installs are supported too; compare the installed path itself,
    # without resolving symlinks into a development checkout.
    assert Path(dliio_mapping.__file__).is_relative_to(prefix), result
    assert Path(_dliio_pose_graph.__file__).is_relative_to(prefix), result
    package = Path(get_package_share_directory('direct_lidar_inertial_odometry'))
    rclpy.init()
    node = rclpy.create_node('dlio_integration_check')
    processes, odometry, maps, static_frames = [], {}, {}, set()
    scopes = ('alpha', 'beta')

    def until(predicate, timeout=30.):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.025)
            if any(process.poll() is not None for process in processes):
                raise RuntimeError('DLIO launch exited early; inspect integration logs')
            if predicate():
                return
        raise RuntimeError('Timed out waiting for the integration contract')

    def receive_tf(message):
        static_frames.update((transform.header.frame_id, transform.child_frame_id)
                             for transform in message.transforms)

    try:
        # Refuse to run this verification on top of another DLIO graph.
        discovery_deadline = time.monotonic()+1.
        until(lambda: time.monotonic() >= discovery_deadline)
        if any(name.startswith('dlio_') and name != node.get_name()
               for name in node.get_node_names()):
            raise RuntimeError('Use a separate ROS_DOMAIN_ID for integration verification')
        subscriptions = [node.create_subscription(TFMessage, '/tf_static', receive_tf,
            QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL))]
        publishers, services = {}, {}
        with ExitStack() as stack:
            for scope in scopes:
                configuration = {
                    'frames/odom': scope+'/odom', 'frames/baselink': scope+'/base',
                    'frames/lidar': scope+'/lidar', 'frames/imu': scope+'/imu', 'frames/map': scope+'/map',
                    'extrinsics/source': 'yaml', 'extrinsics/baselink2imu/t': [0., 0., 0.],
                    'imu/calibration': False, 'odom/debug/dashboard': False,
                    'mapping/fusion_size': 0., 'mapping/publish_rate': 10.,
                }
                config = run/(scope+'.yaml')
                config.write_text(json.dumps({'/**': {'ros__parameters': configuration}}))
                topic = '/'+scope+'/dlio/'
                subscriptions.extend([
                    node.create_subscription(Odometry, topic+'odom_node/odom',
                        lambda message, scope=scope: odometry.update({scope: message}), 10),
                    node.create_subscription(PointCloud2, topic+'map_node/map',
                        lambda message, scope=scope: maps.update({scope: message}),
                        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)),
                ])
                publishers[scope] = node.create_publisher(MappingObservation, topic+'odom_node/mapping_observation', 10)
                services[scope] = node.create_client(MapArchive, topic+'mapping/save_map')
                log = stack.enter_context((run/(scope+'.log')).open('w'))
                process = subprocess.Popen(['ros2', 'launch', 'direct_lidar_inertial_odometry',
                    'dlio.launch.py', 'namespace:='+scope, 'mapper:=persistent',
                    'composed:='+str(scope == 'beta').lower(), 'robot_config:='+str(config),
                    'pointcloud_topic:=lidar/points', 'imu_topic:=imu/data',
                    'archive_directory:='+str(run/scope)], stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
                processes.append(process)
                stack.callback(stop, process)
            until(lambda: all(scope in odometry and publishers[scope].get_subscription_count() == 1
                              and services[scope].service_is_ready() for scope in scopes))
            until(lambda: all((scope+'/base', scope+'/lidar') in static_frames for scope in scopes))
            for index, scope in enumerate(scopes):
                message = odometry[scope]
                assert message.header.frame_id == scope+'/odom', message
                assert message.child_frame_id == scope+'/base', message
                for topic in ('lidar/points', 'imu/data'):
                    inputs = node.get_subscriptions_info_by_topic('/'+scope+'/'+topic)
                    assert any(info.node_namespace == '/'+scope and info.node_name == 'dlio_odom_node'
                               for info in inputs), inputs
                assert node.get_publishers_info_by_topic('/'+scope+'/diagnostics')
                assert node.get_publishers_info_by_topic('/'+scope+'/dlio/mapping/diagnostics')
                cloud = np.array([[index*10+1., 2., 3., 10., 20., 30., 40.]], dtype=np.float32)
                observation = MappingObservation()
                observation.cloud = cloud_message(cloud, 1000000000, scope+'/odom')
                observation.header = observation.cloud.header
                observation.base_frame_id = scope+'/base'
                observation.source_session_id = str(uuid.uuid4())
                observation.registered_pose.orientation.w = 1.
                observation.registration_converged = True
                observation.covariance_model = 'unavailable'
                publishers[scope].publish(observation)
                until(lambda: scope in maps and maps[scope].width == 1)
                assert maps[scope].header.frame_id == scope+'/map'
                np.testing.assert_allclose(decode_cloud(maps[scope], 10, 1000), cloud)
                archive = run/(scope+'.dliomap')
                future = services[scope].call_async(MapArchive.Request(path=str(archive)))
                until(future.done)
                response = future.result()
                assert response.success and response.keyframes == 1 and archive.is_file(), response
            names = dict(node.get_service_names_and_types())
            for scope in scopes:
                assert '/'+scope+'/dlio/mapping/update_pose_graph' in names
            assert not any(name.startswith('/dlio/mapping/') for name in names), names
            assert not node.get_publishers_info_by_topic('/diagnostics')
            result.update(namespaces=list(scopes), synthetic_observations=2,
                          calibration_precedence=True, installed_share=str(package))
        result['passed'] = True
    except BaseException as error:
        result['error'] = str(error)
        raise
    finally:
        result['process_exit_codes'] = [process.poll() for process in processes]
        output.write_text(json.dumps(result, indent=2)+'\n')
        node.destroy_node()
        rclpy.shutdown()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
