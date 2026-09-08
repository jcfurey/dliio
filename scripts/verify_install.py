#!/usr/bin/env python3
"""Exercise installed components, a latched map, and SavePCD without sensor hardware."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import struct
import subprocess
import time

import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from direct_lidar_inertial_odometry.srv import SavePCD
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path.cwd() / 'dliio_run/installed-smoke.json')
    parser.add_argument('--require-livox', action='store_true', help='Also exercise the compiled CustomMsg adapter')
    parser.add_argument('--launch', default='dlio_ouster.launch.py',
        choices=('dlio_ouster.launch.py', 'dlio_composed.launch.py', 'dlio.launch.py'),
        help='Installed preview launch to verify')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'passed': False, 'status': 'starting'}) + '\n')
    expected = {'dlio_odom_node', 'dlio_map_node'}
    if args.launch != 'dlio.launch.py':
        expected.add('dlio_ouster_container' if args.launch == 'dlio_ouster.launch.py' else 'dlio_container')
    rclpy.init()
    node = rclpy.create_node('dlio_handoff_check')
    discovery_deadline = time.monotonic() + 1.
    while time.monotonic() < discovery_deadline:
        rclpy.spin_once(node, timeout_sec=.05)
    if expected.intersection(node.get_node_names()):
        node.destroy_node(); rclpy.shutdown()
        raise RuntimeError('dliio is already running in this ROS domain; use a separate ROS_DOMAIN_ID')
    received = {}
    if args.require_livox:
        from livox_ros_driver2.msg import CustomMsg, CustomPoint
        livox_sub = node.create_subscription(PointCloud2, '/livox2dlio',
            lambda msg: received.update(livox=msg), qos_profile_sensor_data)
        livox_pub = node.create_publisher(CustomMsg, '/livox', qos_profile_sensor_data)
    odom_sub = node.create_subscription(Odometry, '/dlio/odom_node/odom',
        lambda msg: received.update(odom=msg), 10)
    map_sub = node.create_subscription(PointCloud2, '/dlio/map_node/map',
        lambda msg: received.update(map=msg), QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    cloud_pub = node.create_publisher(PointCloud2, '/dlio/odom_node/pointcloud/keyframe', 10)
    service = node.create_client(SavePCD, '/save_pcd')
    with args.output.with_suffix('.log').open('w') as log:
        command = ['ros2', 'launch', 'direct_lidar_inertial_odometry', args.launch, 'rviz:=false']
        if args.launch == 'dlio_ouster.launch.py':
            command += ['profile:=0705', 'mode:=points']
        process = subprocess.Popen(command,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=.05)
                if process.poll() is not None:
                    raise RuntimeError('Launch exited early; inspect the smoke log')
                if expected.issubset(node.get_node_names()) and 'odom' in received and service.service_is_ready() and cloud_pub.get_subscription_count():
                    break
            else:
                raise RuntimeError('Installed components did not become ready')
            cloud = PointCloud2()
            cloud.header.frame_id = 'odom'; cloud.header.stamp.sec = 1
            cloud.height = cloud.width = 1; cloud.point_step = cloud.row_step = 20
            cloud.is_dense = True
            cloud.fields = [PointField(name=name, offset=i * 4, datatype=PointField.FLOAT32, count=1)
                            for i, name in enumerate(['x', 'y', 'z', 'intensity', 'reflectivity'])]
            cloud.data = struct.pack('<5f', 1., 2., 3., 1000., 42.)
            cloud_pub.publish(cloud)
            request = SavePCD.Request(); request.leaf_size = 0.; request.save_path = str(args.output.parent)
            future = service.call_async(request)  # expected rejection also exercises generated type support
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and ('map' not in received or not future.done()):
                rclpy.spin_once(node, timeout_sec=.05)
            if 'map' not in received or not future.done() or future.result().success:
                raise RuntimeError('Map publication or SavePCD validation failed')
            msg = received['map']
            assert msg.header.frame_id == 'odom' and msg.header.stamp.sec == 1
            assert msg.width * msg.height == 1
            values = {f.name: struct.unpack_from('<f', msg.data, f.offset)[0] for f in msg.fields}
            assert values['intensity'] == 1000. and values['reflectivity'] == 42.
            assert not {'t', 'time', 'timestamp'}.intersection(values)
            assert math.isfinite(received['odom'].pose.pose.orientation.w)
            if args.require_livox:
                deadline = time.monotonic() + 5
                while not livox_pub.get_subscription_count() and time.monotonic() < deadline:
                    rclpy.spin_once(node, timeout_sec=.05)
                assert livox_pub.get_subscription_count(), 'Livox adapter was not compiled'
                packet = CustomMsg(); packet.header.stamp.sec = 1700000000
                packet.timebase = 1700000000000000000
                packet.point_num = 1
                packet.points = [CustomPoint(x=1., y=2., z=3., reflectivity=42, offset_time=1000)]
                livox_pub.publish(packet)
                deadline = time.monotonic() + 5
                while 'livox' not in received and time.monotonic() < deadline:
                    rclpy.spin_once(node, timeout_sec=.05)
                converted = received.pop('livox')
                offsets = {f.name: f.offset for f in converted.fields}
                assert converted.width == 1 and converted.header.frame_id == 'os_lidar'
                assert struct.unpack_from('<f', converted.data, offsets['reflectivity'])[0] == 42.
                assert struct.unpack_from('<f', converted.data, offsets['intensity'])[0] == 42.
                assert abs(struct.unpack_from('<d', converted.data, offsets['timestamp'])[0] -
                           (float(packet.timebase) + 1000.)) < 512.
                packet.point_num = 1000  # inconsistent length must be rejected, not read out of bounds
                livox_pub.publish(packet)
                deadline = time.monotonic() + .5
                while time.monotonic() < deadline:
                    rclpy.spin_once(node, timeout_sec=.05)
                assert 'livox' not in received and process.poll() is None
            result = {'ros_distro': os.environ.get('ROS_DISTRO'), 'nodes': sorted(node.get_node_names()),
                      'map_fields': values, 'invalid_export_rejected': True,
                      'livox_adapter_checked': args.require_livox, 'passed': True}
            args.output.write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(result, indent=2))
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM); process.wait(timeout=10)
            node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
