#!/usr/bin/env python3
"""Exercise the installed mapper, bounded eviction, archive services and reload."""
import argparse
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import time
import uuid

import numpy as np
import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from direct_lidar_inertial_odometry.srv import MapArchive
from dliio_mapping.core import FIELDS, Limits, Store, decode_points
from dliio_mapping.node import cloud_message
from dliio_mapping.input import decode_cloud


def stop(process):
    if process and process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path.cwd() / 'dliio_run/mapping-smoke.json')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run = args.output.parent / ('mapping-smoke-' + uuid.uuid4().hex[:8])
    run.mkdir()
    args.output.write_text(json.dumps({'passed': False, 'artifacts': str(run)}) + '\n')
    rclpy.init()
    node = rclpy.create_node('dlio_mapping_check')
    process = None

    def until(predicate, seconds=15):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.025)
            if process and process.poll() is not None:
                raise RuntimeError('Mapper exited; inspect the smoke log')
            if predicate():
                return
        raise RuntimeError('Timed out waiting for the mapper')

    def spin(seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.025)

    try:
        spin(1)
        if 'dlio_mapping_node' in node.get_node_names():
            raise RuntimeError('Mapper already running; use a separate ROS_DOMAIN_ID')
        received = {}
        diagnostics = {}
        def diagnostic(msg):
            for status in msg.status:
                diagnostics.update({value.key: value.value for value in status.values})
        subscriptions = [
            node.create_subscription(PointCloud2, '/mapping_smoke/map', lambda msg: received.update(map=msg),
                                     QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)),
            node.create_subscription(DiagnosticArray, '/dlio/mapping/diagnostics', diagnostic, 10)]
        cloud_pub = node.create_publisher(PointCloud2, '/mapping_smoke/cloud', 10)
        pose_pub = node.create_publisher(PoseStamped, '/mapping_smoke/pose', 10)
        services = {name: node.create_client(MapArchive, '/dlio/mapping/' + name)
                    for name in ('save_map', 'load_map', 'export_pcd')}
        parameter_service = node.create_client(SetParameters, '/dlio_mapping_node/set_parameters')

        def request(operation, path):
            client = services[operation]
            until(client.service_is_ready)
            future = client.call_async(MapArchive.Request(path=str(path)))
            until(future.done, 65)
            return future.result()

        base = ['ros2', 'run', 'direct_lidar_inertial_odometry', 'dlio_mapping_node.py', '--ros-args',
                '-r', 'keyframes:=/mapping_smoke/cloud', '-r', 'keyframe_pose:=/mapping_smoke/pose',
                '-r', 'map:=/mapping_smoke/map', '-p', 'mapping/submap_keyframes:=2',
                '-p', 'mapping/resident_submaps:=2', '-p', 'mapping/max_voxels:=100',
                '-p', 'mapping/pair_timeout:=0.2', '-p', 'mapping/publish_rate:=10.0',
                '-p', 'map/observation/distance:=0.2']
        with (run / 'recording.log').open('w') as log:
            process = subprocess.Popen(base + ['-p', f'mapping/storage_directory:={run}'],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            until(lambda: cloud_pub.get_subscription_count() and pose_pub.get_subscription_count() and
                  services['save_map'].service_is_ready(), 30)
            for i in range(12):
                stamp = 1800000000000000000 + (i + 1) * 100000000
                pose = PoseStamped()
                pose.header.frame_id = 'odom'
                pose.header.stamp.sec, pose.header.stamp.nanosec = divmod(stamp, 1000000000)
                pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = 10. + i, 20., 30.
                pose.pose.orientation.z = pose.pose.orientation.w = 2 ** -.5
                world = np.array([[10+i, 21, 30, 12, 31, np.nan, 43],
                                  [9+i, 20, 30, 22, 51, 62, 73]], dtype=np.float32)
                cloud = cloud_message(world, stamp, 'odom')
                # Alternate arrival order between the two ROS topics.
                if i % 2:
                    pose_pub.publish(pose); cloud_pub.publish(cloud)
                else:
                    cloud_pub.publish(cloud); pose_pub.publish(pose)
                spin(.08)
            until(lambda: int(diagnostics.get('keyframes', 0)) == 12)
            until(lambda: 'map' in received and received['map'].header.stamp == cloud.header.stamp)
            expected_view = decode_cloud(received['map'], 100, 10000)
            assert received['map'].header.frame_id == 'map'
            assert tuple(f.name for f in received['map'].fields) == FIELDS
            assert len(expected_view) == 8 and int(diagnostics['resident_submaps']) == 2
            assert int(diagnostics.get('queue_dropped', 0)) == 0
            assert int(diagnostics.get('processing_errors', 0)) == 0
            # Diagnose an unmatched cloud and reject stale or wrong-frame input.
            cloud_pub.publish(cloud)
            cloud.header.stamp.nanosec += 100000000
            cloud_pub.publish(cloud)
            pose.header.frame_id = 'wrong_frame'
            pose_pub.publish(pose)
            until(lambda: int(diagnostics.get('missing_pose', 0)) == 1 and
                  int(diagnostics.get('invalid_input', 0)) == 1)
            assert int(diagnostics['late_or_duplicate']) == 1
            until(parameter_service.service_is_ready)
            change = Parameter(name='mapping/voxel_size', value=ParameterValue(
                type=ParameterType.PARAMETER_DOUBLE, double_value=.5))
            future = parameter_service.call_async(SetParameters.Request(parameters=[change]))
            until(future.done)
            assert not future.result().results[0].successful
            saved, exported = run / 'saved.dliomap', run / 'all.pcd'
            result = request('save_map', saved)
            assert result.success and result.keyframes == 12 and result.submaps == 6, result.message
            assert request('export_pcd', exported).success
            assert not request('save_map', saved).success
            assert not request('load_map', saved).success
            recording_diagnostics = dict(diagnostics)
            stop(process)
            process = None
        archived = Store(saved, Limits(resident_submaps=2, max_voxels=100))
        try:
            np.testing.assert_array_equal(archived.snapshot(float(recording_diagnostics['fusion_size']) or None), expected_view)
            n, blob, crc = archived.db.execute('SELECT count,points,crc FROM keyframes WHERE id=0').fetchone()
            np.testing.assert_allclose(decode_points(blob, n, crc, 10)[:, :3], [[1, 0, 0], [0, 1, 0]], atol=1e-6)
            assert 'extrinsics/baselink2imu/R' in archived.meta['calibration']
            assert archived.meta['observation_selection']['map/observation/distance'] == .2
            session = archived.meta['session_id']
        finally:
            archived.close()
        header, payload = exported.read_bytes().split(b'DATA binary\n', 1)
        assert b'POINTS 24\n' in header and len(payload) == 24 * 28
        # A fresh process must recover the same window and full export from disk.
        received.clear(); diagnostics.clear()
        spin(.5)
        received.clear(); diagnostics.clear()
        with (run / 'viewer.log').open('w') as log:
            process = subprocess.Popen(base + ['-p', f'mapping/load_path:={saved}'],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            until(lambda: diagnostics.get('read_only') == 'True' and 'map' in received, 30)
            assert diagnostics['session_id'] == session
            np.testing.assert_array_equal(decode_cloud(received['map'], 100, 10000), expected_view)
            assert request('export_pcd', run / 'reloaded.pcd').success
            assert (run / 'reloaded.pcd').read_bytes() == exported.read_bytes()
            broken = run / 'broken.dliomap'
            broken.write_bytes(saved.read_bytes())
            with sqlite3.connect(broken) as db:
                db.execute('UPDATE submaps SET crc=0')
            assert not request('load_map', broken).success
            assert request('export_pcd', run / 'after-failed-load.pcd').success
            assert (run / 'after-failed-load.pcd').read_bytes() == exported.read_bytes()
            assert request('load_map', saved).success
            # Late subscribers receive the existing map even without new input.
            late = {}
            late_sub = node.create_subscription(PointCloud2, '/mapping_smoke/map', lambda msg: late.update(map=msg),
                QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
            until(lambda: 'map' in late)
            np.testing.assert_array_equal(decode_cloud(late['map'], 100, 10000), expected_view)
        result = dict(passed=True, ros_distro=os.environ.get('ROS_DISTRO'), artifacts=str(run),
            keyframes=12, archived_submaps=6, resident_submaps=2, resident_points=8, exported_points=24,
            fields=list(FIELDS), reload_exact=True, failed_load_preserved_map=True,
            recording_diagnostics=recording_diagnostics)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))
    finally:
        stop(process)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
