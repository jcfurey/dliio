#!/usr/bin/env python3
"""Exercise atomic mapping input, corrections, rollback, TF, and offline copies.

Run in a dedicated ROS_DOMAIN_ID with the installed dliio package sourced.
Uses synthetic transforms with independently known geometry; no sensor/bag.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

import numpy as np
import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile
from diagnostic_msgs.msg import DiagnosticArray
from sensor_msgs.msg import PointCloud2
from tf2_msgs.msg import TFMessage
from direct_lidar_inertial_odometry.msg import MappingObservation
from direct_lidar_inertial_odometry.srv import ApplyPoseRevision, MapArchive, RestorePoseRevision
from dliio_mapping.core import Limits, Store, Voxels
from dliio_mapping.input import decode_cloud, decode_pose, pose_message
from dliio_mapping.node import cloud_message


def pose(x=0., yaw=0.):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0, x], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.]])


def placed(local, matrix):
    result = local.copy()
    result[:, :3] = local[:, :3] @ matrix[:3, :3].T + matrix[:3, 3]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run = args.output.parent / ('pose-revisions-' + uuid.uuid4().hex[:8])
    run.mkdir()
    result = dict(passed=False, artifacts=str(run))
    args.output.write_text(json.dumps(result) + '\n')
    rclpy.init()
    node = rclpy.create_node('dliio_pose_revision_check')
    process = None
    received, diagnostics, dynamic, static = {}, {}, {}, []

    def until(predicate, timeout=15.):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.02)
            if process is not None and process.poll() is not None:
                raise RuntimeError(f'Mapper exited; inspect {run / "mapper.log"}')
            if predicate():
                return
        raise RuntimeError('Timed out waiting for revision integration result')

    def diagnostic(message):
        for status in message.status:
            if status.name == 'DLIO Mapping':
                diagnostics.update({value.key: value.value for value in status.values})

    def transform(message, destination):
        for value in message.transforms:
            if value.header.frame_id == 'map' and value.child_frame_id == 'odom':
                if isinstance(destination, list):
                    destination.append(value)
                else:
                    p = pose_message(np.eye(4))
                    p.position.x = value.transform.translation.x
                    p.position.y = value.transform.translation.y
                    p.position.z = value.transform.translation.z
                    p.orientation = value.transform.rotation
                    destination['matrix'] = decode_pose(SimpleNamespace(pose=p))

    def call(client, request):
        until(client.service_is_ready)
        future = client.call_async(request)
        until(future.done, 30.)
        return future.result()

    def matches(expected):
        if 'map' not in received:
            return False
        actual = decode_cloud(received['map'], 100, 10000)
        return actual.shape == expected.shape and np.allclose(actual, expected, atol=1e-5, equal_nan=True)

    try:
        # Give discovery a chance before refusing a conflicting mapper.
        start = time.monotonic()
        until(lambda: time.monotonic() - start > .5)
        if 'dlio_mapping_node' in node.get_node_names():
            raise RuntimeError('Mapper already present; use a dedicated ROS_DOMAIN_ID')
        subscriptions = [
            node.create_subscription(PointCloud2, '/revision_check/map', lambda m: received.update(map=m),
                QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)),
            node.create_subscription(DiagnosticArray, '/dlio/mapping/diagnostics', diagnostic, 10),
            node.create_subscription(TFMessage, '/tf', lambda m: transform(m, dynamic), 50),
            node.create_subscription(TFMessage, '/tf_static', lambda m: transform(m, static),
                QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)),
        ]
        publisher = node.create_publisher(MappingObservation, '/revision_check/observation', 8)
        apply = node.create_client(ApplyPoseRevision, '/dlio/mapping/apply_pose_revision')
        restore = node.create_client(RestorePoseRevision, '/dlio/mapping/restore_pose_revision')
        save = node.create_client(MapArchive, '/dlio/mapping/save_map')
        export = node.create_client(MapArchive, '/dlio/mapping/export_pcd')
        command = ['ros2', 'run', 'direct_lidar_inertial_odometry', 'dlio_mapping_node.py', '--ros-args',
            '-p', 'mapping/transport:=observation', '-p', f'mapping/storage_directory:={run / "archive"}',
            '-p', 'mapping/submap_keyframes:=2', '-p', 'mapping/resident_submaps:=1',
            '-p', 'mapping/fusion_size:=0.1', '-p', 'mapping/publish_rate:=10.0',
            '-r', 'observation:=/revision_check/observation', '-r', 'map:=/revision_check/map']
        with (run / 'mapper.log').open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        until(lambda: publisher.get_subscription_count() > 0 and 'session_id' in diagnostics, 30.)
        session = diagnostics['session_id']
        source = str(uuid.uuid4())
        local = np.array([[1., 0, 0, 2, np.nan, 4, 8], [0., 1, 0, 10, 20, np.nan, 12]], dtype='<f4')
        original_poses = [pose(i*3.) for i in range(5)]
        for i, original in enumerate(original_poses):
            cloud = cloud_message(placed(local, original), (10+i)*1000000000, 'odom')
            observation = MappingObservation(header=cloud.header, cloud=cloud,
                source_session_id=source, observation_id=i, base_frame_id='base_link',
                registered_pose=pose_message(original), covariance_kind=MappingObservation.UNKNOWN,
                covariance_model='synthetic input: uncertainty intentionally unavailable', registration_converged=True)
            publisher.publish(observation)
            until(lambda: diagnostics.get('keyframes') == str(i+1))
        expected_original = Voxels.from_points(placed(local, original_poses[-1]), .1).points()
        until(lambda: matches(expected_original))
        original_path = run / 'original.dliomap'
        assert call(save, MapArchive.Request(path=str(original_path))).success
        original_hash = hashlib.sha256(original_path.read_bytes()).hexdigest()

        corrected = [pose(20+i*3., (i % 3)*np.pi/2) for i in range(5)]
        request = ApplyPoseRevision.Request(session_id=session, expected_revision=0, request_id='known-correction',
            reason='synthetic nonuniform correction', frame_id='map', observation_ids=list(range(5)),
            poses=[pose_message(p) for p in corrected])
        response = call(apply, request)
        assert response.success and response.pose_revision == 1, response.message
        expected_view = Voxels.from_points(placed(local, corrected[-1]), .1).points()
        correction = corrected[-1] @ np.linalg.inv(original_poses[-1])
        until(lambda: matches(expected_view) and diagnostics.get('cached_pose_revision') == '1' and
              'matrix' in dynamic and np.allclose(dynamic['matrix'], correction, atol=1e-9))
        assert not static, 'Mapper published a competing static map -> odom transform'
        response = call(apply, request)
        assert response.success and response.pose_revision == 1, response.message
        request.request_id = 'stale-request'
        assert not call(apply, request).success
        request.request_id = 'known-correction'
        corrected_path = run / 'corrected.dliomap'
        assert call(save, MapArchive.Request(path=str(corrected_path))).success
        exported = run / 'corrected.pcd'
        assert call(export, MapArchive.Request(path=str(exported))).success
        expected_all = Voxels.from_points(np.vstack([placed(local, p) for p in corrected]), .1).points()
        actual_all = np.frombuffer(exported.read_bytes().split(b'DATA binary\n', 1)[1], '<f4').reshape(-1, 7)
        np.testing.assert_allclose(actual_all, expected_all, atol=1e-5)
        loaded = Store(corrected_path, Limits(resident_submaps=1))
        try:
            assert loaded.meta['pose_revision'] == 1
            np.testing.assert_allclose(loaded.snapshot(.1), expected_view, atol=1e-5)
            assert all(json.loads(row[0])['covariance'] is None for row in
                       loaded.db.execute('SELECT json FROM observations'))
        finally:
            loaded.close()

        # The same revision request through the offline copy workflow must
        # preserve the input file and produce equivalent corrected geometry.
        request_path = run / 'revision.json'
        request_path.write_text(json.dumps(dict(session_id=session, expected_revision=0,
            request_id='known-correction', reason='synthetic nonuniform correction', frame_id='map',
            poses=[dict(id=i, pose=p.tolist()) for i, p in enumerate(corrected)])))
        cli = Path(__file__).resolve().with_name('mapping_archive.py')
        offline = run / 'offline.dliomap'
        subprocess.run([sys.executable, str(cli), 'revise', str(original_path), str(offline),
                        '--revision', str(request_path)], check=True, capture_output=True, text=True)
        loaded = Store(offline, Limits(resident_submaps=1))
        try:
            np.testing.assert_allclose(loaded.snapshot(.1), expected_view, atol=1e-5)
        finally:
            loaded.close()
        assert hashlib.sha256(original_path.read_bytes()).hexdigest() == original_hash
        response = call(restore, RestorePoseRevision.Request(session_id=session, expected_revision=1,
                                                            target_revision=0, request_id='rollback'))
        assert response.success and response.pose_revision == 2, response.message
        until(lambda: matches(expected_original) and diagnostics.get('cached_pose_revision') == '2' and
              np.allclose(dynamic['matrix'], np.eye(4), atol=1e-9))
        assert diagnostics.get('processing_errors', '0') == '0'
        assert diagnostics.get('queue_dropped', '0') == '0'
        result.update(passed=True, observations=5, submaps=3, resident_submaps=1,
            nonuniform_rebuild=True, evicted_submaps_exported=True, pose_revision=2,
            dynamic_tf=True, static_tf_absent=True, idempotent_retry=True, stale_request_rejected=True,
            rollback=True, covariance_unknown_preserved=True, offline_copy_verified=True,
            original_archive_unchanged=True, ros_distro=os.environ.get('ROS_DISTRO'))
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
