#!/usr/bin/env python3
"""Exercise real ROS loop injection, GTSAM, map reconstruction, and factor removal.

Use a dedicated ROS_DOMAIN_ID and the installed dliio environment. This is a
known-geometry synthetic check, not a test of place recognition or calibration.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from types import SimpleNamespace
import uuid

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import PointCloud2
from tf2_msgs.msg import TFMessage
from direct_lidar_inertial_odometry.msg import MappingObservation
from direct_lidar_inertial_odometry.srv import MapArchive, UpdatePoseGraph
from dliio_mapping.core import Limits, Store, decode_pose as archived_pose
from dliio_mapping.input import decode_cloud, decode_pose, pose_message
from dliio_mapping.node import cloud_message


def pose(x=0., yaw=0.):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0, x], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.]])


def placed(points, matrix):
    result = points.copy()
    result[:, :3] = points[:, :3] @ matrix[:3, :3].T + matrix[:3, 3]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run = args.output.parent / ('pose-graph-' + uuid.uuid4().hex[:8])
    run.mkdir()
    result = dict(passed=False, artifacts=str(run))
    args.output.write_text(json.dumps(result) + '\n')
    rclpy.init()
    node = rclpy.create_node('dliio_pose_graph_check')
    diagnostics, received, dynamic, static = {}, {}, {}, []
    process = None

    def until(predicate, timeout=30.):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.02)
            if process is not None and process.poll() is not None:
                raise RuntimeError(f'Mapper exited; inspect {run / "mapper.log"}')
            if predicate(): return
        raise RuntimeError('Timed out waiting for pose graph integration result')

    def diagnostic(message):
        for status in message.status:
            if status.name == 'DLIO Mapping':
                diagnostics.update({v.key: v.value for v in status.values})

    def transform(message, output):
        for value in message.transforms:
            if value.header.frame_id == 'map' and value.child_frame_id == 'odom':
                if isinstance(output, list):
                    output.append(value)
                else:
                    p = pose_message(pose())
                    p.position.x, p.position.y, p.position.z = (value.transform.translation.x,
                        value.transform.translation.y, value.transform.translation.z)
                    p.orientation = value.transform.rotation
                    output['matrix'] = decode_pose(SimpleNamespace(pose=p))

    def call(client, request):
        until(client.service_is_ready)
        future = client.call_async(request)
        until(future.done)
        return future.result()

    def matches(expected):
        if 'map' not in received: return False
        actual = decode_cloud(received['map'], 50000, 2000000)
        return actual.shape == expected.shape and np.allclose(actual, expected, atol=1e-5, equal_nan=True)

    try:
        start = time.monotonic()
        until(lambda: time.monotonic() - start > .5)
        if 'dlio_mapping_node' in node.get_node_names():
            raise RuntimeError('Mapper already present; use a dedicated ROS_DOMAIN_ID')
        subscriptions = [
            node.create_subscription(DiagnosticArray, '/dlio/mapping/diagnostics', diagnostic, 10),
            node.create_subscription(PointCloud2, '/graph_check/map', lambda m: received.update(map=m),
                QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)),
            node.create_subscription(TFMessage, '/tf', lambda m: transform(m, dynamic), 50),
            node.create_subscription(TFMessage, '/tf_static', lambda m: transform(m, static),
                QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL))]
        publisher = node.create_publisher(MappingObservation, '/graph_check/observation', 8)
        graph = node.create_client(UpdatePoseGraph, '/dlio/mapping/update_pose_graph')
        save = node.create_client(MapArchive, '/dlio/mapping/save_map')
        command = ['ros2', 'run', 'direct_lidar_inertial_odometry', 'dlio_mapping_node.py', '--ros-args',
            '-p', 'mapping/transport:=observation', '-p', f'mapping/storage_directory:={run / "archive"}',
            '-p', 'mapping/submap_keyframes:=2', '-p', 'mapping/resident_submaps:=1',
            '-p', 'mapping/fusion_size:=0.0', '-p', 'mapping/publish_rate:=10.0',
            '-r', 'observation:=/graph_check/observation', '-r', 'map:=/graph_check/map']
        with (run / 'mapper.log').open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        until(lambda: publisher.get_subscription_count() > 0 and 'session_id' in diagnostics)
        session, source = diagnostics['session_id'], str(uuid.uuid4())
        rng, surfaces = np.random.default_rng(77), []
        for axis, size in enumerate((4., 3., 2.)):
            for sign in (-1, 1):
                xyz = rng.uniform([-4, -3, -2], [4, 3, 2], (600, 3))
                xyz[:, axis] = size * sign
                surfaces.append(xyz)
        xyz = np.vstack(surfaces)
        points = np.column_stack((xyz, rng.uniform(0, 100, (len(xyz), 4)))).astype('<f4')
        points[::9, 4] = np.nan
        originals = []
        for i, x in enumerate([0., 2., 3., 1., 0.]):
            original = pose(x + i*.2, i*.015)
            originals.append(original)
            local = placed(points, np.linalg.inv(pose(x)))
            cloud = cloud_message(placed(local, original), (10+i)*1000000000, 'odom')
            publisher.publish(MappingObservation(header=cloud.header, cloud=cloud,
                source_session_id=source, observation_id=i, base_frame_id='base_link',
                registered_pose=pose_message(original), covariance_kind=MappingObservation.UNKNOWN,
                covariance_model='synthetic registered-pose uncertainty unavailable', registration_converged=True))
            until(lambda: diagnostics.get('keyframes') == str(i+1))
        expected_original = placed(points, originals[-1])
        until(lambda: matches(expected_original))
        original_path = run / 'original.dliomap'
        assert call(save, MapArchive.Request(path=str(original_path))).success
        checksum = hashlib.sha256(original_path.read_bytes()).hexdigest()

        config = dict(odometry_noise=dict(kind='assumed', model='synthetic independent relative increments',
            covariance_floor=np.diag([.2**2]*3 + [.08**2]*3).tolist(),
            covariance_per_second=np.diag([.05**2]*3 + [.01**2]*3).tolist(),
            covariance_per_metre=np.zeros((6, 6)).tolist(), max_gap_seconds=2.),
            validation=dict(min_separation_observations=2, min_separation_seconds=1.))
        initialize = dict(session_id=session, expected_revision=0, request_id='initialize', action='initialize', configuration=config)
        response = call(graph, UpdatePoseGraph.Request(request_json=json.dumps(initialize)))
        assert response.success and response.pose_revision == 1, response.message
        candidate = dict(id='known-revisit', from_id=0, to_id=4, transform=pose().tolist(),
            covariance=np.diag([.03**2]*3 + [.02**2]*3).tolist(), covariance_kind='assumed',
            covariance_model='synthetic relative loop covariance', provenance='Known same-room identity revisit in generated fixture')
        request = dict(session_id=session, expected_revision=1, request_id='bad-cycle', action='add_loop',
                       loop=dict(candidate, transform=pose(10.).tolist(), covariance=(np.eye(6)*1e6).tolist()))
        response = call(graph, UpdatePoseGraph.Request(request_json=json.dumps(request)))
        assert not response.success and response.pose_revision == 1
        assert json.loads(response.report_json)['reason'] == 'absolute_cycle_limit'
        request.update(request_id='known-revisit', loop=candidate)
        response = call(graph, UpdatePoseGraph.Request(request_json=json.dumps(request)))
        assert response.success and response.pose_revision == 2, response.message
        accepted = json.loads(response.report_json)
        retry = call(graph, UpdatePoseGraph.Request(request_json=json.dumps(request)))
        assert retry.success and retry.report_json == response.report_json
        stale = call(graph, UpdatePoseGraph.Request(request_json=json.dumps(dict(request, request_id='stale'))))
        assert not stale.success and not stale.report_json
        corrected_path = run / 'corrected.dliomap'
        assert call(save, MapArchive.Request(path=str(corrected_path))).success
        loaded = Store(corrected_path, Limits(resident_submaps=1))
        try:
            corrected = archived_pose(loaded.db.execute('SELECT pose FROM optimized_poses WHERE id=4').fetchone()[0])
            assert np.linalg.norm(corrected[:3, 3]) < .02
            expected, correction = loaded.snapshot(), corrected @ np.linalg.inv(originals[-1])
            until(lambda: matches(expected) and diagnostics.get('cached_pose_revision') == '2' and
                  'matrix' in dynamic and np.allclose(dynamic['matrix'], correction, atol=1e-9))
            assert loaded.stats()['active_loops'] == 1 and loaded.stats()['graph_attached']
            assert all(json.loads(row[0])['covariance'] is None for row in loaded.db.execute('SELECT json FROM observations'))
        finally:
            loaded.close()
        removal = dict(session_id=session, expected_revision=2, request_id='remove', action='remove_loop',
                       loop_id=candidate['id'], reason='synthetic factor removal and reconstruction check')
        response = call(graph, UpdatePoseGraph.Request(request_json=json.dumps(removal)))
        assert response.success and response.pose_revision == 3, response.message
        until(lambda: matches(expected_original) and diagnostics.get('cached_pose_revision') == '3' and
              diagnostics.get('active_loops') == '0' and np.allclose(dynamic['matrix'], pose(), atol=1e-9))
        assert not static
        assert diagnostics.get('processing_errors', '0') == '0'
        assert diagnostics.get('queue_dropped', '0') == '0'
        assert hashlib.sha256(original_path.read_bytes()).hexdigest() == checksum
        result.update(passed=True, observations=5, submaps=3, resident_submaps=1, pose_revision=3,
            accepted_loop=accepted, inflated_covariance_did_not_bypass_cycle=True, idempotent_retry=True,
            stale_request_rejected=True, corrected_map_and_tf=True, factor_removal_reconstructed_original=True,
            unknown_covariance_preserved=True, original_archive_unchanged=True, ros_distro=os.environ.get('ROS_DISTRO'))
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
