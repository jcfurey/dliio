#!/usr/bin/env python3
"""Check persistent mapping against a real 0705 Ouster packet replay at 1x."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rosbag2_interfaces.srv import Pause
from direct_lidar_inertial_odometry.srv import MapArchive
from dliio_mapping.core import FIELDS, Limits, Store, decode_points
from dliio_mapping.input import decode_cloud


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path)
    parser.add_argument('--seconds', type=float, default=75., help='Sample duration after the first scan')
    parser.add_argument('--mapping-input', choices=('observations', 'keyframes'), default='observations')
    parser.add_argument('--resident-submaps', type=int, default=8)
    parser.add_argument('--submap-keyframes', type=int, default=100)
    parser.add_argument('--require-eviction', action='store_true',
                        help='Fail unless the replay exceeds the resident submap window')
    parser.add_argument('--full', action='store_true', help='Wait for the complete bag player to exit')
    parser.add_argument('--rviz', action='store_true')
    parser.add_argument('--keep-running', action='store_true', help='Keep the successful launch open for viewing')
    parser.add_argument('--output', type=Path, default=Path.cwd() / 'dliio_run/mapping-replay.json')
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds < 10:
        parser.error('--seconds must be finite and at least 10')
    try:
        limits = Limits(resident_submaps=args.resident_submaps, submap_keyframes=args.submap_keyframes)
    except ValueError as error:
        parser.error(str(error))
    args.output = args.output.resolve()
    run = args.output.parent
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'saved.dliomap').exists():
        parser.error('Use a new output directory for each replay')
    args.output.write_text(json.dumps({'passed': False, 'status': 'starting'}) + '\n')
    rclpy.init()
    node = rclpy.create_node('dlio_mapping_replay_check')
    process = None
    passed = False
    frame = 'mapping' if args.mapping_input == 'observations' else 'keyframe'
    pose_name = 'mapping_pose' if frame == 'mapping' else 'keyframe_pose'
    counts, mapping, odometry = Counter(), {}, {}
    stamps = {name: set() for name in ('scan_pose', 'deskewed', 'keyframe', 'keyframe_pose', 'mapping', 'mapping_pose')}
    latest_map = [None]
    first_scan = [None]
    map_peak = [0]
    point_counts = {name: {} for name in ('deskewed', 'keyframe', 'mapping')}
    field_hashes = {}
    started = time.monotonic()
    history = (run / 'mapping-diagnostics.jsonl').open('w')
    peaks = {}

    def stage(name):
        progress = dict(stage=name, elapsed_seconds=time.monotonic() - started)
        args.output.with_suffix('.progress.json').write_text(json.dumps(progress) + '\n')
        print(json.dumps(progress), flush=True)

    def collect(name, message):
        counts[name] += 1
        stamp = message.header.stamp.sec * 1000000000 + message.header.stamp.nanosec
        stamps[name].add(stamp)
        if name in point_counts:
            point_counts[name][stamp] = message.width * message.height
        if name == frame:
            points = decode_cloud(message, Limits().max_input_points, 16777216)
            field_hashes[stamp] = hashlib.sha256(np.ascontiguousarray(points[:, 3:]).tobytes()).hexdigest()
        if name == 'scan_pose' and first_scan[0] is None:
            first_scan[0] = time.monotonic()

    def cloud(message):
        latest_map[0] = message
        map_peak[0] = max(map_peak[0], message.width * message.height)
        counts['map'] += 1

    def diagnostic(target, message):
        for status in message.status:
            target.update({value.key: value.value for value in status.values})
        if target is mapping:
            history.write(json.dumps(dict(elapsed_seconds=time.monotonic() - started,
                                          values=dict(target))) + '\n')
            history.flush()
            for name in ('resident_submaps', 'resident_points', 'resident_array_bytes',
                         'fusion_array_bytes', 'cached_map_bytes', 'published_points',
                         'rss_peak_kib', 'queue_depth', 'pending_pairs'):
                if name in target:
                    peaks[name] = max(peaks.get(name, 0), int(target[name]))

    def until(predicate, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.025)
            if process and process.poll() is not None:
                raise RuntimeError('Replay launch exited early; inspect its log')
            if predicate():
                return
        raise RuntimeError('Replay check timed out')

    def request(operation, destination):
        client = node.create_client(MapArchive, '/dlio/mapping/' + operation)
        until(client.service_is_ready, 10)
        future = client.call_async(MapArchive.Request(path=str(destination)))
        until(future.done, 65)
        response = future.result()
        node.destroy_client(client)
        assert response.success, response.message
        return response

    subscriptions = []
    for name, topic, kind in [
        ('scan_pose', '/dlio/odom_node/scan_pose', PoseStamped),
        ('deskewed', '/dlio/odom_node/pointcloud/deskewed', PointCloud2),
        ('keyframe', '/dlio/odom_node/pointcloud/keyframe', PointCloud2),
        ('keyframe_pose', '/dlio/odom_node/keyframe_pose', PoseStamped),
        ('mapping', '/dlio/odom_node/pointcloud/mapping', PointCloud2),
        ('mapping_pose', '/dlio/odom_node/mapping_pose', PoseStamped)]:
        subscriptions.append(node.create_subscription(kind, topic,
            lambda message, name=name: collect(name, message), QoSProfile(depth=20)))
    subscriptions += [
        node.create_subscription(PointCloud2, '/dlio/map_node/map', cloud,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)),
        node.create_subscription(DiagnosticArray, '/dlio/mapping/diagnostics',
            lambda message: diagnostic(mapping, message), 10),
        node.create_subscription(DiagnosticArray, '/diagnostics',
            lambda message: diagnostic(odometry, message), qos_profile_sensor_data)]
    config = run / 'mapping-check.yaml'
    config.write_text('/**:\n  ros__parameters:\n'
                      f'    mapping/submap_keyframes: {args.submap_keyframes}\n'
                      f'    mapping/resident_submaps: {args.resident_submaps}\n')
    try:
        discovery = time.monotonic() + 1
        while time.monotonic() < discovery:
            rclpy.spin_once(node, timeout_sec=.05)
        if {'dlio_odom_node', 'dlio_mapping_node'} & set(node.get_node_names()):
            raise RuntimeError('dliio already running; use a separate ROS_DOMAIN_ID')
        with args.output.with_suffix('.log').open('w') as log:
            stage('starting')
            process = subprocess.Popen(['ros2', 'launch', 'direct_lidar_inertial_odometry',
                'dlio_mapping.launch.py', 'mode:=packets', 'profile:=0705', 'rate:=1.0',
                f'bag:={args.bag.resolve()}', f'run_dir:={run}', f'mapping_config:={config}',
                f'mapping_input:={args.mapping_input}', 'rviz:=' + str(args.rviz).lower()],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            (run / 'launch.pid').write_text(str(process.pid) + '\n')
            until(lambda: first_scan[0] is not None and 'session_id' in mapping, 45)
            stage('playing')
            if args.full:
                # Launch logs the player's completion on all supported distros.
                metadata = (args.bag / 'metadata.yaml').read_text()
                duration = int(re.search(r'duration:\s+nanoseconds:\s+(\d+)', metadata)[1]) / 1e9
                until(lambda: re.search(r'\[ros2-\d+\]: process has finished cleanly',
                                        args.output.with_suffix('.log').read_text()), duration + 40)
            else:
                until(lambda: time.monotonic() - first_scan[0] >= args.seconds, args.seconds + 10)
                # Export runs on the bounded storage worker. Pause only after
                # sampling at full speed, so export cannot overflow its queue.
                pause = node.create_client(Pause, '/rosbag2_player/pause')
                until(pause.service_is_ready, 5)
                paused = pause.call_async(Pause.Request())
                until(paused.done, 5)
                paused.result()
                node.destroy_client(pause)
                until(lambda: int(mapping.get('keyframes', 0)) == len(stamps[frame]) and
                      int(mapping.get('pending_pairs', -1)) == 0 and int(mapping.get('queue_depth', -1)) == 0, 15)
            if args.full:
                until(lambda: int(mapping.get('keyframes', 0)) == len(stamps[frame]) and
                      int(mapping.get('pending_pairs', -1)) == 0 and int(mapping.get('queue_depth', -1)) == 0, 15)
            required_stamps = set(stamps[frame])
            required_scan_stamps = set(stamps['scan_pose'])
            stage('draining')
            # Freeze the live sample boundary before waiting for separately
            # delivered clouds. Checking a continuously advancing pose set at
            # service completion can mistake an in-flight cloud for a loss.
            until(lambda: required_scan_stamps <= stamps['deskewed'], 10)
            # Wait for the sampled boundary to commit before enqueuing save.
            # Separate cloud/pose callbacks and 1 Hz diagnostics can lag the
            # probe even when no input is lost.
            until(lambda: int(mapping.get('last_stamp_ns', 0)) >= max(required_stamps), 15)
            stage('saving')
            response = request('save_map', run / 'saved.dliomap')
            stage('exporting')
            request('export_pcd', run / 'all.pcd')
            stage('validating_archive')
            archived = Store(run / 'saved.dliomap', limits)
            try:
                archived_stamps = {row[0] for row in archived.db.execute('SELECT stamp_ns FROM keyframes')}
                assert archived_stamps <= stamps[frame] & stamps[pose_name]
                assert required_stamps <= archived_stamps
                if args.full:
                    assert archived_stamps == stamps[frame] == stamps[pose_name]
                assert archived.meta['calibration']['frames/baselink'] == 'dliio_base_link'
                assert archived.meta['keyframes'] == response.keyframes > 0
                assert archived.meta['submaps'] > 0
                if args.require_eviction:
                    assert archived.meta['submaps'] > args.resident_submaps, 'Replay did not exercise eviction'
                assert archived.stats()['resident_submaps'] <= args.resident_submaps
                assert peaks['resident_submaps'] <= args.resident_submaps
                assert len(archived.snapshot()) <= args.resident_submaps * archived.limits.max_voxels
                assert archived.limits.voxel_size == 0., 'Default mapping must retain individual returns'
                archived_input_points = 0
                for stamp, count, blob, crc in archived.db.execute('SELECT stamp_ns,count,points,crc FROM keyframes'):
                    assert count == point_counts[frame][stamp]
                    points = decode_points(blob, count, crc, archived.limits.max_input_points)
                    assert hashlib.sha256(np.ascontiguousarray(points[:, 3:]).tobytes()).hexdigest() == field_hashes[stamp]
                    archived_input_points += count
                fusion_size = float(mapping['fusion_size'])
                assert fusion_size > 0, 'Default output must fuse overlapping keyframes'
                export_points = archived.export_pcd(run / 'reloaded.pcd', fusion_size)
                assert export_points < archived_input_points, 'Output did not co-locate overlapping returns'
                assert export_points >= len(archived.snapshot(fusion_size))
                raw_points = archived.export_pcd(run / 'raw.pcd')
                assert raw_points == archived_input_points, 'Dense source samples were lost'
                archive_stats = archived.stats()
            finally:
                archived.close()
            def digest(path):
                result = hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1048576), b''):
                        result.update(chunk)
                return result.hexdigest()
            # During a partial live sample, later keyframes can arrive between
            # save and export requests. Compare complete exports only at bag end.
            if args.full:
                assert digest(run / 'all.pcd') == digest(run / 'reloaded.pcd')
            msg = latest_map[0]
            assert msg is not None and msg.header.frame_id == 'map'
            assert tuple(field.name for field in msg.fields) == FIELDS
            assert 'Avg Computation Time (ms)' in odometry, 'Missing frontend diagnostics'
            until(lambda: int(mapping.get('keyframes', 0)) >= response.keyframes, 5)
            for name in ('missing_pose', 'missing_cloud', 'queue_dropped', 'processing_errors',
                         'invalid_input', 'late_or_duplicate', 'duplicate_pending'):
                assert int(mapping.get(name, 0)) == 0, (name, mapping.get(name))
            assert counts['scan_pose'] >= (duration - 5 if args.full else args.seconds) * 9
            assert required_scan_stamps <= stamps['deskewed']
            if args.full:
                assert stamps['scan_pose'] <= stamps['deskewed']
            shared = point_counts['deskewed'].keys() & point_counts[frame].keys()
            ratios = [point_counts[frame][stamp] / point_counts['deskewed'][stamp] for stamp in shared]
            assert ratios and min(ratios) > 1., '0705 mapping still receives the registration cloud'
            density = {name: dict(min=min(values.values()), median=float(np.median(list(values.values()))),
                                 max=max(values.values())) for name, values in point_counts.items() if values}
            density['paired_dense_to_registration_ratio'] = dict(min=min(ratios), median=float(np.median(ratios)),
                                                                 max=max(ratios), pairs=len(ratios))
            result = dict(passed=True, ros_distro=os.environ.get('ROS_DISTRO'), rate=1., full_bag=args.full,
                density=density, archived_input_points=archived_input_points, fusion_size=fusion_size,
                configured_limits=dict(resident_submaps=args.resident_submaps,
                    submap_keyframes=args.submap_keyframes), mapping_peaks=peaks,
                eviction_exercised=archive_stats['submaps'] > args.resident_submaps,
                sampled_scan_pairs=len(required_scan_stamps), mapping_input=args.mapping_input,
                counts=dict(counts), timestamp_pairs={name: len(values) for name, values in stamps.items()},
                archive=archive_stats, mapping_diagnostics=mapping, odometry_diagnostics=odometry,
                exported_points=export_points, map_peak_points=map_peak[0],
                reloaded_export_sha256=digest(run / 'reloaded.pcd'), nodes=sorted(node.get_node_names()),
                launch_pid=process.pid, rviz=args.rviz)
            args.output.write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(result, indent=2))
            passed = True
            stage('passed')
    except BaseException as error:
        args.output.write_text(json.dumps(dict(passed=False, error=str(error),
            counts=dict(counts), mapping_diagnostics=mapping, odometry_diagnostics=odometry,
            mapping_peaks=peaks,
            missing_scan_cloud_stamps=sorted(stamps['scan_pose'] - stamps['deskewed']),
            missing_mapping_cloud_stamps=sorted(stamps[pose_name] - stamps[frame]),
            missing_mapping_pose_stamps=sorted(stamps[frame] - stamps[pose_name])), indent=2) + '\n')
        raise
    finally:
        history.close()
        if process and process.poll() is None and not (passed and args.keep_running):
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
