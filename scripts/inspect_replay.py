#!/usr/bin/env python3
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, PointCloud2
from tf2_msgs.msg import TFMessage

ap = argparse.ArgumentParser(description="Sample the local 0705 replay topics and diagnostics")
ap.add_argument("--seconds", type=float, default=15)
ap.add_argument("--output", type=Path, default=Path.cwd() / "dliio_run/probe.json")
args = ap.parse_args()
rclpy.init()
node = rclpy.create_node('dliio_0705_probe')
counts = Counter()
latest = {}
transforms = set()
poses = []
stamps = {name: set() for name in ['deskewed', 'scan_pose', 'keyframe', 'keyframe_pose', 'odom', 'tf']}


def collect(name, msg):
    counts[name] += 1
    latest[name] = msg
    if name in stamps:
        stamps[name].add(msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec)
    if name == 'odom':
        p = msg.pose.pose.position
        poses.append([p.x, p.y, p.z])


subscriptions = []
for name, topic, typ in [
    ('raw', '/ouster/points', PointCloud2),
    ('deskewed', '/dlio/odom_node/pointcloud/deskewed', PointCloud2),
    ('scan_pose', '/dlio/odom_node/scan_pose', PoseStamped),
    ('keyframe', '/dlio/odom_node/pointcloud/keyframe', PointCloud2),
    ('keyframe_pose', '/dlio/odom_node/keyframe_pose', PoseStamped),
    ('imu', '/ouster/imu', Imu),
    ('odom', '/dlio/odom_node/odom', Odometry),
    ('diagnostics', '/diagnostics', DiagnosticArray),
    ('clock', '/clock', Clock),
]:
    subscriptions.append(node.create_subscription(
        typ, topic, lambda msg, name=name: collect(name, msg), qos_profile_sensor_data))
subscriptions.append(node.create_subscription(
    PointCloud2, '/dlio/map_node/map', lambda msg: collect('map', msg),
    QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)))
def collect_tf(msg):
    transforms.update((t.header.frame_id, t.child_frame_id) for t in msg.transforms)
    for t in msg.transforms:
        if t.header.frame_id == 'odom' and t.child_frame_id == 'dliio_base_link':
            stamps['tf'].add(t.header.stamp.sec * 1000000000 + t.header.stamp.nanosec)


for topic, qos in [('/tf', qos_profile_sensor_data),
                   ('/tf_static', QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL))]:
    subscriptions.append(node.create_subscription(TFMessage, topic,
        collect_tf, qos))
start = time.monotonic()
while time.monotonic() - start < args.seconds:
    rclpy.spin_once(node, timeout_sec=0.1)
result = {'duration_seconds': args.seconds, 'counts': dict(counts), 'transforms': sorted(transforms),
          'nodes': sorted(node.get_node_names())}
result['timestamp_pairs'] = {
    f'{left}/{right}': {'matched': len(stamps[left] & stamps[right]),
                       'left_received': len(stamps[left]), 'right_received': len(stamps[right])}
    for left, right in [('deskewed', 'scan_pose'), ('keyframe', 'keyframe_pose'), ('odom', 'tf')]
}
types = {1:'i1', 2:'u1', 3:'i2', 4:'u2', 5:'i4', 6:'u4', 7:'f4', 8:'f8'}
for name in ['raw', 'deskewed', 'keyframe', 'map']:
    if name not in latest:
        continue
    msg = latest[name]
    info = {'points': msg.width * msg.height, 'frame': msg.header.frame_id,
            'stamp': msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9,
            'fields': [f.name for f in msg.fields]}
    for field in msg.fields:
        if field.name not in ['intensity', 'reflectivity', 'intensity_corrected', 'lidar_intensity']:
            continue
        values = np.ndarray((msg.height, msg.width),
            dtype=('>' if msg.is_bigendian else '<') + types[field.datatype],
            buffer=msg.data, offset=field.offset, strides=(msg.row_step, msg.point_step))
        values = values[np.isfinite(values)]
        if values.size:
            info[field.name] = dict(zip(['min', 'median', 'max'], np.quantile(values, [0, .5, 1]).tolist()))
    result[name] = info
if poses:
    result['pose_first'] = poses[0]
    result['pose_last'] = poses[-1]
    result['max_abs_position'] = float(np.abs(poses).max())
if 'diagnostics' in latest:
    result['diagnostics'] = [{
        'name': status.name, 'level': int.from_bytes(status.level, 'little') if isinstance(status.level, bytes) else status.level,
        'message': status.message,
        'values': {kv.key: kv.value for kv in status.values
                   if any(word in kv.key.lower() for word in ['channel', 'field', 'photometric', 'scan', 'drop', 'lag', 'imu', 'flow', 'observer', 'realtime', 'comput', 'degen', 'held', 'rescued'])}
    } for status in latest['diagnostics'].status]
if 'clock' in latest:
    c = latest['clock'].clock
    result['bag_clock'] = c.sec + c.nanosec / 1e9
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
node.destroy_node()
rclpy.shutdown()
