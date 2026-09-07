"""Checked PointCloud2 decoding and bounded exact-timestamp pairing."""
from collections import Counter
import math
import time

import numpy as np

from .core import FIELDS, rigid_pose
from .uncertainty import KINDS, observation_metadata


def stamp_ns(header):
    seconds, nanos = header.stamp.sec, header.stamp.nanosec
    if seconds < 0 or not 0 <= nanos < 1000000000:
        raise ValueError('Invalid measurement timestamp')
    value = seconds * 1000000000 + nanos
    if not 0 < value <= 2 ** 63 - 1:
        raise ValueError('Measurement timestamp must be positive')
    return value


def decode_cloud(message, max_points, max_bytes):
    count = message.width * message.height
    if not 0 < count <= max_points or len(message.data) > max_bytes:
        raise ValueError('Empty or oversized keyframe cloud')
    if (message.point_step <= 0 or message.row_step < message.width * message.point_step or
            len(message.data) != message.height * message.row_step):
        raise ValueError('Invalid PointCloud2 layout')
    types = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}
    names = set()
    fields = {}
    for field in message.fields:
        if field.name in names or field.datatype not in types or field.count <= 0:
            raise ValueError('Duplicate or invalid PointCloud2 field')
        names.add(field.name)
        dtype = np.dtype(('>' if message.is_bigendian else '<') + types[field.datatype])
        if field.offset < 0 or field.offset + field.count * dtype.itemsize > message.point_step:
            raise ValueError('PointCloud2 field exceeds its record')
        if field.name in FIELDS:
            if field.count != 1:
                raise ValueError('Mapping fields must be scalars')
            fields[field.name] = (field.offset, dtype)
    if not {'x', 'y', 'z'} <= set(fields):
        raise ValueError('Keyframe has no valid XYZ fields')
    points = np.full((count, 7), np.nan, dtype='<f4')
    for column, name in enumerate(FIELDS):
        if name in fields:
            offset, dtype = fields[name]
            view = np.ndarray((message.height, message.width), dtype=dtype, buffer=message.data,
                              offset=offset, strides=(message.row_step, message.point_step))
            points[:, column] = view.reshape(-1)
    points = points[np.isfinite(points[:, :3]).all(axis=1)]
    points[:, 3:][~np.isfinite(points[:, 3:])] = np.nan
    if len(points) == 0:
        raise ValueError('Keyframe contains no finite XYZ points')
    return points


def decode_pose(message):
    p, q = message.pose.position, message.pose.orientation
    values = np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float64)
    norm = np.linalg.norm(values[3:])
    if not np.isfinite(values).all() or abs(norm - 1.) > 1e-3:
        raise ValueError('Keyframe pose must have a finite unit quaternion')
    x, y, z, w = values[3:] / norm
    result = np.eye(4)
    result[:3, :3] = [[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                     [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                     [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]]
    result[:3, 3] = values[:3]
    return rigid_pose(result)


def pose_message(matrix):
    """Encode a checked rigid transform, including rotations near pi."""
    from geometry_msgs.msg import Pose
    matrix = rigid_pose(matrix)
    rotation = matrix[:3, :3]
    # Davenport's symmetric quaternion matrix, in [x,y,z,w] order.
    r = rotation
    k = np.array([[r[0, 0]-r[1, 1]-r[2, 2], r[0, 1]+r[1, 0], r[0, 2]+r[2, 0], r[2, 1]-r[1, 2]],
                  [r[0, 1]+r[1, 0], r[1, 1]-r[0, 0]-r[2, 2], r[1, 2]+r[2, 1], r[0, 2]-r[2, 0]],
                  [r[0, 2]+r[2, 0], r[1, 2]+r[2, 1], r[2, 2]-r[0, 0]-r[1, 1], r[1, 0]-r[0, 1]],
                  [r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1], np.trace(r)]])
    _, vectors = np.linalg.eigh(k)
    q = vectors[:, -1]
    if q[3] < 0:
        q = -q
    message = Pose()
    message.position.x, message.position.y, message.position.z = map(float, matrix[:3, 3])
    message.orientation.x, message.orientation.y, message.orientation.z, message.orientation.w = map(float, q)
    return message


def decode_observation(message, odom_frame, base_frame, max_points, max_bytes):
    from types import SimpleNamespace
    stamp = stamp_ns(message.header)
    if (message.header.frame_id != odom_frame or message.cloud.header != message.header or
            message.base_frame_id != base_frame):
        raise ValueError('Observation pose/cloud frames or measurement timestamps differ')
    kind = message.covariance_kind
    if not 0 <= kind < len(KINDS):
        raise ValueError('Unknown observation covariance kind')
    matrix = np.asarray(message.pose_covariance, dtype=np.float64).reshape(6, 6)
    if kind == 0 and not np.array_equal(matrix, np.zeros((6, 6))):
        raise ValueError('Unavailable covariance must have an all-zero wire payload')
    provenance = observation_metadata(dict(source_session_id=message.source_session_id,
        source_sequence=int(message.observation_id), covariance_kind=KINDS[kind],
        covariance_model=message.covariance_model, covariance=None if kind == 0 else matrix,
        quality=dict(registration_converged=message.registration_converged,
                     degenerate_translation_modes=int(message.degenerate_translation_modes),
                     degenerate_rotation_modes=int(message.degenerate_rotation_modes))))
    return (stamp, decode_pose(SimpleNamespace(pose=message.registered_pose)),
            decode_cloud(message.cloud, max_points, max_bytes), provenance)


class Pairer:
    """Called only by the mutually exclusive ROS input callback group.

    A complete newer pair waits for an older pending pair to complete/expire.
    Advancing never silently substitutes a current pose for a missing one.
    """
    def __init__(self, capacity=8, timeout=2., clock=time.monotonic):
        if type(capacity) is not int or not 1 <= capacity <= 128:
            raise ValueError('Pair capacity must be in [1, 128]')
        if not math.isfinite(timeout) or not .01 <= timeout <= 60:
            raise ValueError('Pair timeout must be in [.01, 60] seconds')
        self.capacity, self.timeout, self.clock = capacity, timeout, clock
        self.pending = {}
        self.last_stamp = 0
        self.counts = Counter()

    def push(self, kind, stamp, message):
        if kind not in ('cloud', 'pose'):
            raise ValueError('Unknown pair input')
        self.counts[kind + '_received'] += 1
        if stamp <= self.last_stamp:
            self.counts['late_or_duplicate'] += 1
            return []
        entry = self.pending.setdefault(stamp, {'arrival': self.clock()})
        if kind in entry:
            self.counts['duplicate_pending'] += 1
            return []
        entry[kind] = message
        return self.drain()

    def drain(self):
        ready = []
        while self.pending:
            stamp = min(self.pending)
            entry = self.pending[stamp]
            if 'cloud' in entry and 'pose' in entry:
                ready.append((stamp, entry['pose'], entry['cloud']))
                self.last_stamp = stamp
                self.counts['paired'] += 1
            elif len(self.pending) > self.capacity or self.clock() - entry['arrival'] >= self.timeout:
                self.counts['missing_cloud' if 'cloud' not in entry else 'missing_pose'] += 1
                self.last_stamp = stamp
            else:
                break
            del self.pending[stamp]
        self.counts['pending_peak'] = max(self.counts['pending_peak'], len(self.pending))
        return ready
