"""ROS wire layout and exact pairing at sensor-message boundaries."""
from array import array
from pathlib import Path
import struct
import sys

import numpy as np
import pytest
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2, PointField

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from dliio_mapping.input import Pairer, decode_cloud, decode_pose, decode_observation, pose_message, stamp_ns


@pytest.mark.parametrize('rotation', [np.eye(3), np.diag([1., -1., -1.]), np.diag([-1., 1., -1.]),
                                    np.diag([-1., -1., 1.]), np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])])
def test_pose_encoding_handles_identity_quarter_turn_and_pi(rotation):
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, [3., 4., 5.]
    message = PoseStamped(pose=pose_message(matrix))
    np.testing.assert_allclose(decode_pose(message), matrix, atol=1e-12)


def observation():
    from types import SimpleNamespace
    from copy import deepcopy
    import uuid
    cloud = wire_cloud()
    cloud.header.frame_id, cloud.header.stamp.sec = 'odom', 12
    return SimpleNamespace(header=deepcopy(cloud.header), cloud=cloud,
        base_frame_id='base', source_session_id=str(uuid.uuid4()), observation_id=4,
        registered_pose=pose_message(np.eye(4)), covariance_kind=0,
        covariance_model='unavailable', pose_covariance=[0.]*36, registration_converged=True,
        degenerate_translation_modes=1, degenerate_rotation_modes=0)


def test_atomic_observation_carries_exact_pose_cloud_quality_and_unknown_covariance():
    message = observation()
    stamp, pose, points, provenance = decode_observation(message, 'odom', 'base', 10, 1000)
    assert stamp == 12000000000 and len(points) == 4
    np.testing.assert_array_equal(pose, np.eye(4))
    assert provenance['source_sequence'] == 4
    assert provenance['covariance'] is None
    assert provenance['quality']['degenerate_translation_modes'] == 1


@pytest.mark.parametrize('mutation', ['stamp', 'cloud_frame', 'pose_frame', 'base_frame', 'covariance', 'kind'])
def test_atomic_observation_rejects_inconsistent_frames_times_and_uncertainty(mutation):
    message = observation()
    if mutation == 'stamp':
        message.cloud.header.stamp.nanosec += 1
    elif mutation == 'cloud_frame':
        message.cloud.header.frame_id = 'lidar'
    elif mutation == 'pose_frame':
        message.header.frame_id = 'map'
    elif mutation == 'base_frame':
        message.base_frame_id = 'lidar'
    elif mutation == 'covariance':
        message.pose_covariance[0] = 1.
    else:
        message.covariance_kind = 5
    with pytest.raises(ValueError):
        decode_observation(message, 'odom', 'base', 10, 1000)


def wire_cloud(big_endian=False):
    message = PointCloud2(height=2, width=2, point_step=16, row_step=36, is_bigendian=big_endian)
    message.fields = [PointField(name=name, offset=i * 4, datatype=7, count=1)
                      for i, name in enumerate(('x', 'y', 'z'))]
    message.fields.append(PointField(name='reflectivity', offset=12, datatype=4, count=1))
    data = bytearray(72)
    for i in range(4):
        struct.pack_into(('>' if big_endian else '<') + 'fffH', data, (i // 2) * 36 + (i % 2) * 16,
                         float(i), float(i + 10), float(i + 20), i + 100)
    message.data = array('B', data)
    return message


@pytest.mark.parametrize('big_endian', [False, True])
def test_padded_rows_endian_and_missing_channels(big_endian):
    result = decode_cloud(wire_cloud(big_endian), 10, 1000)
    np.testing.assert_array_equal(result[:, :3], [[0, 10, 20], [1, 11, 21], [2, 12, 22], [3, 13, 23]])
    np.testing.assert_array_equal(result[:, 4], [100, 101, 102, 103])
    assert np.isnan(result[:, [3, 5, 6]]).all()


@pytest.mark.parametrize('mutation', ['short', 'row', 'field_extent', 'duplicate', 'vector', 'no_x'])
def test_invalid_layout_rejected(mutation):
    message = wire_cloud()
    if mutation == 'short':
        message.data.pop()
    elif mutation == 'row':
        message.row_step = 10
    elif mutation == 'field_extent':
        message.fields[0].offset = 100
    elif mutation == 'duplicate':
        message.fields[0].name = 'y'
    elif mutation == 'vector':
        message.fields[0].count = 2
    elif mutation == 'no_x':
        message.fields[0].name = 'other'
    with pytest.raises(ValueError):
        decode_cloud(message, 10, 1000)


def test_nonfinite_xyz_removed_and_channels_marked_missing():
    message = wire_cloud()
    struct.pack_into('<f', message.data, 0, float('nan'))
    message.fields[3] = PointField(name='reflectivity', offset=12, datatype=7, count=1)
    for offset in (12, 28, 48, 64):
        struct.pack_into('<f', message.data, offset, float('inf'))
    result = decode_cloud(message, 10, 1000)
    assert len(result) == 3
    assert np.isnan(result[:, 4]).all()


def test_input_caps_checked_before_decoding():
    with pytest.raises(ValueError, match='oversized'):
        decode_cloud(wire_cloud(), 3, 1000)
    with pytest.raises(ValueError, match='oversized'):
        decode_cloud(wire_cloud(), 10, 71)


def test_quaternion_and_integer_nanosecond_timestamp():
    message = PoseStamped()
    message.header.stamp.sec, message.header.stamp.nanosec = 1800000000, 123456789
    message.pose.position.x = 10.
    message.pose.orientation.z = message.pose.orientation.w = 2 ** -.5
    assert stamp_ns(message.header) == 1800000000123456789
    np.testing.assert_allclose(decode_pose(message), [[0, -1, 0, 10], [1, 0, 0, 0],
                                                     [0, 0, 1, 0], [0, 0, 0, 1]], atol=1e-15)


@pytest.mark.parametrize('quaternion', [(0, 0, 0, 0), (0, 0, 0, 2), (0, 0, 0, float('nan'))])
def test_invalid_quaternion_rejected(quaternion):
    message = PoseStamped()
    q = message.pose.orientation
    q.x, q.y, q.z, q.w = map(float, quaternion)
    with pytest.raises(ValueError):
        decode_pose(message)


def test_pairer_waits_for_older_pair_and_keeps_exact_identity():
    pairer = Pairer()
    assert pairer.push('cloud', 10, 'cloud10') == []
    assert pairer.push('pose', 20, 'pose20') == []
    assert pairer.push('cloud', 20, 'cloud20') == []
    assert pairer.push('pose', 10, 'pose10') == [(10, 'pose10', 'cloud10'), (20, 'pose20', 'cloud20')]
    assert pairer.push('cloud', 10, 'duplicate') == []
    assert pairer.counts['late_or_duplicate'] == 1


def test_pair_timeout_uses_wall_clock_and_cannot_substitute_pose():
    clock = [0.]
    pairer = Pairer(timeout=1., clock=lambda: clock[0])
    pairer.push('cloud', 10, 'cloud10')
    pairer.push('cloud', 20, 'cloud20')
    pairer.push('pose', 20, 'pose20')
    clock[0] = 2.
    assert pairer.drain() == [(20, 'pose20', 'cloud20')]
    assert pairer.counts['missing_pose'] == 1
    assert pairer.push('pose', 10, 'late10') == []


def test_pair_capacity_and_duplicate_messages_remain_bounded():
    pairer = Pairer(capacity=2)
    for stamp in range(1, 100):
        pairer.push('pose', stamp, str(stamp))
        pairer.push('pose', stamp, 'duplicate')
        assert len(pairer.pending) <= 2
    assert pairer.counts['missing_cloud'] == 97
    assert pairer.counts['duplicate_pending'] == 99
    assert pairer.push('cloud', 98, 'c98') == [(98, '98', 'c98')]
