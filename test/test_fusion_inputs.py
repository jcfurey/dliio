"""Tilt preparation preserves 3D measurements, frames and covariance meaning."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from dliio_fusion.inputs import StampGate, pose_covariance, tilt_quaternion


@pytest.mark.parametrize('yaw', [-179., -90., 0., 90., 179.])
def test_tilt_is_heading_invariant_and_reproduces_world_up(yaw):
    matrix = Rotation.from_euler('xyz', [12., -8., yaw], degrees=True).as_matrix()
    result = Rotation.from_quat(tilt_quaternion(matrix[2]))
    np.testing.assert_allclose(result.as_matrix()[2], matrix[2], atol=1.e-14)
    np.testing.assert_allclose(result.as_euler('xyz', degrees=True), [12., -8., 0.], atol=1.e-12)


@pytest.mark.parametrize('direction', [[0, 0, 9.81], [0, 0, 0], [np.nan, 0, 1], [1, 0, 0], [0, 1]])
def test_invalid_or_singular_tilt_is_rejected(direction):
    with pytest.raises(ValueError):
        tilt_quaternion(direction)


def test_published_covariance_keeps_correlations_and_assumed_policy_is_explicit():
    expected = np.eye(6)*.1
    expected[0, 4] = expected[4, 0] = -.01
    np.testing.assert_array_equal(pose_covariance(expected.ravel(), 'published', .3, .1), expected)
    assumed = pose_covariance(np.zeros(36), 'assumed', .3, .1)
    np.testing.assert_allclose(np.diag(assumed), [.09]*3+[.01]*3)
    for invalid in (np.zeros(36), np.full(36, np.nan), np.diag([-1., 1, 1, 1, 1, 1])):
        with pytest.raises(ValueError):
            pose_covariance(invalid, 'published', .3, .1)
    with pytest.raises(ValueError):
        pose_covariance(expected, 'automatic', .3, .1)


def test_stamp_gate_rejects_duplicates_old_stamps_and_decimates_without_retiming():
    gate = StampGate(.2)
    assert not gate.ready(0)
    assert gate.ready(1_000_000_000)
    gate.accept(1_000_000_000)
    assert not gate.ready(1_000_000_000)
    assert not gate.ready(900_000_000)
    assert not gate.ready(1_199_999_999)
    assert gate.ready(1_200_000_000)
