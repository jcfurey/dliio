"""Uncertainty conventions, correlation, and measurement-provenance boundaries."""
from pathlib import Path
import sys
import uuid

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from dliio_mapping.uncertainty import (covariance, gtsam_covariance, inverse_covariance,
                                      observation_metadata, relative_covariance)


def metadata():
    return dict(source_session_id=str(uuid.uuid4()), source_sequence=0, covariance_kind='unknown',
                covariance_model='registered-pose covariance unavailable', covariance=None,
                quality=dict(registration_converged=True, degenerate_translation_modes=1,
                             degenerate_rotation_modes=0))


@pytest.mark.parametrize('matrix', [np.eye(3), np.full((6, 6), np.nan), np.full((6, 6), np.inf),
                                  np.diag([1., 1., 1., 1., 1., -.1]),
                                  np.eye(6) + np.triu(np.ones((6, 6)), 1)])
def test_invalid_covariance_is_rejected(matrix):
    with pytest.raises(ValueError):
        covariance(matrix)


def test_unknown_is_not_a_zero_noise_measurement():
    value = metadata()
    assert observation_metadata(value)['covariance'] is None
    value['covariance'] = np.zeros((6, 6))
    with pytest.raises(ValueError, match='Unknown covariance'):
        observation_metadata(value)
    with pytest.raises(ValueError, match='positive definite'):
        covariance(np.zeros((6, 6)), positive=True)


def test_full_covariance_and_explicit_model_survive_serialization():
    value = metadata()
    rng = np.random.default_rng(32)
    a = rng.normal(size=(6, 6))
    sigma = a @ a.T
    value.update(covariance_kind='assumed', covariance_model='fixture sensor model', covariance=sigma)
    result = observation_metadata(value)
    np.testing.assert_allclose(result['covariance'], sigma)
    assert result['covariance_kind'] == 'assumed'
    converted = gtsam_covariance(sigma)
    assert converted[0, 3] == sigma[3, 0]
    np.testing.assert_allclose(gtsam_covariance(converted), sigma)


def test_relative_uncertainty_requires_a_correlation_decision():
    with pytest.raises(ValueError, match='cross-covariance'):
        relative_covariance(np.eye(4), np.eye(4), np.eye(6), np.eye(6))
    independent = relative_covariance(np.eye(4), np.eye(4), np.eye(6), np.eye(6), independent=True)
    np.testing.assert_allclose(independent, 2*np.eye(6))
    shared = relative_covariance(np.eye(4), np.eye(4), np.eye(6), np.eye(6), cross_covariance=np.eye(6))
    np.testing.assert_allclose(shared, np.zeros((6, 6)), atol=1e-14)
    with pytest.raises(ValueError, match='semidefinite'):
        relative_covariance(np.eye(4), np.eye(4), np.eye(6), np.eye(6), cross_covariance=2*np.eye(6))


def test_relative_rotation_uncertainty_has_lever_arm_and_cross_terms():
    target = np.eye(4)
    target[0, 3] = 2.
    sigma = np.zeros((6, 6))
    sigma[5, 5] = .01
    actual = relative_covariance(np.eye(4), target, sigma, np.zeros((6, 6)), independent=True)
    # Yaw at the source rotates a 2 m baseline: dy = -2*d_yaw;
    # relative yaw = -d_yaw. Their covariance is therefore positive.
    expected = np.zeros((6, 6))
    expected[1, 1], expected[5, 5] = .04, .01
    expected[1, 5] = expected[5, 1] = .02
    np.testing.assert_allclose(actual, expected)


def test_inverse_covariance_preserves_rotation_translation_coupling():
    pose = np.array([[0., -1, 0, 3], [1, 0, 0, 4], [0, 0, 1, 0], [0, 0, 0, 1]])
    sigma = np.diag([.1, .2, .3, .01, .02, .03])
    inverted = inverse_covariance(pose, sigma)
    assert np.linalg.norm(inverted[:3, 3:]) > 0
    np.testing.assert_allclose(inverse_covariance(np.linalg.inv(pose), inverted), sigma, atol=1e-14)


@pytest.mark.parametrize('change', [dict(source_sequence=True), dict(source_sequence=-1),
                                   dict(source_session_id='new-run'), dict(covariance_kind='confident'),
                                   dict(covariance_model=''), dict(quality={})])
def test_invalid_observation_provenance(change):
    value = metadata()
    value.update(change)
    with pytest.raises(ValueError):
        observation_metadata(value)
