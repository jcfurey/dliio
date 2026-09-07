"""Explicit uncertainty contracts for mapping and relative-pose measurements.

All matrices here use RIGHT perturbations T_true = T_est * Exp(delta), with
delta ordered [tx, ty, tz, rx, ry, rz], in metres and radians. This is a local
SE(3) tangent, not the fixed/world-axis covariance of ROS PoseWithCovariance.
No routine treats an unknown covariance as a perfect measurement.
"""
import uuid

import numpy as np

KINDS = ('unknown', 'assumed', 'conditional', 'calibrated')


def covariance(matrix, *, positive=False, dimension=6):
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (dimension, dimension) or not np.isfinite(value).all():
        raise ValueError('Covariance must be a finite square matrix of the declared dimension')
    if not np.allclose(value, value.T, atol=1e-12, rtol=1e-8):
        raise ValueError('Covariance must be symmetric')
    value = (value + value.T) * .5
    eigenvalues = np.linalg.eigvalsh(value)
    tolerance = 1e-12 * max(1., float(np.max(np.abs(eigenvalues))))
    if eigenvalues[0] < -tolerance or (positive and eigenvalues[0] <= 0.):
        raise ValueError('Covariance must be positive definite' if positive else 'Covariance must be positive semidefinite')
    return value


def adjoint(pose):
    from .core import rigid_pose
    pose = rigid_pose(pose)
    x, y, z = pose[:3, 3]
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    result = np.zeros((6, 6))
    result[:3, :3] = result[3:, 3:] = pose[:3, :3]
    result[:3, 3:] = skew @ pose[:3, :3]
    return result


def inverse_covariance(pose, matrix):
    jacobian = -adjoint(pose)
    return covariance(jacobian @ covariance(matrix) @ jacobian.T)


def relative_covariance(from_pose, to_pose, from_covariance, to_covariance,
                        *, cross_covariance=None, independent=False):
    """Propagate covariance of inverse(from_pose) * to_pose.

    Independence must be explicitly asserted. Successive LIO poses normally
    share measurements, so two marginal covariances alone are insufficient.
    """
    from .core import rigid_pose
    relative = np.linalg.inv(rigid_pose(from_pose)) @ rigid_pose(to_pose)
    a, b = covariance(from_covariance), covariance(to_covariance)
    if cross_covariance is None:
        if not independent:
            raise ValueError('Relative covariance requires cross-covariance or explicit independence')
        cross = np.zeros((6, 6))
    else:
        if independent:
            raise ValueError('Specify cross-covariance or independence, not both')
        cross = np.asarray(cross_covariance, dtype=np.float64)
        if cross.shape != (6, 6) or not np.isfinite(cross).all():
            raise ValueError('Invalid cross-covariance')
    covariance(np.block([[a, cross], [cross.T, b]]), dimension=12)
    jacobian = -adjoint(np.linalg.inv(relative))
    return covariance(jacobian @ a @ jacobian.T + b + jacobian @ cross + cross.T @ jacobian.T)


def gtsam_covariance(matrix):
    """Reorder an already RIGHT/local covariance; this does not change frames."""
    order = [3, 4, 5, 0, 1, 2]
    return covariance(matrix)[np.ix_(order, order)]


def observation_metadata(value=None):
    """Validate provenance before an observation can change the archive."""
    if value is None:
        return dict(source_session_id=None, source_sequence=None, covariance_kind='unknown',
                    covariance_model='legacy PoseStamped; registered-pose covariance unavailable',
                    covariance=None, quality={})
    result = dict(value)
    session = result.get('source_session_id')
    if not isinstance(session, str) or str(uuid.UUID(session)) != session:
        raise ValueError('Observation source session must be a canonical UUID')
    sequence = result.get('source_sequence')
    if type(sequence) is not int or not 0 <= sequence < 2 ** 63:
        raise ValueError('Invalid observation source sequence')
    kind, model = result.get('covariance_kind'), result.get('covariance_model')
    if kind not in KINDS or not isinstance(model, str) or not 1 <= len(model) <= 256:
        raise ValueError('Covariance requires an explicit kind and model')
    if kind == 'unknown':
        if result.get('covariance') is not None:
            raise ValueError('Unknown covariance must be absent, not a zero or guessed matrix')
    else:
        result['covariance'] = covariance(result.get('covariance')).tolist()
    quality = result.get('quality')
    if not isinstance(quality, dict) or set(quality) != {
            'registration_converged', 'degenerate_translation_modes', 'degenerate_rotation_modes'}:
        raise ValueError('Observation quality fields are required')
    if type(quality['registration_converged']) is not bool:
        raise ValueError('Registration status must be boolean')
    for key in ('degenerate_translation_modes', 'degenerate_rotation_modes'):
        if type(quality[key]) is not int or not 0 <= quality[key] <= 3:
            raise ValueError('Invalid count of degenerate modes')
    return {key: result[key] for key in ('source_session_id', 'source_sequence',
            'covariance_kind', 'covariance_model', 'covariance', 'quality')}
