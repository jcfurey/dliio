"""Frame-preserving covariance selection and heading-free tilt conversion."""
import math

import numpy as np
from scipy.spatial.transform import Rotation


def positive_sigma(value):
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Assumed standard deviations must be finite and positive')
    return value


def pose_covariance(values, policy, position_sigma, angle_sigma):
    """Never claim that substituting a noise model calibrates a covariance."""
    if policy == 'assumed':
        return np.diag([positive_sigma(position_sigma)**2]*3 + [positive_sigma(angle_sigma)**2]*3)
    if policy != 'published':
        raise ValueError('Covariance policy must be published or assumed')
    result = np.asarray(values, float).reshape(6, 6)
    if not np.isfinite(result).all() or not np.allclose(result, result.T, atol=1.e-10, rtol=0):
        raise ValueError('Published pose covariance must be finite and symmetric')
    if np.linalg.eigvalsh(result)[0] < 1.e-9:
        raise ValueError('Published pose covariance must be positive definite and nonzero')
    return result.copy()  # Preserve every off-diagonal term.


def tilt_quaternion(direction):
    """World +Z expressed in body axes -> body-to-world tilt, arbitrary yaw=0.

    No heading observation is created. The consumer must fuse roll/pitch only.
    Near vertical pitch is outside this Euler-angle EKF experiment's envelope.
    """
    up = np.asarray(direction, float)
    if up.shape != (3,) or not np.isfinite(up).all() or not .9 <= np.linalg.norm(up) <= 1.1:
        raise ValueError('Gravity input must be a finite near-unit world-up direction')
    up = up/np.linalg.norm(up)
    roll = math.atan2(up[1], up[2])
    pitch = math.asin(float(np.clip(-up[0], -1., 1.)))
    if abs(pitch) > math.radians(80):
        raise ValueError('Tilt is too close to the Euler singularity')
    return Rotation.from_euler('xyz', [roll, pitch, 0.]).as_quat()


class StampGate:
    """Bound repeated correlated input; reject old/duplicate acquisition stamps."""
    def __init__(self, period):
        self.period_ns = round(positive_sigma(period)*1.e9)
        self.previous = None

    def ready(self, stamp):
        return stamp > 0 and (self.previous is None or stamp-self.previous >= self.period_ns)

    def accept(self, stamp):
        self.previous = stamp
