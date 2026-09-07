"""Deterministic room geometry and full-covariance graph fixtures."""
import numpy as np
from scipy.spatial.transform import Rotation


def pose(x=0., yaw=0.):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec([0., 0., yaw]).as_matrix()
    result[0, 3] = x
    return result


def placed(points, matrix):
    result = points.copy()
    result[:, :3] = points[:, :3] @ matrix[:3, :3].T + matrix[:3, 3]
    return result


def room():
    rng = np.random.default_rng(815)
    points = []
    for axis, size in enumerate((4., 3., 2.)):
        for sign in (-1, 1):
            xyz = rng.uniform([-4, -3, -2], [4, 3, 2], (600, 3))
            xyz[:, axis] = sign * size
            points.append(xyz)
    xyz = np.vstack(points)
    channels = rng.uniform(1, 100, (len(xyz), 4))
    channels[::7, 1] = np.nan
    return np.column_stack((xyz, channels)).astype('<f4')


def configuration():
    return dict(odometry_noise=dict(kind='assumed', model='synthetic independent relative increments',
        covariance_floor=np.diag([.2**2]*3 + [.08**2]*3).tolist(),
        covariance_per_second=np.diag([.05**2]*3 + [.01**2]*3).tolist(),
        covariance_per_metre=np.zeros((6, 6)).tolist(), max_gap_seconds=2.),
        validation=dict(min_separation_observations=2, min_separation_seconds=1.))


def loop(transform=None):
    return dict(id='revisit-0-4', from_id=0, to_id=4, transform=(np.eye(4) if transform is None else transform).tolist(),
        covariance=np.diag([.03**2]*3 + [.02**2]*3).tolist(), covariance_kind='assumed',
        covariance_model='synthetic loop noise', provenance='Known identity revisit in deterministic synthetic room')
