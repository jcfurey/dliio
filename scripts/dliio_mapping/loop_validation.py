"""Geometry gates for externally registered loop candidates, not place retrieval.

A low registration residual alone cannot establish that repetitive scenes are
the same place. Candidates must carry external verification provenance. These
additional gates deliberately use absolute distances as well as covariance;
inflating uncertainty cannot make a geometrically wrong transform acceptable.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Validation:
    min_separation_observations: int = 30
    min_separation_seconds: float = 10.
    voxel_size: float = .15
    min_range: float = .7
    max_range: float = 80.
    max_points: int = 6000
    match_radius: float = .4
    min_overlap: float = .65
    max_inlier_p95: float = .2
    min_inliers: int = 100
    normal_neighbors: int = 16
    max_normal_variance: float = .03
    min_observability_ratio: float = .001
    max_cycle_translation: float = 3.
    max_cycle_rotation: float = .5
    max_odometry_squared_error: float = 36.
    max_loop_squared_error: float = 16.8
    max_loop_translation: float = .25
    max_loop_rotation: float = .1
    max_solution_translation: float = 20.
    max_solution_rotation: float = 1.

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'Validation {name} must be finite and positive')
        for name, lower, upper in (('min_separation_observations', 2, 5000),
                ('max_points', 100, 20000), ('min_inliers', 12, self.max_points),
                ('normal_neighbors', 6, 64)):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f'Validation {name} must be an integer in [{lower}, {upper}]')
        if (self.min_range >= self.max_range or not .001 <= self.voxel_size <= 10. or
                self.max_inlier_p95 > self.match_radius or
                any(getattr(self, key) > 1 for key in
                    ('min_overlap', 'max_normal_variance', 'min_observability_ratio')) or
                any(getattr(self, key) > np.pi for key in
                    ('max_cycle_rotation', 'max_loop_rotation', 'max_solution_rotation'))):
            raise ValueError('Inconsistent validation distances, ratios, or angles')


class RejectedLoop(ValueError):
    def __init__(self, reason, metrics=None):
        super().__init__(reason)
        self.metrics = metrics or {}


def displacement(pose):
    from .core import rigid_pose
    value = rigid_pose(pose)
    angle = math.acos(float(np.clip((np.trace(value[:3, :3]) - 1) / 2, -1, 1)))
    return float(np.linalg.norm(value[:3, 3])), angle


def check_cycle(measured, predicted, settings):
    translation, rotation = displacement(np.linalg.inv(predicted) @ measured)
    metrics = dict(cycle_translation=translation, cycle_rotation=rotation)
    if translation > settings.max_cycle_translation or rotation > settings.max_cycle_rotation:
        raise RejectedLoop('absolute_cycle_limit', metrics)
    return metrics


def _sample(points, settings):
    xyz = np.asarray(points[:, :3], dtype=np.float64)
    radii = np.linalg.norm(xyz, axis=1)
    xyz = xyz[np.isfinite(xyz).all(axis=1) & (radii >= settings.min_range) & (radii <= settings.max_range)]
    _, selected = np.unique(np.floor(xyz / settings.voxel_size).astype(np.int64), axis=0, return_index=True)
    # np.unique sorts the spatial cells; evenly sample this order deterministically.
    if len(selected) > settings.max_points:
        selected = selected[np.linspace(0, len(selected)-1, settings.max_points, dtype=int)]
    return xyz[selected]


def validate_geometry(target_points, source_points, measured, settings):
    """Z_ij maps source j into target i, both original individual local clouds.

    Nearest-neighbour overlap is evaluated in BOTH directions. Point-to-plane
    observability uses right perturbations in j, with rotation columns scaled
    by RMS source range so eigenvalue ratios do not compare metres to radians.
    The Hessian checks shape only; it is not a calibrated covariance estimate.
    """
    from .core import rigid_pose
    measured = rigid_pose(measured)
    target, source = _sample(target_points, settings), _sample(source_points, settings)
    metrics = dict(target_points=len(target), source_points=len(source))
    if min(len(target), len(source)) < max(settings.min_inliers, settings.normal_neighbors):
        raise RejectedLoop('insufficient_geometry', metrics)
    transformed = source @ measured[:3, :3].T + measured[:3, 3]
    target_tree = cKDTree(target)
    forward, matches = target_tree.query(transformed, workers=1)
    reverse, _ = cKDTree(transformed).query(target, workers=1)
    mask = forward <= settings.match_radius
    reverse_mask = reverse <= settings.match_radius
    metrics.update(forward_overlap=float(mask.mean()), reverse_overlap=float(reverse_mask.mean()))
    if min(metrics['forward_overlap'], metrics['reverse_overlap']) < settings.min_overlap:
        raise RejectedLoop('bidirectional_overlap', metrics)
    if min(int(mask.sum()), int(reverse_mask.sum())) < settings.min_inliers:
        raise RejectedLoop('insufficient_inliers', metrics)
    metrics.update(forward_p95=float(np.quantile(forward[mask], .95)),
                   reverse_p95=float(np.quantile(reverse[reverse_mask], .95)))
    if max(metrics['forward_p95'], metrics['reverse_p95']) > settings.max_inlier_p95:
        raise RejectedLoop('geometric_residual', metrics)

    # Distinct target support prevents a dense source cluster from manufacturing
    # many independent normal constraints on the same small target patch.
    indices = np.flatnonzero(mask)
    _, first = np.unique(matches[mask], return_index=True)
    indices = indices[first]
    _, neighbors = target_tree.query(target[matches[indices]], k=settings.normal_neighbors, workers=1)
    patches = target[neighbors]
    centered = patches - patches.mean(axis=1, keepdims=True)
    eigenvalues, eigenvectors = np.linalg.eigh(np.einsum('nki,nkj->nij', centered, centered))
    planes = (eigenvalues[:, 1] > 1e-6) & (eigenvalues[:, 0] /
             np.maximum(eigenvalues.sum(axis=1), 1e-12) <= settings.max_normal_variance)
    normals = eigenvectors[planes, :, 0] @ measured[:3, :3]
    support = source[indices[planes]]
    metrics['plane_support'] = len(support)
    if len(support) < settings.min_inliers:
        raise RejectedLoop('insufficient_plane_support', metrics)
    length = max(settings.voxel_size, float(np.sqrt(np.mean(np.sum(support**2, axis=1)))))
    jacobian = np.column_stack((normals, np.cross(support, normals) / length))
    spectrum = np.linalg.eigvalsh(jacobian.T @ jacobian / len(jacobian))
    ratio = float(max(0., spectrum[0]) / max(spectrum[-1], 1e-12))
    metrics.update(observability_ratio=ratio, scaled_information_eigenvalues=spectrum.tolist(),
                   characteristic_length=length)
    if ratio < settings.min_observability_ratio:
        raise RejectedLoop('unobservable_geometry', metrics)
    return metrics
