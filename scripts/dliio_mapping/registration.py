"""Bounded local registration and competing-basin checks for loop retrieval.

This module estimates a transform, never a calibrated covariance. A caller
must separately check original held observations and apply the graph gates.
"""
import math
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .core import rigid_pose


def voxels(points, size, maximum=30000):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Registration needs finite XYZ points')
    if not len(points):
        return points.copy()
    _, inverse, counts = np.unique(np.floor(points/size).astype(np.int64), axis=0,
                                   return_inverse=True, return_counts=True)
    result = np.column_stack([np.bincount(inverse, weights=points[:, i])/counts for i in range(3)])
    if len(result) > maximum:
        result = result[np.linspace(0, len(result)-1, maximum, dtype=int)]
    return result


def transformed(points, pose):
    return points @ pose[:3, :3].T + pose[:3, 3]


def rigid_fit(source, target):
    a, b = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source-a).T @ (target-b))
    result = np.eye(4)
    result[:3, :3] = vt.T @ np.diag([1., 1., np.linalg.det(vt.T @ u.T)]) @ u.T
    result[:3, 3] = b-result[:3, :3] @ a
    return result


def normals(points, tree):
    _, indices = tree.query(points, k=min(20, len(points)), workers=1)
    patches = points[indices]
    patches -= patches.mean(axis=1, keepdims=True)
    eigenvalues, eigenvectors = np.linalg.eigh(np.einsum('nki,nkj->nij', patches, patches))
    valid = ((eigenvalues[:, 1] > 1e-6) &
             (eigenvalues[:, 0]/np.maximum(eigenvalues.sum(1), 1e-12) < .04))
    return eigenvectors[:, :, 0], valid


def _fit_python(source, target, initial, *, iterations=70, fine=False, max_queries=6000):
    """Symmetric Huber point-to-plane registration with bounded SE(3) steps.

    Both scans contribute equally regardless of point density. Reverse terms
    differentiate the moving source normal as well as its point: their rotation
    lever arm is the fixed target point. Normals need not be oriented.
    """
    result = rigid_pose(initial).copy()
    if min(len(source), len(target)) < 100:
        raise ValueError('Insufficient registration support')
    source_tree, target_tree = cKDTree(source), cKDTree(target)
    source_normals, source_planar = normals(source, source_tree)
    target_normals, target_planar = normals(target, target_tree)
    # Bound each iteration's queries while keeping both reference surfaces and
    # their normal neighbourhoods intact. This also prevents scan density from
    # dictating CPU cost or forward/reverse residual weight during live replay.
    source_ids = np.linspace(0, len(source)-1, min(max_queries, len(source)), dtype=int)
    target_ids = np.linspace(0, len(target)-1, min(max_queries, len(target)), dtype=int)
    source_query, target_query = source[source_ids], target[target_ids]
    for iteration in range(iterations):
        moved = transformed(source_query, result)
        moved_normals = source_normals @ result[:3, :3].T
        forward, forward_indices = target_tree.query(moved, workers=1)
        reverse, reverse_indices = source_tree.query(transformed(target_query, np.linalg.inv(result)), workers=1)
        radius = .4 if fine else 2.5 if iteration < 15 else 1. if iteration < 35 else .4
        forward_mask = ((forward < radius) & target_planar[forward_indices] & source_planar[source_ids] &
            (abs(np.sum(moved_normals[source_ids]*target_normals[forward_indices], axis=1)) > .85))
        reverse_mask = ((reverse < radius) & source_planar[reverse_indices] & target_planar[target_ids] &
            (abs(np.sum(moved_normals[reverse_indices]*target_normals[target_ids], axis=1)) > .85))
        forward_count, reverse_count = int(forward_mask.sum()), int(reverse_mask.sum())
        if min(forward_count, reverse_count) < 100:
            raise ValueError('Insufficient symmetric registration correspondences')
        a = np.vstack((moved[forward_mask], transformed(source[reverse_indices[reverse_mask]], result)))
        b = np.vstack((target[forward_indices[forward_mask]], target_query[reverse_mask]))
        n = np.vstack((target_normals[forward_indices[forward_mask]], moved_normals[reverse_indices[reverse_mask]]))
        residual = np.sum((a-b)*n, axis=1)
        weights = np.sqrt(np.minimum(1., .1/np.maximum(abs(residual), 1e-12)))
        weights[:forward_count] /= math.sqrt(forward_count)
        weights[forward_count:] /= math.sqrt(reverse_count)
        center = (a.mean(0)+b.mean(0))/2
        rotational = np.vstack((a[:forward_count], b[forward_count:]))
        jacobian = np.column_stack((n, np.cross(rotational-center, n)))
        delta = np.linalg.lstsq(jacobian*weights[:, None], -residual*weights, rcond=1e-5)[0]
        delta *= min(1., .5/max(np.linalg.norm(delta[:3]), 1e-12),
                     math.radians(5)/max(np.linalg.norm(delta[3:]), 1e-12))
        step = np.eye(4)
        step[:3, :3] = Rotation.from_rotvec(delta[3:]).as_matrix()
        step[:3, 3] = center + delta[:3] - step[:3, :3] @ center
        result = step @ result
        if (fine or iteration >= 35) and np.linalg.norm(delta) < 1e-4:
            break
    return rigid_pose(result)


def _backend(value):
    if value not in ('native', 'python'):
        raise ValueError('Registration backend must be native or python')
    return value


def fit(source, target, initial, *, iterations=70, fine=False, max_queries=6000, backend='native'):
    """Native bounded fit, or the retained NumPy/SciPy reference for comparison."""
    if _backend(backend) == 'python':
        return _fit_python(source, target, initial, iterations=iterations, fine=fine, max_queries=max_queries)
    from _dliio_pose_graph import PreparedLoopCloud
    return PreparedLoopCloud(source).fit(PreparedLoopCloud(target), rigid_pose(initial),
                                         iterations, fine, max_queries)


def rotation_angle(matrix):
    return math.acos(float(np.clip((np.trace(matrix)-1)/2, -1., 1.)))


def score(source, target, pose):
    moved = transformed(source, pose)
    a, _ = cKDTree(target).query(moved, workers=1)
    b, _ = cKDTree(moved).query(target, workers=1)
    return float(np.sqrt((np.mean(np.minimum(a, 1.)**2)+np.mean(np.minimum(b, 1.)**2))/2))


def feature_suggestions(source, target, prior):
    from .features import seeds
    return seeds(voxels(voxels(source, .15), .3, 12000),
                 voxels(voxels(target, .15), .3, 12000), prior)


def register(source, target, prior, *, ambiguity_margin=.02, offsets=(-8., -4., 0., 4., 8.),
             feature_hint=None, backend='native'):
    """Select using fitting geometry; reject similarly good distinct basins."""
    source, target = voxels(source, .15), voxels(target, .15)
    coarse_source, coarse_target = voxels(source, .3, 12000), voxels(target, .3, 12000)
    if min(len(source), len(target)) < 100:
        raise ValueError('Insufficient registration support')
    backend = _backend(backend)
    prepare_start = time.monotonic()
    if backend == 'native':
        from _dliio_pose_graph import PreparedLoopCloud
        # Four immutable owned contexts per pair, shared by every start and the
        # reverse check. Lifetime is local, so a pose revision or changed window
        # cannot accidentally reuse stale transformed coordinates.
        source_fine, target_fine = PreparedLoopCloud(source), PreparedLoopCloud(target)
        source_coarse, target_coarse = PreparedLoopCloud(coarse_source), PreparedLoopCloud(coarse_target)
        coarse_fit = lambda initial: source_coarse.fit(target_coarse, initial)
        fine_fit = lambda initial: source_fine.fit(target_fine, initial, iterations=35, fine=True)
        score_fit = lambda pose: source_fine.score(target_fine, pose)
        reverse_fit = lambda initial: target_fine.fit(source_fine, initial, iterations=35, fine=True)
    else:
        coarse_fit = lambda initial: _fit_python(coarse_source, coarse_target, initial)
        fine_fit = lambda initial: _fit_python(source, target, initial, iterations=35, fine=True)
        score_fit = lambda pose: score(source, target, pose)
        reverse_fit = lambda initial: _fit_python(target, source, initial, iterations=35, fine=True)
    preparation_seconds = time.monotonic()-prepare_start
    prior = rigid_pose(prior)
    from .features import seeds as feature_seeds
    feature_initials, feature_report = (feature_seeds(coarse_source, coarse_target, prior)
        if feature_hint is None else feature_hint)
    _, _, vt = np.linalg.svd(target-target.mean(0), full_matrices=False)
    seeds = [prior.copy(), *feature_initials]
    center = prior.copy()
    center[:3, 3] = target.mean(0)-source.mean(0) @ prior[:3, :3].T
    for offset in offsets:
        initial = center.copy()
        initial[:3, 3] += vt[0]*offset
        seeds.append(initial)
    candidates = []
    fit_start = time.monotonic()
    for initial in seeds:
        try:
            pose = fine_fit(coarse_fit(initial))
            candidates.append((score_fit(pose), pose))
        except ValueError:
            continue
    if not candidates:
        raise ValueError('No registration had sufficient support')
    candidates.sort(key=lambda item: item[0])
    best_score, best = candidates[0]
    for competing_score, competing in candidates[1:]:
        difference = np.linalg.inv(best) @ competing
        distinct = np.linalg.norm(difference[:3, 3]) > .5 or rotation_angle(difference[:3, :3]) > math.radians(3)
        if distinct and competing_score < best_score + ambiguity_margin:
            raise ValueError('Ambiguous registration basins')
    reverse = reverse_fit(np.linalg.inv(best))
    cycle = best @ reverse
    if np.linalg.norm(cycle[:3, 3]) > .2 or rotation_angle(cycle[:3, :3]) > math.radians(2):
        raise ValueError('Forward/reverse registration disagreement')
    return best, dict(training_score=best_score, starts=len(seeds), fitted_starts=len(candidates), **feature_report,
                      registration_backend=backend, preparation_seconds=preparation_seconds,
                      fit_seconds=time.monotonic()-fit_start,
                      competing_scores=[item[0] for item in candidates[1:]],
                      reverse_cycle_translation=float(np.linalg.norm(cycle[:3, 3])),
                      reverse_cycle_rotation=rotation_angle(cycle[:3, :3]))
