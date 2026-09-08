from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from dliio_mapping.loop_validation import Validation, RejectedLoop, check_cycle, validate_geometry
from graph_fixtures import placed, pose, room


@pytest.mark.parametrize('yaw', [0., np.pi/2, np.pi])
def test_true_revisit_including_reverse_traversal(yaw):
    target = room()
    measured = pose(.3, yaw)
    source = placed(target, np.linalg.inv(measured))
    report = validate_geometry(target, source, measured, Validation())
    assert report['forward_overlap'] > .98
    assert report['reverse_overlap'] > .98
    assert report['observability_ratio'] > .001


def test_partial_source_cannot_use_one_way_overlap():
    target = room()
    source = target[target[:, 0] == 4]
    with pytest.raises(RejectedLoop, match='bidirectional_overlap'):
        validate_geometry(target, source, pose(), Validation())


@pytest.mark.parametrize('axes', [[2], [1, 2]])
def test_plane_and_axially_unconstrained_corridor_fail_observability(axes):
    rng = np.random.default_rng(42)
    points = []
    for axis in axes:
        for sign in (-1, 1):
            xyz = rng.uniform(-6, 6, (1200, 3))
            xyz[:, axis] = 2 * sign
            points.append(xyz)
    points = np.vstack(points)
    with pytest.raises(RejectedLoop, match='unobservable_geometry'):
        validate_geometry(points, points, pose(), Validation())


def test_small_body_returns_and_duplicate_points_cannot_manufacture_support():
    for points in (np.full((1000, 3), .1), np.tile([1., 1., 1.], (1000, 1))):
        with pytest.raises(RejectedLoop, match='insufficient_geometry'):
            validate_geometry(points, points, pose(), Validation())


def test_wrong_transform_and_absolute_cycle_are_independent_of_covariance():
    points = room()
    with pytest.raises(RejectedLoop):
        validate_geometry(points, points, pose(1.), Validation())
    with pytest.raises(RejectedLoop, match='absolute_cycle_limit'):
        check_cycle(pose(10.), pose(), Validation())
    with pytest.raises(RejectedLoop, match='absolute_cycle_limit'):
        check_cycle(pose(0., np.pi), pose(), Validation())


def test_query_budget_does_not_erase_the_reference_surface():
    target = room()
    measured = pose(.3, .8)
    source = placed(target, np.linalg.inv(measured))
    settings = Validation(max_points=200, min_inliers=100)
    result = validate_geometry(target, source, measured, settings)
    assert result['target_points'] == result['source_points'] == 200
    assert result['target_reference_points'] > 2000
    assert result['forward_overlap'] > .98
    assert result['reverse_p95'] < .15


@pytest.mark.parametrize('changes', [dict(max_points=True), dict(min_overlap=0.), dict(voxel_size=float('nan')),
    dict(max_cycle_rotation=4.), dict(min_inliers=100000), dict(min_separation_observations=1)])
def test_invalid_validation_policy(changes):
    with pytest.raises(ValueError):
        replace(Validation(), **changes)


def sampled_surfaces(phase):
    """Independent tangential samples of six known physical planes."""
    points = []
    for axis, size in enumerate((4., 3., 2.)):
        tangents = [i for i in range(3) if i != axis]
        sizes = (4., 3., 2.)
        coordinates = [np.arange(-sizes[i]+.4+phase, sizes[i]-.4, .4) for i in tangents]
        a, b = np.meshgrid(*coordinates)
        for sign in (-1, 1):
            xyz = np.empty((a.size, 3))
            xyz[:, axis] = sign*size
            xyz[:, tangents[0]], xyz[:, tangents[1]] = a.ravel(), b.ravel()
            points.append(xyz)
    return np.vstack(points)


def surface_settings():
    return Validation(surface_validation=True, max_inlier_p95=.35, min_overlap=.6,
                      normal_neighbors=20, max_normal_variance=.04)


def test_surface_validation_handles_independent_tangential_sampling():
    target, source = sampled_surfaces(0.), sampled_surfaces(.2)
    with pytest.raises(RejectedLoop, match='geometric_residual'):
        validate_geometry(target, source, pose(), Validation())
    result = validate_geometry(target, source, pose(), surface_settings())
    for direction in ('forward', 'reverse'):
        evidence = result[direction+'_surface']
        # Neighbourhood PCA includes some edges; the scene's true planes still
        # agree far better than the 0.28 m tangential point spacing.
        assert evidence['p95'] < .09
        assert evidence['observability_ratio'] > .001
        assert evidence['plane_support'] >= 100


@pytest.mark.parametrize('shift', [[.3, 0, 0], [0, 0, .3], [2., 0, 0], [-2., 0, 0]])
def test_surface_checks_reject_wrong_pose_despite_more_point_distance_tolerance(shift):
    wrong = np.eye(4)
    wrong[:3, 3] = shift
    with pytest.raises(RejectedLoop):
        validate_geometry(sampled_surfaces(0.), sampled_surfaces(.2), wrong, surface_settings())


def test_surface_validation_still_requires_full_pose_observability():
    wall = room()
    wall = wall[wall[:, 0] == 4., :3]
    with pytest.raises(RejectedLoop, match='unobservable_geometry'):
        validate_geometry(wall, wall, pose(), surface_settings())


@pytest.mark.parametrize('changes', [dict(surface_validation=1), dict(min_normal_alignment=1.1),
    dict(min_surface_fraction=0.), dict(max_surface_rms=.2)])
def test_invalid_surface_policy(changes):
    with pytest.raises(ValueError):
        replace(surface_settings(), **changes)
