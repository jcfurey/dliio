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


@pytest.mark.parametrize('changes', [dict(max_points=True), dict(min_overlap=0.), dict(voxel_size=float('nan')),
    dict(max_cycle_rotation=4.), dict(min_inliers=100000), dict(min_separation_observations=1)])
def test_invalid_validation_policy(changes):
    with pytest.raises(ValueError):
        replace(Validation(), **changes)
