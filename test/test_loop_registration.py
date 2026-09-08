"""Native proposals must preserve the reference objective and independent gates."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from _dliio_pose_graph import PreparedLoopCloud
from dliio_mapping.registration import fit, score, transformed
from dliio_mapping.loop_validation import RejectedLoop, Validation, validate_geometry
from graph_fixtures import room


def scene():
    # Break exact grid ties so this tests the numerical objective, not arbitrary
    # ordering of equally distant neighbours in two different tree libraries.
    target = room()[:, :3].astype(float)
    target += np.random.default_rng(872).normal(0, 1.e-4, target.shape)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec([.04, -.08, .06]).as_matrix()
    pose[:3, 3] = [.9, -.3, .4]
    return transformed(target[::2], np.linalg.inv(pose)), target, pose


def test_native_matches_reference_and_known_transform():
    source, target, expected = scene()
    native = fit(source, target, np.eye(4))
    reference = fit(source, target, np.eye(4), backend='python')
    np.testing.assert_allclose(native, reference, atol=2.e-5, rtol=0)
    np.testing.assert_allclose(native, expected, atol=.004, rtol=0)
    a, b = PreparedLoopCloud(source), PreparedLoopCloud(target)
    for pose in (expected, np.eye(4), native):
        assert a.score(b, pose) == pytest.approx(score(source, target, pose), abs=1.e-12)


def test_owned_contexts_are_reusable_and_safe_for_concurrent_queries():
    source, target, _ = scene()
    a, b = PreparedLoopCloud(source), PreparedLoopCloud(target)
    expected = a.fit(b, np.eye(4))
    source[:] = np.nan
    target[:] = 12345.
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: a.fit(b, np.eye(4)), range(4)))
    for result in results:
        np.testing.assert_array_equal(result, expected)


def test_native_cannot_admit_an_unobservable_plane():
    rng = np.random.default_rng(122)
    plane = np.column_stack((rng.uniform(-5., 5., (3000, 2)), np.ones(3000)))
    initial = np.eye(4)
    initial[2, 3] = .2
    pose = fit(plane, plane, initial)
    assert abs(pose[2, 3]) < 1.e-6
    with pytest.raises(RejectedLoop, match='unobservable_geometry'):
        validate_geometry(plane, plane, pose, Validation())


@pytest.mark.parametrize('points', [np.zeros((99, 3)), np.zeros((30001, 3)),
                                  np.zeros((100, 2)), np.full((100, 3), np.nan),
                                  np.full((100, 3), 1.e7)])
def test_invalid_support_is_rejected_before_native_queries(points):
    with pytest.raises(ValueError):
        PreparedLoopCloud(points)


def test_native_rejects_unrelated_support_and_invalid_solver_inputs():
    source, target, _ = scene()
    a, b = PreparedLoopCloud(source), PreparedLoopCloud(target+100.)
    with pytest.raises(ValueError, match='correspondences'):
        a.fit(b, np.eye(4))
    for kwargs in ({'iterations': 0}, {'iterations': 101}, {'max_queries': 99}, {'max_queries': 6001}):
        with pytest.raises(ValueError, match='bounded'):
            a.fit(a, np.eye(4), **kwargs)
    huge = np.eye(4)
    huge[0, 3] = 1.e308
    for pose in (np.zeros((4, 4)), np.full((4, 4), np.nan), np.diag([-1., 1., 1., 1.]), huge):
        with pytest.raises(ValueError, match='rigid'):
            a.fit(a, pose)
