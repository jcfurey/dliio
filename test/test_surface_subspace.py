"""Projected factors constrain measured modes without inventing axial certainty."""
import copy
import json
from pathlib import Path
import sys
import uuid

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import _dliio_pose_graph as native
from dliio_mapping.core import Limits, Store, decode_pose
from dliio_mapping.graph import loop_measurement
from dliio_mapping.loop_inputs import window_cloud
from dliio_mapping.loop_inputs import support_indices
from dliio_mapping.loop_validation import Validation, RejectedLoop
from dliio_mapping.surface_subspace import propose_subspace, projection, subspace, validate_geometry, _jacobian
from dliio_mapping.registration import fit, voxels
from graph_fixtures import configuration, pose, placed


def exp(delta):
    matrix = np.zeros((4, 4))
    x, y, z = delta[3:]
    matrix[:3, :3] = [[0., -z, y], [z, 0., -x], [-y, x, 0.]]
    matrix[:3, 3] = delta[:3]
    return expm(matrix)


@pytest.mark.parametrize('reverse', [False, True])
def test_surface_jacobian_includes_moving_normal_and_right_local_scaling(reverse):
    measured = exp([2., -.3, .6, .3, -.2, .15])
    source = np.array([[.2, .5, 1.]])
    target = np.array([[1.2, .3, 2.]])
    local_normal = np.array([[.3, .4, .5]]); local_normal /= np.linalg.norm(local_normal)
    normal = local_normal @ measured[:3, :3].T if reverse else local_normal
    actual = _jacobian(source, target, normal, measured, 9., reverse)[0]
    def residual(delta):
        pose = measured @ exp(delta/np.array([1., 1., 1., 9., 9., 9.]))
        n = local_normal @ pose[:3, :3].T if reverse else normal
        return float(np.sum((source @ pose[:3, :3].T+pose[:3, 3]-target)*n))
    expected = []
    for axis in np.eye(6):
        expected.append((residual(axis*1e-6)-residual(-axis*1e-6))/2e-6)
    np.testing.assert_allclose(actual, expected, atol=1e-9)


def test_native_projected_factor_matches_independent_numerical_solution():
    poses = [pose(), pose(1., .3), pose(2., .6)]
    measured = [np.linalg.inv(poses[0]) @ poses[1], np.linalg.inv(poses[1]) @ poses[2], pose(1.8, .8)]
    rotation = Rotation.from_rotvec([.2, .4, -.3]).as_matrix()
    transform = np.zeros((6, 6)); transform[:3, :3] = transform[3:, 3:] = rotation
    projector = transform @ np.diag([0., 1., 1., 0., 1., 1.]) @ transform.T
    scaling = np.diag([1., 1., 1., 8., 8., 8.])
    projectors = [np.eye(6), np.eye(6), np.linalg.inv(scaling) @ projector @ scaling]
    covariances = [np.diag([.09]*3+[.04]*3)]*2 + [np.diag([.01]*3+[.01/64]*3)]
    args = (poses, [0, 1, 0], [1, 2, 2], measured, covariances, [False, False, True])
    result = native.optimize(*args, projections=projectors)
    assert result['converged']
    roots = [np.linalg.cholesky(c) for c in covariances]
    pairs = [(0, 1), (1, 2), (0, 2)]
    def residuals(x):
        values = [poses[0], poses[1] @ exp(x[:6]), poses[2] @ exp(x[6:])]
        return np.concatenate([np.linalg.solve(root, p @ native.residual(z, values[i], values[j]))
                               for (i, j), z, root, p in zip(pairs, measured, roots, projectors)])
    reference = least_squares(residuals, np.zeros(12), xtol=1e-12, gtol=1e-12, ftol=1e-12)
    for i in (1, 2):
        np.testing.assert_allclose(result['poses'][i], poses[i] @ exp(reference.x[(i-1)*6:i*6]), atol=2e-6)
    expected = []
    for (i, j), z, root, p in zip(pairs, measured, roots, projectors):
        r = np.linalg.solve(root, p @ native.residual(z, result['poses'][i], result['poses'][j]))
        expected.append(r @ r)
    np.testing.assert_allclose(result['squared_errors'], expected, atol=1e-10)


def test_free_translation_has_exactly_zero_information():
    original = [pose(), pose(1.), pose(2.)]
    p = np.diag([0., 1., 1., 0., 1., 1.])
    for axial_distance in (-5., 0., 5.):
        result = native.optimize(original, [0, 1, 0], [1, 2, 2],
            [pose(1.), pose(1.), pose(axial_distance)], [np.eye(6)]*3,
            [False, False, True], projections=[np.eye(6), np.eye(6), p])
        np.testing.assert_allclose(result['poses'], original, atol=1e-9)
        assert max(result['squared_errors']) < 1e-20


@pytest.mark.parametrize('p', [np.zeros((6, 6)), np.full((6, 6), np.nan), np.eye(6)*1e5])
def test_native_rejects_invalid_projection(p):
    with pytest.raises(ValueError):
        native.optimize([pose(), pose(1.)], [0], [1], [pose(1.)], [np.eye(6)], [True], projections=[p])


def test_odometry_cannot_be_projected_or_projection_count_truncated():
    args = ([pose(), pose(1.)], [0], [1], [pose(1.)], [np.eye(6)], [False])
    for projections in ([np.diag([0., 1., 1., 1., 1., 1.])], [np.eye(6)]*2):
        with pytest.raises(ValueError):
            native.optimize(*args, projections=projections)


def tunnel():
    rng = np.random.default_rng(8821)
    result = []
    for axis, distance in ((1, 3.), (2, 2.)):
        for sign in (-1., 1.):
            points = rng.uniform([-12., -3., -2.], [12., 3., 2.], (3500, 3))
            points[:, axis] = sign*distance
            result.append(points)
    return np.vstack(result)


def test_tunnel_projection_discards_axial_translation_and_rejects_invented_modes():
    points = tunnel(); settings = Validation(surface_validation=True, max_inlier_p95=.3)
    value = propose_subspace(points, points, np.eye(4), settings, ratio=.01)
    p = projection(value)
    assert np.linalg.norm(p @ np.array([1., 0., 0., 0., 0., 0.])) < .02
    validate_geometry(points, points, np.eye(4), settings, value, verify_projection=True)
    forged = copy.deepcopy(value)
    forged['projection'] = np.diag([1., 0., 1., 1., 0., 1.]).tolist()
    with pytest.raises(RejectedLoop, match='unsupported_surface_projection'):
        validate_geometry(points, points, np.eye(4), settings, forged, verify_projection=True)
    # A vertical displacement must not disappear into nearest-neighbour gating.
    wrong = pose(); wrong[2, 3] = .3
    with pytest.raises(RejectedLoop):
        validate_geometry(points, points, wrong, settings, value)


def test_observable_registration_corrects_height_without_walking_down_tunnel():
    target = tunnel()
    truth = pose(2., .08); truth[2, 3] = .3
    source = placed(target, np.linalg.inv(truth))
    initial = truth.copy(); initial[0, 3] = 5.; initial[2, 3] = 1.
    fitted = fit(voxels(source, .15), voxels(target, .15), initial, observable_ratio=.01)
    assert abs(fitted[2, 3]-.3) < .02
    assert abs(fitted[0, 3]-initial[0, 3]) < .15
    assert np.linalg.norm(fitted[:3, :3]-truth[:3, :3]) < .02


@pytest.mark.parametrize('ratio', [-1., 1e-8, .2, float('nan')])
def test_invalid_observable_registration_ratio_is_rejected(ratio):
    points = tunnel()
    with pytest.raises(ValueError, match='ratio'):
        fit(points, points, np.eye(4), observable_ratio=ratio)


def test_single_fitting_scan_still_requires_two_independent_held_observations():
    support = dict(from_observations=[2], held_from_observations=[1, 3],
                   to_observations=[13], held_to_observations=[12, 14])
    assert support_indices(support, 16, 2, 13, allow_single_fitting=True) == support
    with pytest.raises(ValueError): support_indices(support, 16, 2, 13)
    for edit in ('missing_held', 'reused_held'):
        bad = copy.deepcopy(support)
        bad['held_to_observations'] = [12] if edit == 'missing_held' else [12, 13]
        with pytest.raises(ValueError): support_indices(bad, 16, 2, 13, allow_single_fitting=True)


def test_partial_loop_commit_reload_removal_preserve_evidence(tmp_path, monkeypatch):
    meta = dict(odom_frame='odom', base_frame='base', map_frame='map', map_to_odom=np.eye(4).tolist(), calibration={})
    store = Store(tmp_path/'working.dliomap', Limits(submap_keyframes=4, resident_submaps=1), metadata=meta)
    session = str(uuid.uuid4())
    points = np.column_stack((tunnel(), np.ones((14000, 4)))).astype('<f4')
    def request(action, **fields):
        return dict(session_id=store.meta['session_id'], expected_revision=store.meta['pose_revision'],
                    request_id=str(uuid.uuid4()), action=action, **fields)
    try:
        for i in range(16):
            original = pose(i*.04); original[2, 3] = np.clip((i-5)/5., 0., 1.)*.6
            store.ingest((i+1)*1000000000, original, placed(points, original), observation=dict(
                source_session_id=session, source_sequence=i, covariance_kind='unknown', covariance=None,
                covariance_model='synthetic', quality=dict(registration_converged=True,
                    degenerate_translation_modes=1, degenerate_rotation_modes=0)))
        originals = store.db.execute('SELECT * FROM keyframes').fetchall()
        config = dict(configuration(), loop_windows=dict(max_span_seconds=4., validation=dict(
            surface_validation=True, max_inlier_p95=.3)))
        assert store.update_graph(request('initialize', configuration=config))['status'] == 'accepted'
        settings = Validation(**config['loop_windows']['validation'])
        support = dict(from_observations=[0, 2, 4], held_from_observations=[1, 3, 5],
                       to_observations=[10, 12, 14], held_to_observations=[11, 13, 15])
        target = window_cloud(store, 2, support['from_observations'], 4., settings)
        source = window_cloud(store, 13, support['to_observations'], 4., settings)
        value = propose_subspace(target, source, np.eye(4), settings)
        length = value['characteristic_length']
        candidate = dict(id='projected-return', from_id=2, to_id=13, transform=np.eye(4).tolist(), support=support,
            surface_subspace=value, covariance=np.diag([.03**2]*3+[(.03/length)**2]*3).tolist(),
            covariance_kind='assumed', covariance_model='assumed retained modes', provenance='synthetic same tunnel')
        original_cloud = store._graph_cloud
        def wrong_held(identifier):
            result = original_cloud(identifier).copy()
            if identifier in support['held_to_observations']:
                result[:, 2] += .5
            return result
        monkeypatch.setattr(store, '_graph_cloud', wrong_held)
        rejected = store.update_graph(request('add_loop', loop=candidate))
        assert rejected['status'] == 'rejected'
        assert store.meta['pose_revision'] == 1 and store.stats()['active_loops'] == 0
        monkeypatch.setattr(store, '_graph_cloud', original_cloud)
        update = request('add_loop', loop=candidate)
        result = store.update_graph(update)
        assert result['status'] == 'accepted', result
        assert store.update_graph(update) == result
        a, b = [decode_pose(store.db.execute('SELECT pose FROM optimized_poses WHERE id=?', (i,)).fetchone()[0]) for i in (2, 13)]
        assert abs(b[2, 3]-a[2, 3]) < .02
        assert store.db.execute('SELECT * FROM keyframes').fetchall() == originals
        saved = tmp_path/'sealed.dliomap'; store.save(saved)
        loaded = Store(saved, store.limits)
        assert loaded.stats()['graph_attached'] and loaded.stats()['active_loops'] == 1
        loaded.close()
        assert store.update_graph(request('remove_loop', loop_id=candidate['id'], reason='synthetic reversal'))['status'] == 'accepted'
        assert store.stats()['active_loops'] == 0
        for original, current in store.db.execute('SELECT k.pose,p.pose FROM keyframes k JOIN optimized_poses p USING(id)'):
            np.testing.assert_allclose(decode_pose(original), decode_pose(current), atol=1e-9)
        for edit in ('no_support', 'calibrated', 'anisotropic', 'full_rank'):
            bad = copy.deepcopy(candidate)
            if edit == 'no_support': bad.pop('support')
            if edit == 'calibrated': bad['covariance_kind'] = 'calibrated'
            if edit == 'anisotropic': bad['covariance'][0][0] *= 2
            if edit == 'full_rank': bad['surface_subspace']['projection'] = np.eye(6).tolist()
            with pytest.raises(ValueError): loop_measurement(bad, 16)
    finally:
        store.close()
