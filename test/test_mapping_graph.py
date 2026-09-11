"""Real GTSAM solves, loop rejection, removal, and durable graph/map agreement."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import uuid

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.optimize import least_squares

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import _dliio_pose_graph as native
from dliio_mapping.core import Limits, Store, decode_pose, snapshot_database
from dliio_mapping.graph import configuration as checked_configuration
from graph_fixtures import configuration, loop, placed, pose, room

META = dict(odom_frame='odom', base_frame='base', map_frame='map', map_to_odom=np.eye(4).tolist(), calibration={})


def request(store, action, **kwargs):
    return dict(session_id=store.meta['session_id'], expected_revision=store.meta['pose_revision'],
                request_id=str(uuid.uuid4()), action=action, **kwargs)


def database(store):
    names = ['metadata', 'keyframes', 'observations', 'optimized_poses', 'submaps', 'pose_revisions', 'revision_poses',
             'graph_state', 'graph_loops', 'graph_requests']
    return {name: store.db.execute(f'SELECT * FROM {name}').fetchall() for name in names}


def exported(store, path):
    store.export_pcd(path)
    return np.frombuffer(path.read_bytes().split(b'DATA binary\n', 1)[1], '<f4').reshape(-1, 7)


@pytest.fixture
def store(tmp_path):
    value = Store(tmp_path / 'live.dliomap', Limits(submap_keyframes=2, resident_submaps=1), metadata=META)
    points = room()
    for i, distance in enumerate([0., 2., 3., 1., 0.]):
        truth = pose(distance)
        original = pose(distance + i*.2, i*.015)
        local = placed(points, np.linalg.inv(truth))
        value.ingest((i+1)*1000000000, original, placed(local, original))
    yield value
    value.close()


def initialize(store):
    result = store.update_graph(request(store, 'initialize', configuration=configuration()))
    assert result['status'] == 'accepted', result
    return result


def test_initialize_retains_odometry_and_declares_process_noise(store):
    before = database(store)
    result = initialize(store)
    assert result['pose_revision'] == 1
    assert result['metrics']['optimization']['final_error'] < 1e-12
    for original, optimized in store.db.execute('SELECT k.pose,p.pose FROM keyframes k JOIN optimized_poses p ON p.id=k.id'):
        np.testing.assert_allclose(decode_pose(original), decode_pose(optimized), atol=1e-12)
    assert database(store)['keyframes'] == before['keyframes']
    assert store.stats()['graph_attached']
    assert 'assumed' in database(store)['graph_state'][0][1]


def test_loop_closes_drift_and_rebuilds_evicted_geometry_then_removal_recovers_original(store, tmp_path):
    before = database(store)
    original_cloud = exported(store, tmp_path / 'original.pcd')
    initialize(store)
    update = request(store, 'add_loop', loop=loop())
    result = store.update_graph(update)
    assert result['status'] == 'accepted', result
    assert result['pose_revision'] == 2
    assert store.stats()['active_loops'] == 1
    final = decode_pose(store.db.execute('SELECT pose FROM optimized_poses WHERE id=4').fetchone()[0])
    assert np.linalg.norm(final[:3, 3]) < .02
    changed = exported(store, tmp_path / 'corrected.pcd')
    assert not np.allclose(changed[:, :3], original_cloud[:, :3])
    np.testing.assert_array_equal(changed[:, 3:], original_cloud[:, 3:])
    assert store.stats()['resident_submaps'] == 1
    assert database(store)['keyframes'] == before['keyframes']
    assert database(store)['observations'] == before['observations']
    assert store.update_graph(update) == result
    removal = store.update_graph(request(store, 'remove_loop', loop_id='revisit-0-4', reason='synthetic factor removal'))
    assert removal['status'] == 'accepted', removal
    assert store.stats()['active_loops'] == 0
    np.testing.assert_allclose(exported(store, tmp_path / 'removed.pcd'), original_cloud, atol=1e-6)
    assert store.db.execute('SELECT added_revision,removed_revision FROM graph_loops').fetchone() == (2, 3)
    store.save(tmp_path / 'saved.dliomap')
    with_loaded = Store(tmp_path / 'saved.dliomap', store.limits)
    try:
        assert with_loaded.stats()['graph_attached']
        assert len(database(with_loaded)['graph_requests']) == 3
    finally:
        with_loaded.close()


@pytest.mark.parametrize('bad', ['cycle', 'geometry', 'temporal'])
def test_rejected_loop_only_appends_audit_record_and_retry_is_idempotent(store, bad):
    config = configuration()
    if bad == 'temporal':
        config['validation']['min_separation_seconds'] = 10.
    assert store.update_graph(request(store, 'initialize', configuration=config))['status'] == 'accepted'
    candidate = loop(pose(10. if bad == 'cycle' else 1.) if bad != 'temporal' else None)
    if bad == 'temporal':
        candidate['to_id'] = 2
    before = database(store)
    update = request(store, 'add_loop', loop=candidate)
    result = store.update_graph(update)
    assert result['status'] == 'rejected', result
    after = database(store)
    assert len(after.pop('graph_requests')) == 2
    before.pop('graph_requests')
    assert before == after
    assert store.update_graph(update) == result


@pytest.mark.parametrize('change', ['unknown', 'zero', 'negative', 'asymmetric', 'nonfinite', 'inverse_ids', 'adjacent', 'no_provenance'])
def test_invalid_measurement_never_enters_graph(store, change):
    initialize(store)
    candidate = loop()
    if change == 'unknown': candidate['covariance_kind'] = 'unknown'
    if change == 'zero': candidate['covariance'] = np.zeros((6, 6)).tolist()
    if change == 'negative': candidate['covariance'][0][0] = -1.
    if change == 'asymmetric': candidate['covariance'][0][3] = .001
    if change == 'nonfinite': candidate['covariance'][0][0] = float('nan')
    if change == 'inverse_ids': candidate['from_id'], candidate['to_id'] = 4, 0
    if change == 'adjacent': candidate['to_id'] = 1
    if change == 'no_provenance': candidate['provenance'] = ''
    before = database(store)
    with pytest.raises(ValueError):
        store.update_graph(request(store, 'add_loop', loop=candidate))
    assert database(store) == before


def test_sessions_revision_conflicts_duplicate_edges_and_removed_ids(store):
    initialize(store)
    update = request(store, 'add_loop', loop=loop())
    assert store.update_graph(update)['status'] == 'accepted'
    before = database(store)
    attempts = [dict(update, session_id=str(uuid.uuid4())), dict(update, request_id='stale'),
                dict(update, loop=dict(loop(), covariance_model='changed')),
                request(store, 'add_loop', loop=dict(loop(), id='same-pair'))]
    for attempt in attempts:
        with pytest.raises(ValueError): store.update_graph(attempt)
        assert database(store) == before
    store.update_graph(request(store, 'remove_loop', loop_id=loop()['id'], reason='remove'))
    with pytest.raises(ValueError, match='already exists'):
        store.update_graph(request(store, 'add_loop', loop=loop()))


@pytest.mark.parametrize('failure', ['rebuild', 'ledger'])
def test_transaction_failure_rolls_back_graph_map_and_resident_state(store, monkeypatch, failure):
    initialize(store)
    before = database(store)
    cached = store.snapshot(.1).copy()
    def fail(*args): raise RuntimeError('injected transaction failure')
    monkeypatch.setattr(store, '_reconstruct_submap' if failure == 'rebuild' else '_record_graph_request', fail)
    with pytest.raises(RuntimeError, match='injected'):
        store.update_graph(request(store, 'add_loop', loop=loop()))
    assert database(store) == before
    np.testing.assert_array_equal(store.snapshot(.1), cached)


def test_external_rollback_detaches_graph_until_explicit_resolve(store):
    initialize(store)
    store.update_graph(request(store, 'add_loop', loop=loop()))
    store.restore_revision(store.meta['session_id'], 2, 0, request_id='manual-rollback')
    assert not store.stats()['graph_attached']
    assert store.stats()['active_loops'] == 1
    with pytest.raises(ValueError, match='detached'):
        store.update_graph(request(store, 'remove_loop', loop_id=loop()['id'], reason='remove'))
    assert store.update_graph(request(store, 'optimize'))['status'] == 'accepted'
    assert store.stats()['graph_attached']


def test_new_observations_extend_graph_with_time_scaled_odometry(store):
    initialize(store)
    store.ingest(6000000000, pose(1.), placed(room(), pose(1.)))
    result = store.update_graph(request(store, 'optimize'))
    assert result['status'] == 'accepted'
    assert store.stats()['graph_covered_observations'] == 6


def test_time_gap_rejected_and_initialization_noise_must_be_explicit(store):
    config = configuration()
    config['odometry_noise']['max_gap_seconds'] = .5
    result = store.update_graph(request(store, 'initialize', configuration=config))
    assert result['status'] == 'rejected' and result['reason'] == 'odometry_time_gap'
    assert store.meta['pose_revision'] == 0
    config['odometry_noise']['kind'] = 'calibrated'
    with pytest.raises(ValueError, match='assumed'): checked_configuration(config)


def test_loop_batch_has_one_revision_and_preserves_removal_history(store, tmp_path):
    initialize(store)
    before_revisions = store.db.execute('SELECT count(*) FROM pose_revisions').fetchone()[0]
    second = dict(loop(pose(-1.)), id='second-return', from_id=1, to_id=3)
    result = store.update_graph(request(store, 'add_loops', loops=[loop(), second]))
    assert result['status'] == 'accepted', result
    assert store.meta['pose_revision'] == 2
    assert store.stats()['active_loops'] == 2
    assert store.db.execute('SELECT count(*) FROM pose_revisions').fetchone()[0] == before_revisions + 1
    snapshot = tmp_path/'batch.dliomap'
    store.save(snapshot)
    loaded = Store(snapshot, store.limits)
    assert loaded.stats()['active_loops'] == 2
    loaded.close()
    for identifier in (loop()['id'], second['id']):
        assert store.update_graph(request(store, 'remove_loop', loop_id=identifier, reason='test removal'))['status'] == 'accepted'
    for original, optimized in store.db.execute('SELECT k.pose,o.pose FROM keyframes k JOIN optimized_poses o USING(id)'):
        np.testing.assert_allclose(decode_pose(original), decode_pose(optimized), atol=1e-7)


def test_rejected_batch_never_commits_its_successful_prefix(store):
    initialize(store)
    second = dict(loop(pose(20.)), id='false-return', from_id=1, to_id=3)
    before = database(store)
    result = store.update_graph(request(store, 'add_loops', loops=[loop(), second]))
    assert result['status'] == 'rejected'
    after = database(store)
    after.pop('graph_requests'); before.pop('graph_requests')
    assert before == after


def test_v2_upgrade_preserves_original_archive(store, tmp_path):
    source = tmp_path / 'version2.dliomap'
    store.save(source)
    with sqlite3.connect(source) as db:
        for table in ('graph_loops', 'graph_state', 'graph_requests'): db.execute(f'DROP TABLE {table}')
        db.execute('PRAGMA user_version=2')
        db.execute("UPDATE metadata SET json=json_set(json,'$.version',2)")
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    Store(source, store.limits).close()
    working = tmp_path / 'upgraded.dliomap'
    snapshot_database(source, working)
    upgraded = Store(working, store.limits, editable=True)
    try:
        assert upgraded.meta['version'] == 3
        initialize(upgraded)
    finally:
        upgraded.close()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == checksum


def test_offline_cli_retains_rejected_attempt_and_never_modifies_source(store, tmp_path):
    initialize(store)
    source, dest = tmp_path / 'source.dliomap', tmp_path / 'rejected.dliomap'
    store.save(source)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    req = tmp_path / 'request.json'
    req.write_text(json.dumps(request(store, 'add_loop', loop=loop(pose(10.)))))
    result = subprocess.run([sys.executable, str(SCRIPTS / 'mapping_archive.py'), 'graph', str(source), str(dest),
                             '--request', str(req)], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)['graph']['status'] == 'rejected'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    loaded = Store(dest, store.limits)
    try:
        assert loaded.meta['pose_revision'] == 1
        assert len(database(loaded)['graph_requests']) == 2
    finally:
        loaded.close()


def test_abrupt_death_after_factor_insertion_recovers_previous_graph_and_map(store, tmp_path):
    initialize(store)
    path = tmp_path / 'crash.dliomap'
    store.save(path)
    req = tmp_path / 'request.json'
    req.write_text(json.dumps(request(store, 'add_loop', loop=loop())))
    code = """
import json, os, sys
from pathlib import Path
from dliio_mapping.core import Limits, Store
value = Store(Path(sys.argv[1]), Limits(), editable=True)
value._record_graph_request = lambda *args: os._exit(23)
value.update_graph(json.loads(Path(sys.argv[2]).read_text()))
"""
    env = dict(os.environ, PYTHONPATH=str(SCRIPTS) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    result = subprocess.run([sys.executable, '-c', code, str(path), str(req)], env=env, capture_output=True, text=True)
    assert result.returncode == 23, result.stderr
    recovered = tmp_path / 'recovered.dliomap'
    snapshot_database(path, recovered)
    loaded = Store(recovered, store.limits)
    try:
        assert loaded.meta['pose_revision'] == 1
        assert loaded.stats()['active_loops'] == 0
        assert loaded.stats()['graph_attached']
        assert len(database(loaded)['graph_requests']) == 1
    finally:
        loaded.close()


def exp(delta):
    twist = np.zeros((4, 4))
    x, y, z = delta[3:]
    twist[:3, :3] = [[0, -z, y], [z, 0, -x], [-y, x, 0]]
    twist[:3, 3] = delta[:3]
    return expm(twist)


@pytest.mark.parametrize('delta', [[.2, -.1, .3, 0, 0, 0], [0, 0, 0, .4, -.2, 1.2], [.2, -.3, .1, .3, .4, -.7]])
def test_native_residual_uses_right_local_translation_first_with_lever_arm(delta):
    a, z = pose(5., .9), pose(2., -.6)
    b = a @ z @ exp(delta)
    np.testing.assert_allclose(native.residual(z, a, b), delta, atol=1e-12)


def test_native_full_covariance_matches_independent_numerical_optimizer():
    poses = [pose(), pose(1., .3), pose(2., .6)]
    measured = [np.linalg.inv(poses[0]) @ poses[1], np.linalg.inv(poses[1]) @ poses[2], pose(1.8, .8)]
    a = np.diag([.3, .2, .2, .15, .15, .2]); a[0, 5] = .12; a[1, 3] = -.08
    covariances = [a @ a.T, a @ a.T, np.diag([.08**2]*3 + [.07**2]*3)]
    result = native.optimize(poses, [0, 1, 0], [1, 2, 2], measured, covariances, [False, False, True])
    assert result['converged']
    pairs = [(0, 1), (1, 2), (0, 2)]
    roots = [np.linalg.cholesky(c) for c in covariances]
    def residuals(x):
        values = [poses[0], poses[1] @ exp(x[:6]), poses[2] @ exp(x[6:])]
        return np.concatenate([np.linalg.solve(root, native.residual(z, values[i], values[j]))
                               for (i, j), z, root in zip(pairs, measured, roots)])
    reference = least_squares(residuals, np.zeros(12), xtol=1e-12, gtol=1e-12, ftol=1e-12)
    for index in (1, 2):
        np.testing.assert_allclose(result['poses'][index], poses[index] @ exp(reference.x[(index-1)*6:index*6]), atol=1e-6)
    expected = [float(native.residual(z, result['poses'][i], result['poses'][j]) @
                     np.linalg.solve(c, native.residual(z, result['poses'][i], result['poses'][j])))
                for (i, j), z, c in zip(pairs, measured, covariances)]
    np.testing.assert_allclose(result['squared_errors'], expected, atol=1e-10)


@pytest.mark.parametrize('matrix', [np.zeros((6, 6)), -np.eye(6), np.full((6, 6), np.nan)])
def test_native_rejects_invalid_covariance(matrix):
    with pytest.raises(ValueError): native.optimize([pose(), pose(1.)], [0], [1], [pose(1.)], [matrix], [False])


def test_native_rejects_disconnected_chain():
    with pytest.raises(ValueError, match='disconnected'):
        native.optimize([pose(), pose(1.), pose(2.)], [0], [2], [pose(2.)], [np.eye(6)], [True])


@pytest.mark.parametrize('mode', ['inflated_loop', 'overconfident_odometry'])
def test_robust_kernel_and_covariance_cannot_hide_an_unsatisfied_loop(store, mode):
    config, candidate = configuration(), loop()
    if mode == 'inflated_loop':
        candidate['covariance'] = (np.eye(6) * 1e6).tolist()
    else:
        config['odometry_noise']['covariance_floor'] = (np.eye(6) * 1e-10).tolist()
        config['odometry_noise']['covariance_per_second'] = np.zeros((6, 6)).tolist()
    assert store.update_graph(request(store, 'initialize', configuration=config))['status'] == 'accepted'
    result = store.update_graph(request(store, 'add_loop', loop=candidate))
    assert result['status'] == 'rejected'
    assert result['reason'] in ('postfit_absolute_loop_limit', 'graph_residual_limit')
    assert store.meta['pose_revision'] == 1 and store.stats()['active_loops'] == 0


@pytest.mark.parametrize('edit', [
    "UPDATE graph_state SET configuration=json_set(configuration,'$.validation.max_cycle_translation',20)",
    "UPDATE graph_loops SET json=json_set(json,'$.covariance_model','tampered')",
    "DELETE FROM graph_requests WHERE json_extract(json,'$.request.action')='add_loop'",
])
def test_load_reconciles_factor_state_with_transaction_history(store, tmp_path, edit):
    initialize(store)
    assert store.update_graph(request(store, 'add_loop', loop=loop()))['status'] == 'accepted'
    path = tmp_path / 'corrupt.dliomap'
    store.save(path)
    with sqlite3.connect(path) as db:
        db.execute(edit)
    with pytest.raises(ValueError):
        Store(path, store.limits)


def test_removing_one_loop_retains_the_other_constraint(store, tmp_path):
    initialize(store)
    assert store.update_graph(request(store, 'add_loop', loop=loop()))['status'] == 'accepted'
    second = dict(loop(pose(1.)), id='second', to_id=3)
    response = store.update_graph(request(store, 'add_loop', loop=second))
    assert response['status'] == 'accepted', response
    response = store.update_graph(request(store, 'remove_loop', loop_id=loop()['id'], reason='remove one of two'))
    assert response['status'] == 'accepted', response
    assert store.stats()['active_loops'] == 1
    remaining = decode_pose(store.db.execute('SELECT pose FROM optimized_poses WHERE id=3').fetchone()[0])
    np.testing.assert_allclose(remaining[0, 3], 1., atol=.02)
    path = tmp_path / 'one-loop.dliomap'
    store.save(path)
    loaded = Store(path, store.limits)
    try:
        assert loaded.stats()['active_loops'] == 1 and loaded.stats()['graph_attached']
    finally:
        loaded.close()


def test_explicit_failed_registration_rejects_graph_but_keeps_original_observations(tmp_path):
    store = Store(tmp_path / 'quality.dliomap', Limits(), metadata=META)
    source = str(uuid.uuid4())
    try:
        for i in range(3):
            provenance = dict(source_session_id=source, source_sequence=i, covariance=None, covariance_kind='unknown',
                covariance_model='synthetic unavailable registered-pose covariance', quality=dict(
                    registration_converged=i != 1, degenerate_translation_modes=0, degenerate_rotation_modes=0))
            store.ingest((i+1)*1000000000, pose(i), placed(room(), pose(i)), observation=provenance)
        result = store.update_graph(request(store, 'initialize', configuration=configuration()))
        assert result['status'] == 'rejected' and result['reason'] == 'unconverged_odometry_observation'
        assert result['metrics']['observation_id'] == 1
        assert store.meta['pose_revision'] == 0 and store.meta['keyframes'] == 3
        path = tmp_path / 'quality-saved.dliomap'
        store.save(path)
        Store(path, store.limits).close()
    finally:
        store.close()


def noise_update(candidate, scale):
    return dict(id=candidate['id'], covariance=(np.asarray(candidate['covariance'])*scale**2).tolist(),
                covariance_kind='assumed', covariance_model='explicit synthetic reweighting')


def test_noise_reweighting_is_durable_and_preserves_measurement_history(store, tmp_path):
    initialize(store)
    candidate = loop()
    assert store.update_graph(request(store, 'add_loop', loop=candidate))['status'] == 'accepted'
    originals = database(store)['keyframes']
    update = request(store, 'reweight_loops', updates=[noise_update(candidate, .5)], reason='synthetic sensitivity test')
    result = store.update_graph(update)
    assert result['status'] == 'accepted' and result['pose_revision'] == 3
    assert store.update_graph(update) == result
    retained, added, removed = store.db.execute('SELECT json,added_revision,removed_revision FROM graph_loops').fetchone()
    retained = json.loads(retained)
    for key in ('id', 'from_id', 'to_id', 'transform', 'provenance'):
        assert retained[key] == candidate[key]
    assert (added, removed) == (2, None) and database(store)['keyframes'] == originals
    saved = tmp_path/'reweighted.dliomap'; store.save(saved)
    loaded = Store(saved, store.limits)
    assert loaded.stats()['graph_attached'] and loaded.stats()['active_loops'] == 1
    loaded.close()
    with sqlite3.connect(saved) as db:
        db.execute("UPDATE graph_loops SET json=json_set(json,'$.covariance_model','forged')")
    with pytest.raises(ValueError, match='request history'):
        Store(saved, store.limits)


def test_rejected_noise_update_leaves_graph_and_map_unchanged(store):
    initialize(store)
    candidate = loop()
    assert store.update_graph(request(store, 'add_loop', loop=candidate))['status'] == 'accepted'
    before = database(store)
    result = store.update_graph(request(store, 'reweight_loops', updates=[noise_update(candidate, 1e5)],
                                        reason='intentionally invalid weakening'))
    assert result['status'] == 'rejected' and result['reason'] == 'postfit_absolute_loop_limit'
    after = database(store)
    before.pop('graph_requests'); after.pop('graph_requests')
    assert after == before


@pytest.mark.parametrize('bad', ['unknown', 'duplicate', 'geometry', 'indefinite', 'empty_reason'])
def test_invalid_noise_update_cannot_change_measurements(store, bad):
    initialize(store)
    candidate = loop()
    assert store.update_graph(request(store, 'add_loop', loop=candidate))['status'] == 'accepted'
    update = noise_update(candidate, .5)
    if bad == 'unknown': update['id'] = 'missing-loop'
    if bad == 'geometry': update['transform'] = pose(2.).tolist()
    if bad == 'indefinite': update['covariance'][0][0] = -1.
    changes = [update, update] if bad == 'duplicate' else [update]
    before = database(store)
    with pytest.raises(ValueError):
        store.update_graph(request(store, 'reweight_loops', updates=changes,
                                    reason='' if bad == 'empty_reason' else 'synthetic malformed update'))
    assert database(store) == before
