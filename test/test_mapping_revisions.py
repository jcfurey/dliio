"""Known nonuniform corrections, disk eviction, rollback, and crash boundaries."""
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import uuid

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from dliio_mapping.core import Limits, Store, Voxels, decode_points, decode_pose, snapshot_database

META = dict(odom_frame='odom', base_frame='base', map_frame='map',
            map_to_odom=np.eye(4).tolist(), calibration={})
LOCAL = np.array([[1., 0, 0, 2, np.nan, 4, 8], [0, 1, 0, 10, 20, np.nan, 12]], dtype='<f4')


def pose(x=0., yaw=0.):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0, x], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.]])


def placed(local, matrix):
    points = local.copy()
    points[:, :3] = local[:, :3] @ matrix[:3, :3].T + matrix[:3, 3]
    return points


def pcd(store, path, leaf=None):
    store.export_pcd(path, leaf)
    return np.frombuffer(path.read_bytes().split(b'DATA binary\n', 1)[1], '<f4').reshape(-1, 7)


def apply(store, poses, **kwargs):
    return store.apply_revision(store.meta['session_id'], store.meta['pose_revision'],
                                list(enumerate(poses)), request_id=kwargs.pop('request_id', str(uuid.uuid4())),
                                reason='known-transform fixture', **kwargs)


@pytest.fixture(params=[0., .05])
def store(tmp_path, request):
    value = Store(tmp_path / 'live.dliomap',
                  Limits(voxel_size=request.param, submap_keyframes=2, resident_submaps=1), metadata=META)
    for i in range(5):
        value.ingest(i+1, pose(i*3.), placed(LOCAL, pose(i*3.)))
    yield value
    value.close()


def rows(store):
    return {table: store.db.execute(f'SELECT * FROM {table}').fetchall() for table in
            ('metadata', 'keyframes', 'observations', 'optimized_poses', 'submaps', 'pose_revisions', 'revision_poses')}


def test_nonuniform_correction_rebuilds_evicted_submaps_and_preserves_sources(store, tmp_path):
    before = rows(store)
    corrected = [pose(20+i*3., np.pi/2 if i % 2 else 0.) for i in range(5)]
    store.snapshot(.1)  # prime fusion cache
    assert apply(store, corrected) == 1
    assert store.stats()['resident_submaps'] == 1
    np.testing.assert_allclose(store.snapshot(.1), Voxels.from_points(placed(LOCAL, corrected[-1]), .1).points(), atol=1e-6)
    actual = pcd(store, tmp_path / 'corrected.pcd', .1)
    expected = Voxels.from_points(np.vstack([placed(LOCAL, p) for p in corrected]), .1).points()
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    after = rows(store)
    assert after['keyframes'] == before['keyframes']
    assert after['observations'] == before['observations']
    store.save(tmp_path / 'saved.dliomap')
    loaded = Store(tmp_path / 'saved.dliomap', store.limits)
    try:
        assert loaded.meta['pose_revision'] == 1
        np.testing.assert_allclose(pcd(loaded, tmp_path / 'reloaded.pcd', .1), expected, atol=1e-6)
        np.testing.assert_array_equal(loaded.snapshot(.1), store.snapshot(.1))
    finally:
        loaded.close()


def test_retry_is_idempotent_and_stale_or_reused_requests_do_not_change_map(store):
    corrected = [pose(10+i) for i in range(5)]
    kwargs = dict(session_id=store.meta['session_id'], expected_revision=0,
                  poses=list(enumerate(corrected)), request_id='one', reason='fixture')
    assert store.apply_revision(**kwargs) == 1
    before = rows(store)
    assert store.apply_revision(**kwargs) == 1
    assert rows(store) == before
    for change in (dict(request_id='two'), dict(session_id=str(uuid.uuid4())), dict(reason='different')):
        with pytest.raises(ValueError):
            store.apply_revision(**dict(kwargs, **change))
        assert rows(store) == before


@pytest.mark.parametrize('ids', [[1], [0, 0], [0, 2], [True], [0, 1, 2, 3, 4, 5], []])
def test_invalid_coverage_is_rejected_atomically(store, ids):
    before = rows(store)
    with pytest.raises(ValueError):
        store.apply_revision(store.meta['session_id'], 0, [(i, pose()) for i in ids],
                             request_id='bad', reason='invalid coverage')
    assert rows(store) == before


def test_partial_optimization_preserves_odometric_tail_and_future_observations(store, tmp_path):
    corrected = [pose(10., np.pi/2), pose(20., np.pi/2)]
    apply(store, corrected)
    # Last optimized raw pose was at x=3. Later raw motion along +x must now
    # travel along +y, including observations ingested after the revision.
    store.ingest(6, pose(15.), placed(LOCAL, pose(15.)))
    actual = decode_pose(store.db.execute('SELECT pose FROM optimized_poses WHERE id=5').fetchone()[0])
    np.testing.assert_allclose(actual[:3, 3], [20., 12., 0.], atol=1e-12)
    assert store.meta['keyframes'] == 6
    store.save(tmp_path / 'tail.dliomap')
    Store(tmp_path / 'tail.dliomap', store.limits).close()


def test_rollback_is_a_new_revision_and_recovers_original_geometry(store, tmp_path):
    expected = pcd(store, tmp_path / 'original.pcd', .1)
    apply(store, [pose(20+i*3., np.pi/2) for i in range(5)])
    assert store.restore_revision(store.meta['session_id'], 1, 0, request_id='undo') == 2
    np.testing.assert_array_equal(pcd(store, tmp_path / 'restored.pcd', .1), expected)
    assert store.db.execute('SELECT count(*) FROM pose_revisions').fetchone()[0] == 3
    np.testing.assert_allclose(store.meta['map_to_odom'], np.eye(4), atol=1e-12)


def test_failed_rebuild_rolls_back_database_and_resident_geometry(store):
    cached = store.snapshot(.1).copy()
    before = rows(store)
    store.db.execute("CREATE TRIGGER fail_rebuild BEFORE UPDATE ON submaps WHEN OLD.id=2 "
                     "BEGIN SELECT RAISE(ABORT, 'injected rebuild failure'); END")
    with pytest.raises(sqlite3.IntegrityError, match='injected'):
        apply(store, [pose(20+i*3.) for i in range(5)])
    assert rows(store) == before
    np.testing.assert_array_equal(store.snapshot(.1), cached)


def test_abrupt_worker_exit_recovers_last_committed_revision(store, tmp_path):
    working = tmp_path / 'interrupted.dliomap'
    store.save(working)
    script = '''
import os, sys
import numpy as np
from dliio_mapping.core import Store, Limits
store = Store(sys.argv[1], Limits(), editable=True)
store.db.create_function('exit_worker', 0, lambda: os._exit(23))
store.db.execute('CREATE TRIGGER fail_mid_rebuild BEFORE UPDATE ON submaps WHEN OLD.id=1 '
                 'BEGIN SELECT exit_worker(); END')
poses = []
for i in range(store.meta['keyframes']):
    pose = np.eye(4)
    pose[0,3] = 30+i
    poses.append((i, pose))
store.apply_revision(store.meta['session_id'], 0, poses, request_id='interrupted', reason='crash fixture')
'''
    completed = subprocess.run([sys.executable, '-c', script, str(working)],
                               cwd=Path(__file__).resolve().parents[1] / 'scripts',
                               capture_output=True, text=True, timeout=15)
    assert completed.returncode == 23, completed.stderr
    recovered = tmp_path / 'recovered.dliomap'
    snapshot_database(working, recovered)
    loaded = Store(recovered, store.limits)
    try:
        assert loaded.meta['pose_revision'] == 0
        assert loaded.db.execute('SELECT count(*) FROM pose_revisions').fetchone()[0] == 1
        np.testing.assert_array_equal(loaded.snapshot(), store.snapshot())
        assert loaded.db.execute('SELECT * FROM optimized_poses').fetchall() == rows(store)['optimized_poses']
    finally:
        loaded.close()


def test_voxel_rebuild_uses_raw_sample_weights_and_active_append(tmp_path):
    store = Store(tmp_path / 'weighted.dliomap', Limits(voxel_size=.1), metadata=META)
    a = np.array([[.01, 0, 0, 2, np.nan, 4, 6], [.02, 0, 0, 4, 20, 8, 10]], dtype='<f4')
    b = np.array([[.03, 0, 0, 12, 40, np.nan, 20]], dtype='<f4')
    try:
        store.ingest(1, pose(), a)
        store.ingest(2, pose(), b)
        apply(store, [pose(1.), pose(1.)])
        np.testing.assert_allclose(store.snapshot()[0, 3:], [6., 30., 6., 12.])
        store.ingest(3, pose(), b)
        np.testing.assert_allclose(store.snapshot()[0, 3:], [7.5, 100/3, 6., 14.], rtol=1e-6)
    finally:
        store.close()


def test_corrected_voxel_capacity_failure_does_not_publish_partial_revision(tmp_path):
    store = Store(tmp_path / 'small.dliomap', Limits(voxel_size=.1, max_voxels=1), metadata=META)
    try:
        for i in range(2):
            store.ingest(i+1, pose(), LOCAL[:1])
        before = rows(store)
        with pytest.raises(ValueError, match='capacity'):
            apply(store, [pose(), pose(3.)])
        assert rows(store) == before
    finally:
        store.close()


def test_v1_read_compatibility_and_copy_upgrade(store, tmp_path):
    original = store.snapshot().copy()
    path = tmp_path / 'legacy.dliomap'
    store.save(path)
    with sqlite3.connect(path) as db:
        for table in ('revision_poses', 'pose_revisions', 'observations', 'optimized_poses'):
            db.execute(f'DROP TABLE {table}')
        db.execute('PRAGMA user_version=1')
        db.execute("UPDATE metadata SET json=json_set(json,'$.version',1)")
    legacy = Store(path, store.limits)
    try:
        assert legacy.meta['version'] == 1
        np.testing.assert_array_equal(legacy.snapshot(), original)
        legacy.save(tmp_path / 'editable.dliomap')
    finally:
        legacy.close()
    editable = Store(tmp_path / 'editable.dliomap', store.limits, editable=True)
    try:
        assert editable.meta['version'] == 2
        apply(editable, [pose(20+i*3.) for i in range(5)])
        editable.save(tmp_path / 'upgraded.dliomap')
    finally:
        editable.close()
    Store(tmp_path / 'upgraded.dliomap', store.limits).close()
    assert sqlite3.connect(path).execute('PRAGMA user_version').fetchone()[0] == 1


def test_observation_provenance_survives_revision_and_detects_source_restart(tmp_path):
    from test_mapping_uncertainty import metadata
    store = Store(tmp_path / 'observations.dliomap', Limits(), metadata=META)
    try:
        info = metadata()
        store.ingest(1, pose(), LOCAL, observation=info)
        info['source_sequence'] = 4  # a transport gap is allowed and visible in the stored IDs
        store.ingest(2, pose(), LOCAL, observation=info)
        before = rows(store)
        with pytest.raises(ValueError, match='sequence'):
            store.ingest(3, pose(), LOCAL, observation=info)
        assert rows(store) == before
        info['source_session_id'] = str(uuid.uuid4())
        with pytest.raises(ValueError, match='source session'):
            store.ingest(3, pose(), LOCAL, observation=info)
        apply(store, [pose(), pose(1.)])
        assert rows(store)['observations'] == before['observations']
        store.save(tmp_path / 'provenance.dliomap')
        Store(tmp_path / 'provenance.dliomap', store.limits).close()
    finally:
        store.close()
