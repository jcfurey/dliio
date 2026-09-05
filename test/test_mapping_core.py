"""Archive invariants and independently calculated geometry, without ROS."""
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from dliio_mapping.core import FIELDS, Limits, Store, Voxels, WindowFusion, decode_points, decode_pose, snapshot_database

META = dict(odom_frame='odom', base_frame='base', map_frame='map',
            map_to_odom=np.eye(4).tolist(), calibration={'extrinsics/baselink2imu/t': [.01, .02, .03]})


def points(x=0.):
    return np.array([[x, 0, 0, 12, 31, np.nan, 43],
                     [x + 1, 0, 0, 22, 51, 62, 73]], dtype=np.float32)


@pytest.fixture(params=[0., .2])
def store(tmp_path, request):
    result = Store(tmp_path / 'working.dliomap',
                   Limits(voxel_size=request.param, submap_keyframes=2, resident_submaps=2), metadata=META)
    yield result
    result.close()


def test_keyframes_are_local_and_each_pose_applied_once(store, tmp_path):
    pose = np.array([[0, -1, 0, 10], [1, 0, 0, 20], [0, 0, 1, 30], [0, 0, 0, 1.]])
    world = np.array([[10, 21, 30, 10, 50, 20, 40], [9, 20, 30, 30, 70, 40, 60]], dtype=np.float32)
    assert store.ingest(100, pose, world) == 0
    archived_pose, count, blob, crc = store.db.execute('SELECT pose,count,points,crc FROM keyframes').fetchone()
    np.testing.assert_array_equal(decode_pose(archived_pose), pose)
    local = decode_points(blob, count, crc, 10)
    np.testing.assert_allclose(local[:, :3], [[1, 0, 0], [0, 1, 0]], atol=1e-6)
    np.testing.assert_array_equal(local[:, 3:], world[:, 3:])
    np.testing.assert_allclose(sorted(store.snapshot()[:, :3].tolist()), sorted(world[:, :3].tolist()))
    output = tmp_path / 'map.pcd'
    assert store.export_pcd(output) == 2
    header, payload = output.read_bytes().split(b'DATA binary\n', 1)
    assert ('FIELDS ' + ' '.join(FIELDS)).encode() in header
    exported = np.frombuffer(payload, '<f4').reshape(-1, 7)
    np.testing.assert_array_equal(sorted(exported.tolist()), sorted(world.tolist()))


def test_voxels_weight_samples_per_channel_and_floor_negative_coordinates():
    a = np.array([[-.01, 0, 0, 2, np.nan, 4, 8], [-.02, 0, 0, 4, 20, 6, np.nan]], dtype=np.float32)
    b = np.array([[-.03, 0, 0, 12, 40, np.nan, 12], [.01, 0, 0, 100, 100, 100, 100]], dtype=np.float32)
    result = Voxels.from_points(a, .1).merged(Voxels.from_points(b, .1)).points()
    np.testing.assert_allclose(result, [[-.02, 0, 0, 6, 30, 5, 10], [.01, 0, 0, 100, 100, 100, 100]])


def test_incremental_fusion_reverses_eviction_and_reuses_storage():
    fusion = WindowFusion(.1)
    a = np.array([[-.01, 0, 0, 2, np.nan, 4, 8], [-.02, 0, 0, 4, 20, 6, np.nan]], dtype=np.float32)
    b = np.array([[-.03, 0, 0, 12, 40, np.nan, 12], [.01, 0, 0, 100, 100, 100, 100]], dtype=np.float32)
    fusion.update(a); fusion.update(b)
    np.testing.assert_allclose(fusion.points(), [[-.02, 0, 0, 6, 30, 5, 10], [.01, 0, 0, 100, 100, 100, 100]])
    fusion.update(a, remove=True)
    np.testing.assert_allclose(fusion.points(), b)
    capacity = fusion.nbytes
    for i in range(100):
        fusion.update(b, remove=True)
        assert len(fusion.points()) == 0
        b[:, 0] += 10
        fusion.update(b)
        np.testing.assert_allclose(fusion.points(), b)
    assert fusion.nbytes == capacity


def test_incremental_snapshot_matches_full_rebuild_across_rollovers(store):
    rng = np.random.default_rng(414)
    for i in range(20):
        pose = np.eye(4)
        pose[:3, 3] = [.3 * i, .02, -.01]
        world = rng.uniform(-1, 1, (80, 7)).astype('<f4')
        world[::3, 4] = np.nan
        store.ingest(i + 1, pose, world)
        expected = Voxels.from_points(store.snapshot(), .1).points()
        np.testing.assert_allclose(store.snapshot(.1), expected, rtol=1e-6, atol=1e-6)
        # Changing leaf size must rebuild the cache from retained sources.
        if i % 5 == 0:
            np.testing.assert_allclose(store.snapshot(.2), Voxels.from_points(store.snapshot(), .2).points(), atol=1e-6)


def test_resident_memory_stops_growing_while_archive_retains_all_frames(store, tmp_path):
    plateau = []
    for i in range(120):
        store.ingest(i + 1, np.eye(4), points(float(i * 2)))
        if i >= 4 and i % 2:
            plateau.append(store.stats()['resident_array_bytes'])
    assert len(set(plateau)) == 1
    assert store.stats()['resident_submaps'] == 2
    assert store.stats()['submaps'] == 60
    assert store.stats()['keyframes'] == 120
    assert len(store.snapshot()) == 8
    assert store.export_pcd(tmp_path / 'all.pcd') == 240
    assert store.db.execute('SELECT count(*) FROM keyframes').fetchone()[0] == 120


@pytest.mark.parametrize('leaf', [0., .2])
def test_voxel_capacity_rolls_submap_and_rejects_oversize_frame(tmp_path, leaf):
    store = Store(tmp_path / 'small.dliomap', Limits(voxel_size=leaf, max_voxels=2), metadata=META)
    try:
        store.ingest(1, np.eye(4), points())
        store.ingest(2, np.eye(4), points(5))
        assert store.stats()['submaps'] == 2
        with pytest.raises(ValueError, match='capacity'):
            store.ingest(3, np.eye(4), np.vstack([points(10), points(20)]))
        assert store.stats()['keyframes'] == 2
    finally:
        store.close()


def test_unvoxelized_map_keeps_coincident_returns_and_each_channel(tmp_path):
    store = Store(tmp_path / 'dense.dliomap', Limits(), metadata=META)
    expected = np.array([[1, 2, 3, 10, 20, np.nan, 30],
                         [1.001, 2, 3, 40, 50, 60, 70],
                         [1, 2, 3, 80, 90, 100, 110]], dtype='<f4')
    try:
        store.ingest(1, np.eye(4), expected[:2])
        pose = np.array([[0, -1, 0, 10], [1, 0, 0, 20], [0, 0, 1, 30], [0, 0, 0, 1.]])
        store.ingest(2, pose, expected[2:])
        np.testing.assert_array_equal(store.snapshot(), expected)
        saved, exported = tmp_path / 'saved.dliomap', tmp_path / 'all.pcd'
        store.save(saved)
        # Loading honors the archive's resolution rather than the viewer default.
        loaded = Store(saved, Limits(voxel_size=.2))
        try:
            assert loaded.limits.voxel_size == 0
            np.testing.assert_array_equal(loaded.snapshot(), expected)
            assert loaded.export_pcd(exported) == 3
            payload = exported.read_bytes().split(b'DATA binary\n', 1)[1]
            np.testing.assert_array_equal(np.frombuffer(payload, '<f4').reshape(-1, 7), expected)
        finally:
            loaded.close()
    finally:
        store.close()


def test_global_fusion_crosses_submaps_and_preserves_sample_weights(tmp_path):
    store = Store(tmp_path / 'working.dliomap', Limits(submap_keyframes=1), metadata=META)
    a = np.array([[.003, .004, 0, 2, 20, np.nan, 4], [.005, .004, 0, 4, np.nan, 8, 6],
                  [.03, .004, 0, 100, 100, 100, 100]], dtype='<f4')
    b = np.array([[.007, .004, 0, 12, 40, 10, np.nan]], dtype='<f4')
    pose = np.array([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1.]])
    expected = [[.005, .004, 0, 6, 30, 9, 5], [.03, .004, 0, 100, 100, 100, 100]]
    try:
        store.ingest(1, np.eye(4), a)
        store.ingest(2, pose, b)
        assert store.meta['submaps'] == 2
        assert len(store.snapshot()) == 4
        np.testing.assert_allclose(store.snapshot(.02), expected, atol=1e-6)
        output = tmp_path / 'fused.pcd'
        assert store.export_pcd(output, .02) == 2
        payload = output.read_bytes().split(b'DATA binary\n', 1)[1]
        np.testing.assert_allclose(np.frombuffer(payload, '<f4').reshape(-1, 7), expected, atol=1e-6)
        assert store.export_pcd(tmp_path / 'raw.pcd') == 4
        assert store.db.execute('SELECT sum(count) FROM keyframes').fetchone()[0] == 4
        store.save(tmp_path / 'saved.dliomap')
        loaded = Store(tmp_path / 'saved.dliomap', Limits())
        try:
            loaded.export_pcd(tmp_path / 'loaded.pcd', .02)
            assert output.read_bytes() == (tmp_path / 'loaded.pcd').read_bytes()
        finally:
            loaded.close()
        assert not list(tmp_path.glob('.dliio-fusion-*'))
    finally:
        store.close()


def test_global_fusion_keeps_distinct_cells_and_missing_channels(tmp_path):
    store = Store(tmp_path / 'working.dliomap', Limits(submap_keyframes=1, resident_submaps=1), metadata=META)
    a = np.array([[-.001, 0, 0, 2, np.nan, np.nan, 10], [.001, 0, 0, 4, 20, np.nan, 30]], dtype='<f4')
    try:
        store.ingest(1, np.eye(4), a)
        store.ingest(2, np.eye(4), a)
        assert store.stats()['resident_submaps'] == 1
        output = tmp_path / 'all.pcd'
        assert store.export_pcd(output, .02) == 2  # includes evicted submap, globally deduplicated
        payload = output.read_bytes().split(b'DATA binary\n', 1)[1]
        np.testing.assert_array_equal(np.frombuffer(payload, '<f4').reshape(-1, 7), a)
        np.testing.assert_array_equal(store.snapshot(.02), a)
    finally:
        store.close()


def test_failed_global_fusion_cleans_scratch_files_and_preserves_destination(store, tmp_path):
    store.ingest(1, np.eye(4), points())
    store.db.execute('UPDATE submaps SET crc=0')
    output = tmp_path / 'fused.pcd'
    with pytest.raises(ValueError, match='checksum'):
        store.export_pcd(output, .02)
    assert not output.exists()
    assert not list(tmp_path.glob('.dliio-fusion-*'))
    assert not list(tmp_path.glob('.fused.pcd.*'))


@pytest.mark.parametrize('stamp', [0, -1, 100, 99, 2 ** 63, 100.5, True])
def test_timestamp_rejection_leaves_archive_unchanged(store, stamp):
    store.ingest(100, np.eye(4), points())
    before = store.snapshot()
    with pytest.raises(ValueError):
        store.ingest(stamp, np.eye(4), points(2))
    assert store.meta['keyframes'] == 1
    np.testing.assert_array_equal(before, store.snapshot())


def test_transaction_failure_rolls_back_submap_and_metadata(store):
    store.ingest(1, np.eye(4), points())
    before = store.snapshot()
    store.db.execute("CREATE TRIGGER inject_failure BEFORE INSERT ON keyframes BEGIN SELECT RAISE(ABORT, 'injected'); END")
    with pytest.raises(sqlite3.IntegrityError):
        store.ingest(2, np.eye(4), points(2))
    assert store.meta['keyframes'] == 1
    assert json.loads(store.db.execute('SELECT json FROM metadata').fetchone()[0])['keyframes'] == 1
    assert store.db.execute('SELECT last_keyframe FROM submaps').fetchone()[0] == 0
    np.testing.assert_array_equal(store.snapshot(), before)


def test_save_includes_committed_wal_preserves_identity_and_reloads(store, tmp_path):
    for i in range(9):
        store.ingest(i + 1, np.eye(4), points(i * 2))
    saved = tmp_path / 'saved with spaces.dliomap'
    store.save(saved)
    loaded = Store(saved, store.limits)
    try:
        assert loaded.meta['sealed'] is True
        assert loaded.limits.voxel_size == store.limits.voxel_size
        assert loaded.meta['session_id'] == store.meta['session_id']
        assert loaded.meta['calibration'] == META['calibration']
        assert loaded.meta['keyframes'] == 9
        np.testing.assert_array_equal(loaded.snapshot(), store.snapshot())
        for table in ('keyframes', 'submaps'):
            assert loaded.db.execute(f'SELECT * FROM {table}').fetchall() == store.db.execute(f'SELECT * FROM {table}').fetchall()
        with pytest.raises(ValueError, match='read-only'):
            loaded.ingest(10, np.eye(4), points())
        store.export_pcd(tmp_path / 'live.pcd')
        loaded.export_pcd(tmp_path / 'loaded.pcd')
        assert (tmp_path / 'live.pcd').read_bytes() == (tmp_path / 'loaded.pcd').read_bytes()
    finally:
        loaded.close()
    assert store.meta['sealed'] is False
    assert not Path(str(saved) + '-wal').exists()


def test_recovery_snapshots_closed_working_database(store, tmp_path):
    store.ingest(1, np.eye(4), points())
    with pytest.raises(ValueError, match='sealed'):
        Store(store.path, store.limits)
    destination = tmp_path / 'recovered.dliomap'
    snapshot_database(store.path, destination)
    loaded = Store(destination, store.limits)
    loaded.close()


@pytest.mark.parametrize('operation', ['save', 'export_pcd'])
def test_outputs_never_overwrite_existing_files(store, tmp_path, operation):
    store.ingest(1, np.eye(4), points())
    destination = tmp_path / 'existing'
    destination.write_bytes(b'keep me')
    with pytest.raises(ValueError, match='new absolute'):
        getattr(store, operation)(destination)
    assert destination.read_bytes() == b'keep me'
    assert not list(tmp_path.glob('.existing.*'))


@pytest.mark.parametrize('sql', [
    'PRAGMA user_version=99',
    'UPDATE keyframes SET crc=0',
    'UPDATE keyframes SET points=zeroblob(28)',
    'UPDATE keyframes SET count=1000001',
    'UPDATE keyframes SET submap_id=123',
    'UPDATE submaps SET stamp_ns=999',
    'UPDATE submaps SET first_keyframe=1',
    'UPDATE submaps SET crc=0',
    "UPDATE metadata SET json=json_set(json, '$.keyframes', 999)",
    "UPDATE metadata SET json=json_set(json, '$.map_to_odom[0][3]', 10)",
])
def test_corrupt_archives_fail_validation(store, tmp_path, sql):
    store.ingest(1, np.eye(4), points())
    destination = tmp_path / 'broken.dliomap'
    store.save(destination)
    with sqlite3.connect(destination) as connection:
        connection.execute(sql)
    with pytest.raises((ValueError, sqlite3.DatabaseError)):
        Store(destination, store.limits)


def test_empty_archive_roundtrip(store, tmp_path):
    store.save(tmp_path / 'empty.dliomap')
    loaded = Store(tmp_path / 'empty.dliomap', store.limits)
    try:
        assert loaded.snapshot().shape == (0, 7)
        with pytest.raises(ValueError, match='empty'):
            loaded.export_pcd(tmp_path / 'empty.pcd')
        assert not (tmp_path / 'empty.pcd').exists()
    finally:
        loaded.close()


@pytest.mark.parametrize('changes', [{'voxel_size': float('nan')}, {'voxel_size': -.01},
                                    {'voxel_size': .0001}, {'voxel_size': float('inf')}, {'resident_submaps': 0},
                                    {'max_voxels': 1000001}, {'max_input_points': True}])
def test_invalid_limits(changes):
    with pytest.raises(ValueError):
        Limits(**changes)
