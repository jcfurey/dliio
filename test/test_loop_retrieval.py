"""Automatic candidates need geometric evidence; proposals never mutate a map."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from dliio_mapping.live_loops import LoopReadView
from dliio_mapping.loop_validation import RejectedLoop, Validation, validate_geometry
from dliio_mapping.registration import register, rotation_angle, transformed
from dliio_mapping.retrieval import propose
from graph_fixtures import room
from test_loop_inputs import make_store, window_config


def test_symmetric_registration_handles_unequal_density_and_partial_view():
    target = room()[:, :3].astype(float)
    # The second pass samples fewer surfaces and has a known SE(3) displacement.
    expected = np.eye(4)
    expected[:3, :3] = Rotation.from_rotvec([.04, -.08, .06]).as_matrix()
    expected[:3, 3] = [.9, -.3, .4]
    source = transformed(target[::2], np.linalg.inv(expected))
    measured, metrics = register(source, target, np.eye(4), offsets=(-2., 0., 2.))
    error = np.linalg.inv(expected) @ measured
    assert np.linalg.norm(error[:3, 3]) < .025
    assert rotation_angle(error[:3, :3]) < .005
    assert metrics['reverse_cycle_translation'] < .02


def test_plane_cannot_become_a_full_pose_loop():
    rng = np.random.default_rng(150)
    points = np.column_stack((rng.uniform(-10, 10, (8000, 2)), np.ones(8000)))
    with pytest.raises(RejectedLoop, match='unobservable_geometry'):
        validate_geometry(points, points, np.eye(4), Validation())


def test_registration_rejects_empty_or_unrelated_support():
    with pytest.raises(ValueError, match='support'):
        register(np.empty((0, 3)), room()[:, :3], np.eye(4))


def test_native_features_reject_invalid_dimensions_and_preserve_descriptor_order():
    import _dliio_pose_graph as native
    points = room()[:, :3].astype(float)
    descriptors = native.fpfh(points)
    assert descriptors.shape == (len(points), 33)
    assert np.isfinite(descriptors).all()
    for invalid in (points[:, :2], points[:50], points*float('nan')):
        with pytest.raises(ValueError):
            native.fpfh(invalid)


def config():
    return dict(graph=window_config(),
        retrieval=dict(anchor_seconds=3., anchor_distance=.05, exclusion_seconds=6.,
            min_path_separation=.1, search_radius=3., candidate_separation_seconds=2.,
            candidates=2, window_half_seconds=2., max_loops=2, ambiguity_margin=.01,
            held_translation=.2, held_rotation=.035),
        loop_noise=dict(kind='assumed', model='Synthetic independent loop test noise',
                        covariance=np.diag([.1**2]*3+[.03**2]*3).tolist()))


def test_retrieval_reduces_synthetic_drift_without_writing_archive(tmp_path):
    store = make_store(tmp_path)
    try:
        before = list(store.db.iterdump())
        report, corrected = propose(store, config())
        assert report['loops'], report['candidates']
        assert corrected[-1][0, 3] < .35
        assert list(store.db.iterdump()) == before
        for loop in report['loops']:
            assert loop['covariance_kind'] == 'assumed'
            assert len({i for ids in loop['support'].values() for i in ids}) == sum(map(len, loop['support'].values()))
    finally:
        store.close()


def test_live_read_view_is_consistent_across_writer_commits(tmp_path):
    store = make_store(tmp_path)
    try:
        view = LoopReadView(store.path)
        try:
            before = view.db.execute('SELECT json FROM metadata WHERE id=1').fetchone()[0]
            # A separate commit must not appear halfway through proposal work.
            metadata = json.loads(before)
            metadata['snapshot_test_marker'] = True
            with store.db:
                store.db.execute('UPDATE metadata SET json=? WHERE id=1', (json.dumps(metadata),))
            assert view.db.execute('SELECT json FROM metadata WHERE id=1').fetchone()[0] == before
            # Registration can stay open while the recorder checkpoints every
            # committed WAL page. A disk BEGIN used to pin those pages here.
            assert not view.source.in_transaction
            assert store.db.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone() == (0, 0, 0)
            assert view._graph_cloud(0).shape[1] == 7
            with pytest.raises(Exception, match='readonly'):
                view.db.execute('DELETE FROM keyframes')
        finally:
            view.close()
    finally:
        store.close()


def test_live_view_rejects_changed_original_observation(tmp_path):
    store = make_store(tmp_path)
    view = LoopReadView(store.path)
    try:
        with store.db:
            store.db.execute('UPDATE keyframes SET crc=crc+1 WHERE id=0')
        with pytest.raises(ValueError, match='changed after loop snapshot'):
            view._graph_cloud(0)
    finally:
        view.close()
        store.close()


def test_candidate_thread_count_does_not_change_the_geometric_solution(tmp_path):
    store = make_store(tmp_path)
    try:
        cfg = config()
        cfg['retrieval']['registration_workers'] = 1
        serial, a = propose(store, cfg)
        cfg['retrieval']['registration_workers'] = 2
        concurrent, b = propose(store, cfg)
        assert [x['id'] for x in serial['loops']] == [x['id'] for x in concurrent['loops']]
        np.testing.assert_allclose(a, b, atol=1e-10, rtol=0)
        assert store.meta['pose_revision'] == 0
    finally:
        store.close()


def test_features_retrieve_true_room_when_drift_makes_a_different_scene_nearest(tmp_path):
    import uuid
    from dliio_mapping.core import Store, Limits
    from graph_fixtures import pose, placed
    store = Store(tmp_path/'places.dliomap', Limits(submap_keyframes=4, resident_submaps=1),
        metadata=dict(odom_frame='odom', base_frame='base', map_frame='map',
                      map_to_odom=np.eye(4).tolist(), calibration={}))
    rng = np.random.default_rng(228)
    decoy = rng.normal(size=(3600, 3))
    decoy *= 5./np.linalg.norm(decoy, axis=1)[:, None]
    decoy = np.column_stack((decoy, np.ones((len(decoy), 4))))
    source = str(uuid.uuid4())
    try:
        for i in range(21):
            original = pose(.02*i if i < 7 else 2.+.01*i)
            local = decoy if 7 <= i < 14 else room()
            store.ingest((i+1)*1000000000, original, placed(local, original),
                observation=dict(source_session_id=source, source_sequence=i,
                    covariance_kind='unknown', covariance=None, covariance_model='Synthetic uncalibrated drift',
                    quality=dict(registration_converged=True,
                                 degenerate_translation_modes=0, degenerate_rotation_modes=0)))
        cfg = config()
        cfg['retrieval'].update(candidates=1, candidate_pool=8, search_radius=10.)
        report, _ = propose(store, cfg, sources=[17])
        assert report['loops'], report['candidates']
        # The pose-nearest earlier scene is the sphere at x~2; only the earlier
        # room near x~0 shares the actual retained scan geometry.
        assert report['loops'][0]['from_id'] <= 5
    finally:
        store.close()


def test_graph_rejection_still_allows_another_geometrically_valid_candidate(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    try:
        solve = store._solve_graph
        attempted = []
        def reject_first(stamps, originals, loops, graph):
            if loops:
                attempted.append(loops[-1]['from_id'])
                if len(attempted) == 1:
                    raise RejectedLoop('graph_residual_limit', dict(injected_conflicting_factor=True))
            return solve(stamps, originals, loops, graph)
        monkeypatch.setattr(store, '_solve_graph', reject_first)
        before = list(store.db.iterdump())
        report, corrected = propose(store, config(), sources=[12])
        assert len(attempted) == 2 and attempted[0] != attempted[1]
        assert report['loops'][0]['from_id'] == attempted[1]
        assert any(row.get('reason') == 'graph_residual_limit' for row in report['candidates'])
        assert corrected[-1][0, 3] < .35
        assert list(store.db.iterdump()) == before
    finally:
        store.close()


def test_mature_motion_anchors_stay_stable_when_observations_arrive(tmp_path):
    from dliio_mapping.retrieval import anchor_ids, settings
    from graph_fixtures import pose, placed
    store = make_store(tmp_path)
    try:
        cfg = settings(config())
        before = anchor_ids(store, cfg)
        source = json.loads(store.db.execute('SELECT json FROM observations WHERE id=0').fetchone()[0])['source_session_id']
        for i in range(16, 25):
            original = pose(.5 + (i-16)*.1)
            store.ingest((i+1)*1000000000, original, placed(room(), original),
                observation=dict(source_session_id=source, source_sequence=i,
                    covariance_kind='unknown', covariance=None, covariance_model='Synthetic unknown registration',
                    quality=dict(registration_converged=i != 18,
                                 degenerate_translation_modes=0, degenerate_rotation_modes=0)))
        after = anchor_ids(store, cfg)
        assert after[:len(before)] == before and len(after) > len(before)
        assert 18 not in after and max(after) <= 22  # Failed and immature observations stay ineligible.
    finally:
        store.close()
