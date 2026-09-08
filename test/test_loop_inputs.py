"""Failed observations stay honest; window loops must pass independent support."""
import json
from pathlib import Path
import sys
import uuid

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from dliio_mapping.core import Limits, Store
from dliio_mapping.graph import configuration as checked_configuration
from dliio_mapping.loop_inputs import support_indices
from graph_fixtures import configuration, loop, placed, pose, room


def policy():
    return dict(kind='bounded_motion_prior', model='explicitly assumed synthetic failed-solve motion noise',
                covariance_floor=np.diag([.25**2]*3+[.08**2]*3).tolist(),
                max_consecutive=3, max_translation_rate=3., max_rotation_rate=2.)


def request(store, action, **extra):
    return dict(session_id=store.meta['session_id'], expected_revision=store.meta['pose_revision'],
                request_id=str(uuid.uuid4()), action=action, **extra)


def make_store(tmp_path, bad=()):
    value = Store(tmp_path/'map.dliomap', Limits(submap_keyframes=4, resident_submaps=1),
                  metadata=dict(odom_frame='odom', base_frame='base', map_frame='map',
                                map_to_odom=np.eye(4).tolist(), calibration={}))
    session = str(uuid.uuid4())
    for i in range(16):
        truth = pose(.1*min(i, 15-i))
        original = pose(truth[0, 3]+i*.04)
        value.ingest((i+1)*1000000000, original, placed(placed(room(), np.linalg.inv(truth)), original),
                     observation=dict(source_session_id=session, source_sequence=i,
                        covariance_kind='unknown', covariance=None, covariance_model='synthetic unknown registration',
                        quality=dict(registration_converged=i not in bad,
                                     degenerate_translation_modes=0, degenerate_rotation_modes=0)))
    return value


def window_loop():
    return dict(loop(), id='window-return', from_id=2, to_id=13,
                support=dict(from_observations=[0, 2, 4], held_from_observations=[1, 3, 5],
                             to_observations=[10, 12, 14], held_to_observations=[11, 13, 15]))


def window_config():
    return dict(configuration(), loop_windows=dict(max_span_seconds=4., validation={}))


def test_opt_in_motion_prior_keeps_original_failure_flags_and_weakens_incident_edges(tmp_path):
    store = make_store(tmp_path, bad=[7, 8])
    try:
        originals = store.db.execute('SELECT * FROM observations').fetchall()
        rejected = store.update_graph(request(store, 'initialize', configuration=configuration()))
        assert rejected['reason'] == 'unconverged_odometry_observation'
        accepted = store.update_graph(request(store, 'initialize', configuration=dict(configuration(), failure_policy=policy())))
        assert accepted['status'] == 'accepted', accepted
        metrics = accepted['metrics']['optimization']
        assert metrics['unconverged_observations'] == [7, 8]
        assert metrics['assumed_motion_prior_edges'] == [7, 8, 9]
        assert store.db.execute('SELECT * FROM observations').fetchall() == originals
    finally:
        store.close()


def test_long_failed_run_remains_rejected(tmp_path):
    store = make_store(tmp_path, bad=[6, 7, 8, 9])
    try:
        result = store.update_graph(request(store, 'initialize', configuration=dict(configuration(), failure_policy=policy())))
        assert result['reason'] == 'unconverged_run_limit'
        assert store.meta['pose_revision'] == 0
    finally:
        store.close()


def test_window_loop_rebuilds_and_validates_both_observation_splits(tmp_path):
    store = make_store(tmp_path)
    try:
        assert store.update_graph(request(store, 'initialize', configuration=window_config()))['status'] == 'accepted'
        result = store.update_graph(request(store, 'add_loop', loop=window_loop()))
        assert result['status'] == 'accepted', result
        assert result['metrics']['fitting']['forward_overlap'] > .98
        assert result['metrics']['held']['forward_overlap'] > .98
        source = store.db.execute('SELECT pose FROM keyframes WHERE id=13').fetchone()[0]
        corrected = store.db.execute('SELECT pose FROM optimized_poses WHERE id=13').fetchone()[0]
        assert np.frombuffer(corrected, '<f8').reshape(4, 4)[0, 3] < np.frombuffer(source, '<f8').reshape(4, 4)[0, 3]-.2
    finally:
        store.close()


def test_bad_held_observation_cannot_hide_behind_a_good_fit(tmp_path):
    store = make_store(tmp_path, bad=[11])
    try:
        config = dict(window_config(), failure_policy=policy())
        assert store.update_graph(request(store, 'initialize', configuration=config))['status'] == 'accepted'
        result = store.update_graph(request(store, 'add_loop', loop=window_loop()))
        assert result['reason'] == 'unverified_loop_support'
        assert store.stats()['active_loops'] == 0
    finally:
        store.close()


def test_support_cannot_reuse_fitting_observations_as_validation():
    candidate = window_loop()
    candidate['support']['held_from_observations'] = [0, 1]
    with pytest.raises(ValueError, match='disjoint'):
        support_indices(candidate['support'], 16, 2, 13)


@pytest.mark.parametrize('key,value', [('max_consecutive', True), ('max_translation_rate', float('nan')),
                                      ('covariance_floor', np.zeros((6, 6)).tolist()), ('model', '')])
def test_invalid_failure_noise_cannot_be_enabled(key, value):
    p = policy(); p[key] = value
    with pytest.raises(ValueError):
        checked_configuration(dict(configuration(), failure_policy=p))
