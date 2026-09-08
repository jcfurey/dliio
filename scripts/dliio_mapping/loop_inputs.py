"""Explicit graph failure policy and provenance of multi-observation loops.

Original registration results are never relabelled. An optional bounded-motion
prior weakens edges touching an unconverged observation; it does not turn that
registration into a valid loop measurement. Window support is always rebuilt
from immutable individual observations in the declared anchor frame.
"""
import math
import json

import numpy as np

from .loop_validation import RejectedLoop, Validation
from .uncertainty import covariance

SUPPORT_KEYS = ('from_observations', 'to_observations',
                'held_from_observations', 'held_to_observations')
MAX_SUPPORT_OBSERVATIONS = 64


def failure_policy(value):
    required = {'kind', 'model', 'covariance_floor', 'max_consecutive',
                'max_translation_rate', 'max_rotation_rate'}
    if not isinstance(value, dict) or set(value) != required or value['kind'] != 'bounded_motion_prior':
        raise ValueError('Declare a bounded_motion_prior failure policy with its full limits and noise')
    if not isinstance(value['model'], str) or not 1 <= len(value['model'].strip()) <= 256:
        raise ValueError('Failed-registration motion prior needs an explicit assumed noise model')
    result = dict(value, covariance_floor=covariance(value['covariance_floor'], positive=True).tolist())
    if type(value['max_consecutive']) is not int or not 1 <= value['max_consecutive'] <= 100:
        raise ValueError('max_consecutive must be an integer in [1, 100]')
    for key in ('max_translation_rate', 'max_rotation_rate'):
        if type(value[key]) not in (float, int) or not math.isfinite(value[key]) or value[key] <= 0:
            raise ValueError('Failed-registration motion bounds must be finite and positive')
    return result


def window_configuration(value):
    if not isinstance(value, dict) or set(value) != {'max_span_seconds', 'validation'}:
        raise ValueError('Window configuration needs max_span_seconds and validation')
    span = value['max_span_seconds']
    if type(span) not in (float, int) or not math.isfinite(span) or not 0 < span <= 30:
        raise ValueError('Loop support span must be in (0, 30] seconds')
    from dataclasses import asdict
    return dict(max_span_seconds=span, validation=asdict(Validation(**value['validation'])))


def support_indices(value, count, from_id, to_id):
    if not isinstance(value, dict) or set(value) != set(SUPPORT_KEYS):
        raise ValueError('Window loops need disjoint fitting and held observation IDs for both anchors')
    result = {}
    for key in SUPPORT_KEYS:
        ids = value[key]
        if (not isinstance(ids, list) or not 2 <= len(ids) <= MAX_SUPPORT_OBSERVATIONS or
                any(type(i) is not int or not 0 <= i < count for i in ids) or ids != sorted(set(ids))):
            raise ValueError('Loop support IDs must be ordered, unique, bounded and known')
        result[key] = list(ids)
    all_ids = [i for ids in result.values() for i in ids]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError('Fitting, held and opposite-visit observations must be disjoint')
    earlier = result['from_observations'] + result['held_from_observations']
    later = result['to_observations'] + result['held_to_observations']
    if from_id not in earlier or to_id not in later or max(earlier) >= min(later):
        raise ValueError('Window support must contain its anchors and be ordered by visit')
    return result


def window_cloud(store, anchor, identifiers, maximum_span, settings):
    from .core import decode_pose
    selected_ids = sorted(set([anchor, *identifiers]))
    rows = {i: (stamp, decode_pose(blob)) for i, stamp, blob in store.db.execute(
        'SELECT id,stamp_ns,pose FROM keyframes WHERE id IN (' +
        ','.join('?' for _ in selected_ids) + ')', selected_ids)}
    stamp, pose = rows[anchor]
    inverse = np.linalg.inv(pose)
    chunks = []
    for identifier in identifiers:
        sample_stamp, local_pose = rows[identifier]
        if abs(sample_stamp - stamp) * 1e-9 > maximum_span:
            raise ValueError('Loop support exceeds its declared time span')
        quality = json.loads(store.db.execute('SELECT json FROM observations WHERE id=?',
                                             (identifier,)).fetchone()[0])['quality']
        if quality.get('registration_converged') is not True:
            raise RejectedLoop('unverified_loop_support', dict(observation_id=identifier))
        local = store._graph_cloud(identifier)[:, :3]
        transform = inverse @ local_pose
        xyz = local @ transform[:3, :3].T + transform[:3, 3]
        radius = np.linalg.norm(xyz, axis=1)
        xyz = xyz[np.isfinite(xyz).all(axis=1) & (radius >= settings.min_range) & (radius <= settings.max_range)]
        # Bound intermediate memory per observation before combining the window.
        if len(xyz):
            _, selected = np.unique(np.floor(xyz/settings.voxel_size).astype(np.int64), axis=0, return_index=True)
            chunks.append(xyz[selected])
    return np.vstack(chunks) if chunks else np.empty((0, 3))


def validate_support(store, candidate, configuration):
    from .loop_validation import validate_geometry
    settings = Validation(**configuration['validation'])
    support = candidate['support']
    result = {}
    for prefix, target_key, source_key in (
            ('fitting', 'from_observations', 'to_observations'),
            ('held', 'held_from_observations', 'held_to_observations')):
        target = window_cloud(store, candidate['from_id'], support[target_key],
                              configuration['max_span_seconds'], settings)
        source = window_cloud(store, candidate['to_id'], support[source_key],
                              configuration['max_span_seconds'], settings)
        try:
            result[prefix] = validate_geometry(target, source, candidate['transform'], settings)
        except RejectedLoop as error:
            raise RejectedLoop(prefix + '_' + str(error), {**result, prefix: error.metrics}) from error
    return result
