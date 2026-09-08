"""Automatic, bounded revisit proposals from an immutable mapping archive.

Spatial proximity only proposes candidates. Fitting, competing solutions,
separate held observations and the actual graph validator decide admission.
Accepted factors are trial-solved in time order before a single atomic batch.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np

from .graph import configuration, encoded, loop_measurement
from .loop_inputs import window_cloud
from .loop_validation import RejectedLoop, Validation
from .registration import feature_suggestions, fit, register, rotation_angle, voxels
from .uncertainty import covariance


@dataclass(frozen=True)
class Retrieval:
    anchor_seconds: float = 15.
    anchor_distance: float = 2.
    exclusion_seconds: float = 60.
    min_path_separation: float = 30.
    search_radius: float = 18.
    candidate_separation_seconds: float = 10.
    candidates: int = 3
    candidate_pool: int = 12
    registration_workers: int = 2
    registration_backend: str = 'native'
    window_half_seconds: float = 3.
    max_loops: int = 32
    ambiguity_margin: float = .02
    held_translation: float = .2
    held_rotation: float = .035

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key == 'registration_backend':
                if value not in ('native', 'python'):
                    raise ValueError('Retrieval registration_backend must be native or python')
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'Retrieval {key} must be finite and positive')
        if type(self.candidates) is not int or not 1 <= self.candidates <= 10:
            raise ValueError('Retrieval candidates must be in [1, 10]')
        if type(self.candidate_pool) is not int or not self.candidates <= self.candidate_pool <= 32:
            raise ValueError('Retrieval candidate_pool must be between candidates and 32')
        if type(self.registration_workers) is not int or not 1 <= self.registration_workers <= 4:
            raise ValueError('Retrieval registration_workers must be in [1, 4]')
        if type(self.max_loops) is not int or not 1 <= self.max_loops <= 64:
            raise ValueError('Retrieval max_loops must be in [1, 64]')
        if self.window_half_seconds > 15 or self.exclusion_seconds <= 2*self.window_half_seconds:
            raise ValueError('Retrieval windows must be bounded and temporally independent')


def settings(value):
    if not isinstance(value, dict) or set(value) != {'graph', 'retrieval', 'loop_noise'}:
        raise ValueError('Loop closure needs graph, retrieval and explicit loop_noise settings')
    graph = configuration(value['graph'])
    retrieval = Retrieval(**value['retrieval'])
    if 'loop_windows' not in graph or retrieval.window_half_seconds > graph['loop_windows']['max_span_seconds']:
        raise ValueError('Retrieval windows must fit the graph support contract')
    noise = value['loop_noise']
    if not isinstance(noise, dict) or set(noise) != {'kind', 'model', 'covariance'} or noise['kind'] != 'assumed':
        raise ValueError('Automatic loop noise must be explicitly labelled assumed')
    if not isinstance(noise['model'], str) or not 1 <= len(noise['model'].strip()) <= 256:
        raise ValueError('Automatic loop noise needs a model description')
    noise = dict(noise, covariance=covariance(noise['covariance'], positive=True).tolist())
    return dict(graph=graph, retrieval=asdict(retrieval), loop_noise=noise)


class Windows:
    def __init__(self, store, stamps, quality, config, half_width):
        self.store, self.stamps, self.quality = store, np.asarray(stamps, dtype=np.int64), quality
        self.config, self.half_width = config, half_width
        self.validation = Validation(**config['validation'])
        self.cache = OrderedDict()

    def get(self, anchor):
        if anchor in self.cache:
            self.cache.move_to_end(anchor)
            return self.cache[anchor]
        identifiers = np.flatnonzero((abs(self.stamps-self.stamps[anchor])*1e-9 <= self.half_width) & self.quality)
        if anchor not in identifiers or not 4 <= len(identifiers) <= 128:
            raise ValueError('Insufficient verified window observations')
        fitting, held = identifiers[::2].tolist(), identifiers[1::2].tolist()
        clouds = [window_cloud(self.store, anchor, ids, self.config['max_span_seconds'], self.validation)
                  for ids in (fitting, held)]
        result = fitting, held, clouds[0], clouds[1]
        self.cache[anchor] = result
        while len(self.cache) > 8:
            self.cache.popitem(last=False)
        return result


def _registered(selected, source, windows, current, retrieval, feature_hints):
    """Only numeric registration runs concurrently; SQLite stays on its owner.

    Consume results in candidate order so scheduling cannot change admission or
    ambiguity checks. Reference/query arrays are bounded by the window policy.
    """
    if not selected:
        return
    with ThreadPoolExecutor(max_workers=retrieval.registration_workers,
                            thread_name_prefix='loop-fit') as executor:
        pending = []
        for target in selected:
            try:
                target_data, source_data = windows.get(target), windows.get(source)
                prior = np.linalg.inv(current[target]) @ current[source]
                future = executor.submit(register, source_data[2], target_data[2], prior,
                                         ambiguity_margin=retrieval.ambiguity_margin,
                                         backend=retrieval.registration_backend,
                                         feature_hint=feature_hints.get(target))
                pending.append((target, target_data, source_data, future, None))
            except ValueError as error:
                pending.append((target, None, None, None, error))
        for target, target_data, source_data, future, error in pending:
            result = None
            if error is None:
                try:
                    result = future.result()
                except ValueError as failure:
                    error = failure
            yield target, target_data, source_data, result, error


def _rank_candidates(pool, source, windows, current, retrieval):
    """Spatial drift must not make the nearest few poses the only hypotheses.

    Rank the wider bounded pool by geometrically consistent feature inliers;
    retain the nearest spatial hypothesis too. Features only choose which
    candidates to fit. Actual geometric/held/graph admission is unchanged.
    """
    hints, ranked = {}, []
    for order, target in enumerate(pool):
        try:
            hint = feature_suggestions(windows.get(source)[2], windows.get(target)[2],
                                       np.linalg.inv(current[target]) @ current[source])
            hints[target] = hint
            ranked.append((-max(hint[1].get('feature_inlier_counts', [0]) or [0]), order, target))
        except ValueError:
            continue
    ranked.sort()
    if not ranked or ranked[0][0] == 0:
        return pool[:retrieval.candidates], hints
    selected = [target for _, _, target in ranked[:max(1, retrieval.candidates-1)]]
    for target in pool:
        if len(selected) >= retrieval.candidates:
            break
        if target not in selected:
            selected.append(target)
    return selected, hints


def anchor_ids(store, config):
    """Append-stable original-motion anchors with mature, verified support."""
    retrieval = Retrieval(**config['retrieval'])
    stamps, poses = store._graph_originals()
    quality = [json.loads(row[0])['quality'].get('registration_converged') is True
               for row in store.db.execute('SELECT json FROM observations ORDER BY id')]
    result = []
    for i, stamp in enumerate(stamps):
        if not quality[i] or (stamps[-1]-stamp)*1e-9 < retrieval.window_half_seconds:
            continue
        if result and ((stamp-stamps[result[-1]])*1e-9 < retrieval.anchor_seconds or
                       np.linalg.norm(poses[i][:3, 3]-poses[result[-1]][:3, 3]) < retrieval.anchor_distance):
            continue
        result.append(i)
    return result


def propose(store, config, progress=None, *, start_source=0, last_anchor=None, max_sources=None, sources=None):
    """Read-only proposal and trial solve; no archive table or live TF changes."""
    config = settings(config)
    retrieval, graph, noise = Retrieval(**config['retrieval']), config['graph'], config['loop_noise']
    state = store.db.execute('SELECT configuration,pose_revision FROM graph_state WHERE id=1').fetchone()
    if state and (configuration(json.loads(state[0])) != graph or state[1] != store.meta['pose_revision']):
        raise ValueError('Automatic retrieval requires the attached graph and its original configuration')
    stamps, originals = store._graph_originals()
    if sources is not None and (not isinstance(sources, list) or len(sources) > 64 or
            sources != sorted(set(sources)) or any(type(i) is not int or not 0 <= i < len(originals) for i in sources)):
        raise ValueError('Explicit retrieval sources must be 0..64 ordered unique known observation IDs')
    stamps = np.asarray(stamps, dtype=np.int64)
    times = (stamps-stamps[0])*1e-9
    quality = np.array([json.loads(row[0])['quality'].get('registration_converged') is True
                        for row in store.db.execute('SELECT json FROM observations ORDER BY id')])
    active = [json.loads(row[0]) for row in store.db.execute(
        'SELECT json FROM graph_loops WHERE removed_revision IS NULL ORDER BY id')]
    previous_ids = {row[0] for row in store.db.execute('SELECT id FROM graph_loops')}
    active_sources = {loop['to_id'] for loop in active}
    current, baseline = store._solve_graph(stamps, originals, active, graph)
    current = np.asarray(current)
    distance_along_path = np.r_[0., np.cumsum(np.linalg.norm(np.diff(np.asarray(originals)[:, :3, 3], axis=0), axis=1))]
    windows = Windows(store, stamps, quality, graph['loop_windows'], retrieval.window_half_seconds)
    reports, accepted, processed, processed_sources = [], [], 0, []
    config_hash = hashlib.sha256(encoded(config).encode()).hexdigest()
    for source in (range(start_source, len(originals)) if sources is None else sources):
        if len(active)+len(accepted) >= retrieval.max_loops or (max_sources is not None and processed >= max_sources):
            break
        if not quality[source] or times[-1]-times[source] < retrieval.window_half_seconds:
            continue
        if sources is None and last_anchor is not None and (times[source]-times[last_anchor] < retrieval.anchor_seconds or
                np.linalg.norm(originals[source][:3, 3]-originals[last_anchor][:3, 3]) < retrieval.anchor_distance):
            continue
        last_anchor = source
        processed += 1
        processed_sources.append(source)
        if source in active_sources:
            continue
        positions = current[:, :3, 3]
        distance = np.linalg.norm(positions-positions[source], axis=1)
        eligible = np.flatnonzero(quality & (times < times[source]-retrieval.exclusion_seconds) &
                                  (distance_along_path[source]-distance_along_path > retrieval.min_path_separation) &
                                  (distance < retrieval.search_radius))
        selected = []
        for target in eligible[np.argsort(distance[eligible])]:
            if f'retrieved-{target}-{source}' in previous_ids:
                continue
            if any(abs(times[target]-times[previous]) < retrieval.candidate_separation_seconds for previous in selected):
                continue
            selected.append(int(target))
            if len(selected) >= retrieval.candidate_pool:
                break
        selected, feature_hints = _rank_candidates(selected, source, windows, current, retrieval)
        candidates = []
        for target, target_data, source_data, registration_result, error in _registered(
                selected, source, windows, current, retrieval, feature_hints):
            detail = dict(from_id=target, to_id=source, seconds=[float(times[target]), float(times[source])])
            try:
                if error is not None:
                    raise error
                tf, th, target_fit, target_held = target_data
                sf, sh, source_fit, source_held = source_data
                transform, registration = registration_result
                candidate = loop_measurement(dict(id=f'retrieved-{target}-{source}', from_id=target, to_id=source,
                    transform=transform.tolist(), covariance=noise['covariance'], covariance_kind=noise['kind'],
                    covariance_model=noise['model'],
                    provenance=f'Automatic spatial retrieval; distinct fitting/held original observations; '
                               f'bidirectional stability and competing-basin checks; configuration sha256 {config_hash}. '
                               'Conditional geometric evidence, not independent ground truth.',
                    support=dict(from_observations=tf, to_observations=sf,
                                 held_from_observations=th, held_to_observations=sh)), len(originals))
                checked = store._validate_candidate(candidate, current, stamps, graph)
                held_fit = fit(voxels(source_held, .15), voxels(target_held, .15), transform,
                               backend=retrieval.registration_backend)
                change = np.linalg.inv(transform) @ held_fit
                if np.linalg.norm(change[:3, 3]) > retrieval.held_translation or rotation_angle(change[:3, :3]) > retrieval.held_rotation:
                    raise ValueError('Held observations disagree with fitted transform')
                detail.update(status='geometry_passed', candidate=candidate, registration=registration, validation=checked,
                              held_refit_translation=float(np.linalg.norm(change[:3, 3])),
                              held_refit_rotation=rotation_angle(change[:3, :3]))
                candidates.append((registration['training_score'], candidate, detail))
            except (ValueError, RejectedLoop) as error:
                detail.update(status='rejected', reason=str(error), metrics=getattr(error, 'metrics', {}))
            reports.append(detail)
            if progress:
                progress(detail)
        candidates.sort(key=lambda item: item[0])
        if not candidates:
            continue
        for best_score, candidate, detail in candidates:
            corrected = current[candidate['from_id']] @ np.asarray(candidate['transform'])
            ambiguous = False
            for score, alternative, _ in candidates:
                pose = current[alternative['from_id']] @ np.asarray(alternative['transform'])
                difference = np.linalg.inv(corrected) @ pose
                distinct = np.linalg.norm(difference[:3, 3]) > .5 or rotation_angle(difference[:3, :3]) > math.radians(3)
                if distinct and score < best_score + retrieval.ambiguity_margin:
                    ambiguous = True
            if ambiguous:
                detail.update(status='rejected', reason='Ambiguous retrieved places')
                if progress: progress(detail)
                continue
            try:
                poses, metrics = store._solve_graph(stamps, originals, [*active, *accepted, candidate], graph)
            except RejectedLoop as error:
                detail.update(status='rejected', reason=str(error), metrics=error.metrics)
                if progress: progress(detail)
                continue
            accepted.append(candidate)
            current = np.asarray(poses)
            detail.update(status='accepted_for_batch', optimization=metrics)
            if progress: progress(detail)
            break
        if len(accepted) >= retrieval.max_loops:
            break
    return dict(configuration=config, initial_graph=baseline, candidates=reports, loops=accepted,
                source_session_id=store.meta['session_id'], source_pose_revision=store.meta['pose_revision'],
                last_anchor_id=last_anchor, processed_source_ids=processed_sources), current
