"""Auditable loop injection, batch optimization, and atomic graph/map commits.

The archive worker owns this API. Retrieval/registration remains external;
supplied candidates are checked against original clouds before a trial solve.
Rejected trials change only the request ledger. Accepted graph changes share
the SQLite transaction that installs corrected poses and reconstructs the map.
"""
from dataclasses import asdict
import hashlib
import json
import math

import numpy as np

from .loop_validation import Validation, RejectedLoop, check_cycle, displacement, validate_geometry
from .uncertainty import covariance

MAX_GRAPH_POSES = 5000
MAX_LOOPS = 15000
MAX_REQUESTS = 100000
MAX_REQUEST_BYTES = 1024 * 1024


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _text(value, maximum, name):
    if not isinstance(value, str) or not 1 <= len(value) <= maximum or not value.strip():
        raise ValueError(f'{name} must contain 1..{maximum} characters')
    return value


def configuration(value):
    if (not isinstance(value, dict) or not {'odometry_noise', 'validation'} <= set(value) or
            set(value) - {'odometry_noise', 'validation', 'failure_policy', 'loop_windows'}):
        raise ValueError('Graph configuration requires odometry_noise and validation')
    noise = value['odometry_noise']
    if not isinstance(noise, dict) or set(noise) != {'kind', 'model', 'covariance_floor',
            'covariance_per_second', 'covariance_per_metre', 'max_gap_seconds'}:
        raise ValueError('Declare the full relative odometry process-noise model')
    if noise['kind'] != 'assumed':
        raise ValueError('The odometry process-noise model must be explicitly labelled assumed')
    noise = dict(noise, model=_text(noise['model'], 256, 'Odometry noise model'))
    for name in ('covariance_floor', 'covariance_per_second', 'covariance_per_metre'):
        noise[name] = covariance(noise[name], positive=name == 'covariance_floor').tolist()
    gap = noise['max_gap_seconds']
    if type(gap) not in (int, float) or not math.isfinite(gap) or gap <= 0:
        raise ValueError('Odometry maximum gap must be finite and positive')
    try:
        settings = Validation(**value['validation'])
    except TypeError as error:
        raise ValueError('Unknown or invalid loop validation settings') from error
    result = dict(odometry_noise=noise, validation=asdict(settings))
    from .loop_inputs import failure_policy, window_configuration
    if 'failure_policy' in value:
        result['failure_policy'] = failure_policy(value['failure_policy'])
    if 'loop_windows' in value:
        result['loop_windows'] = window_configuration(value['loop_windows'])
    return result


def loop_measurement(value, count):
    from .core import rigid_pose
    required = {'id', 'from_id', 'to_id', 'transform', 'covariance', 'covariance_kind',
                'covariance_model', 'provenance'}
    if not isinstance(value, dict) or not required <= set(value) or set(value) - required - {'support', 'surface_subspace'}:
        raise ValueError('Loop measurement fields are incomplete or unknown')
    value = dict(value)
    _text(value['id'], 128, 'Loop ID')
    a, b = value['from_id'], value['to_id']
    if type(a) is not int or type(b) is not int or not 0 <= a < b - 1 < count - 1:
        raise ValueError('Loop IDs must be ordered, nonadjacent, known archive observations')
    value['transform'] = rigid_pose(value['transform']).tolist()
    value['covariance'] = covariance(value['covariance'], positive=True).tolist()
    if value['covariance_kind'] not in ('assumed', 'conditional', 'calibrated'):
        raise ValueError('Unknown covariance cannot become a loop factor')
    _text(value['covariance_model'], 256, 'Loop covariance model')
    _text(value['provenance'], 1024, 'External loop verification provenance')
    if 'support' in value:
        from .loop_inputs import support_indices
        value['support'] = support_indices(value['support'], count, a, b,
                                           allow_single_fitting='surface_subspace' in value)
    if 'surface_subspace' in value:
        from .surface_subspace import subspace
        if 'support' not in value or value['covariance_kind'] != 'assumed':
            raise ValueError('Projected surface loops require disjoint window support and explicitly assumed noise')
        value['surface_subspace'] = subspace(value['surface_subspace'])
        length = value['surface_subspace']['characteristic_length']
        scale = np.diag([1., 1., 1., length, length, length])
        scaled_covariance = scale @ np.asarray(value['covariance']) @ scale
        variance = scaled_covariance.trace()/6
        if not np.allclose(scaled_covariance, np.eye(6)*variance, atol=1e-12, rtol=1e-8):
            raise ValueError('Projected loop noise must be isotropic in the declared length-scaled tangent')
    return value


def _request(value):
    if not isinstance(value, dict):
        raise ValueError('Graph request must be a JSON object')
    required = {'session_id', 'expected_revision', 'request_id', 'action'}
    action_fields = dict(initialize={'configuration'}, add_loop={'loop'}, add_loops={'loops'},
                         remove_loop={'loop_id', 'reason'}, reweight_loops={'updates', 'reason'}, optimize=set())
    action = value.get('action')
    if not isinstance(action, str) or action not in action_fields or set(value) != required | action_fields[action]:
        raise ValueError('Unknown graph action or unexpected/missing request fields')
    _text(value['request_id'], 128, 'Graph request ID')
    _text(value['session_id'], 36, 'Archive session ID')
    if type(value['expected_revision']) is not int or value['expected_revision'] < 0:
        raise ValueError('Invalid expected graph/map revision')
    if action == 'add_loops' and (not isinstance(value['loops'], list) or not 1 <= len(value['loops']) <= 64):
        raise ValueError('A loop batch must contain 1..64 candidates')
    if action == 'reweight_loops':
        _text(value['reason'], 1024, 'Loop reweighting reason')
        if not isinstance(value['updates'], list) or not 1 <= len(value['updates']) <= 64:
            raise ValueError('Reweight 1..64 explicit active loop IDs')
    serialized = encoded(value)
    if len(serialized.encode('utf8')) > MAX_REQUEST_BYTES:
        raise ValueError('Graph request exceeds 1 MiB')
    return hashlib.sha256(serialized.encode('utf8')).hexdigest()


def reweighted(loops, updates, count):
    """Change only noise; immutable measurements and support stay byte-equivalent."""
    values = {loop['id']: loop for loop in loops}
    seen = set()
    for update in updates:
        if (not isinstance(update, dict) or set(update) != {'id', 'covariance', 'covariance_kind', 'covariance_model'}
                or not isinstance(update['id'], str) or update['id'] in seen or update['id'] not in values):
            raise ValueError('Noise updates require unique active loop IDs and only covariance fields')
        identifier = update['id']
        values[identifier] = loop_measurement(dict(values[identifier], **update), count)
        seen.add(identifier)
    return [values[loop['id']] for loop in loops]


class GraphStore:
    def _create_graph_tables(self):
        for sql in (
            'CREATE TABLE graph_state (id INTEGER PRIMARY KEY CHECK(id=1), configuration TEXT NOT NULL, '
            'pose_revision INTEGER REFERENCES pose_revisions(id), covered_observations INTEGER NOT NULL)',
            'CREATE TABLE graph_loops (id TEXT PRIMARY KEY, from_id INTEGER NOT NULL REFERENCES keyframes(id), '
            'to_id INTEGER NOT NULL REFERENCES keyframes(id), json TEXT NOT NULL, '
            'added_revision INTEGER NOT NULL REFERENCES pose_revisions(id), '
            'removed_revision INTEGER REFERENCES pose_revisions(id))',
            'CREATE UNIQUE INDEX active_loop_pair ON graph_loops(from_id,to_id) WHERE removed_revision IS NULL',
            'CREATE TABLE graph_requests (request_id TEXT PRIMARY KEY, digest TEXT NOT NULL, json TEXT NOT NULL)',
        ):
            self.db.execute(sql)

    def graph_stats(self):
        if self.meta['version'] < 3:
            return dict(graph_initialized=False, graph_attached=False, active_loops=0)
        state = self.db.execute('SELECT pose_revision,covered_observations FROM graph_state WHERE id=1').fetchone()
        return dict(graph_initialized=state is not None,
            graph_attached=state is not None and state[0] == self.meta['pose_revision'],
            graph_covered_observations=state[1] if state else 0,
            active_loops=self.db.execute('SELECT count(*) FROM graph_loops WHERE removed_revision IS NULL').fetchone()[0])

    def _validate_graph(self):
        for table, column, limit in (('graph_state', 'configuration', 16384),
                ('graph_loops', 'json', 16384), ('graph_requests', 'json', MAX_REQUEST_BYTES * 2)):
            if self.db.execute(f'SELECT 1 FROM {table} WHERE length({column})>? LIMIT 1', (limit,)).fetchone():
                raise ValueError('Oversized graph record')
        if self.db.execute('SELECT count(*) FROM graph_requests').fetchone()[0] > MAX_REQUESTS:
            raise ValueError('Graph request ledger exceeds capacity')
        if self.db.execute('SELECT count(*) FROM graph_loops').fetchone()[0] > MAX_LOOPS:
            raise ValueError('Graph loop ledger exceeds capacity')
        state = self.db.execute('SELECT configuration,pose_revision,covered_observations FROM graph_state WHERE id=1').fetchone()
        if state:
            configuration(json.loads(state[0]))
            if (state[1] is not None and state[1] != self.meta['pose_revision']) or not 2 <= state[2] <= min(
                    self.meta['keyframes'], MAX_GRAPH_POSES):
                raise ValueError('Graph solution differs from its committed map revision or coverage')
        for identifier, a, b, payload, added, removed in self.db.execute('SELECT * FROM graph_loops'):
            value = loop_measurement(json.loads(payload), self.meta['keyframes'])
            if (state is None or (value['id'], value['from_id'], value['to_id']) != (identifier, a, b) or
                    not 1 <= added <= self.meta['pose_revision'] or
                    (removed is not None and not added < removed <= self.meta['pose_revision'])):
                raise ValueError('Invalid loop history')
        expected_loops, expected_config, last_solution, coverage = {}, None, None, None
        for identifier, digest, payload in self.db.execute('SELECT * FROM graph_requests ORDER BY rowid'):
            value = json.loads(payload)
            encoded(value)  # JSON's non-standard NaN/Infinity literals are invalid here.
            if (set(value) != {'request', 'report'} or _request(value['request']) != digest or
                    value['request']['request_id'] != identifier or
                    value['request']['session_id'] != self.meta['session_id'] or
                    value['report']['status'] not in ('accepted', 'rejected') or
                    not 0 <= value['report']['pose_revision'] <= self.meta['pose_revision']):
                raise ValueError('Invalid graph request history')
            request, report = value['request'], value['report']
            accepted = report['status'] == 'accepted'
            if (report['action'] != request['action'] or report['request_id'] != identifier or
                    report['pose_revision'] != request['expected_revision'] + int(accepted)):
                raise ValueError('Graph report differs from its request')
            revision_row = self.db.execute('SELECT id,last_keyframe FROM pose_revisions WHERE request_id=?',
                                           ('graph:' + digest,)).fetchone()
            if not accepted:
                if revision_row is not None:
                    raise ValueError('Rejected loop has a committed pose revision')
                continue
            if revision_row is None or revision_row[0] != report['pose_revision']:
                raise ValueError('Accepted graph request has no matching pose revision')
            if last_solution is not None and report['pose_revision'] <= last_solution:
                raise ValueError('Graph solution history is out of order')
            last_solution, coverage = revision_row[0], revision_row[1] + 1
            action = request['action']
            if action == 'initialize':
                if expected_config is not None or request['expected_revision'] != 0:
                    raise ValueError('Invalid graph initialization history')
                expected_config = configuration(request['configuration'])
            elif expected_config is None:
                raise ValueError('Graph action predates initialization')
            if action in ('add_loop', 'add_loops'):
                for value in ([request['loop']] if action == 'add_loop' else request['loops']):
                    loop = loop_measurement(value, self.meta['keyframes'])
                    if loop['id'] in expected_loops:
                        raise ValueError('Loop history reuses an ID')
                    expected_loops[loop['id']] = [loop, last_solution, None]
            elif action == 'remove_loop':
                entry = expected_loops.get(request['loop_id'])
                if entry is None or entry[2] is not None:
                    raise ValueError('Loop removal history has no active factor')
                entry[2] = last_solution
            elif action == 'reweight_loops':
                active = [entry[0] for entry in expected_loops.values() if entry[2] is None]
                for loop in reweighted(active, request['updates'], self.meta['keyframes']):
                    expected_loops[loop['id']][0] = loop
        if (state is None) != (expected_config is None):
            raise ValueError('Graph state has no matching initialization record')
        if state is not None:
            attached_revision = last_solution if last_solution == self.meta['pose_revision'] else None
            if (encoded(configuration(json.loads(state[0]))) != encoded(expected_config) or
                    state[1:] != (attached_revision, coverage)):
                raise ValueError('Graph state differs from request history')
        actual = {identifier: [json.loads(payload), added, removed] for identifier, payload, added, removed in
                  self.db.execute('SELECT id,json,added_revision,removed_revision FROM graph_loops')}
        if actual != expected_loops:
            raise ValueError('Loop factors differ from request history')

    def _graph_originals(self):
        from .core import decode_pose
        if not 2 <= self.meta['keyframes'] <= MAX_GRAPH_POSES:
            raise ValueError(f'Graph requires 2..{MAX_GRAPH_POSES} committed observations')
        rows = self.db.execute('SELECT stamp_ns,pose FROM keyframes ORDER BY id').fetchall()
        return [row[0] for row in rows], [decode_pose(row[1]) for row in rows]

    def _graph_cloud(self, identifier):
        from .core import decode_points
        count, blob, crc = self.db.execute('SELECT count,points,crc FROM keyframes WHERE id=?', (identifier,)).fetchone()
        return decode_points(blob, count, crc, self.limits.max_input_points)

    def _solve_graph(self, stamps, originals, loops, config):
        try:
            import _dliio_pose_graph as native
        except ImportError as error:
            raise RuntimeError('Build dliio and run its installed mapping_archive.py/ROS node to load GTSAM') from error
        settings, noise = Validation(**config['validation']), config['odometry_noise']
        degenerate, unknown_quality = 0, 0
        policy = config.get('failure_policy')
        failed, consecutive = set(), 0
        for identifier, payload in self.db.execute('SELECT id,json FROM observations ORDER BY id'):
            quality = json.loads(payload)['quality']
            if quality.get('registration_converged') is False:
                if policy is None:
                    raise RejectedLoop('unconverged_odometry_observation', dict(observation_id=identifier))
                failed.add(identifier)
                consecutive += 1
                if consecutive > policy['max_consecutive']:
                    raise RejectedLoop('unconverged_run_limit', dict(observation_id=identifier))
            else:
                consecutive = 0
            unknown_quality += quality.get('registration_converged') is None
            degenerate += any(quality.get(key, 0) for key in
                ('degenerate_translation_modes', 'degenerate_rotation_modes'))
        from_ids, to_ids, measurements, matrices, flags, weakened = [], [], [], [], [], []
        for i in range(1, len(originals)):
            seconds = (stamps[i] - stamps[i-1]) * 1e-9
            if seconds > noise['max_gap_seconds']:
                raise RejectedLoop('odometry_time_gap', dict(from_id=i-1, to_id=i, seconds=seconds))
            relative = np.linalg.inv(originals[i-1]) @ originals[i]
            distance = float(np.linalg.norm(relative[:3, 3]))
            matrix = (np.asarray(noise['covariance_floor']) + seconds * np.asarray(noise['covariance_per_second']) +
                      distance * np.asarray(noise['covariance_per_metre']))
            if i in failed or i-1 in failed:
                translation, rotation = displacement(relative)
                if (translation/seconds > policy['max_translation_rate'] or
                        rotation/seconds > policy['max_rotation_rate']):
                    raise RejectedLoop('failed_registration_motion_bound', dict(from_id=i-1, to_id=i))
                matrix += np.asarray(policy['covariance_floor'])
                weakened.append(i)
            from_ids.append(i-1)
            to_ids.append(i)
            measurements.append(relative)
            matrices.append(covariance(matrix, positive=True))
            flags.append(False)
        for loop in loops:
            from_ids.append(loop['from_id'])
            to_ids.append(loop['to_id'])
            measurements.append(np.asarray(loop['transform']))
            matrices.append(np.asarray(loop['covariance']))
            flags.append(True)
        projected = any('surface_subspace' in loop for loop in loops)
        kwargs = {}
        if projected:
            from .surface_subspace import projection
            kwargs['projections'] = [np.eye(6) for _ in range(len(originals)-1)] + [
                projection(loop['surface_subspace']) if 'surface_subspace' in loop else np.eye(6) for loop in loops]
        try:
            result = native.optimize(originals, from_ids, to_ids, measurements, matrices, flags, **kwargs)
        except RuntimeError as error:
            raise RejectedLoop('optimizer_failure', dict(detail=str(error)[:1024])) from error
        metrics = {key: result[key] for key in ('initial_error', 'final_error', 'iterations', 'converged')}
        for key in ('initial_error', 'final_error'):
            if not math.isfinite(metrics[key]):
                metrics[key] = None
        if not result['converged']:
            raise RejectedLoop('optimizer_did_not_converge', metrics)
        errors = np.asarray(result['squared_errors'])
        if not np.isfinite(errors).all():
            raise RejectedLoop('nonfinite_graph_residual')
        odom_max = float(errors[:len(originals)-1].max())
        loop_max = float(errors[len(originals)-1:].max()) if loops else 0.
        metrics.update(max_odometry_squared_error=odom_max, max_loop_squared_error=loop_max,
                       gtsam_version=native.gtsam_version, nodes=len(originals), active_loops=len(loops),
                       odometry_degenerate_observations=degenerate, odometry_unknown_quality_observations=unknown_quality,
                       unconverged_observations=sorted(failed), assumed_motion_prior_edges=weakened)
        if odom_max > settings.max_odometry_squared_error or loop_max > settings.max_loop_squared_error:
            raise RejectedLoop('graph_residual_limit', metrics)
        poses = result['poses']
        if not loops:
            # The original chain satisfies every original relative increment.
            # Preserve its exact pose bytes instead of turning numerical chart
            # roundoff into a dense-map rewrite during initialization/removal.
            if not np.allclose(poses, originals, atol=1e-8, rtol=0):
                raise RejectedLoop('unexpected_unconstrained_solution', metrics)
            poses = originals
        for i in weakened:
            translation, rotation = displacement(np.linalg.inv(poses[i-1]) @ poses[i])
            seconds = (stamps[i]-stamps[i-1])*1e-9
            if (translation/seconds > policy['max_translation_rate'] or
                    rotation/seconds > policy['max_rotation_rate']):
                raise RejectedLoop('postfit_motion_prior_bound', dict(metrics, from_id=i-1, to_id=i))
        for loop in loops:
            error = np.linalg.inv(loop['transform']) @ np.linalg.inv(poses[loop['from_id']]) @ poses[loop['to_id']]
            if 'surface_subspace' in loop:
                from .surface_subspace import projection
                residual = projection(loop['surface_subspace']) @ native.residual(
                    np.asarray(loop['transform']), poses[loop['from_id']], poses[loop['to_id']])
                translation, rotation = np.linalg.norm(residual[:3]), np.linalg.norm(residual[3:])
                surface_settings = Validation(**config['loop_windows']['validation'])
                if (translation > min(settings.max_loop_translation, surface_settings.max_surface_translation_step) or
                        rotation > min(settings.max_loop_rotation, surface_settings.max_surface_rotation_step)):
                    raise RejectedLoop('postfit_projected_loop_limit', dict(metrics, failed_loop=loop['id']))
                # Tangential sliding can reduce full 3D overlap without changing
                # measured modes. Report it; a projected factor cannot certify
                # agreement in its nullspace. Dense map evaluation is separate.
                raw_translation, raw_rotation = displacement(error)
                metrics.setdefault('projected_loop_residuals', {})[loop['id']] = dict(
                    translation=float(translation), rotation=float(rotation),
                    unprojected_translation=raw_translation, unprojected_rotation=raw_rotation)
                continue
            translation, rotation = displacement(error)
            if translation > settings.max_loop_translation or rotation > settings.max_loop_rotation:
                raise RejectedLoop('postfit_absolute_loop_limit', dict(metrics,
                    failed_loop=loop['id'], translation=translation, rotation=rotation))
        maximum_translation, maximum_rotation = 0., 0.
        for original, pose in zip(originals, poses):
            translation, rotation = displacement(np.linalg.inv(original) @ pose)
            maximum_translation, maximum_rotation = max(maximum_translation, translation), max(maximum_rotation, rotation)
        metrics.update(max_solution_translation=maximum_translation, max_solution_rotation=maximum_rotation)
        if maximum_translation > settings.max_solution_translation or maximum_rotation > settings.max_solution_rotation:
            raise RejectedLoop('solution_displacement_limit', metrics)
        return poses, metrics

    def _validate_candidate(self, candidate, current, stamps, config):
        settings = Validation(**config['validation'])
        a, b = candidate['from_id'], candidate['to_id']
        if b-a < settings.min_separation_observations or (stamps[b]-stamps[a])*1e-9 < settings.min_separation_seconds:
            raise RejectedLoop('temporal_exclusion')
        for identifier in (a, b):
            quality = json.loads(self.db.execute('SELECT json FROM observations WHERE id=?',
                                                 (identifier,)).fetchone()[0])['quality']
            if quality.get('registration_converged') is False:
                raise RejectedLoop('unconverged_loop_anchor', dict(observation_id=identifier))
        if 'support' in candidate:
            from .loop_inputs import validate_support
            if 'loop_windows' not in config:
                raise ValueError('Window loops require an explicit loop_windows validation configuration')
            window_settings = Validation(**config['loop_windows']['validation'])
            metrics = check_cycle(candidate['transform'], np.linalg.inv(current[a]) @ current[b], window_settings)
            if 'surface_subspace' in candidate:
                from .surface_subspace import validate_support as validate_projected_support
                metrics.update(validate_projected_support(self, candidate, config['loop_windows']))
            else:
                metrics.update(validate_support(self, candidate, config['loop_windows']))
        else:
            metrics = check_cycle(candidate['transform'], np.linalg.inv(current[a]) @ current[b], settings)
            metrics.update(validate_geometry(self._graph_cloud(a), self._graph_cloud(b), candidate['transform'], settings))
        return metrics

    def update_graph(self, request):
        from .core import decode_pose
        if self.read_only:
            raise ValueError('A loaded map is read-only; update a working archive or an editable copy')
        digest = _request(request)
        if request['session_id'] != self.meta['session_id']:
            raise ValueError('Graph request belongs to a different archive session')
        previous = self.db.execute('SELECT digest,json FROM graph_requests WHERE request_id=?',
                                   (request['request_id'],)).fetchone()
        if previous:
            report = json.loads(previous[1])['report']
            if previous[0] != digest or report['pose_revision'] != self.meta['pose_revision']:
                raise ValueError('Graph request ID was reused or has been superseded')
            return report
        if request['expected_revision'] != self.meta['pose_revision']:
            raise ValueError('Stale graph/map revision')
        if self.db.execute('SELECT count(*) FROM graph_requests').fetchone()[0] >= MAX_REQUESTS:
            raise ValueError('Graph request ledger is full')
        state = self.db.execute('SELECT configuration,pose_revision FROM graph_state WHERE id=1').fetchone()
        action = request['action']
        if action == 'initialize':
            if state or self.meta['pose_revision'] != 0:
                raise ValueError('Initialize the graph once, before applying external pose revisions')
            config = configuration(request['configuration'])
        else:
            if state is None:
                raise ValueError('Initialize the graph with an explicit odometry noise model first')
            if state[1] is None and action != 'optimize':
                raise ValueError('External pose revision detached the graph; explicitly optimize to reattach it')
            config = configuration(json.loads(state[0]))
        settings = Validation(**config['validation'])
        stamps, originals = self._graph_originals()
        loops = [json.loads(row[0]) for row in self.db.execute(
            'SELECT json FROM graph_loops WHERE removed_revision IS NULL ORDER BY id')]
        candidates, metrics = [], {}
        if action in ('add_loop', 'add_loops'):
            candidates = [loop_measurement(value, len(originals)) for value in
                          ([request['loop']] if action == 'add_loop' else request['loops'])]
            identifiers, pairs = set(), {(value['from_id'], value['to_id']) for value in loops}
            for candidate in candidates:
                if (candidate['id'] in identifiers or
                        self.db.execute('SELECT 1 FROM graph_loops WHERE id=?', (candidate['id'],)).fetchone()):
                    raise ValueError('Loop ID already exists, including removed loop history')
                pair = candidate['from_id'], candidate['to_id']
                if pair in pairs:
                    raise ValueError('An active loop already connects these observations')
                identifiers.add(candidate['id']); pairs.add(pair)
            if self.db.execute('SELECT count(*) FROM graph_loops').fetchone()[0] + len(candidates) > MAX_LOOPS:
                raise ValueError('Graph loop ledger is full')
        if action == 'remove_loop':
            _text(request['loop_id'], 128, 'Loop ID')
            _text(request['reason'], 1024, 'Loop removal reason')
            if not any(value['id'] == request['loop_id'] for value in loops):
                raise ValueError('Cannot remove an unknown or inactive loop')
            loops = [value for value in loops if value['id'] != request['loop_id']]
        if action == 'reweight_loops':
            loops = reweighted(loops, request['updates'], len(originals))
        report = dict(action=action, request_id=request['request_id'], pose_revision=self.meta['pose_revision'])
        try:
            if candidates:
                current = [decode_pose(row[0]) for row in self.db.execute('SELECT pose FROM optimized_poses ORDER BY id')]
                trials = []
                for candidate in candidates:
                    checked = self._validate_candidate(candidate, current, stamps, config)
                    loops.append(candidate)
                    current, solved = self._solve_graph(stamps, originals, loops, config)
                    trials.append(dict(id=candidate['id'], **checked))
                poses = current
                metrics.update(trials[0] if action == 'add_loop' else dict(candidates=trials))
            else:
                poses, solved = self._solve_graph(stamps, originals, loops, config)
            metrics['optimization'] = solved
        except RejectedLoop as error:
            metrics.update(error.metrics)
            report.update(status='rejected', reason=str(error), metrics=metrics)
            self.db.execute('BEGIN IMMEDIATE')
            try:
                self._record_graph_request(request, digest, report)
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise
            return report

        report.update(status='accepted', reason='validated graph solution', metrics=metrics,
                      pose_revision=self.meta['pose_revision'] + 1)

        def commit(revision):
            if action == 'initialize':
                self.db.execute('INSERT INTO graph_state VALUES(1,?,?,?)', (encoded(config), revision, len(poses)))
            else:
                self.db.execute('UPDATE graph_state SET pose_revision=?,covered_observations=? WHERE id=1',
                                (revision, len(poses)))
            for candidate in candidates:
                self.db.execute('INSERT INTO graph_loops VALUES(?,?,?,?,?,NULL)',
                    (candidate['id'], candidate['from_id'], candidate['to_id'], encoded(candidate), revision))
            if action == 'remove_loop':
                self.db.execute('UPDATE graph_loops SET removed_revision=? WHERE id=?', (revision, request['loop_id']))
            if action == 'reweight_loops':
                changed_ids = {update['id'] for update in request['updates']}
                for loop in loops:
                    if loop['id'] in changed_ids:
                        self.db.execute('UPDATE graph_loops SET json=? WHERE id=?', (encoded(loop), loop['id']))
            self._record_graph_request(request, digest, report)

        self.apply_revision(self.meta['session_id'], request['expected_revision'], list(enumerate(poses)),
                            request_id='graph:' + digest, reason='pose graph ' + action + ': ' + request['request_id'],
                            _graph_commit=commit)
        return report

    def _record_graph_request(self, request, digest, report):
        self.db.execute('INSERT INTO graph_requests VALUES(?,?,?)',
                        (request['request_id'], digest, encoded(dict(request=request, report=report))))
