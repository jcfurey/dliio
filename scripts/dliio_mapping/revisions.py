"""Transactional pose revisions; immutable keyframes are the reconstruction source.

The storage worker is the sole writer. SQLite readers retain their previous
snapshot until commit; resident geometry is replaced only after commit. A
revision covers an ordered prefix, allowing new odometry observations to
arrive while an optimizer works on an earlier archive snapshot.
"""
from collections import OrderedDict
import hashlib
import json

import numpy as np

MAX_REVISION_POSES = 100000


class RevisionStore:
    def _create_revision_tables(self):
        # Individual execute calls preserve the caller's transaction.
        for sql in (
            'CREATE TABLE optimized_poses (id INTEGER PRIMARY KEY REFERENCES keyframes(id), pose BLOB NOT NULL)',
            'CREATE TABLE observations (id INTEGER PRIMARY KEY REFERENCES keyframes(id), json TEXT NOT NULL)',
            'CREATE TABLE pose_revisions (id INTEGER PRIMARY KEY, request_id TEXT UNIQUE NOT NULL, '
            'digest TEXT NOT NULL, last_keyframe INTEGER NOT NULL, correction BLOB NOT NULL, reason TEXT NOT NULL)',
            'CREATE TABLE revision_poses (revision INTEGER NOT NULL REFERENCES pose_revisions(id), '
            'id INTEGER NOT NULL REFERENCES keyframes(id), pose BLOB NOT NULL, PRIMARY KEY(revision,id))',
        ):
            self.db.execute(sql)
        self.db.execute('INSERT INTO pose_revisions VALUES(0,?,?,?,?,?)',
                        ('initial', '', -1, np.eye(4, dtype='<f8').tobytes(), 'original odometry'))

    def _upgrade(self):
        from .core import VERSION
        from .uncertainty import observation_metadata
        if self.meta['version'] == VERSION:
            return
        updated = dict(self.meta, version=VERSION)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            if self.meta['version'] == 1:
                self._create_revision_tables()
                self.db.execute('INSERT INTO optimized_poses SELECT id,pose FROM keyframes')
                self.db.execute('INSERT INTO observations SELECT id,? FROM keyframes',
                                (json.dumps(observation_metadata(), allow_nan=False),))
            self._create_graph_tables()
            self.db.execute(f'PRAGMA user_version={VERSION}')
            self.db.execute('UPDATE metadata SET json=? WHERE id=1', (json.dumps(updated, allow_nan=False),))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        self.meta = updated

    def _validate_revisions(self):
        from .core import decode_pose
        from .uncertainty import observation_metadata
        n, revision = self.meta['keyframes'], self.meta['pose_revision']
        for table, column, maximum in (('optimized_poses', 'pose', 128), ('revision_poses', 'pose', 128),
                                       ('pose_revisions', 'correction', 128), ('observations', 'json', 16384)):
            comparison = '>' if column == 'json' else '!='
            if self.db.execute(f'SELECT 1 FROM {table} WHERE length({column}) {comparison} ? LIMIT 1',
                               (maximum,)).fetchone():
                raise ValueError('Invalid or oversized revision/provenance record')
        if self.db.execute('SELECT count(*) FROM optimized_poses').fetchone()[0] != n:
            raise ValueError('Optimized pose coverage differs from archive observations')
        for identifier, blob in self.db.execute('SELECT id,pose FROM optimized_poses ORDER BY id'):
            if not 0 <= identifier < n:
                raise ValueError('Unknown optimized observation ID')
            decode_pose(blob)
        if self.db.execute('SELECT count(*) FROM observations').fetchone()[0] != n:
            raise ValueError('Observation provenance coverage differs from archive')
        previous = None
        for (encoded,) in self.db.execute('SELECT json FROM observations ORDER BY id'):
            value = json.loads(encoded)
            if value.get('source_session_id') is None:
                if value != observation_metadata():
                    raise ValueError('Invalid legacy observation metadata')
            else:
                observation_metadata(value)
            if previous is not None and (previous['source_session_id'] != value['source_session_id'] or
                    (value['source_sequence'] is not None and value['source_sequence'] <= previous['source_sequence'])):
                raise ValueError('Archived observation source session/sequence is inconsistent')
            previous = value
        if self.db.execute('SELECT min(id),max(id),count(*) FROM pose_revisions').fetchone() != (0, revision, revision + 1):
            raise ValueError('Pose revision history is incomplete')
        row = self.db.execute('SELECT last_keyframe,correction FROM pose_revisions WHERE id=?', (revision,)).fetchone()
        if row is None or not -1 <= row[0] < n or (revision > 0 and row[0] < 0):
            raise ValueError('Invalid current pose revision')
        correction = decode_pose(row[1])
        if not np.allclose(correction, self.meta['map_to_odom'], atol=1e-9, rtol=0):
            raise ValueError('Map correction differs from committed pose revision')
        if revision == 0 and not np.array_equal(correction, np.eye(4)):
            raise ValueError('Initial pose revision must have identity correction')
        if revision > 0:
            if self.db.execute('SELECT count(*) FROM revision_poses WHERE revision=?', (revision,)).fetchone()[0] != row[0] + 1:
                raise ValueError('Revision pose coverage is incomplete')
            anchor = self.db.execute('SELECT r.pose,k.pose FROM revision_poses r JOIN keyframes k ON k.id=r.id '
                                    'WHERE r.revision=? AND r.id=?', (revision, row[0])).fetchone()
            if anchor is None or not np.allclose(decode_pose(anchor[0]) @ np.linalg.inv(decode_pose(anchor[1])),
                                                 correction, atol=1e-9, rtol=0):
                raise ValueError('Map correction differs from the revision anchor')
        # Reconstruct the expected effective trajectory from the committed
        # prefix plus its odometric tail. This catches stale pose-table edits.
        for identifier, original, actual in self.db.execute(
                'SELECT k.id,k.pose,p.pose FROM keyframes k JOIN optimized_poses p ON p.id=k.id ORDER BY k.id'):
            if identifier <= row[0]:
                saved = self.db.execute('SELECT pose FROM revision_poses WHERE revision=? AND id=?',
                                        (revision, identifier)).fetchone()
                if saved is None:
                    raise ValueError('Revision pose coverage is incomplete')
                expected = decode_pose(saved[0])
            else:
                expected = correction @ decode_pose(original)
            if not np.allclose(decode_pose(actual), expected, atol=1e-9, rtol=0):
                raise ValueError('Optimized pose differs from committed revision')

    def _reconstruct_submap(self, identifier):
        from .core import Samples, Voxels, decode_points, decode_pose, transform
        anchor, accumulator = None, None
        for pose, count, blob, crc in self.db.execute(
                'SELECT p.pose,k.count,k.points,k.crc FROM keyframes k '
                'JOIN optimized_poses p ON p.id=k.id WHERE k.submap_id=? ORDER BY k.id', (identifier,)):
            pose = decode_pose(pose)
            local = decode_points(blob, count, crc, self.limits.max_input_points)
            # Validate the eventual world output before committing caches. Even
            # the finest supported fusion grid must represent the correction.
            world = transform(local, pose)
            if np.abs(world[:, :3].astype(np.float64)).max() / .001 >= 2 ** 62:
                raise ValueError('Corrected map exceeds the supported fusion coordinate range')
            if anchor is None:
                anchor = pose
                values = local
            else:
                values = transform(local, np.linalg.inv(anchor) @ pose)
            incoming = (Voxels.from_points(values, self.limits.voxel_size)
                        if self.limits.voxel_size else Samples(values))
            if accumulator is not None and not self.limits.voxel_size and len(accumulator) + len(incoming) > self.limits.max_voxels:
                raise ValueError('Corrected submap exceeds point capacity')
            accumulator = incoming if accumulator is None else accumulator.merged(incoming)
            if len(accumulator) > self.limits.max_voxels:
                raise ValueError('Corrected submap exceeds point capacity')
        if anchor is None:
            raise ValueError('Cannot reconstruct an empty submap')
        return anchor, accumulator

    def _restore_active(self):
        if self.meta['submaps']:
            self.active_id = self.meta['submaps'] - 1
            _, self.active = self._reconstruct_submap(self.active_id)
            self.active_count = self.db.execute('SELECT count(*) FROM keyframes WHERE submap_id=?',
                                               (self.active_id,)).fetchone()[0]

    def apply_revision(self, session_id, expected_revision, poses, *, request_id, reason, _graph_commit=None):
        """Apply complete optimized poses for IDs 0..N, with an odometric tail.

        poses is an ordered sequence of (archive_id, T_map_base). Retries with
        an identical request ID/payload are idempotent while that revision is
        current. Original clouds/poses/provenance are never modified.
        """
        from .core import decode_pose, encode_points, rigid_pose, validate_metadata
        if self.read_only:
            raise ValueError('A loaded map is read-only; revise an editable copy')
        if session_id != self.meta['session_id']:
            raise ValueError('Pose revision belongs to a different archive session')
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError('Invalid expected pose revision')
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128 or request_id == 'initial':
            raise ValueError('Revision request ID must contain 1..128 characters')
        if not isinstance(reason, str) or not 1 <= len(reason) <= 1024:
            raise ValueError('Revision must carry a reason/provenance of 1..1024 characters')
        if not 1 <= len(poses) <= min(self.meta['keyframes'], MAX_REVISION_POSES):
            raise ValueError('Revision must contain a nonempty bounded prefix of archived poses')
        checked = []
        for expected, (identifier, pose) in enumerate(poses):
            if type(identifier) is not int or identifier != expected:
                raise ValueError('Revision IDs must be the complete ordered prefix 0..N without duplicates')
            checked.append(rigid_pose(pose).astype('<f8').tobytes())
        hasher = hashlib.sha256(json.dumps([session_id, expected_revision, reason]).encode())
        for blob in checked:
            hasher.update(blob)
        digest = hasher.hexdigest()
        previous = self.db.execute('SELECT id,digest FROM pose_revisions WHERE request_id=?', (request_id,)).fetchone()
        if previous is not None:
            if previous != (self.meta['pose_revision'], digest):
                raise ValueError('Revision request ID was reused or has been superseded')
            return self.meta['pose_revision']
        if expected_revision != self.meta['pose_revision']:
            raise ValueError('Stale pose revision; obtain the current revision before optimizing')
        last = len(checked) - 1
        prior_last = self.db.execute('SELECT last_keyframe FROM pose_revisions WHERE id=?', (expected_revision,)).fetchone()[0]
        if last < prior_last:
            raise ValueError('Revision coverage predates the current optimized prefix')
        original = decode_pose(self.db.execute('SELECT pose FROM keyframes WHERE id=?', (last,)).fetchone()[0])
        correction = rigid_pose(decode_pose(checked[-1]) @ np.linalg.inv(original))
        revision = expected_revision + 1
        updated = dict(self.meta, pose_revision=revision, map_to_odom=correction.tolist())
        validate_metadata(updated)
        resident, active = self.resident.copy(), self.active
        changed_submaps = set()
        def update_pose(identifier, blob):
            previous, submap = self.db.execute('SELECT p.pose,k.submap_id FROM optimized_poses p '
                'JOIN keyframes k ON k.id=p.id WHERE p.id=?', (identifier,)).fetchone()
            if blob != previous:
                changed_submaps.add(submap)
                self.db.execute('UPDATE optimized_poses SET pose=? WHERE id=?', (blob, identifier))
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self.db.execute('INSERT INTO pose_revisions VALUES(?,?,?,?,?,?)',
                            (revision, request_id, digest, last, correction.astype('<f8').tobytes(), reason))
            for identifier, blob in enumerate(checked):
                self.db.execute('INSERT INTO revision_poses VALUES(?,?,?)', (revision, identifier, blob))
                update_pose(identifier, blob)
            for identifier, blob in self.db.execute('SELECT id,pose FROM keyframes WHERE id>? ORDER BY id', (last,)):
                pose = rigid_pose(correction @ decode_pose(blob))
                update_pose(identifier, pose.astype('<f8').tobytes())
            # Exact unchanged pose bytes need no cloud reads/writes. In
            # particular, graph initialization only records its noise model
            # and history instead of rebuilding gigabytes of identical points.
            for identifier in sorted(changed_submaps):
                anchor, rebuilt = self._reconstruct_submap(identifier)
                points = rebuilt.points()
                blob, crc = encode_points(points)
                self.db.execute('UPDATE submaps SET pose=?,count=?,points=?,crc=? WHERE id=?',
                                (anchor.astype('<f8').tobytes(), len(points), blob, crc, identifier))
                if identifier in resident:
                    resident[identifier] = (anchor, points)
                if identifier == self.active_id:
                    active = rebuilt
            self.db.execute('UPDATE metadata SET json=? WHERE id=1', (json.dumps(updated, allow_nan=False),))
            if _graph_commit is not None:
                _graph_commit(revision)
            else:
                # An external correction/rollback does not remove factors. An
                # explicit graph optimize is required before using them again.
                self.db.execute('UPDATE graph_state SET pose_revision=NULL WHERE id=1')
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        self.meta, self.resident, self.active = updated, resident, active
        if changed_submaps.intersection(resident):
            self.fusion = None
        self.active_id = self.meta['submaps'] - 1
        self.active_count = self.db.execute('SELECT count(*) FROM keyframes WHERE submap_id=?',
                                           (self.active_id,)).fetchone()[0]
        return revision

    def restore_revision(self, session_id, expected_revision, target_revision, *, request_id):
        """Rollback is a new revision, never a deletion of correction history."""
        from .core import decode_pose
        if type(target_revision) is not int or not 0 <= target_revision <= self.meta['pose_revision']:
            raise ValueError('Unknown rollback revision')
        if self.meta['keyframes'] > MAX_REVISION_POSES:
            raise ValueError('Rollback exceeds the revision pose limit')
        row = self.db.execute('SELECT last_keyframe,correction FROM pose_revisions WHERE id=?', (target_revision,)).fetchone()
        if row is None:
            raise ValueError('Missing rollback revision')
        correction = decode_pose(row[1])
        poses = []
        for identifier, blob in self.db.execute('SELECT id,pose FROM keyframes ORDER BY id'):
            if identifier <= row[0]:
                previous = self.db.execute('SELECT pose FROM revision_poses WHERE revision=? AND id=?',
                                           (target_revision, identifier)).fetchone()
                if previous is None:
                    raise ValueError('Incomplete rollback trajectory')
                pose = decode_pose(previous[0])
            else:
                pose = correction @ decode_pose(blob)
            poses.append((identifier, pose))
        return self.apply_revision(session_id, expected_revision, poses, request_id=request_id,
                                   reason=f'rollback to pose revision {target_revision}')
