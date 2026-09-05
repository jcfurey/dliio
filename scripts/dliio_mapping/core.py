"""Mapping storage and geometry. No ROS executor, sensor driver, or pickle data.

One worker owns each Store. SQLite holds all keyframes/submaps on disk; only a
bounded window of submap points and one active voxel accumulator stay in RAM.
"""
from collections import OrderedDict
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import uuid
import zlib

import numpy as np

FIELDS = ('x', 'y', 'z', 'intensity', 'reflectivity', 'intensity_corrected', 'lidar_intensity')
APPLICATION_ID = 0x444C4D50
VERSION = 1


def validate_metadata(metadata):
    # Version 1 has no optimized trajectory; accepting a nonidentity correction
    # here would silently display the archive in the wrong frame.
    for name in ('odom_frame', 'base_frame', 'map_frame'):
        value = metadata.get(name)
        if not isinstance(value, str) or not value or value.startswith('/'):
            raise ValueError(f'Invalid {name}')
    if metadata['map_frame'] == metadata['odom_frame']:
        raise ValueError('Map and odom frames must differ')
    if (metadata.get('pose_revision', 0) != 0 or
            not np.array_equal(rigid_pose(metadata.get('map_to_odom')), np.eye(4))):
        raise ValueError('Version 1 requires an identity map-to-odom transform')
    if not isinstance(metadata.get('calibration'), dict):
        raise ValueError('Archive must contain calibration metadata')
    encoded = json.dumps(metadata, allow_nan=False)
    if len(encoded.encode('utf8')) > 1048576:
        raise ValueError('Archive metadata exceeds 1 MiB')


@dataclass(frozen=True)
class Limits:
    voxel_size: float = .2
    submap_keyframes: int = 20
    resident_submaps: int = 8
    max_voxels: int = 100000
    max_input_points: int = 200000

    def __post_init__(self):
        if not math.isfinite(self.voxel_size) or not .001 <= self.voxel_size <= 100:
            raise ValueError('voxel_size must be finite and between .001 and 100 metres')
        for name, maximum in [('submap_keyframes', 1000), ('resident_submaps', 64),
                              ('max_voxels', 1000000), ('max_input_points', 1000000)]:
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f'{name} must be an integer in [1, {maximum}]')


def rigid_pose(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('Pose must be a finite 4x4 matrix')
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9, rtol=0) or
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0) or
            abs(np.linalg.det(rotation) - 1.) > 1e-6):
        raise ValueError('Pose must contain a proper rigid rotation')
    return matrix.copy()


def transform(points, pose, inverse=False):
    result = np.array(points, dtype='<f4', copy=True)
    xyz = points[:, :3].astype(np.float64)
    if inverse:
        xyz = (xyz - pose[:3, 3]) @ pose[:3, :3]
    else:
        xyz = xyz @ pose[:3, :3].T + pose[:3, 3]
    result[:, :3] = xyz
    if not np.isfinite(result[:, :3]).all():
        raise ValueError('Transformed XYZ is not representable as float32')
    return result


def encode_points(points):
    data = np.ascontiguousarray(points, dtype='<f4').tobytes()
    return data, zlib.crc32(data)


def decode_points(data, count, checksum, maximum):
    if type(count) is not int or not 0 < count <= maximum or len(data) != count * 28:
        raise ValueError('Invalid or oversized archived point array')
    if zlib.crc32(data) != checksum:
        raise ValueError('Archived point checksum mismatch')
    points = np.frombuffer(data, dtype='<f4').reshape(count, 7)
    if not np.isfinite(points[:, :3]).all() or np.isinf(points[:, 3:]).any():
        raise ValueError('Invalid archived scalar values')
    return points


def decode_pose(data):
    if len(data) != 128:
        raise ValueError('Invalid archived pose size')
    return rigid_pose(np.frombuffer(data, dtype='<f8').reshape(4, 4))


class Voxels:
    """Per-field counts preserve missing channels without treating them as zero."""
    def __init__(self, keys, sums, counts):
        self.keys, self.sums, self.counts = keys, sums, counts

    @staticmethod
    def from_points(points, leaf):
        scaled = np.floor(points[:, :3].astype(np.float64) / leaf)
        if not np.isfinite(scaled).all() or np.abs(scaled).max() >= 2 ** 62:
            raise ValueError('Voxel coordinates are out of range')
        keys, inverse = np.unique(scaled.astype(np.int64), axis=0, return_inverse=True)
        valid = np.isfinite(points)
        values = np.where(valid, points, 0).astype(np.float64)
        sums = np.column_stack([np.bincount(inverse, weights=values[:, i], minlength=len(keys)) for i in range(7)])
        counts = np.column_stack([np.bincount(inverse, weights=valid[:, i], minlength=len(keys)) for i in range(7)])
        return Voxels(keys, sums, counts)

    def merged(self, other):
        keys, inverse = np.unique(np.vstack((self.keys, other.keys)), axis=0, return_inverse=True)
        sums = np.vstack((self.sums, other.sums))
        counts = np.vstack((self.counts, other.counts))
        return Voxels(keys,
            np.column_stack([np.bincount(inverse, weights=sums[:, i], minlength=len(keys)) for i in range(7)]),
            np.column_stack([np.bincount(inverse, weights=counts[:, i], minlength=len(keys)) for i in range(7)]))

    def points(self):
        values = np.full_like(self.sums, np.nan)
        np.divide(self.sums, self.counts, out=values, where=self.counts != 0)
        return values.astype('<f4')

    @property
    def nbytes(self):
        return self.keys.nbytes + self.sums.nbytes + self.counts.nbytes


def atomic_output(path, write):
    """Install a completed output without overwriting an existing destination."""
    path = Path(path).expanduser()
    if not path.is_absolute() or not path.parent.is_dir() or path.exists():
        raise ValueError('Output must be a new absolute filename in an existing directory')
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    os.close(fd)
    try:
        write(Path(temporary))
        with open(temporary, 'rb') as stream:
            os.fsync(stream.fileno())
        os.link(temporary, path)  # atomic no-clobber publication, on the same filesystem
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        for suffix in ('', '-journal', '-wal', '-shm'):
            Path(temporary + suffix).unlink(missing_ok=True)


class Store:
    def __init__(self, path, limits, *, metadata=None):
        self.path = Path(path).expanduser().resolve()
        self.limits = limits
        self.resident = OrderedDict()
        self.active = None
        self.active_count = 0
        self.active_id = None
        self.read_only = metadata is None
        if self.read_only:
            self.db = sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True, isolation_level=None)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
            self.db = sqlite3.connect(self.path, isolation_level=None)
        try:
            self.db.execute('PRAGMA cache_size=-8192')
            self.db.execute('PRAGMA temp_store=FILE')
            self.db.execute('PRAGMA mmap_size=0')
            self.db.execute('PRAGMA foreign_keys=ON')
            if self.read_only:
                self._load()
            else:
                self._create(metadata)
        except BaseException:
            self.db.close()
            raise

    def _create(self, metadata):
        self.meta = dict(metadata, format='dliio-mapping', version=VERSION,
            session_id=str(uuid.uuid4()), fields=list(FIELDS), coordinates='keyframe_base',
            voxel_size=self.limits.voxel_size, submap_keyframes=self.limits.submap_keyframes,
            keyframes=0, submaps=0, last_stamp_ns=0, sealed=False, pose_revision=0)
        validate_metadata(self.meta)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript(f'''
            PRAGMA application_id={APPLICATION_ID};
            PRAGMA user_version={VERSION};
            CREATE TABLE metadata (id INTEGER PRIMARY KEY CHECK(id=1), json TEXT NOT NULL);
            CREATE TABLE submaps (id INTEGER PRIMARY KEY, pose BLOB NOT NULL,
                first_keyframe INTEGER NOT NULL, last_keyframe INTEGER NOT NULL,
                stamp_ns INTEGER NOT NULL, count INTEGER NOT NULL, points BLOB NOT NULL, crc INTEGER NOT NULL);
            CREATE TABLE keyframes (id INTEGER PRIMARY KEY, stamp_ns INTEGER UNIQUE NOT NULL,
                submap_id INTEGER NOT NULL REFERENCES submaps(id), pose BLOB NOT NULL,
                count INTEGER NOT NULL, points BLOB NOT NULL, crc INTEGER NOT NULL);
            CREATE INDEX keyframes_submap ON keyframes(submap_id);
        ''')
        self.db.execute('INSERT INTO metadata VALUES(1, ?)', (json.dumps(self.meta, allow_nan=False),))

    def _load(self):
        if (self.db.execute('PRAGMA application_id').fetchone()[0] != APPLICATION_ID or
                self.db.execute('PRAGMA user_version').fetchone()[0] != VERSION):
            raise ValueError('Unsupported map archive format/version')
        if self.db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('Map archive failed SQLite integrity check')
        row = self.db.execute('SELECT json FROM metadata WHERE id=1').fetchone()
        if row is None or len(row[0]) > 1048576:
            raise ValueError('Invalid archive metadata')
        self.meta = json.loads(row[0])
        if (self.meta.get('format') != 'dliio-mapping' or self.meta.get('version') != VERSION or
                self.meta.get('fields') != list(FIELDS) or self.meta.get('coordinates') != 'keyframe_base' or
                self.meta.get('sealed') is not True):
            raise ValueError('Load a sealed save_map snapshot; use mapping_archive.py snapshot for a working archive')
        uuid.UUID(self.meta['session_id'])
        validate_metadata(self.meta)
        if any(type(self.meta.get(name)) is not int or self.meta[name] < 0
               for name in ('keyframes', 'submaps', 'last_stamp_ns')):
            raise ValueError('Invalid archive counts')
        self.limits = replace(self.limits, voxel_size=self.meta['voxel_size'],
                              submap_keyframes=self.meta['submap_keyframes'])
        # Check lengths in SQLite before fetching blobs into the Python process.
        for table, maximum in [('keyframes', self.limits.max_input_points), ('submaps', self.limits.max_voxels)]:
            bad = self.db.execute(f'SELECT id FROM {table} WHERE count<1 OR count>? OR length(points)!=count*28 '
                                  'OR length(pose)!=128 LIMIT 1', (maximum,)).fetchone()
            if bad is not None:
                raise ValueError(f'Invalid or oversized {table} record')
        previous_stamp = 0
        count = 0
        for identifier, stamp, pose, n, blob, checksum in self.db.execute(
                'SELECT id,stamp_ns,pose,count,points,crc FROM keyframes ORDER BY id'):
            if identifier != count or not previous_stamp < stamp <= 2 ** 63 - 1:
                raise ValueError('Noncontiguous IDs or invalid keyframe timestamp ordering')
            decode_pose(pose)
            decode_points(blob, n, checksum, self.limits.max_input_points)
            previous_stamp = stamp
            count += 1
        if count != self.meta['keyframes'] or previous_stamp != self.meta['last_stamp_ns']:
            raise ValueError('Archive keyframe metadata does not match its records')
        count = 0
        previous_last = -1
        for identifier, pose, first, last, stamp, n, blob, checksum in self.db.execute('SELECT * FROM submaps ORDER BY id'):
            if (identifier != count or first != previous_last + 1 or
                    not 0 <= first <= last < self.meta['keyframes'] or
                    last - first + 1 > self.limits.submap_keyframes):
                raise ValueError('Invalid submap record')
            members = self.db.execute('SELECT min(id),max(id),count(*),max(stamp_ns) FROM keyframes '
                                      'WHERE submap_id=?', (identifier,)).fetchone()
            if members != (first, last, last - first + 1, stamp):
                raise ValueError('Submap membership or timestamp is inconsistent')
            if pose != self.db.execute('SELECT pose FROM keyframes WHERE id=?', (first,)).fetchone()[0]:
                raise ValueError('Submap anchor differs from its first keyframe pose')
            anchor = decode_pose(pose)
            points = decode_points(blob, n, checksum, self.limits.max_voxels)
            self.resident[identifier] = (anchor, points)
            self._evict()
            count += 1
            previous_last = last
        if (count != self.meta['submaps'] or previous_last + 1 != self.meta['keyframes'] or
                self.db.execute('PRAGMA foreign_key_check').fetchone() is not None):
            raise ValueError('Archive submap metadata or keyframe references are inconsistent')

    def _evict(self):
        while len(self.resident) > self.limits.resident_submaps:
            self.resident.popitem(last=False)

    def ingest(self, stamp_ns, pose, world_points):
        if self.read_only:
            raise ValueError('A loaded map is read-only; start a new session for live input')
        if type(stamp_ns) is not int or not self.meta['last_stamp_ns'] < stamp_ns <= 2 ** 63 - 1:
            raise ValueError('Duplicate/backward keyframe timestamp; start a new session')
        pose = rigid_pose(pose)
        points = np.asarray(world_points, dtype='<f4')
        if points.ndim != 2 or points.shape[1] != 7 or not 0 < len(points) <= self.limits.max_input_points:
            raise ValueError('Invalid or oversized keyframe array')
        if not np.isfinite(points[:, :3]).all() or np.isinf(points[:, 3:]).any():
            raise ValueError('Invalid keyframe scalar values')
        local = transform(points, pose, inverse=True)
        fresh = self.active is None or self.active_count >= self.limits.submap_keyframes
        anchor = pose if fresh else self.resident[self.active_id][0]
        incoming = Voxels.from_points(transform(points, anchor, inverse=True), self.limits.voxel_size)
        candidate = incoming if fresh else self.active.merged(incoming)
        if len(candidate.keys) > self.limits.max_voxels and not fresh:
            fresh, anchor = True, pose
            candidate = Voxels.from_points(local, self.limits.voxel_size)
        if len(candidate.keys) > self.limits.max_voxels:
            raise ValueError('Single keyframe exceeds submap voxel capacity')
        identifier = self.meta['keyframes']
        submap_id = self.meta['submaps'] if fresh else self.active_id
        first = identifier if fresh else identifier - self.active_count
        centers = candidate.points()
        local_blob, local_crc = encode_points(local)
        map_blob, map_crc = encode_points(centers)
        updated = dict(self.meta, keyframes=identifier + 1, submaps=self.meta['submaps'] + int(fresh), last_stamp_ns=stamp_ns)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self.db.execute('INSERT INTO submaps VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET '
                'last_keyframe=excluded.last_keyframe,stamp_ns=excluded.stamp_ns,count=excluded.count,points=excluded.points,crc=excluded.crc',
                (submap_id, anchor.astype('<f8').tobytes(), first, identifier, stamp_ns, len(centers), map_blob, map_crc))
            self.db.execute('INSERT INTO keyframes VALUES(?,?,?,?,?,?,?)',
                (identifier, stamp_ns, submap_id, pose.astype('<f8').tobytes(), len(local), local_blob, local_crc))
            self.db.execute('UPDATE metadata SET json=? WHERE id=1', (json.dumps(updated, allow_nan=False),))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        # Mutate resident state only after the complete keyframe transaction commits.
        self.meta = updated
        self.active, self.active_id = candidate, submap_id
        self.active_count = 1 if fresh else self.active_count + 1
        self.resident[submap_id] = (anchor, centers)
        self._evict()
        return identifier

    def snapshot(self):
        if not self.resident:
            return np.empty((0, 7), dtype='<f4')
        return np.vstack([transform(points, pose) for pose, points in self.resident.values()])

    def stats(self):
        return dict(session_id=self.meta['session_id'], keyframes=self.meta['keyframes'], submaps=self.meta['submaps'],
            resident_submaps=len(self.resident), resident_points=sum(len(p) for _, p in self.resident.values()),
            resident_array_bytes=sum(p.nbytes + pose.nbytes for pose, p in self.resident.values()) +
                                 (self.active.nbytes if self.active is not None else 0),
            last_stamp_ns=self.meta['last_stamp_ns'], read_only=self.read_only, archive=str(self.path))

    def save(self, destination):
        snapshot_database(self.path, destination)

    def export_pcd(self, destination, leaf_size=None):
        if leaf_size is not None and (not math.isfinite(leaf_size) or not .001 <= leaf_size <= 100):
            raise ValueError('Export leaf size must be between .001 and 100 metres')

        def chunks():
            for pose, count, blob, crc in self.db.execute('SELECT pose,count,points,crc FROM submaps ORDER BY id'):
                points = transform(decode_points(blob, count, crc, self.limits.max_voxels), decode_pose(pose))
                if leaf_size is not None:
                    points = Voxels.from_points(points, leaf_size).points()
                yield points

        # Two bounded passes permit an ordinary binary PCD header and no full-map allocation.
        count = sum(len(points) for points in chunks())
        if not 0 < count <= 2 ** 32 - 1:
            raise ValueError('Cannot export an empty or oversized PCD')
        def write(path):
            header = ('# .PCD v0.7\nVERSION 0.7\nFIELDS ' + ' '.join(FIELDS) +
                '\nSIZE 4 4 4 4 4 4 4\nTYPE F F F F F F F\nCOUNT 1 1 1 1 1 1 1\n' +
                f'WIDTH {count}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {count}\nDATA binary\n')
            with path.open('wb') as stream:
                stream.write(header.encode('ascii'))
                for points in chunks():
                    stream.write(np.asarray(points, dtype='<f4').tobytes())
        atomic_output(destination, write)
        return count

    def close(self):
        self.db.close()


def snapshot_database(source, destination):
    """SQLite's backup API includes committed WAL pages in a consistent snapshot."""
    source = Path(source).expanduser().resolve()
    def write(path):
        src = sqlite3.connect(source.as_uri() + '?mode=ro', uri=True)
        dst = sqlite3.connect(path, isolation_level=None)
        try:
            if (src.execute('PRAGMA application_id').fetchone()[0] != APPLICATION_ID or
                    src.execute('PRAGMA user_version').fetchone()[0] != VERSION):
                raise ValueError('Not a dliio mapping archive')
            src.backup(dst, pages=256)
            dst.execute('PRAGMA journal_mode=DELETE')
            row = dst.execute('SELECT json FROM metadata WHERE id=1').fetchone()
            metadata = json.loads(row[0])
            metadata['sealed'] = True
            dst.execute('UPDATE metadata SET json=? WHERE id=1', (json.dumps(metadata, allow_nan=False),))
        finally:
            dst.close()
            src.close()
    atomic_output(destination, write)
