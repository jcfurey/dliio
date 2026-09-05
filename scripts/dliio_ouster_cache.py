#!/usr/bin/env python3
"""Decode the 0705 Ouster bag to a deterministic, local odometry test cache.

Supports only RNG19_RFL8_SIG16_NIR16 + legacy IMU, checked explicitly. Layout
and Cartesian conversion follow the vendored Ouster SDK parsing.cpp and XYZ LUT.
Uses the common sensor clock for LiDAR and IMU, retaining bag receive times in
each frame header. This isolates estimator tests from DDS drops and replay load;
it is not an end-to-end test of the ROS driver's host-time stamping mode.
"""
import argparse
import json
import sqlite3
import struct
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('bag', type=Path)
    ap.add_argument('metadata', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--seconds', type=float, default=0)
    args = ap.parse_args()
    meta = json.loads(args.metadata.read_text())
    fmt = meta['lidar_data_format']
    config = meta['config_params']
    assert config['udp_profile_lidar'] == 'RNG19_RFL8_SIG16_NIR16'
    assert config['udp_profile_imu'] == 'LEGACY'
    h, w = fmt['pixels_per_column'], fmt['columns_per_frame']
    cpp = fmt['columns_per_packet']
    assert h == 64 and w == 1024 and cpp == 16
    beam = meta['beam_intrinsics']
    altitude = np.deg2rad(beam['beam_altitude_angles'])[:, None]
    azimuth = -np.deg2rad(beam['beam_azimuth_angles'])[:, None]
    enc = 2 * np.pi * (1 - np.arange(w) / w)[None, :]
    origin = np.array(beam['beam_to_lidar_transform']).reshape(4, 4)[:3, 3]
    assert origin[1] == 0
    direction = np.stack(np.broadcast_arrays(np.cos(enc + azimuth) * np.cos(altitude),
                                             np.sin(enc + azimuth) * np.cos(altitude),
                                             np.sin(altitude)), axis=-1)
    offset = np.stack(np.broadcast_arrays(origin[0] * np.cos(enc), origin[0] * np.sin(enc),
                                          np.full_like(enc, origin[2])), axis=-1)
    shifts = np.array(fmt['pixel_shift_by_row'])[:, None]
    destagger_cols = (np.arange(w)[None, :] - shifts) % w
    rows = np.arange(h)[:, None]
    point = np.dtype({'names': ['x', 'y', 'z', 'intensity', 't', 'reflectivity', 'ring', 'ambient', 'range'],
                      'formats': ['<f4', '<f4', '<f4', '<f4', '<u4', '<u2', '<u2', '<u2', '<u4'],
                      'offsets': [0, 4, 8, 12, 16, 20, 22, 24, 28], 'itemsize': 32})
    pixels = np.zeros((h, w, 3), dtype='<u4')
    stamps = np.zeros(w, dtype='<u8')
    seen = np.zeros(w, dtype=bool)
    cloud = np.zeros((h, w), dtype=point)
    cloud['ring'] = rows
    args.output.mkdir(parents=True, exist_ok=True)
    # Refuse to truncate an existing cache accidentally.
    frames_file = (args.output / 'frames.bin').open('xb')
    imu_file = (args.output / 'imu.bin').open('xb')
    frame_id, frame_rx, first_rx = None, 0, None
    frames = partial = packets = imus = 0
    frame_times = []

    def flush_frame():
        nonlocal frames, partial
        if not seen.all():
            partial += 1
            return
        raw_range = pixels[:, :, 0] & 0x7ffff
        valid = raw_range != 0
        xyz = ((raw_range[:, :, None] - np.linalg.norm(origin)) * direction + offset) * 0.001
        xyz[~valid] = np.nan
        xyz = xyz[rows, destagger_cols]
        for k, name in enumerate(('x', 'y', 'z')):
            cloud[name] = xyz[:, :, k]
        cloud['range'] = raw_range[rows, destagger_cols]
        cloud['reflectivity'] = (pixels[:, :, 1] & 0xff)[rows, destagger_cols]
        cloud['intensity'] = (pixels[:, :, 1] >> 16)[rows, destagger_cols]
        cloud['ambient'] = (pixels[:, :, 2] & 0xffff)[rows, destagger_cols]
        ts = int(stamps[0])
        assert np.all(stamps >= ts) and int(stamps[-1]) - ts < 200_000_000
        cloud['t'] = (stamps[destagger_cols] - ts).astype('<u4')
        frames_file.write(struct.pack('<QQ', ts, frame_rx))
        frames_file.write(cloud.tobytes())
        frame_times.append(ts)
        frames += 1
        if frames % 500 == 0:
            print(f'{frames} complete frames, {partial} partial, {imus} IMU samples', flush=True)

    db = sqlite3.connect(f'file:{args.bag.resolve()}?mode=ro', uri=True)
    ids = {name: tid for tid, name in db.execute('SELECT id,name FROM topics')}
    lid, imu = ids['/ouster/lidar_packets'], ids['/ouster/imu_packets']
    query = 'SELECT topic_id,timestamp,data FROM messages WHERE topic_id IN (?,?) ORDER BY timestamp'
    for tid, rx, cdr in db.execute(query, (lid, imu)):
        if first_rx is None:
            first_rx = rx
        if args.seconds and rx - first_rx > args.seconds * 1e9:
            break
        assert cdr[:4] == b'\x00\x01\x00\x00'
        size = struct.unpack_from('<I', cdr, 4)[0]
        buf = memoryview(cdr)[8:8 + size]
        assert len(buf) == size
        if tid == imu:
            assert size == 48
            ts = struct.unpack_from('<Q', buf, 16)[0]
            vals = np.array(struct.unpack_from('<6f', buf, 24), dtype=float)
            vals[:3] *= 9.80665
            vals[3:] *= np.pi / 180
            imu_file.write(struct.pack('<Q6d', ts, *vals))
            imus += 1
            continue
        assert size == 32 + cpp * (12 + h * 12) + 32
        fid = struct.unpack_from('<H', buf, 2)[0]
        if frame_id != fid:
            if frame_id is not None:
                flush_frame()
            frame_id, frame_rx = fid, rx
            seen.fill(False)
            pixels.fill(0)
        cols = np.ndarray((cpp,), dtype=np.dtype([('ts', '<u8'), ('mid', '<u2'), ('status', '<u2'),
                                                  ('px', '<u4', (h, 3))]), buffer=buf, offset=32)
        for col in cols:
            mid = int(col['mid'])
            assert mid < w
            if int(col['status']) & 1:
                pixels[:, mid] = col['px']
                stamps[mid] = col['ts']
                seen[mid] = True
        packets += 1
    if frame_id is not None:
        flush_frame()
    frames_file.close()
    imu_file.close()
    gaps = np.diff(frame_times) * 1e-9
    summary = dict(bag=str(args.bag.resolve()), metadata=str(args.metadata.resolve()),
                   clock='sensor common oscillator; frame receive time retained separately',
                   height=h, width=w, point_step=32, frame_header_bytes=16,
                   frames=frames, partial_frames=partial, lidar_packets=packets, imu_samples=imus,
                   frame_period_min=float(gaps.min()), frame_period_max=float(gaps.max()),
                   first_stamp_ns=frame_times[0], last_stamp_ns=frame_times[-1])
    (args.output / 'manifest.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
