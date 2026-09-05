#!/usr/bin/env python3
"""Check installed Ouster packet replay, scan delivery, fields, and pose pairing."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path)
    parser.add_argument('--seconds', type=float, default=75.)
    parser.add_argument('--output', type=Path, default=Path.cwd() / 'dliio_run/packet-replay.json')
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds < 10:
        parser.error('--seconds must be finite and at least 10')
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'passed': False, 'status': 'starting'}) + '\n')
    with args.output.with_suffix('.log').open('w') as log:
        process = subprocess.Popen(['ros2', 'launch', 'direct_lidar_inertial_odometry',
            'dlio_ouster.launch.py', 'mode:=packets', 'profile:=0705', 'rviz:=false',
            'rate:=1.0', f'bag:={args.bag.resolve()}', f'run_dir:={args.output.parent}'],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            # Allow component loading, the launch's bag delay, and IMU calibration.
            for _ in range(40):
                if process.poll() is not None:
                    raise RuntimeError('Replay launch exited during startup; inspect its log')
                time.sleep(.5)
            subprocess.run([sys.executable, str(Path(__file__).with_name('inspect_replay.py')),
                '--seconds', str(args.seconds), '--output', str(args.output)],
                stdout=log, stderr=subprocess.STDOUT, check=True, timeout=args.seconds + 30)
            result = json.loads(args.output.read_text())
            assert process.poll() is None, 'Replay launch exited unexpectedly'
            assert result['counts'].get('scan_pose', 0) >= args.seconds * 9, 'Replay did not sustain 9 Hz'
            assert result['counts'].get('imu', 0) >= args.seconds * 80, 'Missing IMU stream'
            for key in ('deskewed/scan_pose', 'keyframe/keyframe_pose'):
                pair = result['timestamp_pairs'][key]
                assert pair['matched'] > 0, f'No paired {key}'
                # At most one message may straddle each sample-window boundary.
                assert pair['left_received'] - pair['matched'] <= 2, f'Unpaired {key}'
                assert pair['right_received'] - pair['matched'] <= 2, f'Unpaired {key}'
            for key in ('deskewed', 'keyframe', 'map'):
                assert result[key]['frame'] == 'odom'
                assert {'intensity', 'reflectivity', 'intensity_corrected', 'lidar_intensity'} <= set(result[key]['fields'])
                assert not {'t', 'time', 'timestamp'} & set(result[key]['fields'])
            result.update(ros_distro=os.environ.get('ROS_DISTRO'), rate=1.0, passed=True)
            args.output.write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps({'ros_distro': result['ros_distro'], 'rate': 1.,
                              'counts': result['counts'], 'passed': True}, indent=2))
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)


if __name__ == '__main__':
    main()
