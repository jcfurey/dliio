#!/usr/bin/env python3
"""Alternate baseline/candidate estimator replays on a fixed Ouster cache."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('results', type=Path)
ap.add_argument('--repetitions', type=int, default=5)
ap.add_argument('--baseline', type=Path, required=True)
ap.add_argument('--candidate', type=Path, required=True)
ap.add_argument('--baseline-build', type=Path, required=True)
ap.add_argument('--candidate-build', type=Path, required=True)
ap.add_argument('--candidate-config', default='flow-signal-refl.yaml')
ap.add_argument('--cpus', help='Optional taskset CPU list, e.g. 4-7; default uses available CPUs')
a = ap.parse_args()
r = a.results.resolve()
env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
           ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST', RMW_IMPLEMENTATION='rmw_cyclonedds_cpp',
           ROS_LOG_DIR=str(r / 'benchmark-ros-logs'))
runs = []
for rep in range(1, a.repetitions + 1):
    for label, executable, build, cfg in [
        ('baseline', a.baseline, a.baseline_build, 'baseline.yaml'),
        ('candidate', a.candidate, a.candidate_build, a.candidate_config),
    ]:
        tag = f'nrun-{label}-{rep}'
        executable, build = executable.resolve(), build.resolve()
        run_env = dict(env, ROS_DOMAIN_ID=str(130 + rep),
                       LD_LIBRARY_PATH=str(build) + ':' + env.get('LD_LIBRARY_PATH', ''))
        affinity = ['taskset', '-c', a.cpus] if a.cpus else []
        cmd = ['/usr/bin/time', '-f', '%M', '-o', str(r / (tag + '.rss')),
               *affinity, str(executable), str(r / 'cache'),
               str(r / (tag + '.csv')), '0', '--ros-args', '--params-file', str(r / cfg)]
        start = time.monotonic()
        with (r / (tag + '.log')).open('x') as log:
            proc = subprocess.run(cmd, env=run_env, stdout=log, stderr=subprocess.STDOUT)
        result = dict(label=label, repetition=rep, returncode=proc.returncode,
                      wall_seconds=time.monotonic() - start, command=cmd,
                      peak_rss_kib=int((r / (tag + '.rss')).read_text().splitlines()[-1]),
                      executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                      library_sha256=hashlib.sha256((build / 'libdlio_components.so').read_bytes()).hexdigest(),
                      config_sha256=hashlib.sha256((r / cfg).read_bytes()).hexdigest())
        runs.append(result)
        (r / 'nrun-manifest.json').write_text(json.dumps(runs, indent=2) + '\n')
        print(f'{tag}: status={proc.returncode}, wall={result["wall_seconds"]:.1f}s', flush=True)
        if proc.returncode not in (0, 3):
            raise SystemExit(f'Unexpected replay failure: {tag}; inspect its log')
