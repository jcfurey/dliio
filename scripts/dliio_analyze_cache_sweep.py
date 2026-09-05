#!/usr/bin/env python3
"""Analyze in-bag Ouster sweep behavior and runtime, without claiming pose GT."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('results', type=Path)
a = ap.parse_args()
r = a.results
manifest = json.loads((r / 'nrun-manifest.json').read_text())
expected = json.loads((r / 'cache/manifest.json').read_text())['frames']
results = []
trajectories = {}
for run in manifest:
    label, rep = run['label'], run['repetition']
    p = np.genfromtxt(r / f'nrun-{label}-{rep}.csv', delimiter=',', names=True)
    xyz = np.c_[p['x'], p['y'], p['z']]
    t = p['stamp'] - p['stamp'][0]
    # World X is the dominant tunnel direction in this recording. Use the
    # same fixed axis for every run; never rotate/scale a candidate to improve
    # its metric. Gaussian sigma=0.5 s separates short-period motion, including
    # real brief motion. It is a behavior metric, not ground-truth error.
    x = p['x']
    dt = float(np.median(np.diff(t)))
    smooth = gaussian_filter1d(x, 0.5 / dt)
    axial_path = float(np.abs(np.diff(x)).sum())
    short_period_path = axial_path - float(np.abs(np.diff(smooth)).sum())
    runtime = p['compute_ms'][t >= 5]
    valid = run['returncode'] == 0 and len(p) == expected and np.isfinite(xyz).all()
    row = dict(label=label, repetition=rep, complete=bool(valid), frames=len(p),
               axial_span_m=float(np.ptp(x)), axial_travel_m=axial_path,
               short_period_axial_travel_m=short_period_path,
               path_3d_m=float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()),
               endpoint_m=xyz[-1].tolist(), compute_p50_ms=float(np.percentile(runtime, 50)),
               compute_p95_ms=float(np.percentile(runtime, 95)),
               compute_over_100ms=int((runtime > 100).sum()),
               peak_rss_mib=run['peak_rss_kib'] / 1024,
               wall_seconds=run['wall_seconds'],
               median_photo_points=float(np.median(p['photo_count'][t >= 5])),
               median_flow_points=float(np.median(p['flow_count'][t >= 5])),
               flow_rms_median=float(np.median(p['flow_rms'][t >= 5])))
    if valid:
        tail = t >= 530
        row['tail_axial_span_m'] = float(np.ptp(x[tail]))
        row['tail_axial_travel_m'] = float(np.abs(np.diff(x[tail])).sum())
        trajectories[(label, rep)] = (t, x)
    results.append(row)
summary = {}
for label in sorted({v['label'] for v in results}):
    rows = [v for v in results if v['label'] == label]
    complete = [v for v in rows if v['complete']]
    entry = dict(completed=len(complete), repetitions=len(rows))
    if complete:
        for key in ['axial_span_m', 'axial_travel_m', 'short_period_axial_travel_m',
                    'compute_p50_ms', 'compute_p95_ms', 'peak_rss_mib',
                    'tail_axial_span_m', 'tail_axial_travel_m']:
            values = [v[key] for v in complete]
            entry[key] = dict(median=float(np.median(values)), min=min(values), max=max(values))
    summary[label] = entry
(r / 'nrun-metrics.json').write_text(json.dumps(dict(runs=results, summary=summary), indent=2) + '\n')
print(json.dumps(summary, indent=2))
os.environ.setdefault('MPLCONFIGDIR', str(r.resolve() / 'mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(3, 1, figsize=(12, 9), layout='constrained')
colors = dict(baseline='#ac4726', candidate='#1769aa')
for (label, rep), (t, x) in trajectories.items():
    for ax in axes:
        ax.plot(t, x, color=colors[label], alpha=0.6, lw=1,
                label=label if rep == 1 else None)
axes[0].set_title('0705 tunnel axis: all completed repetitions (no independent pose ground truth)')
axes[1].set_xlim(0, 60)
axes[1].set_ylim(-13, 2)
axes[1].set_title('First minute')
axes[2].set_xlim(530, 553)
tail_values = [x[t >= 530] for t, x in trajectories.values() if np.any(t >= 530)]
if tail_values:
    low = min(float(x.min()) for x in tail_values)
    high = max(float(x.max()) for x in tail_values)
    margin = max(0.02, 0.05 * (high - low))
    axes[2].set_ylim(low - margin, high + margin)
axes[2].set_title('Final segment')
for ax in axes:
    ax.set_xlabel('Recording time (s)')
    ax.set_ylabel('World X position (m)')
    ax.grid(alpha=0.2)
axes[0].legend()
fig.savefig(r / 'nrun-tunnel-axis.png', dpi=150)
