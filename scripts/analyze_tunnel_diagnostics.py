#!/usr/bin/env python3
"""Audit diagnostic replay counters, coupled geometry, and optional baseline parity.

These are estimator consistency measurements; no trajectory accuracy is inferred.
Run from the workspace with NumPy, and Matplotlib when --plot is requested.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


STATUS = ('disabled', 'no_source', 'no_reference', 'frame_gap', 'invalid_input',
          'invalid_geometry', 'translation_strong', 'multiple_weak_translations',
          'too_few_unique', 'shift_disagreement', 'too_few_inliers', 'low_consensus',
          'accepted')
REJECTIONS = ('nonfinite', 'spacing', 'neighborhood', 'nonplanar', 'axis_normal',
              'reference_support', 'reference_contrast', 'repeated', 'source_support',
              'source_contrast', 'boundary', 'low_correlation', 'ambiguous', 'flat_peak')


def read(path):
    data = np.genfromtxt(path, delimiter=',', names=True, ndmin=1)
    if len(data) < 2 or not np.all(np.diff(data['scan_stamp']) > 0):
        raise ValueError(f'{path}: need at least two scans with increasing measurement timestamps')
    return data


def quantiles(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return np.quantile(values, [0, .5, .95, 1]).tolist() if len(values) else None


def audit(data):
    fields = data.dtype.names
    required = ['texture_status', 'texture_examined', 'geometry_valid', 'geometry_length']
    required += ['texture_rej_' + key for key in REJECTIONS]
    if any(key not in fields for key in required):
        raise ValueError('CSV does not contain the new geometry/texture diagnostics')
    count_keys = required[1:2] + ['texture_candidates', 'texture_supported', 'texture_unique', 'texture_inliers']
    count_keys += ['texture_rej_' + key for key in REJECTIONS]
    for key in count_keys:
        if not np.all(np.isfinite(data[key]) & (data[key] >= 0) & (data[key] == np.floor(data[key]))):
            raise ValueError(f'invalid count: {key}')
    rejected = {key: data['texture_rej_' + key] for key in REJECTIONS}
    identities = [
        (data['texture_examined'], data['texture_candidates'] + sum(rejected[k] for k in REJECTIONS[:7])),
        (data['texture_candidates'], data['texture_supported'] + sum(rejected[k] for k in REJECTIONS[7:10])),
        (data['texture_supported'], data['texture_unique'] + sum(rejected[k] for k in REJECTIONS[10:])),
    ]
    if any(not np.array_equal(left, right) for left, right in identities):
        raise ValueError('texture counter accounting does not balance')
    status = data['texture_status']
    if not np.all(np.isfinite(status) & (status == np.floor(status)) & (status >= 0) & (status < len(STATUS))):
        raise ValueError('invalid texture status')
    accepted = data['texture_valid'] > 0
    if not np.array_equal(accepted, status == STATUS.index('accepted')):
        raise ValueError('acceptance flag and status disagree')
    time = data['scan_stamp'] - data['scan_stamp'][0]
    valid = data['geometry_valid'] > 0
    eigen = np.column_stack([data[f'geometry_eig_{i}'] for i in range(6)])
    mode = np.column_stack([data[f'geometry_mode_{i}'] for i in range(6)])
    if valid.any() and (not np.isfinite(eigen[valid]).all() or np.any(eigen[valid] < 0) or
                        np.any(np.diff(eigen[valid], axis=1) < 0) or
                        not np.allclose(np.linalg.norm(mode[valid], axis=1), 1.0)):
        raise ValueError('invalid geometry spectrum or weak vector')
    ratio = np.full(len(data), np.nan)
    ratio[valid] = eigen[valid, 0] / eigen[valid, 5]
    rotation_share = np.sum(mode[:, :3] ** 2, axis=1)
    translation_share = np.sum(mode[:, 3:] ** 2, axis=1)
    # Descriptive fractions in the declared scaled coordinates, not a gate or
    # an observability probability. A near-repeated smallest eigenvalue makes
    # interpretation of a single vector unstable; report that separately.
    separated = valid & (eigen[:, 1] > 1.2 * np.maximum(eigen[:, 0], 1e-30))
    translating_mode = separated & (translation_share > 1e-12)
    yaw_per_translation = np.full(len(data), np.nan)
    yaw_per_translation[translating_mode] = np.abs(mode[translating_mode, 2]) / (
        data['geometry_length'][translating_mode] * np.sqrt(translation_share[translating_mode]))

    # A weak roll/pitch mode can precede the yaw/translation mode of interest.
    # This conditional view holds world-X/Y rotations fixed. It is NOT an IMU
    # covariance update or marginalization over those two rotations.
    yaw_translation = None
    if 'geometry_h_0_0' in fields:
        h = np.zeros((len(data), 6, 6))
        for i in range(6):
            for j in range(i, 6):
                h[:, i, j] = h[:, j, i] = data[f'geometry_h_{i}_{j}']
        if not np.isfinite(h[valid]).all():
            raise ValueError('nonfinite centered geometry matrix')
        recomputed = np.linalg.eigvalsh(h[valid])
        if not np.allclose(np.maximum(0, recomputed), eigen[valid], rtol=1e-8, atol=1e-8):
            raise ValueError('saved centered matrix does not reproduce the logged spectrum')
        e4, v4 = np.linalg.eigh(h[valid, 2:, 2:])
        Htt = h[valid, 3:, 3:]
        Htz = h[valid, 3:, 2]
        Hzz = h[valid, 2, 2]
        inverse_zz = np.divide(1.0, Hzz, out=np.zeros_like(Hzz), where=Hzz > 0)
        schur = Htt - np.einsum('ni,nj->nij', Htz, Htz) * inverse_zz[:, None, None]
        tt_values = np.linalg.eigvalsh(Htt)
        yaw_retained = np.divide(np.maximum(0, np.linalg.eigvalsh(schur)[:, 0]), tt_values[:, 0],
                                 out=np.full(len(Htt), np.nan), where=tt_values[:, 0] > 1e-9 * tt_values[:, 2])
        v = v4[:, :, 0]
        trans_norm = np.linalg.norm(v[:, 1:], axis=1)
        suitable = (e4[:, 1] > 1.2 * np.maximum(e4[:, 0], 1e-30)) & (trans_norm > 1e-6)
        physical_yaw = np.full(len(e4), np.nan)
        physical_yaw[suitable] = np.abs(v[suitable, 0]) / (
            data['geometry_length'][valid][suitable] * trans_norm[suitable])
        yaw_translation = {}
        ratio4 = np.divide(np.maximum(0, e4[:, 0]), e4[:, 3], out=np.full(len(e4), np.nan), where=e4[:, 3] > 0)
        for key, values in [('ratio', ratio4),
                            ('weak_yaw_share', v[:, 0] ** 2),
                            ('schur_retained', yaw_retained),
                            ('abs_weak_yaw_per_translation_rad_m', physical_yaw)]:
            yaw_translation[key] = np.full(len(data), np.nan)
            yaw_translation[key][valid] = values

    def summarize(mask):
        geometry = mask & valid
        schur = geometry & (data['geometry_schur_retained'] >= 0)
        summary = {
            'scans': int(mask.sum()),
            'texture_status': {name: int(np.sum(mask & (status == i))) for i, name in enumerate(STATUS)},
            'texture_patch_rejections': {key: int(values[mask].sum()) for key, values in rejected.items()},
            'texture_patch_totals': {key: int(data[key][mask].sum()) for key in
                                     ('texture_examined', 'texture_candidates', 'texture_supported', 'texture_unique')},
            'geometry_valid_scans': int(geometry.sum()),
            'geometry_length_m': sorted(set(data['geometry_length'][geometry].tolist())),
            'full_ratio_min_median_p95_max': quantiles(ratio[geometry]),
            'translation_ratio_min_median_p95_max': quantiles(data['geometry_trans_ratio'][geometry]),
            'centered_rotation_ratio_min_median_p95_max': quantiles(data['geometry_rot_ratio'][geometry]),
            'schur_retained_min_median_p95_max': quantiles(data['geometry_schur_retained'][schur]),
            'half_length_ratio_min_median_p95_max': quantiles(data['geometry_half_length_ratio'][geometry]),
            'double_length_ratio_min_median_p95_max': quantiles(data['geometry_double_length_ratio'][geometry]),
            'weak_rotation_share_min_median_p95_max': quantiles(rotation_share[geometry]),
            'weak_world_z_rotation_share_min_median_p95_max': quantiles(mode[geometry, 2] ** 2),
            'separated_weak_mode_scans': int(np.sum(mask & separated)),
            'absolute_weak_world_z_rotation_per_translation_rad_m_min_median_p95_max': quantiles(
                yaw_per_translation[mask & translating_mode]),
            'absolute_weak_correction_m_min_median_p95_max': quantiles(np.abs(data['geometry_correction_projection'][geometry])),
        }
        if yaw_translation is not None:
            summary['yaw_translation_with_world_xy_rotation_fixed'] = {
                key + '_min_median_p95_max': quantiles(values[mask]) for key, values in yaw_translation.items()}
        return summary

    report = {'scans': len(data), 'duration_s': float(time[-1]), 'counter_accounting': 'exact for every scan',
              'all': summarize(np.ones(len(data), dtype=bool)),
              'window_360_500_s': summarize((time >= 360) & (time < 500)),
              'interpretation': {'separated': 'lambda_1 > 1.2 * lambda_0 (descriptive single-vector separation)',
                                 'yaw_per_translation': 'abs(weak_z_rotation) / (L * norm(weak_translation)); no binary mixed-mode cutoff',
                                 'yaw_translation_subsystem': 'principal 4x4 block [L * world-Z rotation, xyz translation]; other rotations held fixed',
                                 'meaning': 'local normal-matrix diagnostics, not an estimator gate or calibrated confidence'}}
    indices = np.flatnonzero(accepted)
    report['first_texture_acceptance_s'] = float(time[indices[0]]) if len(indices) else None
    report['longest_interior_texture_gap'] = None
    if len(indices) > 1:
        j = int(np.argmax(np.diff(time[indices])))
        a, b = indices[j:j + 2]
        report['longest_interior_texture_gap'] = {
            'start_s': float(time[a]), 'end_s': float(time[b]), 'duration_s': float(time[b] - time[a]),
            'interior': summarize((np.arange(len(data)) > a) & (np.arange(len(data)) < b))}
    return report, time, ratio, rotation_share, yaw_translation


def parity(data, baseline):
    if len(data) != len(baseline) or not np.array_equal(data['scan_stamp'], baseline['scan_stamp']):
        raise ValueError('baseline replay has different measurement timestamps or length')
    shared = sorted(set(data.dtype.names) & set(baseline.dtype.names) - {'compute_ms', 'wait_ms'})
    # An enabled/disabled comparison intentionally changes the geometry-only
    # diagnostic columns, while all common estimator and texture values must
    # remain identical. Earlier baselines lack these appended fields entirely.
    shared = [key for key in shared if not key.startswith('geometry_')]
    differences = {}
    for key in shared:
        same = (data[key] == baseline[key]) | (np.isnan(data[key]) & np.isnan(baseline[key]))
        if not same.all():
            first = int(np.flatnonzero(~same)[0])
            differences[key] = {'first_scan': first, 'different_scans': int((~same).sum()),
                                'max_absolute_difference': float(np.nanmax(np.abs(data[key] - baseline[key])))}
    return {'exact': not differences, 'compared_columns': shared, 'differences': differences}


def plot(data, time, ratio, rotation_share, yaw_translation, output):
    os.environ.setdefault('MPLCONFIGDIR', str(output / 'matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True, constrained_layout=True)
    xyz = np.column_stack([data[key] for key in ('x', 'y', 'z')])
    speed = np.linalg.norm(np.diff(xyz, axis=0), axis=1) / np.diff(time)
    axes[0].plot(time[1:], speed, linewidth=.7)
    axes[0].set_ylabel('Estimated speed [m/s]')
    lengths = sorted(set(data['geometry_length'][data['geometry_valid'] > 0].tolist()))
    length_label = '/'.join(f'{value:g}' for value in lengths) or 'unavailable'
    for values, label in [(ratio, f'Full geometry, L={length_label} m'),
                           (data['geometry_trans_ratio'], 'Translation block'),
                           (data['geometry_schur_ratio'], 'Translation after rotation adjusts')]:
        axes[1].semilogy(time, np.where(values > 0, values, np.nan), label=label, linewidth=.7)
    axes[1].set_ylabel('Minimum / reference maximum')
    axes[1].legend(loc='lower left', fontsize=8)
    valid = data['geometry_valid'] > 0
    axes[2].plot(time, np.where(valid, data['geometry_schur_retained'], np.nan), label='Schur minimum / translation minimum', linewidth=.7)
    axes[2].plot(time, np.where(valid, rotation_share, np.nan), label='Weak-mode rotation share', linewidth=.7)
    if yaw_translation is not None:
        axes[2].plot(time, yaw_translation['schur_retained'], label='Translation retained when only yaw adjusts', linewidth=.7)
    axes[2].set_ylim(0, 1.02)
    axes[2].legend(loc='upper left', fontsize=8)
    axes[2].set_ylabel('Fraction')
    edges = np.arange(0, time[-1] + 10, 10.0)
    centers = (edges[:-1] + edges[1:]) / 2
    candidate = np.histogram(time, bins=edges, weights=data['texture_candidates'])[0]
    values = []
    for key in ['texture_rej_repeated', 'texture_rej_source_support', 'texture_rej_source_contrast', 'texture_supported']:
        counts = np.histogram(time, bins=edges, weights=data[key])[0]
        values.append(np.divide(counts, candidate, out=np.zeros_like(counts), where=candidate > 0))
    axes[3].stackplot(centers, values, labels=['Repeated reference', 'Missing source support', 'Low source contrast', 'Supported'],
                      colors=['#8064a2', '#df8a30', '#b5bac1', '#39895a'])
    axes[3].set_ylim(0, 1)
    axes[3].set_ylabel('Candidate patches, 10 s bins')
    axes[3].set_xlabel('Seconds from first registered scan midpoint')
    axes[3].legend(loc='lower left', fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle('September tunnel diagnostic replay — consistency signals, no trajectory ground truth')
    fig.savefig(output / 'diagnostics.png', dpi=160)
    fig.savefig(output / 'diagnostics.svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    data = read(args.csv)
    report, time, ratio, rotation_share, yaw_translation = audit(data)
    report['input'] = {'path': str(args.csv), 'sha256': hashlib.sha256(args.csv.read_bytes()).hexdigest()}
    if args.baseline:
        report['baseline'] = {'path': str(args.baseline), 'sha256': hashlib.sha256(args.baseline.read_bytes()).hexdigest()}
        report['parity'] = parity(data, read(args.baseline))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if args.plot:
        plot(data, time, ratio, rotation_share, yaw_translation, args.output)
    print(json.dumps({'scans': report['scans'], 'counter_accounting': report['counter_accounting'],
                      'parity_exact': report.get('parity', {}).get('exact'),
                      'output': str(args.output)}, indent=2))
    if args.baseline and not report['parity']['exact']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
