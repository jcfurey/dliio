"""Explicit, verified observable surface modes for degenerate window loops.

The scaled point-to-plane Hessian selects directions, not calibrated noise.
Unretained modes have exactly zero information. Residuals and Jacobians use the
same right/source-local tangent as the native graph. Ordinary loops retain all
their existing gates; this opt-in contract requires disjoint window support.
"""
import numpy as np
from scipy.spatial import cKDTree

from .loop_validation import RejectedLoop, Validation, _normals, _sample


def subspace(value):
    if not isinstance(value, dict) or set(value) != {'projection', 'characteristic_length', 'eigenvalue_ratio'}:
        raise ValueError('Declare a scaled surface projection, length and eigenvalue ratio')
    p = np.asarray(value['projection'], dtype=float)
    length, ratio = value['characteristic_length'], value['eigenvalue_ratio']
    if (type(length) not in (float, int) or not np.isfinite(length) or not .15 <= length <= 80 or
            type(ratio) not in (float, int) or not .001 <= ratio <= .1 or
            p.shape != (6, 6) or not np.isfinite(p).all() or
            not np.allclose(p, p.T, atol=1e-9, rtol=0) or
            not np.allclose(p @ p, p, atol=1e-9, rtol=0) or
            not 3 <= np.count_nonzero(np.linalg.eigvalsh(p) > .5) <= 5):
        raise ValueError('Surface subspace must be an orthogonal projector of rank 3..5 with bounded scaling')
    return dict(projection=p.tolist(), characteristic_length=float(length), eigenvalue_ratio=float(ratio))


def projection(value):
    value = subspace(value)
    length = value['characteristic_length']
    scale = np.diag([1., 1., 1., length, length, length])
    return np.linalg.inv(scale) @ np.asarray(value['projection']) @ scale


def _jacobian(source, target, normal, measured, length, reverse):
    normal = normal @ measured[:3, :3]
    # In reverse matching the source normal moves too. Differentiating it
    # changes the rotational lever arm to the fixed target in source axes.
    support = (target-measured[:3, 3]) @ measured[:3, :3] if reverse else source
    return np.column_stack((normal, np.cross(support, normal)/length))


def geometry(target_points, source_points, measured, settings, length=None):
    target, source = [_sample(points, settings, min(settings.max_reference_points, 30000))
                      for points in (target_points, source_points)]
    if min(len(target), len(source)) < max(settings.min_inliers, settings.normal_neighbors):
        raise RejectedLoop('insufficient_geometry')
    length = max(.15, float(np.sqrt(np.mean(np.sum(source**2, axis=1))))) if length is None else length
    tn, tv = _normals(target, settings)
    sn, sv = _normals(source, settings)
    moved, rotated = source @ measured[:3, :3].T + measured[:3, 3], sn @ measured[:3, :3].T
    result = []
    for p, q, pn, qn, pv, qv, reverse in (
            (target, moved, tn, rotated, tv, sv, False), (moved, target, rotated, tn, sv, tv, True)):
        ids = np.linspace(0, len(q)-1, min(settings.max_points, len(q)), dtype=int)
        distances, nearest = cKDTree(p).query(q[ids])
        close = distances <= settings.match_radius
        report = dict(overlap=float(close.mean()), inliers=int(close.sum()))
        if int(close.sum()) < settings.min_inliers or report['overlap'] < settings.min_overlap:
            raise RejectedLoop('bidirectional_overlap', report)
        report['inlier_p95'] = float(np.quantile(distances[close], .95))
        if report['inlier_p95'] > settings.max_inlier_p95:
            raise RejectedLoop('geometric_residual', report)
        valid = close & pv[nearest] & qv[ids] & (
            abs(np.sum(pn[nearest]*qn[ids], axis=1)) >= settings.min_normal_alignment)
        selected = np.flatnonzero(valid)
        _, first = np.unique(nearest[selected], return_index=True)
        selected = selected[first]
        pi, qi = nearest[selected], ids[selected]
        report.update(plane_fraction=float(valid.mean()), plane_support=len(selected))
        if len(selected) < settings.min_inliers or report['plane_fraction'] < settings.min_surface_fraction:
            raise RejectedLoop('insufficient_surface_support', report)
        jacobian = _jacobian(source[pi if reverse else qi], target[qi if reverse else pi],
                             pn[pi], measured, length, reverse)
        residual = np.sum((q[qi]-p[pi])*pn[pi], axis=1)*(-1 if reverse else 1)
        report.update(p95=float(np.quantile(abs(residual), .95)), rms=float(np.sqrt(np.mean(residual**2))))
        if report['p95'] > settings.max_surface_p95 or report['rms'] > settings.max_surface_rms:
            raise RejectedLoop('surface_residual', report)
        result.append((report, jacobian, residual))
    return length, result


def propose_subspace(target, source, measured, settings, ratio=.01):
    length, directions = geometry(target, source, measured, settings)
    jacobian = np.vstack([entry[1] for entry in directions])
    eigenvalues, vectors = np.linalg.eigh(jacobian.T @ jacobian/len(jacobian))
    retained = vectors[:, eigenvalues >= eigenvalues[-1]*ratio]
    value = subspace(dict(projection=(retained @ retained.T).tolist(),
                          characteristic_length=length, eigenvalue_ratio=ratio))
    return value


def validate_geometry(target, source, measured, settings, value, *, verify_projection=False):
    value = subspace(value)
    length, directions = geometry(target, source, measured, settings, value['characteristic_length'])
    p = np.asarray(value['projection'])
    eigenvalues, vectors = np.linalg.eigh(p)
    basis = vectors[:, eigenvalues > .5]
    if verify_projection:
        expected = propose_subspace(target, source, measured, settings, value['eigenvalue_ratio'])
        if (not np.isclose(expected['characteristic_length'], length, rtol=1e-9) or
                not np.allclose(expected['projection'], p, atol=1e-8, rtol=0)):
            raise RejectedLoop('unsupported_surface_projection')
    reports = []
    for report, jacobian, residual in directions:
        h = jacobian.T @ jacobian/len(jacobian)
        spectrum = np.linalg.eigvalsh(basis.T @ h @ basis)
        ratio = float(spectrum[0]/max(np.linalg.eigvalsh(h)[-1], 1e-12))
        delta = basis @ np.linalg.lstsq(jacobian @ basis, -residual, rcond=None)[0]
        report.update(retained_rank=basis.shape[1], retained_observability_ratio=ratio,
            stationarity_translation=float(np.linalg.norm(delta[:3])),
            stationarity_rotation=float(np.linalg.norm(delta[3:])/length))
        # Each direction, including held observations, must support the selected
        # modes. A factor cannot declare confidence in an arbitrary direction.
        if ratio < value['eigenvalue_ratio']*.5:
            raise RejectedLoop('unobservable_retained_modes', report)
        if (report['stationarity_translation'] > settings.max_surface_translation_step or
                report['stationarity_rotation'] > settings.max_surface_rotation_step):
            raise RejectedLoop('surface_not_stationary', report)
        reports.append(report)
    return dict(forward=reports[0], reverse=reports[1])


def validate_support(store, candidate, config, *, pose=None):
    from .loop_inputs import window_cloud
    from .registration import fit, voxels
    import _dliio_pose_graph as native
    settings = Validation(**config['validation'])
    if not settings.surface_validation:
        raise ValueError('Projected loops require explicit surface validation')
    measured = np.asarray(candidate['transform']) if pose is None else pose
    value, reports = candidate['surface_subspace'], {}
    for name, akey, bkey in (('fitting', 'from_observations', 'to_observations'),
                             ('held', 'held_from_observations', 'held_to_observations')):
        target = window_cloud(store, candidate['from_id'], candidate['support'][akey],
                              config['max_span_seconds'], settings)
        source = window_cloud(store, candidate['to_id'], candidate['support'][bkey],
                              config['max_span_seconds'], settings)
        reports[name] = validate_geometry(target, source, measured, settings, value,
                                           verify_projection=pose is None and name == 'fitting')
        if pose is None:
            fitted = fit(voxels(source, settings.voxel_size), voxels(target, settings.voxel_size), measured)
            error = projection(value) @ native.residual(measured, np.eye(4), fitted)
            reports[name]['refit_translation'] = float(np.linalg.norm(error[:3]))
            reports[name]['refit_rotation'] = float(np.linalg.norm(error[3:]))
            if (np.linalg.norm(error[:3]) > settings.max_surface_translation_step or
                    np.linalg.norm(error[3:]) > settings.max_surface_rotation_step):
                raise RejectedLoop('unstable_surface_modes', reports)
    return reports
