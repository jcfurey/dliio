"""PCL FPFH correspondences and deterministic bounded RANSAC pose proposals.

Features only initialize geometric registration. They never bypass fit, held
observations, ambiguity, observability, cycle or graph residual checks.
"""
import numpy as np
from scipy.spatial import cKDTree


def seeds(source, target, prior, *, trials=8192, maximum=3):
    import _dliio_pose_graph as native
    from .registration import rotation_angle
    sf, tf = np.asarray(native.fpfh(source)), np.asarray(native.fpfh(target))
    si = np.flatnonzero(np.isfinite(sf).all(1) & (sf.sum(1) > 0))
    ti = np.flatnonzero(np.isfinite(tf).all(1) & (tf.sum(1) > 0))
    if min(len(si), len(ti)) < 100:
        return [], dict(feature_matches=0, feature_hypotheses=0)
    # Hellinger distance balances histogram bins without a fitted descriptor.
    sf, tf = np.sqrt(np.maximum(sf[si], 0)/300), np.sqrt(np.maximum(tf[ti], 0)/300)
    distances, neighbors = cKDTree(tf).query(sf, k=2, workers=1)
    _, backward = cKDTree(sf).query(tf, workers=1)
    keep = ((distances[:, 0] < .97*distances[:, 1]) &
            (backward[neighbors[:, 0]] == np.arange(len(si))))
    source_matches, target_matches = source[si[keep]], target[ti[neighbors[keep, 0]]]
    count = len(source_matches)
    report = dict(feature_matches=count, feature_hypotheses=0)
    if count < 12:
        return [], report
    # Bound scoring memory while retaining correspondences across the scene.
    if count > 2000:
        selection = np.linspace(0, count-1, 2000, dtype=int)
        source_matches, target_matches = source_matches[selection], target_matches[selection]
        count = len(selection)
    rng = np.random.default_rng(83290)
    candidates = []
    for start in range(0, trials, 256):
        indices = rng.integers(count, size=(min(256, trials-start), 3))
        a, b = source_matches[indices], target_matches[indices]
        sa, sb = np.linalg.norm(a-np.roll(a, 1, axis=1), axis=2), np.linalg.norm(b-np.roll(b, 1, axis=1), axis=2)
        good = ((np.minimum(sa, sb) > .75).all(1) &
                (np.minimum(sa, sb) >= .85*np.maximum(sa, sb)).all(1) &
                (np.linalg.norm(np.cross(a[:, 1]-a[:, 0], a[:, 2]-a[:, 0]), axis=1) > .5))
        a, b = a[good], b[good]
        if not len(a):
            continue
        ca, cb = a.mean(1), b.mean(1)
        u, _, vt = np.linalg.svd(np.einsum('nki,nkj->nij', a-ca[:, None], b-cb[:, None]))
        diagonal = np.tile(np.eye(3), (len(a), 1, 1))
        diagonal[:, 2, 2] = np.linalg.det(vt.transpose(0, 2, 1) @ u.transpose(0, 2, 1))
        rotations = vt.transpose(0, 2, 1) @ diagonal @ u.transpose(0, 2, 1)
        translations = cb-np.einsum('nij,nj->ni', rotations, ca)
        close_rotation = np.array([rotation_angle(prior[:3, :3].T @ r) < .5 for r in rotations])
        rotations, translations = rotations[close_rotation], translations[close_rotation]
        if not len(rotations):
            continue
        residual = np.einsum('nij,kj->nki', rotations, source_matches) + translations[:, None] - target_matches
        inliers = (np.linalg.norm(residual, axis=2) < .4).sum(1)
        report['feature_hypotheses'] += len(inliers)
        for index in np.argsort(inliers)[-maximum:]:
            if inliers[index] < max(12, int(.05*count)):
                continue
            pose = np.eye(4)
            pose[:3, :3], pose[:3, 3] = rotations[index], translations[index]
            candidates.append((int(inliers[index]), pose))
    candidates.sort(key=lambda entry: -entry[0])
    selected, counts = [], []
    for count, pose in candidates:
        if any(np.linalg.norm(pose[:3, 3]-old[:3, 3]) < .5 and
               rotation_angle(pose[:3, :3].T @ old[:3, :3]) < .05 for old in selected):
            continue
        selected.append(pose)
        counts.append(count)
        if len(selected) == maximum:
            break
    report['feature_inlier_counts'] = counts
    return selected, report
