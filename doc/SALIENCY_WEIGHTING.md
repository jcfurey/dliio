# Saliency-weighted point selection (anti-dilution)

Implements direction #1 from `doc/EXPLORATION_2026-06-26.md`: attack the tunnel
degeneracy's recurring **dilution** failure mode at its root, in the geometric
GICP term, instead of adding yet another auxiliary observation that dilutes onto
the walls (all of which — intensity, condScale, near-IR, dense visual, GenZ —
came back A/B-negative).

## Idea

In a tunnel the GICP cost is dominated by the abundant, redundant **planar wall**
points, whose plane normals are perpendicular to the tunnel axis and contribute
nothing to the along-axis DOF. The rare **salient** points — ribs, bolts, junction
boxes, pipe flanges, conduit hangers (edges/corners that vary *along* the axis) —
are what actually constrain it, and they get swamped. So **up-weight the salient
points** and leave the planar walls at baseline, concentrating the term's mass
where the weak-axis information is.

## How

- **Saliency** (`pointSaliency`, `nano_gicp/saliency_weight.h`) is computed per
  point from its local neighborhood-covariance eigenvalues (the LOAM/DAMM-LOAM PCA
  feature test): `saliency = 1 − planarity`, `planarity = (λ1−λ0)/λ2`. Planar
  wall → ~0; edge/rib/corner → ~1. Computed from the **raw** covariance in
  `calculate_covariances`, before the GICP regularization flattens the shape, and
  only for the **source** cloud, and only when the feature is on (the extra
  per-point eigendecomposition is skipped otherwise).
- **Weight** (`saliencyMultiplier`): `1 + (boost−1)·saliency`, applied to the
  per-correspondence metric `M` in `linearize` (scales that point's whole
  contribution to H, b, cost). A planar point keeps weight 1; a maximally-salient
  point gets up to `boost`.
- **Controls:** `setSaliencyWeighting(enabled, boost)`; params
  `odom/saliency/{enabled, boost}`. `boost = 1` (default) → unit weights,
  **bit-identical**, and no saliency computation at all.

Composes with the validated X-ICP ternary gate (the gate decides *which*
directions to trust; this decides *which points* inform them) and the governor.

## Status

Default-off prototype. Kernel unit-tested (`test_saliency_weight.cpp`, 8 cases);
wiring covered by `SaliencyDisabledIsBitIdentical` + `SaliencyOnStillRecoversTransform`
in `test_nano_gicp.cpp`. Build + `colcon test` (ROS 2 Jazzy): 162 tests, 0
failures. **Efficacy on the tunnel is unvalidated** — needs a `boost` sweep A/B on
the 06042026 bag (e.g. `{1, 2, 4, 8}`, with the X-ICP gate on), watching for the
deg=6 rate and worst-case `max|x|`. Risk to watch: over-boosting can amplify
noisy/sparse salient returns, so start low.
