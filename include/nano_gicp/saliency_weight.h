// Saliency-weighted point selection (anti-dilution) for the geometric GICP term.
//
// The recurring failure mode on the 06042026 tunnel is DILUTION: every auxiliary
// observation lever (intensity, condScale, near-IR, dense visual, GenZ) put its
// mass on the dominant wall geometry, never on the weak along-axis DOF, and came
// back A/B-negative (doc/FINDINGS_2026-06-22 / -25). This attacks the root cause
// in the term that DOES have mass on the weak axis when the features exist: the
// geometric GICP. Up-weight the rare SALIENT points (tunnel ribs, bolts, junction
// boxes, pipe flanges, conduit hangers -- edges/corners that vary ALONG the axis)
// and leave the abundant, redundant planar wall points at baseline, so the
// along-axis constraint is not swamped. (doc/EXPLORATION_2026-06-26.md #1; the
// local-covariance shape classification is the LOAM/DAMM-LOAM PCA feature test.)
//
// Pure, type-independent free functions (unit-tested, test/test_saliency_weight.cpp),
// mirroring genzPlaneWeight / softGateKeepFraction.
#pragma once

#include <algorithm>
#include <Eigen/Core>

namespace nano_gicp {

// Point saliency in [0,1] from the local neighborhood-covariance eigenvalues
// (ASCENDING: l0 <= l1 <= l2). saliency = 1 - planarity, planarity = (l1-l0)/l2:
//   planar wall (l0~0, l1~l2)      -> planarity ~1 -> saliency ~0 (baseline weight)
//   edge / rib   (l0~l1~0, l2 big) -> planarity ~0 -> saliency ~1 (boosted)
//   corner/scatter (l0~l1~l2)      -> planarity ~0 -> saliency ~1 (boosted)
// Degenerate (l2 <= 0) -> 0. Result is clamped to [0,1].
inline float pointSaliency(const Eigen::Vector3f& eigvals_ascending) {
  const float l0 = eigvals_ascending(0);
  const float l1 = eigvals_ascending(1);
  const float l2 = eigvals_ascending(2);
  if (!(l2 > 0.f)) { return 0.f; }
  const float planarity = (l1 - l0) / l2;
  float s = 1.f - planarity;
  if (s < 0.f) { s = 0.f; }
  if (s > 1.f) { s = 1.f; }
  return s;
}

// Multiplicative GICP weight for a point of the given saliency. `boost` >= 1 is
// the weight a maximally-salient point (saliency 1) receives; a planar point
// (saliency 0) gets 1.0. boost <= 1 -> always 1.0 (feature OFF / bit-identical).
//   weight = 1 + (boost - 1) * saliency
inline float saliencyMultiplier(float saliency, float boost) {
  if (!(boost > 1.f)) { return 1.f; }
  if (saliency < 0.f) { saliency = 0.f; }
  if (saliency > 1.f) { saliency = 1.f; }
  return 1.f + (boost - 1.f) * saliency;
}

}  // namespace nano_gicp
