// X-ICP ternary localizability classification.
//
// From X-ICP (Tuna, Nubert, Nava, Khattak, Hutter, "X-ICP: Localizability-Aware
// LiDAR Registration for Robust Localization in Extreme Environments," IEEE T-RO
// 2024, vol. 40, pp. 452-471; arXiv:2211.16335). X-ICP makes a TERNARY judgment
// per optimization eigen-direction rather than the binary observable/degenerate
// split of Zhang/Kaess/Singh solution remapping:
//
//   - LOCALIZABLE      (eigval >= kappa_full)            : take the data update.
//   - PARTIAL          (kappa_partial <= eigval < kappa_full) : take a BOUNDED,
//                       controlled update (admit it scaled, don't fully freeze).
//   - NON_LOCALIZABLE  (eigval < kappa_partial)          : hard equality constraint
//                       -- zero update along the eigenvector, hold the prior.
//
// The binary endpoints already exist in nano_gicp.cc's gate (the Zhang remap +
// softGateKeepFraction). This adds the MIDDLE category X-ICP contributes: a
// second threshold and a controlled partial admit across the partial band, so a
// marginal axis is neither fully trusted nor fully frozen. Wiring into the gate
// reuses these two pure functions (see doc/IMPLEMENTATION_NOTES.md); kept as a
// header-only, type-independent kernel so it is unit-tested in isolation
// (test/test_xicp_localizability.cpp), like softGateKeepFraction / governPose.
//
// NOTE on thresholds: X-ICP derives kappa from the localizability CONTRIBUTION of
// the correspondences (alignment strength against each principal direction). Here
// the kappa_* are passed in as absolute eigenvalue thresholds so the classifier
// is agnostic to how the caller computes them (e.g. ratio*lambda_max for the
// partial bar and a looser multiple for the full bar, matching the gate's
// existing degeneracy_thresh_ratio convention).
#pragma once

#include <algorithm>

namespace nano_gicp {

enum class Localizability { NonLocalizable = 0, Partial = 1, Localizable = 2 };

// Classify an eigen-direction by its (block) Hessian eigenvalue against the two
// thresholds. Requires kappa_partial <= kappa_full; if they are mis-ordered or
// equal the classifier collapses to the binary split at kappa_full (no partial
// band), so a caller that sets them equal recovers the existing gate behavior.
inline Localizability xicpCategory(double eigval, double kappa_partial,
                                   double kappa_full) {
  if (kappa_partial > kappa_full) { kappa_partial = kappa_full; }
  if (eigval >= kappa_full) { return Localizability::Localizable; }
  if (eigval >= kappa_partial) { return Localizability::Partial; }
  return Localizability::NonLocalizable;
}

// Fraction of the data update to ADMIT along an eigen-direction, in [0,1]:
//   1            when LOCALIZABLE   (eigval >= kappa_full)
//   0            when NON_LOCALIZABLE (eigval < kappa_partial; hold the prior)
//   linear ramp  across the PARTIAL band (0 at kappa_partial -> 1 at kappa_full)
// This is the controlled-update scale; (1 - scale) is the share of the IMU prior
// held, generalizing the gate's binary `keep` to X-ICP's ternary judgment. A
// degenerate band (kappa_partial >= kappa_full) reduces to a hard step at
// kappa_full. NaN-safe.
inline double xicpPartialScale(double eigval, double kappa_partial,
                               double kappa_full) {
  if (!(kappa_partial < kappa_full)) {           // no/degenerate band -> binary step
    return (eigval >= kappa_full) ? 1.0 : 0.0;
  }
  if (eigval >= kappa_full) { return 1.0; }
  if (eigval <= kappa_partial) { return 0.0; }
  return (eigval - kappa_partial) / (kappa_full - kappa_partial);
}

}  // namespace nano_gicp
