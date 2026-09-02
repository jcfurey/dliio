// GenZ-ICP adaptive point-to-plane / point-to-point blend weight.
//
// Adapted from GenZ-ICP (Lee et al., "GenZ-ICP: Generalizable and Degeneracy-
// Robust LiDAR Odometry Using an Adaptive Weighting," IEEE RA-L 2025,
// arXiv:2411.06766; code: github.com/cocel-postech/genz-icp). GenZ-ICP keeps the
// registration well-posed in degenerate geometry (long corridors / tunnels) by
// COMBINING two metrics that fail in opposite ways: point-to-plane is sharp on
// well-observed surfaces but contributes nothing along an unconstrained axis,
// while point-to-point adds isotropic stiffness that regularizes that axis at the
// cost of accuracy on planes. The residual is a convex blend
//     r = alpha * r_point_to_plane + (1 - alpha) * r_point_to_point
// with alpha driven by how planar / well-conditioned the local geometry is.
//
// GenZ-ICP derives alpha from per-point planarity. Here we drive it from the
// geometric translation-Hessian conditioning the degeneracy gate ALREADY
// eigendecomposes each iteration (lambda_min / lambda_max of H_geo[3:6,3:6]), so
// the blend needs no extra per-point eigenanalysis and reacts to exactly the
// degeneracy the gate measures: a well-conditioned scan keeps pure point-to-plane
// (alpha = 1, bit-identical), an ill-conditioned scan blends point-to-point in.
//
// Pure, type-independent free function so it is cheap to unit-test in isolation
// (see test/test_genz_weight.cpp), mirroring softGateKeepFraction / governPose.
#pragma once

#include <algorithm>

namespace nano_gicp {

// Point-to-plane weight `alpha` in [floor, 1] for a scan whose geometric
// translation block has conditioning `cond_ratio` = lambda_min / lambda_max in
// (0, 1] (1 = isotropic / fully observable, ->0 = a collapsed/degenerate axis).
//
//   cond_ratio >= knee : alpha = 1            (healthy -> pure point-to-plane)
//   0 < cond_ratio < knee : linear ramp        (alpha falls toward `floor`)
//   cond_ratio <= 0 (or NaN guards) : alpha = floor (maximally blended)
//
// `floor` in [0,1] is the smallest plane weight (1 = feature OFF / bit-identical;
// a typical enable is ~0.5 so the worst case is a 50/50 plane+point blend).
// `knee` in (0,1] is the conditioning at which point-to-point starts mixing in.
// (1 - alpha) is the point-to-point share the caller applies to its isotropic
// (identity-metric) term. Clamped and degenerate-input-safe.
inline double genzPlaneWeight(double cond_ratio, double floor, double knee) {
  if (!(floor < 1.0)) { return 1.0; }          // floor >= 1 (or NaN) -> feature off
  if (floor < 0.0) { floor = 0.0; }
  if (!(knee > 0.0)) { return floor; }          // degenerate knee -> always blended
  if (!(cond_ratio > 0.0)) { return floor; }    // collapsed axis (or NaN) -> floor
  if (cond_ratio >= knee) { return 1.0; }       // healthy -> pure point-to-plane
  const double t = cond_ratio / knee;           // in (0,1)
  return floor + (1.0 - floor) * t;             // linear ramp floor -> 1
}

}  // namespace nano_gicp
