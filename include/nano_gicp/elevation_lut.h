// Per-row elevation LUT: validity test + inversion.
//
// OS-series beam elevations are NON-uniform, so the linear el(row) model leaves
// a row-dependent projection bias; the node self-calibrates a per-row LUT
// (row -> mean elevation) from the first organized scan and every projection
// consumer inverts it (elevation -> fractional row + local slope).
//
// Shared here because there are THREE consumers -- the LiDAR frame-to-map term,
// the frame-to-frame flow term, and the node's keyframe-image reference sampler
// -- and they MUST agree. A 2026-07-26 audit found the node's sampler using a
// private copy of the inversion with only a size check, while the GICP path
// validated finiteness + monotonicity first: on a LUT with any NaN row the two
// sides silently chose DIFFERENT projection models (linear vs LUT), so the
// reference brightness was sampled at different pixels than the residual
// projected to. One implementation, one validity predicate, no drift.
//
// Why a LUT row can be NaN: the node builds it once (cached for the run) and
// fills only rows that had at least one valid return in that first organized
// scan; a beam seeing sky/void leaves quiet_NaN behind permanently.
#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace nano_gicp {

// True when `lut` can be used as a projection model for an image with
// `expected_rows` rows: right size, at least two entries, all finite, and
// STRICTLY monotonic. Monotonicity is required, not merely nice: the inversion
// below picks its search orientation from the endpoints and scans half-open
// intervals, which is unsound on a non-monotonic table. Callers that get false
// must fall back to the linear el_a/el_b model.
inline bool elevationLutUsable(const std::vector<float>& lut, int expected_rows) {
  if (expected_rows <= 0) { return false; }
  if (static_cast<int>(lut.size()) != expected_rows) { return false; }
  if (lut.size() < 2) { return false; }
  for (size_t k = 0; k < lut.size(); ++k) {
    if (!std::isfinite(lut[k])) { return false; }
  }
  const bool inc = lut.back() > lut.front();
  for (size_t k = 1; k < lut.size(); ++k) {
    const float a = lut[k - 1], b = lut[k];
    if (inc ? !(b > a) : !(b < a)) { return false; }   // NaN-safe: !(NaN > x) is true
  }
  return true;
}

// Invert a monotonic per-row elevation LUT: given an elevation [rad], return the
// fractional row and the local slope d(el)/d(row) [rad/row] used by the
// projection Jacobian. Returns false if `el` is outside the LUT's coverage (the
// caller drops the point). Handles both increasing- and decreasing-with-row beam
// orderings. Behavior is unchanged from the original in-tree implementation for
// any LUT that passes elevationLutUsable().
inline bool rowFromElevationLut(const std::vector<float>& lut, float el,
                                float& row_out, float& slope_out) {
  const int n = static_cast<int>(lut.size());
  if (n < 2) { return false; }
  const bool inc = lut[n - 1] >= lut[0];
  for (int k = 0; k < n - 1; ++k) {
    const float a = lut[k], b = lut[k + 1];
    const float lo = inc ? a : b;
    const float hi = inc ? b : a;
    if (el >= lo && el <= hi) {          // NaN-safe: comparisons with NaN are false
      const float denom = b - a;
      if (std::abs(denom) < 1e-9f) { continue; }
      row_out = static_cast<float>(k) + (el - a) / denom;
      slope_out = denom;                 // [rad/row] between row k and k+1
      return true;
    }
  }
  return false;
}

}  // namespace nano_gicp
