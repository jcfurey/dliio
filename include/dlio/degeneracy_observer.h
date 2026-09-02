// LODESTAR-flavored degeneracy-aware observer gain (default off).
//
// LODESTAR (Lee, Marsim, Myung, RA-L 2025) makes degenerate ("fixed") states take
// ZERO Kalman gain so a held axis is not corrected by the unreliable measurement
// and instead coasts as a reference anchor. DLIO has no covariance-propagating
// filter to host a Schmidt-Kalman update (see doc/IMPLEMENTATION_NOTES.md §4), but
// its contracting hierarchical observer (Lopez 2023) applies a SCALAR gain to the
// LiDAR pose error in updateState(). The same intent transfers directly: on the
// world-frame eigen-directions the degeneracy gate flagged degenerate AND HELD,
// ATTENUATE the observer's correction so the state rides the IMU-propagated prior
// there instead of absorbing registration noise / coasting bias into the runaway.
//
// This is the contracting-observer analogue of LODESTAR's reduced-gain fixed
// state: a per-axis gain on the held directions, NOT a full ESKF rewrite.
// gain == 1 -> the correction is untouched (feature OFF, bit-identical); gain == 0
// -> the correction along a held axis is fully removed (the axis is "fixed", pure
// IMU coast); 0 < gain < 1 -> partial trust. Pairs with the governor's covariance
// inflation (which marks the same axes untrusted downstream).
//
// Pure, type-independent free function so it is cheap to unit-test in isolation
// (test/test_degeneracy_observer.cpp), mirroring governPose / softGateKeepFraction.
#pragma once

#include <cmath>
#include <vector>
#include <Eigen/Core>

namespace dlio {

// Attenuate the world-frame correction `e` along each held world-frame direction
// in `dirs` by `gain` in [0,1]: the component of `e` along a held axis is scaled
// to `gain * component`. `dirs` are unit eigen-directions (re-normalized
// defensively). gain >= 1 (or empty dirs) returns `e` unchanged -> OFF /
// bit-identical. Held directions from one self-adjoint block are orthonormal, so
// the sequential projection is order-independent and exact.
inline Eigen::Vector3f attenuateAlongHeldAxes(
    Eigen::Vector3f e, const std::vector<Eigen::Vector3d>& dirs, float gain) {
  if (gain >= 1.f || dirs.empty()) { return e; }
  const float g = (gain > 0.f) ? gain : 0.f;     // clamp negative gains to a full freeze
  for (const auto& dd : dirs) {
    const float n2 = static_cast<float>(dd.squaredNorm());
    if (n2 <= 0.f) { continue; }
    const Eigen::Vector3f d = dd.cast<float>() / std::sqrt(n2);
    e -= d * (d.dot(e)) * (1.f - g);             // scale the along-axis component to g*comp
  }
  return e;
}

}  // namespace dlio
