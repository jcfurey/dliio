// PHYSICS FUSE — last-line output-step clamp for the takeoff failure mode.
//
// The governor (degeneracy_governor.h) caps per-scan motion along the HELD
// eigen-directions; it cannot act when the gate misses, when the runaway rides
// an axis the gate never flagged, or when the IMU prior itself blows up (bad
// deskew / bias runaway — the 2026-07-09 leg-1 pose reached 3.3e7 m). This fuse
// clamps the TOTAL per-scan output step, in any direction, to a physical bound:
// with maxStepTrans = 1 m at 10 Hz the published pose can never move faster
// than 10 m/s, so a "takeoff" is impossible at the output level by
// construction. It does NOT fix the estimate — it bounds the damage, flags the
// scan (trip counter -> /diagnostics, keyframe veto), and keeps the trajectory
// physically plausible for downstream consumers (r_l EKF, planners).
//
// Distinct from odom/gicp/maxCorr*: that clamps the GICP CORRECTION vs the IMU
// prior (the prior already contains the motion, so a diverging prior passes it
// untouched). This clamps the OUTPUT pose vs the previous output pose — prior
// divergence included.
//
// Pure geometry (no ROS / node state), directly unit-testable:
// test/test_physics_fuse.cpp. Caps <= 0 disable that block (bit-identical).
#pragma once

#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace dlio {

// Clamp T_new's step relative to T_prev to max_step_trans [m] and
// max_step_rot [rad]. tripped (optional) reports whether any clamp engaged.
// Translation: the step VECTOR is rescaled (direction preserved). Rotation:
// the relative rotation angle is capped about its own axis (slerp-equivalent).
inline Eigen::Matrix4f fusePose(const Eigen::Matrix4f& T_prev,
                                const Eigen::Matrix4f& T_new,
                                float max_step_trans, float max_step_rot,
                                bool* tripped = nullptr) {
  Eigen::Matrix4f T = T_new;
  bool trip = false;

  if (max_step_trans > 0.f) {
    const Eigen::Vector3f p_prev = T_prev.block<3, 1>(0, 3);
    const Eigen::Vector3f dp = T_new.block<3, 1>(0, 3) - p_prev;
    const float n = dp.norm();
    // NaN-safe: a non-finite step fails the <= comparison and is replaced by
    // holding the previous position outright (norm can't be scaled).
    if (!(n <= max_step_trans)) {
      trip = true;
      T.block<3, 1>(0, 3) =
          (n > 0.f && std::isfinite(n)) ? Eigen::Vector3f(p_prev + dp * (max_step_trans / n))
                                        : p_prev;
    }
  }

  if (max_step_rot > 0.f) {
    const Eigen::Matrix3f R_prev = T_prev.block<3, 3>(0, 0);
    const Eigen::Matrix3f R_rel = T_new.block<3, 3>(0, 0) * R_prev.transpose();
    Eigen::AngleAxisf aa(R_rel);
    const float ang = aa.angle();   // in [0, pi] from a rotation matrix
    if (!(ang <= max_step_rot)) {
      trip = true;
      const Eigen::Matrix3f R_capped = std::isfinite(ang)
          ? Eigen::AngleAxisf(max_step_rot, aa.axis()).toRotationMatrix()
          : Eigen::Matrix3f::Identity();
      T.block<3, 3>(0, 0) = R_capped * R_prev;
    }
  }

  if (tripped != nullptr) { *tripped = trip; }
  return T;
}

}  // namespace dlio
