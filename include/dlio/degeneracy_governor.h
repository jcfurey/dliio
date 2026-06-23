// Degeneracy GOVERNOR — fail-safe pose clamp for the held-prior runaway.
//
// The degeneracy gate (nano_gicp) holds the IMU prior on unobservable
// eigen-directions; that prior dead-reckons (accel/gyro bias double-integrates)
// into a km-scale divergence on a long featureless tunnel. This caps the
// per-scan OUTPUT motion along the held world-frame directions, vs the previous
// pose, to a physical per-scan step. Pure geometry (no ROS / node state) so it
// is directly unit-testable; see test/test_degeneracy_governor.cpp.
//
// Frames: trans_dirs / rot_dirs are world-frame unit eigen-directions (the
// optimization step dx is a left/world perturbation, so the gate's eigenvectors
// are world-frame). cap_t [m] / cap_r [rad] are per-scan caps; <= 0 disables
// that block. Empty dir lists -> identity (no-op), so a non-degenerate scan is
// untouched and the result equals T_new exactly.
#pragma once

#include <vector>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace dlio {

inline Eigen::Matrix4f governPose(
    const Eigen::Matrix4f& T_prev, const Eigen::Matrix4f& T_new,
    const std::vector<Eigen::Vector3d>& trans_dirs,
    const std::vector<Eigen::Vector3d>& rot_dirs,
    float cap_t, float cap_r) {
  Eigen::Matrix4f T = T_new;

  // Translation: clamp the world displacement along each held axis.
  if (cap_t > 0.f && !trans_dirs.empty()) {
    const Eigen::Vector3f p_prev = T_prev.block<3, 1>(0, 3);
    Eigen::Vector3f dp = T_new.block<3, 1>(0, 3) - p_prev;
    for (const auto& dd : trans_dirs) {
      const Eigen::Vector3f d = dd.cast<float>().normalized();
      const float c = d.dot(dp);
      const float cl = std::max(-cap_t, std::min(cap_t, c));
      dp += d * (cl - c);
    }
    T.block<3, 1>(0, 3) = p_prev + dp;
  }

  // Rotation: clamp the relative rotation (left/world) about each held axis.
  if (cap_r > 0.f && !rot_dirs.empty()) {
    const Eigen::Matrix3f R_prev = T_prev.block<3, 3>(0, 0);
    const Eigen::Matrix3f R_rel = T_new.block<3, 3>(0, 0) * R_prev.transpose();
    Eigen::AngleAxisf aa(R_rel);
    Eigen::Vector3f rv = aa.axis() * aa.angle();   // world-frame rotation vector
    for (const auto& dd : rot_dirs) {
      const Eigen::Vector3f d = dd.cast<float>().normalized();
      const float c = d.dot(rv);
      const float cl = std::max(-cap_r, std::min(cap_r, c));
      rv += d * (cl - c);
    }
    const float ang = rv.norm();
    const Eigen::Matrix3f R_rel_g = (ang > 1e-9f)
        ? Eigen::AngleAxisf(ang, rv / ang).toRotationMatrix()
        : Eigen::Matrix3f::Identity();
    T.block<3, 3>(0, 0) = R_rel_g * R_prev;
  }

  return T;
}

}  // namespace dlio
