// Unit tests for the degeneracy GOVERNOR fail-safe (dlio::governPose).
//
// The governor caps per-scan output motion along the gate's held world-frame
// eigen-directions, vs the previous pose, to a physical per-scan step. It is
// the fail-safe that converts the held-prior km-scale tunnel runaway into a
// bounded coast. These tests pin: (1) the bit-identical guarantees (caps off /
// no held dirs -> exactly T_new), (2) the clamp only acts ALONG a held axis,
// (3) under-cap motion is preserved, (4) rotation clamps about a held axis.

#include <gtest/gtest.h>
#include "dlio/degeneracy_governor.h"

#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>

using dlio::governPose;

namespace {

Eigen::Matrix4f translated(float x, float y, float z) {
  Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
  T(0, 3) = x; T(1, 3) = y; T(2, 3) = z;
  return T;
}

Eigen::Matrix4f yawed(float rad) {
  Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
  T.block<3, 3>(0, 0) = Eigen::AngleAxisf(rad, Eigen::Vector3f::UnitZ()).toRotationMatrix();
  return T;
}

const std::vector<Eigen::Vector3d> kNoDirs{};
const std::vector<Eigen::Vector3d> kXdir{Eigen::Vector3d::UnitX()};
const std::vector<Eigen::Vector3d> kYdir{Eigen::Vector3d::UnitY()};
const std::vector<Eigen::Vector3d> kZdir{Eigen::Vector3d::UnitZ()};

}  // namespace

// (1a) Caps == 0 -> exactly T_new even with held dirs (bit-identical when off).
TEST(DegeneracyGovernor, ZeroCapsIsNoOp) {
  const Eigen::Matrix4f prev = Eigen::Matrix4f::Identity();
  const Eigen::Matrix4f cur = translated(14.f, 0.f, 0.f);  // a runaway jump
  const Eigen::Matrix4f out = governPose(prev, cur, kXdir, kZdir, 0.f, 0.f);
  EXPECT_TRUE(out.isApprox(cur));
}

// (1b) No held dirs -> exactly T_new (a non-degenerate scan is untouched).
TEST(DegeneracyGovernor, NoHeldDirsIsNoOp) {
  const Eigen::Matrix4f prev = Eigen::Matrix4f::Identity();
  const Eigen::Matrix4f cur = translated(14.f, 2.f, -1.f);
  const Eigen::Matrix4f out = governPose(prev, cur, kNoDirs, kNoDirs, 0.30f, 0.05f);
  EXPECT_TRUE(out.isApprox(cur));
}

// (2) Translation runaway ALONG the held axis is clamped to the cap.
TEST(DegeneracyGovernor, ClampsTranslationAlongHeldAxis) {
  const Eigen::Matrix4f prev = Eigen::Matrix4f::Identity();
  const Eigen::Matrix4f cur = translated(14.f, 0.f, 0.f);  // 14 m forward jump
  const Eigen::Matrix4f out = governPose(prev, cur, kXdir, kNoDirs, 0.30f, 0.f);
  EXPECT_NEAR(out(0, 3), 0.30f, 1e-5f);   // clamped to the cap
  EXPECT_NEAR(out(1, 3), 0.0f, 1e-5f);
  EXPECT_NEAR(out(2, 3), 0.0f, 1e-5f);
}

// (3) Motion OFF the held axis is untouched (the held axis is y, motion is x).
TEST(DegeneracyGovernor, LeavesUnheldAxisAlone) {
  const Eigen::Matrix4f prev = Eigen::Matrix4f::Identity();
  const Eigen::Matrix4f cur = translated(14.f, 0.f, 0.f);  // x motion
  const Eigen::Matrix4f out = governPose(prev, cur, kYdir, kNoDirs, 0.30f, 0.f);
  EXPECT_NEAR(out(0, 3), 14.0f, 1e-4f);   // x preserved (not the held axis)
  EXPECT_NEAR(out(1, 3), 0.0f, 1e-5f);
}

// (4) Under-cap motion along the held axis is preserved (no spurious clamp).
TEST(DegeneracyGovernor, PreservesUnderCapMotion) {
  const Eigen::Matrix4f prev = Eigen::Matrix4f::Identity();
  const Eigen::Matrix4f cur = translated(0.10f, 0.f, 0.f);  // < 0.30 cap
  const Eigen::Matrix4f out = governPose(prev, cur, kXdir, kNoDirs, 0.30f, 0.f);
  EXPECT_TRUE(out.isApprox(cur));
}

// (5) Clamp is relative to the PREVIOUS pose, not the origin.
TEST(DegeneracyGovernor, ClampsRelativeToPrev) {
  const Eigen::Matrix4f prev = translated(100.f, 0.f, 0.f);   // already at 100 m
  const Eigen::Matrix4f cur = translated(114.f, 0.f, 0.f);    // +14 m this scan
  const Eigen::Matrix4f out = governPose(prev, cur, kXdir, kNoDirs, 0.30f, 0.f);
  EXPECT_NEAR(out(0, 3), 100.30f, 1e-3f);   // prev + cap, not 0.30
}

// (6) Yaw runaway about the held rotation axis is clamped to the cap.
TEST(DegeneracyGovernor, ClampsYawAboutHeldAxis) {
  const Eigen::Matrix4f prev = Eigen::Matrix4f::Identity();
  const Eigen::Matrix4f cur = yawed(0.5f);   // 0.5 rad yaw jump
  const Eigen::Matrix4f out = governPose(prev, cur, kNoDirs, kZdir, 0.f, 0.05f);
  Eigen::AngleAxisf aa(out.block<3, 3>(0, 0));
  EXPECT_NEAR(aa.angle(), 0.05f, 1e-4f);
  EXPECT_NEAR(std::abs(aa.axis().z()), 1.0f, 1e-4f);  // about z
}

// (7) A held trans axis does not corrupt rotation (and vice versa).
TEST(DegeneracyGovernor, BlocksAreIndependent) {
  const Eigen::Matrix4f prev = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f cur = yawed(0.5f);
  cur(0, 3) = 14.f;   // both a yaw jump and an x jump
  // Only the translation axis is held -> rotation passes through unchanged.
  const Eigen::Matrix4f out = governPose(prev, cur, kXdir, kNoDirs, 0.30f, 0.05f);
  EXPECT_NEAR(out(0, 3), 0.30f, 1e-5f);                       // x clamped
  const Eigen::Matrix3f R_out = out.block<3, 3>(0, 0);
  const Eigen::Matrix3f R_cur = cur.block<3, 3>(0, 0);
  EXPECT_TRUE(R_out.isApprox(R_cur, 1e-5f));                  // yaw intact
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
