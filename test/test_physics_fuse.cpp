// Unit tests for the PHYSICS FUSE (dlio::fusePose, doc/RUNTIME_GUARDS.md).
//
// The fuse clamps the TOTAL per-scan output step -- any direction, prior
// divergence included -- to a physical bound, making a takeoff impossible at
// the output level by construction. These pin: (1) the bit-identical
// guarantees (caps off / under-cap motion -> exactly T_new, not tripped),
// (2) the translation clamp preserves direction and lands exactly on the cap,
// (3) the rotation clamp caps the relative angle about its own axis,
// (4) non-finite steps hold the previous pose instead of propagating NaN.

#include <gtest/gtest.h>
#include "dlio/physics_fuse.h"

#include <cmath>
#include <limits>
#include <Eigen/Core>
#include <Eigen/Geometry>

using dlio::fusePose;

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

}  // namespace

TEST(PhysicsFuse, CapsOffIsBitIdentical) {
  const Eigen::Matrix4f T_prev = Eigen::Matrix4f::Identity();
  const Eigen::Matrix4f T_new = translated(1e6f, -2e6f, 3e6f);  // absurd step
  bool tripped = true;
  const Eigen::Matrix4f out = fusePose(T_prev, T_new, 0.f, 0.f, &tripped);
  EXPECT_TRUE(out.isApprox(T_new, 0.f));   // exactly untouched
  EXPECT_FALSE(tripped);
}

TEST(PhysicsFuse, UnderCapMotionUntouched) {
  const Eigen::Matrix4f T_prev = translated(10.f, 5.f, -2.f);
  const Eigen::Matrix4f T_new = translated(10.3f, 5.1f, -2.05f);  // |step| ~ 0.32
  bool tripped = true;
  const Eigen::Matrix4f out = fusePose(T_prev, T_new, 1.0f, 0.5f, &tripped);
  EXPECT_TRUE(out.isApprox(T_new, 0.f));
  EXPECT_FALSE(tripped);
}

TEST(PhysicsFuse, TranslationClampPreservesDirection) {
  const Eigen::Matrix4f T_prev = Eigen::Matrix4f::Identity();
  const Eigen::Vector3f dir = Eigen::Vector3f(3.f, -4.f, 12.f).normalized();
  const Eigen::Matrix4f T_new = translated(50.f * dir.x(), 50.f * dir.y(), 50.f * dir.z());
  bool tripped = false;
  const Eigen::Matrix4f out = fusePose(T_prev, T_new, 1.0f, 0.f, &tripped);
  EXPECT_TRUE(tripped);
  const Eigen::Vector3f step = out.block<3, 1>(0, 3);
  EXPECT_NEAR(step.norm(), 1.0f, 1e-5f);                       // exactly the cap
  EXPECT_NEAR(step.normalized().dot(dir), 1.0f, 1e-5f);        // same direction
}

TEST(PhysicsFuse, TakeoffMagnitudeIsBounded) {
  // The 2026-07-09 leg-1 pose reached 3.3e7 m; the fuse must reduce any such
  // step to the cap regardless of magnitude.
  const Eigen::Matrix4f T_prev = Eigen::Matrix4f::Identity();
  const Eigen::Matrix4f T_new = translated(3.3e7f, -5.8e6f, -2.9e7f);
  bool tripped = false;
  const Eigen::Matrix4f out = fusePose(T_prev, T_new, 2.0f, 0.f, &tripped);
  EXPECT_TRUE(tripped);
  const float snorm = out.block<3, 1>(0, 3).norm();
  EXPECT_NEAR(snorm, 2.0f, 1e-3f);
}

TEST(PhysicsFuse, RotationClampCapsAngleAboutOwnAxis) {
  const Eigen::Matrix4f T_prev = yawed(0.2f);
  const Eigen::Matrix4f T_new = yawed(0.2f + 1.5f);  // 1.5 rad relative yaw
  bool tripped = false;
  const Eigen::Matrix4f out = fusePose(T_prev, T_new, 0.f, 0.1f, &tripped);
  EXPECT_TRUE(tripped);
  const Eigen::Matrix3f R_rel =
      out.block<3, 3>(0, 0) * T_prev.block<3, 3>(0, 0).transpose();
  Eigen::AngleAxisf aa(R_rel);
  EXPECT_NEAR(aa.angle(), 0.1f, 1e-5f);                        // exactly the cap
  EXPECT_NEAR(std::abs(aa.axis().z()), 1.0f, 1e-5f);           // still about Z
}

TEST(PhysicsFuse, RotationUnderCapUntouched) {
  const Eigen::Matrix4f T_prev = yawed(0.2f);
  const Eigen::Matrix4f T_new = yawed(0.25f);
  bool tripped = true;
  const Eigen::Matrix4f out = fusePose(T_prev, T_new, 0.f, 0.1f, &tripped);
  EXPECT_TRUE(out.isApprox(T_new, 1e-6f));
  EXPECT_FALSE(tripped);
}

TEST(PhysicsFuse, NonFiniteStepHoldsPreviousPosition) {
  const Eigen::Matrix4f T_prev = translated(1.f, 2.f, 3.f);
  Eigen::Matrix4f T_new = T_prev;
  T_new(0, 3) = std::numeric_limits<float>::quiet_NaN();
  bool tripped = false;
  const Eigen::Matrix4f out = fusePose(T_prev, T_new, 1.0f, 0.f, &tripped);
  EXPECT_TRUE(tripped);
  const Eigen::Vector3f p = out.block<3, 1>(0, 3);
  EXPECT_TRUE(p.allFinite());
  const Eigen::Vector3f p_prev = T_prev.block<3, 1>(0, 3);  // hoisted: macro-comma hazard
  EXPECT_TRUE(p.isApprox(p_prev, 1e-6f));
}

TEST(PhysicsFuse, BothBlocksClampIndependently) {
  const Eigen::Matrix4f T_prev = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f T_new = yawed(2.0f);
  T_new.block<3, 1>(0, 3) = Eigen::Vector3f(30.f, 0.f, 0.f);
  bool tripped = false;
  const Eigen::Matrix4f out = fusePose(T_prev, T_new, 0.5f, 0.05f, &tripped);
  EXPECT_TRUE(tripped);
  const float snorm = out.block<3, 1>(0, 3).norm();  // hoisted: comma in <3,1> breaks the macro
  EXPECT_NEAR(snorm, 0.5f, 1e-5f);
  Eigen::AngleAxisf aa(Eigen::Matrix3f(out.block<3, 3>(0, 0)));
  EXPECT_NEAR(aa.angle(), 0.05f, 1e-5f);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
