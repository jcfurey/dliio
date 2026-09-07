#include <gtest/gtest.h>
#include <limits>
#include "dlio/auxiliary_gravity.h"

TEST(AuxiliaryGravity, RejectsExtrapolationAndGapsAndBadSamples) {
  dlio::GravityBuffer buffer;
  EXPECT_FALSE(buffer.push(0, Eigen::Vector3d::Zero()));
  ASSERT_TRUE(buffer.push(1000000000, Eigen::Vector3d::UnitZ()));
  EXPECT_FALSE(buffer.push(1000000000, Eigen::Vector3d::UnitZ()));
  ASSERT_TRUE(buffer.push(1010000000, Eigen::Vector3d::UnitZ()));
  Eigen::Vector3d direction;
  int64_t lower, upper;
  EXPECT_FALSE(buffer.at(999000000, 20000000, direction, lower, upper));
  EXPECT_FALSE(buffer.at(1011000000, 20000000, direction, lower, upper));
  EXPECT_FALSE(buffer.at(1005000000, 5000000, direction, lower, upper));
  ASSERT_TRUE(buffer.at(1005000000, 20000000, direction, lower, upper));
  EXPECT_EQ(lower, 1000000000); EXPECT_EQ(upper, 1010000000);
  EXPECT_TRUE(direction.isApprox(Eigen::Vector3d::UnitZ()));
}

TEST(AuxiliaryGravity, ReducesTiltAndRespectsRateBound) {
  Eigen::Quaterniond pose(Eigen::AngleAxisd(.3, Eigen::Vector3d::UnitX()));
  auto result = dlio::correctGravity(pose, Eigen::Vector3d::UnitZ(), .3, 1., .05, .6);
  ASSERT_TRUE(result.accepted);
  EXPECT_NEAR(result.disagreement, .3, 1e-12);
  EXPECT_NEAR(result.applied_angle, .015, 1e-12);
  EXPECT_NEAR(Eigen::AngleAxisd(result.orientation).angle(), .285, 1e-12);
  EXPECT_NEAR(result.orientation.norm(), 1., 1e-12);
}

TEST(AuxiliaryGravity, IsEquivariantToUnmeasuredWorldHeading) {
  Eigen::Quaterniond pose(Eigen::AngleAxisd(.2, Eigen::Vector3d(1., 2., .1).normalized()));
  Eigen::Quaterniond yaw(Eigen::AngleAxisd(1.7, Eigen::Vector3d::UnitZ()));
  auto a = dlio::correctGravity(pose, Eigen::Vector3d::UnitZ(), .3, 1., 1., .6);
  auto b = dlio::correctGravity(yaw*pose, Eigen::Vector3d::UnitZ(), .3, 1., 1., .6);
  EXPECT_NEAR((yaw*a.orientation).angularDistance(b.orientation), 0., 1e-12);
  auto upright = dlio::correctGravity(yaw, Eigen::Vector3d::UnitZ(), .3, 1., 1., .6);
  EXPECT_EQ(upright.applied_angle, 0.);
  EXPECT_EQ(yaw.angularDistance(upright.orientation), 0.);
}

TEST(AuxiliaryGravity, RejectsLargeDisagreementAndKeepsMonitorUnchanged) {
  Eigen::Quaterniond pose(Eigen::AngleAxisd(.4, Eigen::Vector3d::UnitX()));
  EXPECT_FALSE(dlio::correctGravity(pose, Eigen::Vector3d::UnitZ(), .3, 1., 1., .2).accepted);
  auto monitor = dlio::correctGravity(pose, Eigen::Vector3d::UnitZ(), .3, 0., 1., .6);
  EXPECT_TRUE(monitor.accepted);
  EXPECT_TRUE((monitor.orientation.coeffs().array() == pose.coeffs().array()).all());
  EXPECT_FALSE(dlio::correctGravity(pose, Eigen::Vector3d::UnitZ(), 0., 1., 1., .6).accepted);
  EXPECT_FALSE(dlio::correctGravity(pose, Eigen::Vector3d::UnitZ(), .3,
      std::numeric_limits<double>::quiet_NaN(), 1., .6).accepted);
}
