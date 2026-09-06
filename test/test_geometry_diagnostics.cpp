#include <gtest/gtest.h>
#include <Eigen/Geometry>
#include <cmath>
#include <limits>
#include "nano_gicp/geometry_diagnostics.h"

namespace {
using nano_gicp::Matrix6d;
using nano_gicp::Vector6d;

// Physical point-to-plane Jacobians in a chosen world frame, independently
// assembled from points and normals. No centered-Hessian transform here.
Matrix6d scene(const Eigen::Matrix3d& rotation, const Eigen::Vector3d& origin,
               double spatial_scale = 1.0) {
  Matrix6d h = Matrix6d::Zero();
  for (int i = 1; i < 40; ++i) {
    const Eigen::Vector3d local(4.0 * std::sin(i * 0.7), std::cos(i * 0.3), 2.0 * std::sin(i * 1.3));
    const Eigen::Vector3d normal = rotation * Eigen::Vector3d(std::sin(i * 0.4), std::cos(i * 0.9), 0.7).normalized();
    const Eigen::Vector3d point = rotation * (spatial_scale * local) + origin;
    Vector6d j;
    j.head<3>() = point.cross(normal);
    j.tail<3>() = normal;
    h.noalias() += j * j.transpose();
  }
  return h;
}
}  // namespace

TEST(GeometryDiagnostics, FindsCoupledYawTranslationNullModeDespiteHealthyBlocks) {
  Matrix6d h = Matrix6d::Identity();
  h(2, 3) = h(3, 2) = -1.0;  // residual dx - dyaw; other coordinates independent
  const auto d = nano_gicp::analyzeGeometry(h, Eigen::Vector3d::Zero(), 1.0);
  ASSERT_TRUE(d.valid);
  EXPECT_DOUBLE_EQ(d.rotation_ratio, 1.0);
  EXPECT_DOUBLE_EQ(d.translation_ratio, 1.0);
  EXPECT_NEAR(d.eigenvalues(0), 0.0, 1e-14);
  EXPECT_NEAR(d.eigenvalues(5), 2.0, 1e-14);
  EXPECT_NEAR(std::abs(d.weakest(2)), std::sqrt(0.5), 1e-14);
  EXPECT_NEAR(d.weakest(2), d.weakest(3), 1e-14);
  EXPECT_NEAR(d.schur_ratio, 0.0, 1e-14);
  EXPECT_NEAR(d.schur_retained, 0.0, 1e-14);
}

TEST(GeometryDiagnostics, CenteringRemovesWorldOriginLeverArmWithoutChangingInput) {
  const auto base = nano_gicp::analyzeGeometry(scene(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero()),
      Eigen::Vector3d::Zero(), 5.0);
  ASSERT_TRUE(base.valid);
  const Eigen::Vector3d origin(500.0, -1000.0, 42.0);
  const Matrix6d h = scene(Eigen::Matrix3d::Identity(), origin), saved = h;
  const auto shifted = nano_gicp::analyzeGeometry(h, origin, 5.0);
  ASSERT_TRUE(shifted.valid);
  EXPECT_TRUE((h.array() == saved.array()).all());
  EXPECT_TRUE(base.eigenvalues.isApprox(shifted.eigenvalues, 1e-8));
  EXPECT_TRUE(base.hessian.isApprox(shifted.hessian, 1e-8));
  EXPECT_NEAR(std::abs(base.weakest.dot(shifted.weakest)), 1.0, 1e-8);
  EXPECT_NEAR(base.schur_retained, shifted.schur_retained, 1e-8);
  const auto uncentered = nano_gicp::analyzeGeometry(h, Eigen::Vector3d::Zero(), 5.0);
  ASSERT_TRUE(uncentered.valid);
  EXPECT_LT(uncentered.eigenvalues(0) / uncentered.eigenvalues(5),
            0.01 * base.eigenvalues(0) / base.eigenvalues(5));
}

TEST(GeometryDiagnostics, WorldRotationRotatesBothPartsOfTheWeakMode) {
  const Eigen::Matrix3d R = Eigen::AngleAxisd(0.7, Eigen::Vector3d(1.0, 2.0, -3.0).normalized()).toRotationMatrix();
  const auto a = nano_gicp::analyzeGeometry(scene(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero()),
      Eigen::Vector3d::Zero(), 5.0);
  const auto b = nano_gicp::analyzeGeometry(scene(R, Eigen::Vector3d(30.0, -20.0, 7.0)),
      Eigen::Vector3d(30.0, -20.0, 7.0), 5.0);
  ASSERT_TRUE(a.valid && b.valid);
  Vector6d expected;
  expected.head<3>() = R * a.weakest.head<3>();
  expected.tail<3>() = R * a.weakest.tail<3>();
  EXPECT_TRUE(a.eigenvalues.isApprox(b.eigenvalues, 1e-10));
  EXPECT_NEAR(std::abs(expected.dot(b.weakest)), 1.0, 1e-10);
  EXPECT_NEAR(a.schur_retained, b.schur_retained, 1e-10);
}

TEST(GeometryDiagnostics, ConsistentLengthUnitsAndScaleSensitivity) {
  const auto h = scene(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
  const auto meters = nano_gicp::analyzeGeometry(h, Eigen::Vector3d::Zero(), 5.0);
  const auto centimeters = nano_gicp::analyzeGeometry(
      scene(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), 100.0), Eigen::Vector3d::Zero(), 500.0);
  ASSERT_TRUE(meters.valid && centimeters.valid);
  EXPECT_TRUE(meters.eigenvalues.isApprox(centimeters.eigenvalues, 1e-12));
  EXPECT_TRUE(meters.hessian.isApprox(centimeters.hessian, 1e-12));
  const auto half = nano_gicp::analyzeGeometry(h, Eigen::Vector3d::Zero(), 2.5);
  const auto twice = nano_gicp::analyzeGeometry(h, Eigen::Vector3d::Zero(), 10.0);
  EXPECT_NEAR(meters.half_length_ratio, half.eigenvalues(0) / half.eigenvalues(5), 1e-14);
  EXPECT_NEAR(meters.double_length_ratio, twice.eigenvalues(0) / twice.eigenvalues(5), 1e-14);
}

TEST(GeometryDiagnostics, SingularRotationBlockUsesRankAwareSchurComplement) {
  Matrix6d h = Matrix6d::Identity();
  h(0, 0) = 0.0;
  h(2, 3) = h(3, 2) = -0.5;
  const auto d = nano_gicp::analyzeGeometry(h, Eigen::Vector3d::Zero(), 1.0);
  ASSERT_TRUE(d.valid);
  EXPECT_NEAR(d.rotation_ratio, 0.0, 1e-14);
  EXPECT_NEAR(d.schur_ratio, 0.75, 1e-14);
  EXPECT_NEAR(d.schur_retained, 0.75, 1e-14);
}

TEST(GeometryDiagnostics, InvalidOrEmptyGeometryCannotReportConfidence) {
  const auto check = [](const Matrix6d& h, const Eigen::Vector3d& center, double length) {
    const auto d = nano_gicp::analyzeGeometry(h, center, length);
    EXPECT_FALSE(d.valid);
    EXPECT_LT(d.eigenvalues(0), 0.0);
    EXPECT_LT(d.schur_ratio, 0.0);
  };
  check(Matrix6d::Zero(), Eigen::Vector3d::Zero(), 5.0);
  check(-Matrix6d::Identity(), Eigen::Vector3d::Zero(), 5.0);
  check(Matrix6d::Identity(), Eigen::Vector3d::Zero(), 0.0);
  check(Matrix6d::Identity(), Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN()), 5.0);
  Matrix6d h = Matrix6d::Identity();
  h(0, 1) = h(1, 0) = 2.0;
  check(h, Eigen::Vector3d::Zero(), 5.0);
}
