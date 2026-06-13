// Unit tests for NanoGICP: alignment on well-conditioned geometry, the
// block-wise degeneracy gate on a planar (tunnel-like) target, and the
// small-cloud covariance guard.

#include <gtest/gtest.h>

#include <cmath>

#include "dlio/dlio.h"
#include "nano_gicp/nano_gicp.h"

namespace {

using Cloud = pcl::PointCloud<dlio::Point>;

dlio::Point makePoint(float x, float y, float z) {
  dlio::Point p;
  p.x = x; p.y = y; p.z = z;
  p.intensity = 0.f;
  p.reflectivity = 0.f;
  return p;
}

// Grid on the z=0 plane, `half` meters in each direction, `step` spacing.
Cloud::Ptr makePlane(float half, float step) {
  auto cloud = std::make_shared<Cloud>();
  for (float x = -half; x <= half; x += step) {
    for (float y = -half; y <= half; y += step) {
      cloud->push_back(makePoint(x, y, 0.f));
    }
  }
  return cloud;
}

// Three orthogonal planes meeting at the origin (a room corner): fully
// constrains all six degrees of freedom.
Cloud::Ptr makeCorner(float extent, float step) {
  auto cloud = std::make_shared<Cloud>();
  for (float a = step; a <= extent; a += step) {
    for (float b = step; b <= extent; b += step) {
      cloud->push_back(makePoint(a, b, 0.f));  // floor
      cloud->push_back(makePoint(0.f, a, b));  // wall x=0
      cloud->push_back(makePoint(a, 0.f, b));  // wall y=0
    }
  }
  return cloud;
}

Cloud::Ptr transformCloud(const Cloud::ConstPtr& in, const Eigen::Matrix4f& T) {
  auto out = std::make_shared<Cloud>();
  pcl::transformPointCloud(*in, *out, T);
  return out;
}

nano_gicp::NanoGICP<dlio::Point, dlio::Point> makeGICP() {
  nano_gicp::NanoGICP<dlio::Point, dlio::Point> gicp;
  gicp.setMaxCorrespondenceDistance(1.0f);
  gicp.setMaximumIterations(32);
  gicp.setCorrespondenceRandomness(16);
  gicp.setRegularizationMethod(nano_gicp::RegularizationMethod::PLANE);
  return gicp;
}

}  // namespace

TEST(NanoGICP, RecoversSmallTransformOnCornerGeometry) {
  auto target = makeCorner(1.0f, 0.05f);

  Eigen::Matrix4f T_true = Eigen::Matrix4f::Identity();
  T_true.block<3, 3>(0, 0) =
      Eigen::AngleAxisf(0.02f, Eigen::Vector3f::UnitZ()).toRotationMatrix();
  T_true.block<3, 1>(0, 3) = Eigen::Vector3f(0.03f, -0.02f, 0.04f);

  // source = T_true^-1 * target, so the recovered transform should be T_true
  auto source = transformCloud(target, T_true.inverse());

  auto gicp = makeGICP();
  gicp.setInputTarget(target);
  gicp.setInputSource(source);

  Cloud aligned;
  gicp.align(aligned);

  ASSERT_TRUE(gicp.hasConverged());
  EXPECT_EQ(gicp.lastDegenerateDirections(), 0);

  Eigen::Matrix4f T = gicp.getFinalTransformation();
  EXPECT_LT((T.block<3, 1>(0, 3) - T_true.block<3, 1>(0, 3)).norm(), 0.02f);
  Eigen::AngleAxisf rot_err(
      Eigen::Matrix3f(T.block<3, 3>(0, 0).transpose() * T_true.block<3, 3>(0, 0)));
  EXPECT_LT(std::abs(rot_err.angle()), 0.02f);
}

TEST(NanoGICP, GatesUnobservableDirectionsOnPlanarGeometry) {
  // A single plane is the canonical degenerate target: in-plane translation
  // (2 dof) and rotation about the normal (1 dof) are unobservable -- the
  // tunnel scenario in miniature.
  auto target = makePlane(2.0f, 0.05f);

  Eigen::Matrix4f T_shift = Eigen::Matrix4f::Identity();
  T_shift(0, 3) = 0.4f;  // large in-plane shift the registration cannot observe
  auto source = transformCloud(target, T_shift);

  auto gicp = makeGICP();
  gicp.setInputTarget(target);
  gicp.setInputSource(source);

  Cloud aligned;
  gicp.align(aligned);

  // The gate must flag the unobservable directions...
  EXPECT_GE(gicp.lastDegenerateDirections(), 2);
  // ...and hold the prior (identity guess) there rather than inventing motion.
  Eigen::Matrix4f T = gicp.getFinalTransformation();
  float translation_norm = T.block<3, 1>(0, 3).norm();
  EXPECT_LT(translation_norm, 0.05f);
}

TEST(NanoGICP, DegeneracyGateCanBeDisabled) {
  auto target = makePlane(1.0f, 0.05f);
  auto gicp = makeGICP();
  gicp.setDegeneracyThreshRatio(0.f);
  gicp.setInputTarget(target);
  gicp.setInputSource(target);

  Cloud aligned;
  gicp.align(aligned);
  EXPECT_EQ(gicp.lastDegenerateDirections(), 0);
}

TEST(NanoGICP, TinyCloudDoesNotCrashCovarianceEstimation) {
  // Fewer points than kCorrespondences: previously read uninitialized
  // kd-tree result slots (out-of-bounds indices).
  auto tiny = std::make_shared<Cloud>();
  tiny->push_back(makePoint(0.f, 0.f, 0.f));
  tiny->push_back(makePoint(0.1f, 0.f, 0.f));
  tiny->push_back(makePoint(0.f, 0.1f, 0.f));

  auto gicp = makeGICP();
  EXPECT_NO_THROW({
    gicp.setInputTarget(tiny);
    gicp.setInputSource(tiny);
  });
  EXPECT_EQ(gicp.getSourceCovariances().size(), tiny->size());
}

// Covariance regularization (synthetic plane): PLANE forces the scale-free
// (1,1,1e-3) disc; MIN_EIG only clamps the smallest eigenvalue, leaving the two
// in-plane eigenvalues at the data variance. Locks the "honest regularization"
// behavior (these were silently identical before the PLANE/MIN_EIG split).
TEST(NanoGICP, PlaneRegularizationProducesUnitDisc) {
  auto cloud = makePlane(2.0f, 0.04f);   // z=0 plane
  nano_gicp::NanoGICP<dlio::Point, dlio::Point> gicp;
  gicp.setCorrespondenceRandomness(16);
  gicp.setRegularizationMethod(nano_gicp::RegularizationMethod::PLANE);
  gicp.setInputSource(cloud);

  const auto& covs = gicp.getSourceCovariances();
  ASSERT_EQ(covs.size(), cloud->size());
  int checked = 0;
  for (size_t i = 0; i < covs.size(); i += 37) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(covs[i].block<3,3>(0,0));
    Eigen::Vector3f ev = es.eigenvalues();  // ascending
    EXPECT_NEAR(ev(2), 1.0f, 1e-4) << "i=" << i;   // two unit eigenvalues
    EXPECT_NEAR(ev(1), 1.0f, 1e-4) << "i=" << i;
    EXPECT_NEAR(ev(0), 1e-3f, 1e-4) << "i=" << i;  // flattened normal
    ++checked;
  }
  EXPECT_GT(checked, 5);
}

TEST(NanoGICP, MinEigRegularizationKeepsInPlaneVariance) {
  auto cloud = makePlane(2.0f, 0.04f);
  nano_gicp::NanoGICP<dlio::Point, dlio::Point> gicp;
  gicp.setCorrespondenceRandomness(16);
  gicp.setRegularizationMethod(nano_gicp::RegularizationMethod::MIN_EIG);
  gicp.setInputSource(cloud);

  const auto& covs = gicp.getSourceCovariances();
  ASSERT_EQ(covs.size(), cloud->size());
  bool any_non_unit_inplane = false;
  for (size_t i = 0; i < covs.size(); i += 37) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(covs[i].block<3,3>(0,0));
    Eigen::Vector3f ev = es.eigenvalues();
    EXPECT_GE(ev(0), 1e-3f - 1e-5f);              // smallest clamped up to >= 1e-3
    if (std::abs(ev(2) - 1.0f) > 1e-2f) { any_non_unit_inplane = true; }
  }
  // unlike PLANE, the in-plane eigenvalues are the data variance, not 1.0
  EXPECT_TRUE(any_non_unit_inplane);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
