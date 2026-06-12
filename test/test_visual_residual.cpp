// Unit tests for the direct visual (camera) photometric residual in NanoGICP.
//
// These run with NO real camera, image, or extrinsic: synthetic pinhole images
// + known transforms prove the math is correct before any bag replay.
//   1. Residual is ~0 at the truth pose (projection + reference consistency).
//   2. The analytic 6-DOF Jacobian matches a finite-difference of the cost
//      (the GN gradient satisfies dcost/dxi = 2*b exactly) -- guards the
//      sign/convention, the single highest-risk bug.
//   3. A descent direction: -H^-1 b reduces the cost (sign sanity).
//   4. The visual term adds Hessian stiffness to the targeted translation axis
//      (the mechanism that re-opens the degeneracy gate in a tunnel).
//   5. Disabled => no contribution (off-by-default regression guard).

#include <gtest/gtest.h>

#include <cmath>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include "dlio/dlio.h"
#include "nano_gicp/nano_gicp.h"

namespace {

using Cloud = pcl::PointCloud<dlio::Point>;

// Subclass to reach the protected accumulator and reset state per call.
class TestableGICP : public nano_gicp::NanoGICP<dlio::Point, dlio::Point> {
 public:
  // Returns (H, b, cost) for the visual term alone at pose `trans`.
  double visualSystem(const Eigen::Isometry3f& trans,
                      Eigen::Matrix<double, 6, 6>& H,
                      Eigen::Matrix<double, 6, 1>& b) {
    H.setZero();
    b.setZero();
    double cost = 0.0;
    this->accumulateVisualResidual(trans, &H, &b, &cost);
    return cost;
  }
};

dlio::Point makePoint(float x, float y, float z) {
  dlio::Point p;
  p.x = x; p.y = y; p.z = z;
  p.intensity = 0.f;
  p.reflectivity = 0.f;
  return p;
}

// A linear-ramp image I(u,v) = au + bv + c. Bilinear sampling and central
// differences are then EXACT, so the numeric Jacobian check is limited only by
// the perspective non-linearity (O(eps^2) truncation), not interpolation error.
cv::Mat makeRampImage(int w, int h, float a, float b, float c) {
  cv::Mat img(h, w, CV_32FC1);
  for (int v = 0; v < h; ++v) {
    float* row = img.ptr<float>(v);
    for (int u = 0; u < w; ++u) {
      row[u] = a * static_cast<float>(u) + b * static_cast<float>(v) + c;
    }
  }
  return img;
}

// Camera at (0,0,height) looking straight down (-z_world), right-handed:
// cam_x = world_x, cam_y = -world_y, cam_z = -world_z. Returns world->camera.
Eigen::Isometry3f lookingDownCamera(float height) {
  Eigen::Matrix3f R_wc;  // columns = camera axes expressed in world
  R_wc.col(0) = Eigen::Vector3f(1.f, 0.f, 0.f);
  R_wc.col(1) = Eigen::Vector3f(0.f, -1.f, 0.f);
  R_wc.col(2) = Eigen::Vector3f(0.f, 0.f, -1.f);
  Eigen::Isometry3f T_wc = Eigen::Isometry3f::Identity();
  T_wc.linear() = R_wc;
  T_wc.translation() = Eigen::Vector3f(0.f, 0.f, height);
  return T_wc.inverse();  // world -> camera
}

// A patch of points spread around the origin on/near the z=0 plane.
Cloud::Ptr makeWorldPoints() {
  auto cloud = std::make_shared<Cloud>();
  for (float x = -1.0f; x <= 1.0f + 1e-3f; x += 0.25f) {
    for (float y = -1.0f; y <= 1.0f + 1e-3f; y += 0.25f) {
      cloud->push_back(makePoint(x, y, 0.f));
    }
  }
  return cloud;
}

// Apply a left perturbation exp(xi^) to a pose, matching computeTransformation's
// prerotate-then-pretranslate convention. `k` selects the DOF (0-2 rotation,
// 3-5 translation); `eps` its magnitude.
Eigen::Isometry3f perturbLeft(const Eigen::Isometry3f& trans, int k, float eps) {
  Eigen::Isometry3f out = trans;
  if (k < 3) {
    Eigen::Vector3f axis = Eigen::Vector3f::Zero();
    axis(k) = 1.f;
    out.prerotate(Eigen::AngleAxisf(eps, axis));
  } else {
    Eigen::Vector3f t = Eigen::Vector3f::Zero();
    t(k - 3) = eps;
    out.pretranslate(t);
  }
  return out;
}

constexpr float kFx = 1094.2f, kFy = 1092.2f, kCx = 320.f, kCy = 240.f;
constexpr int kW = 640, kH = 480;

}  // namespace

TEST(VisualResidual, ResidualIsZeroAtTruth) {
  // Identical images + identical current/previous camera + identity correction
  // => the reference and the warp sample the same pixel => r == 0 for all.
  TestableGICP gicp;
  gicp.setVisualEnabled(true);
  gicp.setVisualWeight(1.0f);
  gicp.setVisualHuberDelta(0.f);
  gicp.setVisualIntrinsics(kFx, kFy, kCx, kCy);

  cv::Mat img = makeRampImage(kW, kH, 0.01f, 0.007f, 0.2f);
  Eigen::Isometry3f T_cw = lookingDownCamera(5.f);
  gicp.setVisualCurrentFrame(img, T_cw);
  gicp.setVisualPreviousFrame(img, T_cw);
  gicp.setInputSource(makeWorldPoints());

  Eigen::Matrix<double, 6, 6> H;
  Eigen::Matrix<double, 6, 1> b;
  double cost = gicp.visualSystem(Eigen::Isometry3f::Identity(), H, b);

  EXPECT_GT(gicp.lastVisualCount(), 10);
  EXPECT_NEAR(cost, 0.0, 1e-9);
  // b is Sigma w*J*r; with r==0 at truth it vanishes up to float roundoff in
  // the two (current vs previous) projection paths -- well below any real
  // gradient (O(0.1-1) in the other tests).
  EXPECT_LT(b.norm(), 1e-4);
  EXPECT_NEAR(gicp.lastVisualRms(), 0.f, 1e-5f);
}

TEST(VisualResidual, AnalyticJacobianMatchesFiniteDifference) {
  // Current and previous cameras differ by a small motion so the residual (and
  // hence the gradient) is non-trivial at the identity correction.
  TestableGICP gicp;
  gicp.setVisualEnabled(true);
  gicp.setVisualWeight(1.0f);
  gicp.setVisualHuberDelta(0.f);  // constant weight => dcost/dxi = 2b exactly
  gicp.setVisualIntrinsics(kFx, kFy, kCx, kCy);

  cv::Mat img = makeRampImage(kW, kH, 0.013f, 0.009f, 0.15f);
  Eigen::Isometry3f T_cur = lookingDownCamera(5.f);
  Eigen::Isometry3f T_prev = lookingDownCamera(5.f);
  // Nudge the previous camera (translation + small yaw) in its own frame.
  T_prev.pretranslate(Eigen::Vector3f(0.05f, -0.03f, 0.02f));
  T_prev.prerotate(Eigen::AngleAxisf(0.01f, Eigen::Vector3f::UnitZ()));
  gicp.setVisualCurrentFrame(img, T_cur);
  gicp.setVisualPreviousFrame(img, T_prev);
  gicp.setInputSource(makeWorldPoints());

  Eigen::Matrix<double, 6, 6> H;
  Eigen::Matrix<double, 6, 1> b;
  gicp.visualSystem(Eigen::Isometry3f::Identity(), H, b);
  const int count0 = gicp.lastVisualCount();
  ASSERT_GT(count0, 10);

  const float eps = 1e-3f;
  Eigen::Matrix<double, 6, 1> num_grad;
  Eigen::Matrix<double, 6, 6> Hd;
  Eigen::Matrix<double, 6, 1> bd;
  for (int k = 0; k < 6; ++k) {
    double cost_plus =
        gicp.visualSystem(perturbLeft(Eigen::Isometry3f::Identity(), k, eps), Hd, bd);
    ASSERT_EQ(gicp.lastVisualCount(), count0) << "point set changed at +eps, dof " << k;
    double cost_minus =
        gicp.visualSystem(perturbLeft(Eigen::Isometry3f::Identity(), k, -eps), Hd, bd);
    ASSERT_EQ(gicp.lastVisualCount(), count0) << "point set changed at -eps, dof " << k;
    num_grad(k) = (cost_plus - cost_minus) / (2.0 * eps);
  }

  // dcost/dxi = 2 b  (Gauss-Newton gradient). Compare vectors.
  Eigen::Matrix<double, 6, 1> analytic = 2.0 * b;
  EXPECT_LT((num_grad - analytic).norm(), 0.02 * analytic.norm() + 1e-6)
      << "num=" << num_grad.transpose() << "\nana=" << analytic.transpose();
}

TEST(VisualResidual, GaussNewtonStepDescendsCost) {
  // -H^-1 b must reduce the cost; a flipped Jacobian sign would ascend.
  TestableGICP gicp;
  gicp.setVisualEnabled(true);
  gicp.setVisualWeight(1.0f);
  gicp.setVisualHuberDelta(0.f);
  gicp.setVisualIntrinsics(kFx, kFy, kCx, kCy);

  cv::Mat img = makeRampImage(kW, kH, 0.013f, 0.009f, 0.15f);
  Eigen::Isometry3f T_cur = lookingDownCamera(5.f);
  Eigen::Isometry3f T_prev = lookingDownCamera(5.f);
  T_prev.pretranslate(Eigen::Vector3f(0.08f, -0.05f, 0.0f));
  gicp.setVisualCurrentFrame(img, T_cur);
  gicp.setVisualPreviousFrame(img, T_prev);
  gicp.setInputSource(makeWorldPoints());

  Eigen::Matrix<double, 6, 6> H;
  Eigen::Matrix<double, 6, 1> b;
  double cost0 = gicp.visualSystem(Eigen::Isometry3f::Identity(), H, b);

  H.diagonal().array() += 1e-6;  // mild damping for the solve
  Eigen::Matrix<double, 6, 1> dx = H.ldlt().solve(-b);

  Eigen::Isometry3f stepped = Eigen::Isometry3f::Identity();
  const Eigen::Vector3f rot = dx.head<3>().cast<float>();
  if (rot.norm() > 1e-12f) stepped.prerotate(Eigen::AngleAxisf(rot.norm(), rot / rot.norm()));
  stepped.pretranslate(dx.tail<3>().cast<float>());

  Eigen::Matrix<double, 6, 6> H1;
  Eigen::Matrix<double, 6, 1> b1;
  double cost1 = gicp.visualSystem(stepped, H1, b1);
  EXPECT_LT(cost1, cost0);
}

TEST(VisualResidual, AddsStiffnessToTargetedTranslationAxis) {
  // Camera looking down with an image gradient along u only => the residual is
  // sensitive primarily to world-x translation. The translation Hessian block
  // H[3:6,3:6] must put its dominant mass on the x DOF -- exactly the in-plane
  // direction a featureless LiDAR plane leaves unobservable.
  TestableGICP gicp;
  gicp.setVisualEnabled(true);
  gicp.setVisualWeight(1.0f);
  gicp.setVisualHuberDelta(0.f);
  gicp.setVisualIntrinsics(kFx, kFy, kCx, kCy);

  cv::Mat img = makeRampImage(kW, kH, 0.02f, 0.0f, 0.2f);  // gradient in u only
  Eigen::Isometry3f T_cw = lookingDownCamera(10.f);        // far => near-orthographic
  gicp.setVisualCurrentFrame(img, T_cw);
  gicp.setVisualPreviousFrame(img, T_cw);
  // Points near x=0 so the perspective z-coupling stays small.
  auto cloud = std::make_shared<Cloud>();
  for (float x = -0.3f; x <= 0.3f + 1e-3f; x += 0.1f)
    for (float y = -0.3f; y <= 0.3f + 1e-3f; y += 0.1f)
      cloud->push_back(makePoint(x, y, 0.f));
  gicp.setInputSource(cloud);

  Eigen::Matrix<double, 6, 6> H;
  Eigen::Matrix<double, 6, 1> b;
  gicp.visualSystem(Eigen::Isometry3f::Identity(), H, b);

  Eigen::Matrix3d Htt = H.block<3, 3>(3, 3);
  // x-translation diagonal dominates y (no v-gradient) and z (looming, small far away).
  EXPECT_GT(Htt(0, 0), 1e3 * Htt(1, 1) + 1e-9);
  EXPECT_GT(Htt(0, 0), Htt(2, 2));
}

TEST(VisualResidual, DisabledContributesNothing) {
  TestableGICP gicp;
  gicp.setVisualEnabled(false);  // off
  gicp.setVisualWeight(1.0f);
  gicp.setVisualIntrinsics(kFx, kFy, kCx, kCy);
  cv::Mat img = makeRampImage(kW, kH, 0.01f, 0.01f, 0.2f);
  gicp.setVisualCurrentFrame(img, lookingDownCamera(5.f));
  gicp.setVisualPreviousFrame(img, lookingDownCamera(5.f));
  gicp.setInputSource(makeWorldPoints());

  Eigen::Matrix<double, 6, 6> H;
  Eigen::Matrix<double, 6, 1> b;
  double cost = gicp.visualSystem(Eigen::Isometry3f::Identity(), H, b);
  EXPECT_EQ(cost, 0.0);
  EXPECT_EQ(H.norm(), 0.0);
  EXPECT_EQ(b.norm(), 0.0);
  EXPECT_EQ(gicp.lastVisualCount(), 0);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
