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
  void useAnalyticTunnelCovariances() {
    // Exact plane covariances isolate the degeneracy gate from finite-sample
    // normal estimation at the synthetic tunnel's wall/ceiling corners.
    source_covs_.clear();
    for (const auto& p : *input_) {
      Eigen::Vector3f n = std::abs(p.y) >= std::abs(p.z)
          ? Eigen::Vector3f::UnitY() : Eigen::Vector3f::UnitZ();
      Eigen::Matrix4f c = Eigen::Matrix4f::Identity();
      c.block<3, 3>(0, 0) -= 0.999f * n * n.transpose();
      source_covs_.push_back(c);
    }
    setTargetCovariances(std::make_shared<nano_gicp::CovarianceList>(source_covs_));
  }
  bool gradientAt(int index, Eigen::Vector3f& gradient) {
    return estimate_spatial_intensity_gradient(index, gradient);
  }
  double registrationSystem(const Eigen::Isometry3f& trans,
                            Eigen::Matrix<double, 6, 6>& H,
                            Eigen::Matrix<double, 6, 1>& b,
                            Eigen::Matrix<double, 6, 6>& H_geo) {
    calculate_target_intensity_gradients();
    update_correspondences(trans);
    double cost = 0.0;
    linearize(trans, &H, &b, &cost, &H_geo);
    return cost;
  }
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

  // Returns (H, b, cost) for the frame-to-MAP term alone at pose `trans`.
  double visualMapSystem(const Eigen::Isometry3f& trans,
                         Eigen::Matrix<double, 6, 6>& H,
                         Eigen::Matrix<double, 6, 1>& b) {
    H.setZero();
    b.setZero();
    double cost = 0.0;
    this->accumulateVisualMapResidual(trans, &H, &b, &cost);
    return cost;
  }

  // Returns (H, b, cost) for the COIN-LIO LiDAR intensity term alone.
  double lidarMapSystem(const Eigen::Isometry3f& trans,
                        Eigen::Matrix<double, 6, 6>& H,
                        Eigen::Matrix<double, 6, 1>& b) {
    H.setZero();
    b.setZero();
    double cost = 0.0;
    this->accumulateLidarMapResidual(trans, &H, &b, &cost);
    return cost;
  }

  // Returns (H, b, cost) for the frame-to-FRAME LiDAR flow term alone.
  double lidarFlowSystem(const Eigen::Isometry3f& trans,
                         Eigen::Matrix<double, 6, 6>& H,
                         Eigen::Matrix<double, 6, 1>& b) {
    H.setZero();
    b.setZero();
    double cost = 0.0;
    this->accumulateLidarFlowResidual(trans, &H, &b, &cost);
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

// Bilinear sample matching NanoGICP's internal sampler, so a reference computed
// here equals the residual's I_mov at the truth pose (r == 0 exactly).
float bilinearSampleRamp(const cv::Mat& img, float u, float v) {
  int x0 = static_cast<int>(std::floor(u)), y0 = static_cast<int>(std::floor(v));
  float ax = u - x0, ay = v - y0;
  const float* r0 = img.ptr<float>(y0);
  const float* r1 = img.ptr<float>(y0 + 1);
  float top = r0[x0] * (1.f - ax) + r0[x0 + 1] * ax;
  float bot = r1[x0] * (1.f - ax) + r1[x0 + 1] * ax;
  return top * (1.f - ay) + bot * ay;
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

// ---- Frame-to-MAP camera term ----------------------------------------------
namespace {
// Build a VisualRefList for a target/map cloud: sample the reference brightness
// from `img` at each point's projection under world->cam `T_cw`, store the
// keyframe-camera ray, mark in-FOV points valid.
nano_gicp::VisualRefList makeMapRefs(const Cloud& target, const cv::Mat& img,
                                     const Eigen::Isometry3f& T_cw) {
  nano_gicp::VisualRefList refs(target.size());
  for (size_t i = 0; i < target.size(); ++i) {
    Eigen::Vector3f p_w(target[i].x, target[i].y, target[i].z);
    Eigen::Vector3f Pc = T_cw * p_w;
    nano_gicp::VisualRef r;
    r.p_kf_cam = Pc;
    r.valid = 0;
    if (Pc.z() > 1e-3f) {
      float u = kFx * Pc.x() / Pc.z() + kCx;
      float v = kFy * Pc.y() / Pc.z() + kCy;
      if (u >= 2.f && u <= img.cols - 3.f && v >= 2.f && v <= img.rows - 3.f) {
        r.ref = 0.f;  // each test sets the reference along its own sampling path
        r.valid = 1;
      }
    }
    refs[i] = r;
  }
  return refs;
}
}  // namespace

TEST(VisualMapResidual, ResidualIsZeroAtTruth) {
  TestableGICP gicp;
  gicp.setVisualMapWeight(1.0f);
  gicp.setVisualMapViewAngleMax(3.14f);  // disable viewpoint gating for this test
  gicp.setVisualHuberDelta(0.f);
  gicp.setVisualIntrinsics(kFx, kFy, kCx, kCy);

  cv::Mat img = makeRampImage(kW, kH, 0.012f, 0.008f, 0.2f);
  Eigen::Isometry3f T_cw = lookingDownCamera(5.f);
  gicp.setVisualCurrentFrame(img, T_cw);

  auto target = makeWorldPoints();
  gicp.setInputTarget(target);

  // Reference sampled exactly along the residual's own path at trans=Identity,
  // so r==0 by construction.
  auto refs = std::make_shared<nano_gicp::VisualRefList>(makeMapRefs(*target, img, T_cw));
  for (size_t i = 0; i < target->size(); ++i) {
    auto& r = (*refs)[i];
    if (!r.valid) continue;
    Eigen::Vector3f Pc = T_cw * Eigen::Vector3f(target->at(i).x, target->at(i).y, target->at(i).z);
    float u = kFx * Pc.x() / Pc.z() + kCx, v = kFy * Pc.y() / Pc.z() + kCy;
    r.ref = bilinearSampleRamp(img, u, v);
  }
  gicp.setTargetVisualRefs(refs);

  Eigen::Matrix<double, 6, 6> H;
  Eigen::Matrix<double, 6, 1> b;
  double cost = gicp.visualMapSystem(Eigen::Isometry3f::Identity(), H, b);
  EXPECT_GT(gicp.lastVisualMapCount(), 10);
  EXPECT_NEAR(cost, 0.0, 1e-9);
  EXPECT_LT(b.norm(), 1e-4);
}

TEST(VisualMapResidual, AnalyticJacobianMatchesFiniteDifference) {
  TestableGICP gicp;
  gicp.setVisualMapWeight(1.0f);
  gicp.setVisualMapViewAngleMax(3.14f);
  gicp.setVisualHuberDelta(0.f);
  gicp.setVisualIntrinsics(kFx, kFy, kCx, kCy);

  cv::Mat img = makeRampImage(kW, kH, 0.012f, 0.008f, 0.2f);
  Eigen::Isometry3f T_cw = lookingDownCamera(5.f);
  gicp.setVisualCurrentFrame(img, T_cw);
  auto target = makeWorldPoints();
  gicp.setInputTarget(target);

  // References sampled at a SHIFTED pose so the residual is non-trivial at I.
  Eigen::Isometry3f T_cw_ref = T_cw;
  T_cw_ref.pretranslate(Eigen::Vector3f(0.05f, -0.03f, 0.02f));
  auto refs = std::make_shared<nano_gicp::VisualRefList>(makeMapRefs(*target, img, T_cw));
  for (size_t i = 0; i < target->size(); ++i) {
    auto& r = (*refs)[i];
    if (!r.valid) continue;
    Eigen::Vector3f Pc = T_cw_ref * Eigen::Vector3f(target->at(i).x, target->at(i).y, target->at(i).z);
    if (Pc.z() <= 1e-3f) { r.valid = 0; continue; }
    float u = kFx * Pc.x() / Pc.z() + kCx, v = kFy * Pc.y() / Pc.z() + kCy;
    r.ref = bilinearSampleRamp(img, u, v);
  }
  gicp.setTargetVisualRefs(refs);

  Eigen::Matrix<double, 6, 6> H;
  Eigen::Matrix<double, 6, 1> b;
  gicp.visualMapSystem(Eigen::Isometry3f::Identity(), H, b);
  const int count0 = gicp.lastVisualMapCount();
  ASSERT_GT(count0, 10);

  const float eps = 1e-3f;
  Eigen::Matrix<double, 6, 1> num_grad;
  Eigen::Matrix<double, 6, 6> Hd; Eigen::Matrix<double, 6, 1> bd;
  for (int k = 0; k < 6; ++k) {
    double cp = gicp.visualMapSystem(perturbLeft(Eigen::Isometry3f::Identity(), k, eps), Hd, bd);
    ASSERT_EQ(gicp.lastVisualMapCount(), count0) << "point set changed +eps dof " << k;
    double cm = gicp.visualMapSystem(perturbLeft(Eigen::Isometry3f::Identity(), k, -eps), Hd, bd);
    ASSERT_EQ(gicp.lastVisualMapCount(), count0) << "point set changed -eps dof " << k;
    num_grad(k) = (cp - cm) / (2.0 * eps);
  }
  Eigen::Matrix<double, 6, 1> analytic = 2.0 * b;
  EXPECT_LT((num_grad - analytic).norm(), 0.02 * analytic.norm() + 1e-6)
      << "num=" << num_grad.transpose() << "\nana=" << analytic.transpose();
}

TEST(VisualMapResidual, GaussNewtonStepDescendsCost) {
  TestableGICP gicp;
  gicp.setVisualMapWeight(1.0f);
  gicp.setVisualMapViewAngleMax(3.14f);
  gicp.setVisualHuberDelta(0.f);
  gicp.setVisualIntrinsics(kFx, kFy, kCx, kCy);

  cv::Mat img = makeRampImage(kW, kH, 0.012f, 0.008f, 0.2f);
  Eigen::Isometry3f T_cw = lookingDownCamera(5.f);
  gicp.setVisualCurrentFrame(img, T_cw);
  auto target = makeWorldPoints();
  gicp.setInputTarget(target);

  Eigen::Isometry3f T_cw_ref = T_cw;
  T_cw_ref.pretranslate(Eigen::Vector3f(0.08f, -0.05f, 0.0f));
  auto refs = std::make_shared<nano_gicp::VisualRefList>(makeMapRefs(*target, img, T_cw));
  for (size_t i = 0; i < target->size(); ++i) {
    auto& r = (*refs)[i];
    if (!r.valid) continue;
    Eigen::Vector3f Pc = T_cw_ref * Eigen::Vector3f(target->at(i).x, target->at(i).y, target->at(i).z);
    if (Pc.z() <= 1e-3f) { r.valid = 0; continue; }
    float u = kFx * Pc.x() / Pc.z() + kCx, v = kFy * Pc.y() / Pc.z() + kCy;
    r.ref = bilinearSampleRamp(img, u, v);
  }
  gicp.setTargetVisualRefs(refs);

  Eigen::Matrix<double, 6, 6> H; Eigen::Matrix<double, 6, 1> b;
  double cost0 = gicp.visualMapSystem(Eigen::Isometry3f::Identity(), H, b);
  H.diagonal().array() += 1e-6;
  Eigen::Matrix<double, 6, 1> dx = H.ldlt().solve(-b);
  Eigen::Isometry3f stepped = Eigen::Isometry3f::Identity();
  const Eigen::Vector3f rot = dx.head<3>().cast<float>();
  if (rot.norm() > 1e-12f) stepped.prerotate(Eigen::AngleAxisf(rot.norm(), rot / rot.norm()));
  stepped.pretranslate(dx.tail<3>().cast<float>());
  Eigen::Matrix<double, 6, 6> H1; Eigen::Matrix<double, 6, 1> b1;
  double cost1 = gicp.visualMapSystem(stepped, H1, b1);
  EXPECT_LT(cost1, cost0);  // a flipped Jacobian sign would ascend
}

// ---- COIN-LIO LiDAR intensity-image term ------------------------------------
namespace {
constexpr int kLW = 256, kLH = 64;
const float kLAzA = 2.f * static_cast<float>(M_PI) / kLW, kLAzB = -static_cast<float>(M_PI);
const float kLElA = 0.8f / kLH, kLElB = -0.4f;

void projectL(const Eigen::Vector3f& P, float& u, float& v) {
  const float rxy = std::sqrt(P.x() * P.x() + P.y() * P.y());
  u = (std::atan2(P.y(), P.x()) - kLAzB) / kLAzA;
  v = (std::atan2(P.z(), rxy) - kLElB) / kLElA;
}

// Target points generated by back-projecting interior pixels to a range, so they
// land cleanly inside the lidar image (lidar frame == world frame for the test).
Cloud::Ptr makeLidarTarget() {
  auto cloud = std::make_shared<Cloud>();
  for (int col = 40; col <= 210; col += 14) {
    for (int row = 12; row <= 50; row += 6) {
      const float az = kLAzA * col + kLAzB;
      const float el = kLElA * row + kLElB;
      const float R = 6.f + 0.02f * col;
      dlio::Point p;
      p.x = R * std::cos(el) * std::cos(az);
      p.y = R * std::cos(el) * std::sin(az);
      p.z = R * std::sin(el);
      p.reflectivity = 0.f;
      cloud->push_back(p);
    }
  }
  return cloud;
}
}  // namespace

TEST(LidarMapResidual, ResidualIsZeroAtTruth) {
  TestableGICP gicp;
  gicp.setLidarMapWeight(1.0f);
  gicp.setLidarProjection(kLAzA, kLAzB, kLElA, kLElB);
  gicp.setLidarFrame(Eigen::Isometry3f::Identity());  // world == lidar
  cv::Mat img = makeRampImage(kLW, kLH, 0.02f, 0.03f, 0.1f);
  gicp.setLidarImage(img);

  auto target = makeLidarTarget();
  // reflectivity (0..255) = image value at the point's own projection * 255, so
  // the residual (I_mov - reflectivity/255) is exactly 0 at trans = Identity.
  for (auto& p : target->points) {
    float u, v; projectL(Eigen::Vector3f(p.x, p.y, p.z), u, v);
    p.reflectivity = bilinearSampleRamp(img, u, v) * 255.f;
  }
  gicp.setInputTarget(target);

  Eigen::Matrix<double, 6, 6> H; Eigen::Matrix<double, 6, 1> b;
  double cost = gicp.lidarMapSystem(Eigen::Isometry3f::Identity(), H, b);
  EXPECT_GT(gicp.lastLidarMapCount(), 20);
  EXPECT_NEAR(cost, 0.0, 1e-9);
  // b is float roundoff amplified by the count-normalization factor
  // (kLidarRefCount/count ~ 11x here); still far below any real gradient (O(0.1+)).
  EXPECT_LT(b.norm(), 2e-3);
}

// setLidarImageScale retargets the per-channel full-scale normalization (the
// residual is I_image - reflectivity/scale). Default 255 is exercised by
// ResidualIsZeroAtTruth above (bit-identical); this checks a near-IR-like scale.
TEST(LidarMapResidual, ImageScaleRetargetsReference) {
  cv::Mat img = makeRampImage(kLW, kLH, 0.02f, 0.03f, 0.1f);
  auto setup = [&](TestableGICP& g, float refl_scale) {
    g.setLidarMapWeight(1.0f);
    g.setLidarProjection(kLAzA, kLAzB, kLElA, kLElB);
    g.setLidarFrame(Eigen::Isometry3f::Identity());          // world == lidar
    g.setLidarImage(img);
    auto target = makeLidarTarget();
    // reference = image value at the point's own projection, scaled by refl_scale,
    // so the residual is 0 iff the term normalizes by the same refl_scale.
    for (auto& p : target->points) {
      float u, v; projectL(Eigen::Vector3f(p.x, p.y, p.z), u, v);
      p.reflectivity = bilinearSampleRamp(img, u, v) * refl_scale;
    }
    g.setInputTarget(target);
  };
  Eigen::Matrix<double, 6, 6> H; Eigen::Matrix<double, 6, 1> b;

  // A near-IR-like reference (sample * 2000) is zero ONLY when scale = 2000.
  TestableGICP big; setup(big, 2000.f); big.setLidarImageScale(2000.f);
  double cost_big = big.lidarMapSystem(Eigen::Isometry3f::Identity(), H, b);
  ASSERT_GT(big.lastLidarMapCount(), 20);
  EXPECT_NEAR(cost_big, 0.0, 1e-9);

  // The same sample*2000 reference at the DEFAULT 255 scale is far from zero,
  // proving the scale is actually applied (not ignored).
  TestableGICP mis; setup(mis, 2000.f);                      // default scale 255
  double cost_mis = mis.lidarMapSystem(Eigen::Isometry3f::Identity(), H, b);
  ASSERT_GT(mis.lastLidarMapCount(), 20);
  EXPECT_GT(cost_mis, 0.1);
}

// Keyframe-image references (INTENSITY_AUDIT_2026-07-09): when a size-matched
// refs list is set, the reference brightness comes from it -- NOT from the
// target's .reflectivity field. Field deliberately garbage; refs = the image
// value at each point's own projection, so cost is 0 iff the refs are used.
TEST(LidarMapResidual, KeyframeRefsReplaceFieldReference) {
  TestableGICP gicp;
  gicp.setLidarMapWeight(1.0f);
  gicp.setLidarProjection(kLAzA, kLAzB, kLElA, kLElB);
  gicp.setLidarFrame(Eigen::Isometry3f::Identity());
  cv::Mat img = makeRampImage(kLW, kLH, 0.02f, 0.03f, 0.1f);
  gicp.setLidarImage(img);

  auto target = makeLidarTarget();
  auto refs = std::make_shared<std::vector<float>>(target->size(), -1.f);
  for (size_t i = 0; i < target->size(); ++i) {
    auto& p = target->points[i];
    float u, v; projectL(Eigen::Vector3f(p.x, p.y, p.z), u, v);
    (*refs)[i] = bilinearSampleRamp(img, u, v);   // already /scale units
    p.reflectivity = 12345.f;                     // garbage: must be ignored
  }
  gicp.setInputTarget(target);
  gicp.setTargetLidarRefs(refs);

  Eigen::Matrix<double, 6, 6> H; Eigen::Matrix<double, 6, 1> b;
  double cost = gicp.lidarMapSystem(Eigen::Isometry3f::Identity(), H, b);
  ASSERT_GT(gicp.lastLidarMapCount(), 20);
  EXPECT_NEAR(cost, 0.0, 1e-9);                   // refs used, field ignored
}

// A refs list whose size doesn't match the target must be IGNORED (stale refs
// after a submap swap): the term falls back to the field reference.
TEST(LidarMapResidual, KeyframeRefsSizeMismatchFallsBackToField) {
  cv::Mat img = makeRampImage(kLW, kLH, 0.02f, 0.03f, 0.1f);
  auto setup = [&](TestableGICP& g) {
    g.setLidarMapWeight(1.0f);
    g.setLidarProjection(kLAzA, kLAzB, kLElA, kLElB);
    g.setLidarFrame(Eigen::Isometry3f::Identity());
    g.setLidarImage(img);
    auto target = makeLidarTarget();
    for (auto& p : target->points) {
      float u, v; projectL(Eigen::Vector3f(p.x, p.y, p.z), u, v);
      p.reflectivity = bilinearSampleRamp(img, u, v) * 255.f;   // exact field ref
    }
    g.setInputTarget(target);
    return target;
  };
  Eigen::Matrix<double, 6, 6> H; Eigen::Matrix<double, 6, 1> b;

  TestableGICP base; setup(base);
  const double cost_base = base.lidarMapSystem(Eigen::Isometry3f::Identity(), H, b);
  const int count_base = base.lastLidarMapCount();

  TestableGICP stale;
  auto target = setup(stale);
  auto bad = std::make_shared<std::vector<float>>(target->size() + 7, 0.9f);  // wrong size
  stale.setTargetLidarRefs(bad);
  const double cost_stale = stale.lidarMapSystem(Eigen::Isometry3f::Identity(), H, b);
  EXPECT_EQ(stale.lastLidarMapCount(), count_base);
  EXPECT_NEAR(cost_stale, cost_base, 1e-12);      // bit-identical fallback
}

// Invalid entries (< 0: the point didn't project into its keyframe's image)
// are skipped, not fed to the residual as garbage.
TEST(LidarMapResidual, KeyframeRefsInvalidEntriesSkipped) {
  TestableGICP gicp;
  gicp.setLidarMapWeight(1.0f);
  gicp.setLidarProjection(kLAzA, kLAzB, kLElA, kLElB);
  gicp.setLidarFrame(Eigen::Isometry3f::Identity());
  cv::Mat img = makeRampImage(kLW, kLH, 0.02f, 0.03f, 0.1f);
  gicp.setLidarImage(img);
  auto target = makeLidarTarget();
  gicp.setInputTarget(target);
  auto refs = std::make_shared<std::vector<float>>(target->size(), -1.f);  // ALL invalid
  gicp.setTargetLidarRefs(refs);
  Eigen::Matrix<double, 6, 6> H; Eigen::Matrix<double, 6, 1> b;
  double cost = gicp.lidarMapSystem(Eigen::Isometry3f::Identity(), H, b);
  EXPECT_EQ(gicp.lastLidarMapCount(), 0);
  EXPECT_NEAR(cost, 0.0, 1e-12);
  EXPECT_LT(b.norm(), 1e-12);
}

TEST(LidarMapResidual, AnalyticJacobianMatchesFiniteDifference) {
  TestableGICP gicp;
  gicp.setLidarMapWeight(1.0f);
  gicp.setLidarProjection(kLAzA, kLAzB, kLElA, kLElB);
  gicp.setLidarFrame(Eigen::Isometry3f::Identity());
  cv::Mat img = makeRampImage(kLW, kLH, 0.02f, 0.03f, 0.1f);
  gicp.setLidarImage(img);

  auto target = makeLidarTarget();
  // Reference sampled at a SHIFTED pose so the residual is non-trivial at I.
  Eigen::Isometry3f Tshift = Eigen::Isometry3f::Identity();
  Tshift.pretranslate(Eigen::Vector3f(0.1f, 0.05f, 0.03f));
  for (auto& p : target->points) {
    Eigen::Vector3f Ps = Tshift * Eigen::Vector3f(p.x, p.y, p.z);
    float u, v; projectL(Ps, u, v);
    p.reflectivity = bilinearSampleRamp(img, u, v) * 255.f;
  }
  gicp.setInputTarget(target);

  Eigen::Matrix<double, 6, 6> H; Eigen::Matrix<double, 6, 1> b;
  gicp.lidarMapSystem(Eigen::Isometry3f::Identity(), H, b);
  const int count0 = gicp.lastLidarMapCount();
  ASSERT_GT(count0, 20);

  const float eps = 1e-3f;
  Eigen::Matrix<double, 6, 1> num_grad;
  Eigen::Matrix<double, 6, 6> Hd; Eigen::Matrix<double, 6, 1> bd;
  for (int k = 0; k < 6; ++k) {
    double cp = gicp.lidarMapSystem(perturbLeft(Eigen::Isometry3f::Identity(), k, eps), Hd, bd);
    ASSERT_EQ(gicp.lastLidarMapCount(), count0) << "point set changed +eps dof " << k;
    double cm = gicp.lidarMapSystem(perturbLeft(Eigen::Isometry3f::Identity(), k, -eps), Hd, bd);
    ASSERT_EQ(gicp.lastLidarMapCount(), count0) << "point set changed -eps dof " << k;
    num_grad(k) = (cp - cm) / (2.0 * eps);
  }
  Eigen::Matrix<double, 6, 1> analytic = 2.0 * b;
  EXPECT_LT((num_grad - analytic).norm(), 0.03 * analytic.norm() + 1e-6)
      << "num=" << num_grad.transpose() << "\nana=" << analytic.transpose();
}

TEST(LidarMapResidual, GaussNewtonStepDescendsCost) {
  TestableGICP gicp;
  gicp.setLidarMapWeight(1.0f);
  gicp.setLidarProjection(kLAzA, kLAzB, kLElA, kLElB);
  gicp.setLidarFrame(Eigen::Isometry3f::Identity());
  cv::Mat img = makeRampImage(kLW, kLH, 0.02f, 0.03f, 0.1f);
  gicp.setLidarImage(img);

  auto target = makeLidarTarget();
  Eigen::Isometry3f Tshift = Eigen::Isometry3f::Identity();
  Tshift.pretranslate(Eigen::Vector3f(0.15f, 0.08f, 0.0f));
  for (auto& p : target->points) {
    Eigen::Vector3f Ps = Tshift * Eigen::Vector3f(p.x, p.y, p.z);
    float u, v; projectL(Ps, u, v);
    p.reflectivity = bilinearSampleRamp(img, u, v) * 255.f;
  }
  gicp.setInputTarget(target);

  Eigen::Matrix<double, 6, 6> H; Eigen::Matrix<double, 6, 1> b;
  double cost0 = gicp.lidarMapSystem(Eigen::Isometry3f::Identity(), H, b);
  H.diagonal().array() += 1e-6;
  Eigen::Matrix<double, 6, 1> dx = H.ldlt().solve(-b);
  Eigen::Isometry3f stepped = Eigen::Isometry3f::Identity();
  const Eigen::Vector3f rot = dx.head<3>().cast<float>();
  if (rot.norm() > 1e-12f) stepped.prerotate(Eigen::AngleAxisf(rot.norm(), rot / rot.norm()));
  stepped.pretranslate(dx.tail<3>().cast<float>());
  Eigen::Matrix<double, 6, 6> H1; Eigen::Matrix<double, 6, 1> b1;
  double cost1 = gicp.lidarMapSystem(stepped, H1, b1);
  EXPECT_LT(cost1, cost0);
}

// Occlusion gate: a map point whose range disagrees with the range image at its
// pixel is rejected; a consistent one is kept.
TEST(LidarMapResidual, RangeImageRejectsOccludedPoints) {
  TestableGICP gicp;
  gicp.setLidarMapWeight(1.0f);
  gicp.setLidarProjection(kLAzA, kLAzB, kLElA, kLElB);
  gicp.setLidarFrame(Eigen::Isometry3f::Identity());
  cv::Mat img = makeRampImage(kLW, kLH, 0.02f, 0.03f, 0.1f);
  gicp.setLidarImage(img);

  auto target = makeLidarTarget();
  for (auto& p : target->points) {
    float u, v; projectL(Eigen::Vector3f(p.x, p.y, p.z), u, v);
    p.reflectivity = bilinearSampleRamp(img, u, v) * 255.f;
  }
  gicp.setInputTarget(target);

  // Baseline (no range image): all points counted.
  Eigen::Matrix<double, 6, 6> H; Eigen::Matrix<double, 6, 1> b;
  gicp.lidarMapSystem(Eigen::Isometry3f::Identity(), H, b);
  const int count_all = gicp.lastLidarMapCount();
  ASSERT_GT(count_all, 20);

  // Range image holding each point's TRUE range -> nothing rejected.
  cv::Mat rng(kLH, kLW, CV_32FC1, cv::Scalar(0.f));
  for (const auto& p : target->points) {
    float u, v; projectL(Eigen::Vector3f(p.x, p.y, p.z), u, v);
    // The full interpolation/gradient footprint must contain real returns.
    for (int row = static_cast<int>(v) - 1; row <= static_cast<int>(v) + 2; ++row) {
      for (int col = static_cast<int>(u) - 1; col <= static_cast<int>(u) + 2; ++col) {
        rng.at<float>(row, col) = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
      }
    }
  }
  gicp.setLidarRangeImage(rng);
  gicp.setLidarRangeConsistency(0.5f, 0.1f);
  gicp.lidarMapSystem(Eigen::Isometry3f::Identity(), H, b);
  EXPECT_EQ(gicp.lastLidarMapCount(), count_all);

  // Range image claiming a much NEARER surface everywhere -> all occluded out.
  cv::Mat rng_near(kLH, kLW, CV_32FC1, cv::Scalar(1.0f));  // 1 m vs points at 6-10 m
  gicp.setLidarRangeImage(rng_near);
  gicp.lidarMapSystem(Eigen::Isometry3f::Identity(), H, b);
  EXPECT_EQ(gicp.lastLidarMapCount(), 0);
}

// Elevation LUT projection reproduces the linear model when the LUT is the exact
// linear elevations (residual still zero at truth).
TEST(LidarMapResidual, ElevationLutMatchesLinearAtTruth) {
  TestableGICP gicp;
  gicp.setLidarMapWeight(1.0f);
  gicp.setLidarProjection(kLAzA, kLAzB, kLElA, kLElB);
  gicp.setLidarFrame(Eigen::Isometry3f::Identity());
  cv::Mat img = makeRampImage(kLW, kLH, 0.02f, 0.03f, 0.1f);
  gicp.setLidarImage(img);

  std::vector<float> lut(kLH);
  for (int row = 0; row < kLH; ++row) { lut[row] = kLElA * row + kLElB; }
  gicp.setLidarElevationLut(lut);

  auto target = makeLidarTarget();
  for (auto& p : target->points) {
    float u, v; projectL(Eigen::Vector3f(p.x, p.y, p.z), u, v);
    p.reflectivity = bilinearSampleRamp(img, u, v) * 255.f;
  }
  gicp.setInputTarget(target);

  Eigen::Matrix<double, 6, 6> H; Eigen::Matrix<double, 6, 1> b;
  double cost = gicp.lidarMapSystem(Eigen::Isometry3f::Identity(), H, b);
  EXPECT_GT(gicp.lastLidarMapCount(), 20);
  EXPECT_NEAR(cost, 0.0, 1e-9);
}

// The projection Jacobian's elevation term uses inv_el_eff = 1/slope taken from
// the per-row elevation LUT (the path for non-uniform OS beams). The other
// LidarMapResidual tests don't pin this: AnalyticJacobianMatchesFiniteDifference
// uses the LINEAR elevation model (inv_el_eff = 1/el_a), and
// ElevationLutMatchesLinearAtTruth only checks the residual under a LINEAR LUT.
// This finite-differences the full 6-DOF Jacobian with a genuinely NON-UNIFORM
// LUT active, so a wrong sign/scale of inv_el_eff -- the load-bearing constraint
// in a degenerate tunnel -- would be caught.
TEST(LidarMapResidual, AnalyticJacobianMatchesFiniteDifferenceWithNonUniformLut) {
  TestableGICP gicp;
  gicp.setLidarMapWeight(1.0f);
  gicp.setLidarProjection(kLAzA, kLAzB, kLElA, kLElB);  // azimuth model + linear fallback
  gicp.setLidarFrame(Eigen::Isometry3f::Identity());
  cv::Mat img = makeRampImage(kLW, kLH, 0.02f, 0.03f, 0.1f);
  gicp.setLidarImage(img);

  // Strictly-increasing but NON-LINEAR per-row elevation LUT (slope varies with
  // row via a smooth convex warp), spanning ~the linear range so makeLidarTarget
  // elevations fall well inside its coverage.
  std::vector<float> lut(kLH);
  const float span = kLElA * (kLH - 1);
  for (int row = 0; row < kLH; ++row) {
    const float t = static_cast<float>(row) / (kLH - 1);
    const float g = (t + 0.4f * t * t) / 1.4f;   // g(0)=0, g(1)=1, slope rises with row
    lut[row] = kLElB + span * g;
  }
  gicp.setLidarElevationLut(lut);

  // Mirror the code's LUT projection (linear azimuth + LUT-inverse elevation) to
  // sample the reference at a small shift, so the residual is non-trivial at I.
  // (The FD-vs-analytic comparison validates inv_el_eff regardless, but a small
  // residual keeps the first-order/central-difference agreement tight.)
  auto projectLut = [&](const Eigen::Vector3f& P, float& u, float& v) -> bool {
    const float rxy = std::sqrt(P.x() * P.x() + P.y() * P.y());
    u = (std::atan2(P.y(), P.x()) - kLAzB) / kLAzA;
    const float el = std::atan2(P.z(), rxy);
    for (int k = 0; k + 1 < kLH; ++k) {                 // LUT increasing
      const float a = lut[k], bb = lut[k + 1];
      if (el >= a && el <= bb) {
        const float denom = bb - a;
        if (std::abs(denom) < 1e-9f) continue;
        v = static_cast<float>(k) + (el - a) / denom;
        return true;
      }
    }
    return false;
  };

  auto target = makeLidarTarget();
  Eigen::Isometry3f Tshift = Eigen::Isometry3f::Identity();
  Tshift.pretranslate(Eigen::Vector3f(0.1f, 0.05f, 0.03f));
  for (auto& p : target->points) {
    Eigen::Vector3f Ps = Tshift * Eigen::Vector3f(p.x, p.y, p.z);
    float u, v;
    p.reflectivity = projectLut(Ps, u, v) ? bilinearSampleRamp(img, u, v) * 255.f : 0.f;
  }
  gicp.setInputTarget(target);

  Eigen::Matrix<double, 6, 6> H; Eigen::Matrix<double, 6, 1> b;
  gicp.lidarMapSystem(Eigen::Isometry3f::Identity(), H, b);
  const int count0 = gicp.lastLidarMapCount();
  ASSERT_GT(count0, 20) << "non-uniform LUT projection dropped too many points";

  const float eps = 1e-3f;
  Eigen::Matrix<double, 6, 1> num_grad;
  Eigen::Matrix<double, 6, 6> Hd; Eigen::Matrix<double, 6, 1> bd;
  for (int k = 0; k < 6; ++k) {
    double cp = gicp.lidarMapSystem(perturbLeft(Eigen::Isometry3f::Identity(), k, eps), Hd, bd);
    ASSERT_EQ(gicp.lastLidarMapCount(), count0) << "point set changed +eps dof " << k;
    double cm = gicp.lidarMapSystem(perturbLeft(Eigen::Isometry3f::Identity(), k, -eps), Hd, bd);
    ASSERT_EQ(gicp.lastLidarMapCount(), count0) << "point set changed -eps dof " << k;
    num_grad(k) = (cp - cm) / (2.0 * eps);
  }
  Eigen::Matrix<double, 6, 1> analytic = 2.0 * b;
  EXPECT_LT((num_grad - analytic).norm(), 0.04 * analytic.norm() + 1e-6)
      << "non-uniform-LUT Jacobian mismatch\nnum=" << num_grad.transpose()
      << "\nana=" << analytic.transpose();
}

// --- Dense source cloud for the frame-to-frame term (setVisualSource) ---
//
// The f2f term iterates the source cloud only to PROJECT points for brightness.
// A dense source (the full deskewed scan) keeps the narrow camera FOV populated
// where the voxelised input_ collapses to ~0. Null source -> iterate input_
// (bit-identical). See doc/FINDINGS_2026-06-22.md Part 1/4.

namespace {
// Build a small f2f scene; returns the configured gicp ready for visualSystem.
void setupF2F(TestableGICP& g) {
  g.setVisualEnabled(true);
  g.setVisualWeight(1.0f);
  g.setVisualHuberDelta(0.f);
  g.setVisualIntrinsics(kFx, kFy, kCx, kCy);
  cv::Mat cur = makeRampImage(kW, kH, 0.01f, 0.007f, 0.2f);
  cv::Mat prev = makeRampImage(kW, kH, 0.01f, 0.007f, 0.25f);  // != cur -> nonzero r
  g.setVisualCurrentFrame(cur, lookingDownCamera(5.f));
  g.setVisualPreviousFrame(prev, lookingDownCamera(5.f));
}
// A dense in-FOV cloud (looking-down camera at height 5): n x n grid on z=0.
Cloud::Ptr makeDenseGrid(int n) {
  auto c = std::make_shared<Cloud>();
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      c->push_back(makePoint(-1.f + 2.f * i / (n - 1), -1.f + 2.f * j / (n - 1), 0.f));
  return c;
}
}  // namespace

// Null dense source -> the term iterates input_, bit-identical.
TEST(VisualDenseSource, NullSourceIsBitIdentical) {
  auto input = makeWorldPoints();
  TestableGICP a; setupF2F(a); a.setInputSource(input);
  Eigen::Matrix<double, 6, 6> Ha; Eigen::Matrix<double, 6, 1> ba;
  double ca = a.visualSystem(Eigen::Isometry3f::Identity(), Ha, ba);

  TestableGICP b; setupF2F(b); b.setInputSource(input);
  b.setVisualSource(nullptr, 4000);                  // explicitly off
  Eigen::Matrix<double, 6, 6> Hb; Eigen::Matrix<double, 6, 1> bb;
  double cb = b.visualSystem(Eigen::Isometry3f::Identity(), Hb, bb);

  EXPECT_EQ(a.lastVisualCount(), b.lastVisualCount());
  EXPECT_DOUBLE_EQ(ca, cb);
  EXPECT_TRUE(Ha.isApprox(Hb, 0.0));
  EXPECT_TRUE(ba.isApprox(bb, 0.0));
}

// A dense source overrides a sparse input_: the iterated/kept count reflects the
// dense cloud, not the tiny registration cloud (the de-starvation mechanism).
TEST(VisualDenseSource, DenseOverridesSparseInput) {
  auto sparse = std::make_shared<Cloud>();          // 4 in-FOV points
  sparse->push_back(makePoint(0.f, 0.f, 0.f));
  sparse->push_back(makePoint(0.2f, 0.f, 0.f));
  sparse->push_back(makePoint(0.f, 0.2f, 0.f));
  sparse->push_back(makePoint(-0.2f, -0.2f, 0.f));

  TestableGICP base; setupF2F(base); base.setInputSource(sparse);
  Eigen::Matrix<double, 6, 6> H; Eigen::Matrix<double, 6, 1> bvec;
  base.visualSystem(Eigen::Isometry3f::Identity(), H, bvec);
  const int sparse_count = base.lastVisualCount();
  EXPECT_LE(sparse_count, 4);

  TestableGICP dense; setupF2F(dense); dense.setInputSource(sparse);
  dense.setVisualSource(makeDenseGrid(9), 4000);     // 81 in-FOV points, no stride
  dense.visualSystem(Eigen::Isometry3f::Identity(), H, bvec);
  EXPECT_GT(dense.lastVisualCount(), sparse_count);  // de-starved
  EXPECT_GE(dense.lastVisualCount(), 60);            // ~81 in-FOV
}

// maxPoints strides the dense source down, bounding per-iteration cost.
TEST(VisualDenseSource, StrideCapsIteration) {
  auto grid = makeDenseGrid(11);                     // 121 in-FOV points
  TestableGICP full; setupF2F(full); full.setInputSource(makeWorldPoints());
  full.setVisualSource(grid, 4000);                  // no stride
  Eigen::Matrix<double, 6, 6> H; Eigen::Matrix<double, 6, 1> bvec;
  full.visualSystem(Eigen::Isometry3f::Identity(), H, bvec);
  EXPECT_GE(full.lastVisualCount(), 110);            // ~121 kept

  TestableGICP capped; setupF2F(capped); capped.setInputSource(makeWorldPoints());
  capped.setVisualSource(grid, 30);                  // stride = 121/30 = 4 -> ~31 kept
  capped.visualSystem(Eigen::Isometry3f::Identity(), H, bvec);
  EXPECT_LT(capped.lastVisualCount(), full.lastVisualCount());
  EXPECT_LE(capped.lastVisualCount(), 40);           // <= ~maxPoints
}

// ---- Frame-to-FRAME LiDAR flow term: numerical correctness ------------------
//
// doc/LIDAR_FLOW_TERM.md calls the Jacobian "correct by construction" but states
// a sign/frame slip "can only be ruled out on the bag" -- until now the only
// flow tests were crash/bounded smoke checks. These pin the two properties that
// actually catch a sign or frame error without a bag: the residual vanishes at
// the true pose, and the analytic Jacobian matches a finite difference of the
// cost. Uses the image-to-image reference mode (the default since the 2026-07-09
// intensity audit), so both sides of the residual are image samples.
namespace {

// Set up a flow problem where the PREVIOUS scan sits at a known offset from the
// current one, both looking at the same textured cylinder.
void setupFlow(TestableGICP& g, const cv::Mat& img, const Eigen::Isometry3f& T_lw_prev,
               const Cloud::Ptr& src) {
  g.setLidarProjection(kLAzA, kLAzB, kLElA, kLElB);
  g.setLidarImage(img);                       // current image = reference side
  g.setLidarFrame(Eigen::Isometry3f::Identity());   // world == current lidar
  g.setLidarFlowPrev(img, T_lw_prev);         // same texture seen from prev pose
  g.setLidarFlowMode(true, 0);                // image-to-image, single pixel
  g.setLidarFlowWeight(1.0f);
  g.setInputSource(src);
}

}  // namespace

// With the previous scan at the SAME pose as the current one, every point
// projects to the same pixel in both images, so the residual is identically
// zero at trans = Identity. A flipped residual sign or a swapped frame would
// still be zero here -- that is what the finite-difference test below is for --
// but a projection/frame mismatch shows up immediately as a nonzero cost.
TEST(LidarFlowResidual, ResidualIsZeroWhenFramesCoincide) {
  TestableGICP gicp;
  cv::Mat img = makeRampImage(kLW, kLH, 0.02f, 0.03f, 0.1f);
  auto src = makeLidarTarget();
  setupFlow(gicp, img, Eigen::Isometry3f::Identity(), src);

  Eigen::Matrix<double, 6, 6> H; Eigen::Matrix<double, 6, 1> b;
  const double cost = gicp.lidarFlowSystem(Eigen::Isometry3f::Identity(), H, b);
  EXPECT_GT(gicp.lastLidarFlowCount(), 20);   // the term actually engaged
  EXPECT_NEAR(cost, 0.0, 1e-9);
  EXPECT_LT(b.norm(), 1e-3);                  // no gradient at the optimum
  EXPECT_LT(gicp.lastLidarFlowRms(), 1e-4);
}

TEST(LidarFlowResidual, MeanSubtractedRampHasNoMotionInformation) {
  // Translating a linear brightness ramp changes only the patch mean. After
  // removing that mean, its Hessian must not claim any motion information.
  TestableGICP g;
  auto img = makeRampImage(kLW, kLH, 0.002f, 0.003f, 0.1f);
  setupFlow(g, img, Eigen::Isometry3f::Identity(), makeLidarTarget());
  g.setLidarFlowMode(true, 2);
  Eigen::Matrix<double, 6, 6> H;
  Eigen::Matrix<double, 6, 1> b;
  g.lidarFlowSystem(Eigen::Isometry3f::Identity(), H, b);
  ASSERT_GT(g.lastLidarFlowCount(), 20);
  EXPECT_LT(H.norm(), 1e-5);
}

TEST(LidarFlowResidual, RejectsOcclusionAndMissingGradientSupport) {
  TestableGICP g;
  auto img = makeRampImage(kLW, kLH, 0.002f, 0.003f, 0.1f);
  setupFlow(g, img, Eigen::Isometry3f::Identity(), makeLidarTarget());
  cv::Mat range(kLH, kLW, CV_32FC1, cv::Scalar(0.1f));
  g.setLidarFlowPrev(img, Eigen::Isometry3f::Identity(), range);
  g.setLidarRangeConsistency(0.05f, 0.01f);
  Eigen::Matrix<double, 6, 6> H;
  Eigen::Matrix<double, 6, 1> b;
  EXPECT_EQ(g.lidarFlowSystem(Eigen::Isometry3f::Identity(), H, b), 0.0);
  EXPECT_EQ(g.lastLidarFlowCount(), 0);

  // Even with a permissive depth tolerance, missing pixels around an otherwise
  // valid center cannot be treated as texture edges.
  range.setTo(10.f);
  for (int col = 0; col < range.cols; col += 2) { range.col(col).setTo(0.f); }
  g.setLidarFlowPrev(img, Eigen::Isometry3f::Identity(), range);
  g.setLidarRangeConsistency(100.f, 1.f);
  EXPECT_EQ(g.lidarFlowSystem(Eigen::Isometry3f::Identity(), H, b), 0.0);
  EXPECT_EQ(g.lastLidarFlowCount(), 0);
}

namespace {
struct TunnelFrame {
  Cloud::Ptr cloud = std::make_shared<Cloud>();
  cv::Mat image{64, 1024, CV_32FC1, cv::Scalar(0.f)};
  cv::Mat range{64, 1024, CV_32FC1, cv::Scalar(0.f)};
};

TunnelFrame paintedTunnel(float sensor_x, bool painted) {
  TunnelFrame out;
  for (int row = 0; row < 64; ++row) {
    const float el = 0.36f - 0.72f * row / 63.f;
    for (int col = 0; col < 1024; ++col) {
      const float az = -2.f * M_PI * col / 1024.f;
      Eigen::Vector3f ray(std::cos(el) * std::cos(az), std::cos(el) * std::sin(az), std::sin(el));
      const float range = 1.f / std::max(std::abs(ray.y()), std::abs(ray.z()));
      if (range > 10.f) { continue; }
      const Eigen::Vector3f p = range * ray;
      const float world_x = p.x() + sensor_x;
      const float value = painted ? 0.5f + 0.2f * std::sin(4.f * world_x)
          + 0.15f * std::sin(7.1f * world_x + 2.f * p.z())
          + 0.1f * std::cos(2.1f * world_x + 3.f * p.y()) : 0.5f;
      out.image.at<float>(row, col) = value;
      out.range.at<float>(row, col) = range;
      if (row % 2 == 0 && col % 8 == 0) {
        auto pt = makePoint(p.x(), p.y(), p.z());
        pt.reflectivity = value * 255.f;
        out.cloud->push_back(pt);
      }
    }
  }
  return out;
}

Eigen::Matrix4f alignPaintedTunnel(float motion, bool painted, bool flow,
                                  float prior, int* rescued) {
  auto prev = paintedTunnel(0.f, painted);
  auto cur = paintedTunnel(motion, painted);
  TestableGICP g;
  g.setNumThreads(1);
  g.setRegularizationMethod(nano_gicp::RegularizationMethod::PLANE);
  g.setDegeneracyThreshRatio(0.005f);
  g.setMaximumIterations(30);
  g.setTransformationEpsilon(1e-5f);
  g.setRotationEpsilon(1e-5f);
  g.setMaxCorrespondenceDistance(0.5f);
  g.setPhotometricWeight(0.f);
  g.setPhotometricHuberDelta(0.1f);
  g.setLidarProjection(-2.f * M_PI / 1024.f, 0.f, -0.72f / 63.f, 0.36f);
  g.setLidarImage(cur.image);
  g.setLidarRangeImage(cur.range);
  g.setLidarFrame(Eigen::Isometry3f::Identity());
  g.setLidarFlowPrev(prev.image, Eigen::Isometry3f::Identity(), prev.range);
  g.setLidarFlowMode(true, 0);
  g.setLidarFlowWeight(flow ? 100.f : 0.f);
  g.setVisualGateMaxStep(0.15f, 0.05f);
  g.setInputSource(cur.cloud);
  g.setInputTarget(prev.cloud);
  g.useAnalyticTunnelCovariances();
  Cloud aligned;
  Eigen::Matrix4f guess = Eigen::Matrix4f::Identity();
  guess(0, 3) = prior;
  g.align(aligned, guess);
  *rescued = g.lastVisualRescuedDirections();
  return g.getFinalTransformation();
}
}  // namespace

TEST(LidarTunnelOdometry, PaintConstrainsForwardReverseAndStationaryMotion) {
  // Geometry is an infinite square tunnel: moving along x does not change it.
  // Paint is attached to WORLD coordinates; it supplies the missing constraint.
  // A directional velocity clamp would fail the reverse and stationary cases.
  for (float motion : {-0.08f, 0.f, 0.08f}) {
    SCOPED_TRACE(motion);
    int rescued = 0;
    const auto pose = alignPaintedTunnel(motion, true, true, 0.03f, &rescued);
    EXPECT_GT(rescued, 0);
    EXPECT_NEAR(pose(0, 3), motion, 0.01f);
    EXPECT_LT((pose.block<2, 1>(1, 3).norm()), 0.005f);
  }
}

TEST(LidarTunnelOdometry, NoTextureOrDisabledFlowKeepsWeakAxisPrior) {
  for (bool painted : {false, true}) {
    int rescued = 0;
    const auto pose = alignPaintedTunnel(0.08f, painted, !painted, 0.03f, &rescued);
    EXPECT_EQ(rescued, 0);
    EXPECT_NEAR(pose(0, 3), 0.03f, 0.003f);
  }
}

TEST(LidarTunnelOdometry, PlanarGraffitiHasATangentialPhotometricGradient) {
  auto cloud = std::make_shared<Cloud>();
  for (int row = -4; row <= 4; ++row) {
    for (int col = -4; col <= 4; ++col) {
      auto p = makePoint(0.1f * col, 0.1f * row, 1.f);
      p.reflectivity = 255.f * (0.5f + 0.2f * p.x - 0.1f * p.y);
      cloud->push_back(p);
    }
  }
  TestableGICP g;
  g.setPhotometricChannel(true);
  g.setPhotometricScale(255.f);
  g.setInputTarget(cloud);
  Eigen::Vector3f gradient;
  ASSERT_TRUE(g.gradientAt(40, gradient));
  EXPECT_NEAR(gradient.x(), 0.2f, 1e-4f);
  EXPECT_NEAR(gradient.y(), -0.1f, 1e-4f);
  EXPECT_NEAR(gradient.z(), 0.f, 1e-6f);
}

TEST(LidarTunnelOdometry, AppearanceDoesNotMaskGeometricDegeneracyAndCostMatchesGradient) {
  auto target = std::make_shared<Cloud>();
  for (int row = -4; row <= 4; ++row) {
    for (int col = -4; col <= 4; ++col) {
      auto p = makePoint(0.1f * col, 0.1f * row, 1.f);
      p.reflectivity = 255.f * (0.5f + 0.2f * p.x - 0.1f * p.y);
      target->push_back(p);
    }
  }
  auto source = std::make_shared<Cloud>(*target);
  for (auto& p : *source) { p.reflectivity += 3.f; }
  TestableGICP g;
  g.setNumThreads(1);
  g.setPhotometricChannel(true);
  g.setPhotometricScale(255.f);
  g.setPhotometricHuberDelta(0.f);
  g.setInputSource(source);
  g.setInputTarget(target);
  g.useAnalyticTunnelCovariances();
  Eigen::Matrix<double, 6, 6> H0, H, geo0, geo;
  Eigen::Matrix<double, 6, 1> b0, b;
  const auto pose = Eigen::Isometry3f::Identity();
  g.setPhotometricWeight(0.f);
  g.registrationSystem(pose, H0, b0, geo0);
  g.setPhotometricWeight(1000.f);
  g.registrationSystem(pose, H, b, geo);
  EXPECT_LT((geo - H0).norm(), 1e-8);
  EXPECT_GT((H - geo).norm(), 100.f);
  Eigen::Matrix<double, 6, 1> numeric;
  constexpr float eps = 1e-4f;
  for (int k = 0; k < 6; ++k) {
    Eigen::Matrix<double, 6, 6> temp, temp_geo;
    Eigen::Matrix<double, 6, 1> temp_b;
    const double cp = g.registrationSystem(perturbLeft(pose, k, eps), temp, temp_b, temp_geo);
    const double cm = g.registrationSystem(perturbLeft(pose, k, -eps), temp, temp_b, temp_geo);
    numeric(k) = (cp - cm) / (2.0 * eps);
  }
  EXPECT_LT((numeric - 2.0 * b).norm(), 0.005 * b.norm());
}

// THE sign/frame test: the analytic gradient must agree with a central finite
// difference of the term's own cost, per DOF. A flipped Jacobian sign or a
// left/right perturbation mix-up inverts one or more components and fails here.
TEST(LidarFlowResidual, AnalyticJacobianMatchesFiniteDifference) {
  TestableGICP gicp;
  cv::Mat img = makeRampImage(kLW, kLH, 0.02f, 0.03f, 0.1f);
  auto src = makeLidarTarget();
  // Previous scan displaced, so the residual is non-trivial at Identity.
  Eigen::Isometry3f T_lw_prev = Eigen::Isometry3f::Identity();
  T_lw_prev.pretranslate(Eigen::Vector3f(0.12f, -0.07f, 0.04f));
  setupFlow(gicp, img, T_lw_prev, src);

  Eigen::Matrix<double, 6, 6> H; Eigen::Matrix<double, 6, 1> b;
  const double c0 = gicp.lidarFlowSystem(Eigen::Isometry3f::Identity(), H, b);
  ASSERT_GT(gicp.lastLidarFlowCount(), 20);
  ASSERT_GT(c0, 1e-9);                        // non-trivial working point

  // House convention (see VisualResidual.AnalyticJacobianMatchesFiniteDifference
  // above): the accumulators build b = +J^T r and the solver descends via
  // solve(-b), so the gradient of the term's cost is +2b.
  const float eps = 1e-4f;
  Eigen::Matrix<double, 6, 1> num_grad = Eigen::Matrix<double, 6, 1>::Zero();
  for (int k = 0; k < 6; ++k) {
    Eigen::Matrix<double, 6, 1> dx = Eigen::Matrix<double, 6, 1>::Zero();
    dx(k) = eps;
    auto step = [&](double s) {
      Eigen::Isometry3f T = Eigen::Isometry3f::Identity();
      const Eigen::Vector3f rot = (s * dx.head<3>()).cast<float>();
      if (rot.norm() > 1e-12f) { T.prerotate(Eigen::AngleAxisf(rot.norm(), rot / rot.norm())); }
      T.pretranslate((s * dx.tail<3>()).cast<float>());
      return T;
    };
    Eigen::Matrix<double, 6, 6> Hp, Hm; Eigen::Matrix<double, 6, 1> bp, bm;
    const double cp = gicp.lidarFlowSystem(step(+1.0), Hp, bp);
    const double cm = gicp.lidarFlowSystem(step(-1.0), Hm, bm);
    num_grad(k) = (cp - cm) / (2.0 * eps);
  }
  const Eigen::Matrix<double, 6, 1> analytic = 2.0 * b;
  // Tight: the cost is a bilinear image sample so the difference carries some
  // interpolation noise, but agreement is otherwise to several digits. A sign
  // slip in ANY dof blows this up by 2x the component.
  EXPECT_LT((num_grad - analytic).norm(), 0.02 * analytic.norm() + 1e-6)
      << "num=" << num_grad.transpose() << "\nana=" << analytic.transpose();
}

// Descent check: stepping along the term's own Gauss-Newton direction must
// LOWER its cost. This is the property a flipped sign destroys outright.
TEST(LidarFlowResidual, GaussNewtonStepDecreasesCost) {
  TestableGICP gicp;
  cv::Mat img = makeRampImage(kLW, kLH, 0.02f, 0.03f, 0.1f);
  auto src = makeLidarTarget();
  Eigen::Isometry3f T_lw_prev = Eigen::Isometry3f::Identity();
  T_lw_prev.pretranslate(Eigen::Vector3f(0.10f, 0.05f, -0.03f));
  setupFlow(gicp, img, T_lw_prev, src);

  Eigen::Matrix<double, 6, 6> H; Eigen::Matrix<double, 6, 1> b;
  const double cost0 = gicp.lidarFlowSystem(Eigen::Isometry3f::Identity(), H, b);
  ASSERT_GT(cost0, 1e-9);
  H.diagonal().array() += 1e-6;
  const Eigen::Matrix<double, 6, 1> dx = H.ldlt().solve(-b);
  Eigen::Isometry3f stepped = Eigen::Isometry3f::Identity();
  const Eigen::Vector3f rot = dx.head<3>().cast<float>();
  if (rot.norm() > 1e-12f) { stepped.prerotate(Eigen::AngleAxisf(rot.norm(), rot / rot.norm())); }
  stepped.pretranslate(dx.tail<3>().cast<float>());

  Eigen::Matrix<double, 6, 6> H1; Eigen::Matrix<double, 6, 1> b1;
  const double cost1 = gicp.lidarFlowSystem(stepped, H1, b1);
  EXPECT_LT(cost1, cost0);                    // a flipped Jacobian would ascend
}

// Legacy point-field reference mode must remain wired: with imageRef off the
// reference is the source point's own .reflectivity, so a cylinder whose points
// carry exactly their own image sample is again zero-residual at coincidence.
TEST(LidarFlowResidual, FieldReferenceModeStillZeroAtCoincidence) {
  TestableGICP gicp;
  cv::Mat img = makeRampImage(kLW, kLH, 0.02f, 0.03f, 0.1f);
  auto src = makeLidarTarget();
  for (auto& p : src->points) {
    float u, v; projectL(Eigen::Vector3f(p.x, p.y, p.z), u, v);
    p.reflectivity = bilinearSampleRamp(img, u, v) * 255.f;   // matches /255 scale
  }
  gicp.setLidarProjection(kLAzA, kLAzB, kLElA, kLElB);
  gicp.setLidarFrame(Eigen::Isometry3f::Identity());
  gicp.setLidarFlowPrev(img, Eigen::Isometry3f::Identity());
  gicp.setLidarFlowMode(false, 0);            // legacy: point-field reference
  gicp.setLidarFlowWeight(1.0f);
  gicp.setInputSource(src);

  Eigen::Matrix<double, 6, 6> H; Eigen::Matrix<double, 6, 1> b;
  const double cost = gicp.lidarFlowSystem(Eigen::Isometry3f::Identity(), H, b);
  EXPECT_GT(gicp.lastLidarFlowCount(), 20);
  EXPECT_NEAR(cost, 0.0, 1e-9);
}

TEST(LidarImageChannels, DedicatedChannelFeedsMapAndFlowIndependentlyOfReflectivity) {
  TestableGICP g;
  const cv::Mat img = makeRampImage(kLW, kLH, 0.02f, 0.03f, 0.1f);
  auto cloud = makeLidarTarget();
  for (auto& p : *cloud) {
    float u, v; projectL(Eigen::Vector3f(p.x, p.y, p.z), u, v);
    p.lidar_intensity = bilinearSampleRamp(img, u, v) * 4096.f;
    p.reflectivity = 20.f; // independent calibrated measurement
  }
  g.setLidarImageUseDedicatedChannel(true);
  g.setLidarImageScale(4096.f);
  g.setLidarProjection(kLAzA, kLAzB, kLElA, kLElB);
  g.setLidarFrame(Eigen::Isometry3f::Identity());
  g.setLidarImage(img);
  g.setLidarFlowPrev(img, Eigen::Isometry3f::Identity());
  g.setLidarFlowMode(false, 0);
  g.setLidarFlowWeight(1.f);
  g.setLidarMapWeight(1.f);
  g.setInputSource(cloud);
  g.setInputTarget(cloud);
  Eigen::Matrix<double, 6, 6> H;
  Eigen::Matrix<double, 6, 1> b;
  EXPECT_NEAR(g.lidarMapSystem(Eigen::Isometry3f::Identity(), H, b), 0.0, 1e-9);
  EXPECT_GT(g.lastLidarMapCount(), 20);
  EXPECT_NEAR(g.lidarFlowSystem(Eigen::Isometry3f::Identity(), H, b), 0.0, 1e-9);
  EXPECT_GT(g.lastLidarFlowCount(), 20);
  g.setLidarImageUseDedicatedChannel(false);
  EXPECT_GT(g.lidarMapSystem(Eigen::Isometry3f::Identity(), H, b), 0.1);
  EXPECT_GT(g.lidarFlowSystem(Eigen::Isometry3f::Identity(), H, b), 0.1);
}

TEST(LidarImageChannels, ClockwiseScanProjectsBothHalvesAcrossAtan2Seam) {
  TestableGICP g;
  const float az_step = -2.f * static_cast<float>(M_PI) / kLW;
  const cv::Mat img = makeRampImage(kLW, kLH, 0.002f, 0.003f, 0.1f);
  auto cloud = std::make_shared<Cloud>();
  for (int col = 40; col <= 210; col += 14) {
    for (int row = 12; row <= 50; row += 6) {
      const float az = az_step * col, el = kLElA * row + kLElB;
      dlio::Point p;
      p.x = 6.f * std::cos(el) * std::cos(az);
      p.y = 6.f * std::cos(el) * std::sin(az);
      p.z = 6.f * std::sin(el);
      p.reflectivity = img.at<float>(row, col) * 255.f;
      cloud->push_back(p);
    }
  }
  g.setLidarProjection(az_step, 0.f, kLElA, kLElB);
  g.setLidarFrame(Eigen::Isometry3f::Identity());
  g.setLidarImage(img);
  g.setLidarFlowPrev(img, Eigen::Isometry3f::Identity());
  g.setLidarFlowMode(true, 0);
  g.setLidarFlowWeight(1.f);
  g.setLidarMapWeight(1.f);
  g.setInputSource(cloud);
  g.setInputTarget(cloud);
  Eigen::Matrix<double, 6, 6> H;
  Eigen::Matrix<double, 6, 1> b;
  EXPECT_NEAR(g.lidarMapSystem(Eigen::Isometry3f::Identity(), H, b), 0.0, 1e-9);
  EXPECT_EQ(g.lastLidarMapCount(), static_cast<int>(cloud->size()));
  EXPECT_NEAR(g.lidarFlowSystem(Eigen::Isometry3f::Identity(), H, b), 0.0, 1e-9);
  EXPECT_EQ(g.lastLidarFlowCount(), static_cast<int>(cloud->size()));
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
