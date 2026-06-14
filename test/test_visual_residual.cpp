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
    rng.at<float>(static_cast<int>(std::lround(v)), static_cast<int>(std::lround(u))) =
        std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
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

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
