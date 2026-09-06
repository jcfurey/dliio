#include <gtest/gtest.h>
#include <Eigen/Geometry>
#include <cmath>
#include "dlio/dlio.h"
#include "nano_gicp/nano_gicp.h"
#include "nano_gicp/surface_texture.h"

namespace {
using nano_gicp::TextureCloud;

float paint(float x, float z, int pattern) {
  if (pattern == 0) { return 0.4f; }
  if (pattern == 2) { return 0.4f + 0.15f * std::sin(2.f * M_PI * x / 0.24f); }
  if (pattern == 3) { return 0.4f + 0.03f * x; }
  return 0.4f + 0.12f * std::sin(5.3f * x + 1.7f * z) +
      0.08f * std::cos(10.1f * x - 2.3f * z) + 0.06f * std::sin(2.7f * x * x + z);
}

TextureCloud::Ptr walls(float motion, int pattern = 1, float gain = 1.f, float offset = 0.f,
                        float phase = 0.f) {
  auto cloud = std::make_shared<TextureCloud>();
  for (float side : {-1.5f, 1.5f}) {
    for (int row = -32; row <= 32; ++row) {
      for (int col = -90; col <= 90; ++col) {
        const float x = col * 0.035f + phase, z = row * 0.035f + phase;
        pcl::PointXYZI p;
        p.x = x - motion; p.y = side; p.z = z;
        p.intensity = gain * paint(x, z, pattern) + offset;
        cloud->push_back(p);
      }
    }
  }
  return cloud;
}

TextureCloud::Ptr movingHead(float motion, float phase) {
  auto cloud = std::make_shared<TextureCloud>();
  for (int scan = 0; scan < 3; ++scan) {
    for (int ring = 0; ring < 16; ++ring) {
      const float elevation = (-15.f + 2.f * ring) * M_PI / 180.f;
      for (int col = 0; col < 720; ++col) {
        const float azimuth = col * 2.f * M_PI / 720.f;
        const Eigen::AngleAxisf head(phase + 0.35f * (scan + col / 720.f), Eigen::Vector3f::UnitY());
        const Eigen::Vector3f origin = head * Eigen::Vector3f(0.1f, 0.f, 0.05f);
        const Eigen::Vector3f native_ray(std::cos(elevation) * std::cos(azimuth),
            std::cos(elevation) * std::sin(azimuth), std::sin(elevation));
        const Eigen::Vector3f ray = head * native_ray;
        float range = 100.f;
        for (int axis : {1, 2}) {
          if (std::abs(ray(axis)) > 1e-4f) {
            const float wall = std::copysign(1.5f, ray(axis));
            range = std::min(range, (wall - origin(axis)) / ray(axis));
          }
        }
        if (range > 7.f) { continue; }
        // Native return -> time-specific head transform -> common vehicle
        // frame. Different head phases have different sampling/coverage.
        const Eigen::Vector3f point = origin + head * (range * native_ray);
        pcl::PointXYZI p;
        p.getVector3fMap() = point;
        p.intensity = paint(point.x() + motion, point.z() + 0.37f * point.y(), 1);
        cloud->push_back(p);
      }
    }
  }
  return cloud;
}
}  // namespace

TEST(SurfaceTexture, ForwardReverseAndStationaryDespiteBrightnessAndSamplingChanges) {
  const auto reference = walls(0.f);
  for (float motion : {-0.15f, 0.f, 0.17f}) {
    SCOPED_TRACE(motion);
    const auto source = walls(motion, 1, 1.6f, 0.08f, 0.013f);
    const auto match = nano_gicp::matchSurfaceTexture(source, reference,
        Eigen::Isometry3f::Identity(), Eigen::Vector3f::UnitX());
    ASSERT_TRUE(match.valid) << match.candidates << ' ' << match.supported << ' ' << match.unique;
    EXPECT_GE(match.inliers, 8);
    EXPECT_NEAR(match.shift, motion, 0.015f);
    EXPECT_GT(match.correlation, 0.9f);
  }
}

TEST(SurfaceTexture, MovingHeadUsesCompensated3DSurfaces) {
  const auto reference = movingHead(0.f, -0.4f);
  for (float motion : {-0.12f, 0.f, 0.12f}) {
    SCOPED_TRACE(motion);
    const auto match = nano_gicp::matchSurfaceTexture(movingHead(motion, 0.3f), reference,
        Eigen::Isometry3f::Identity(), Eigen::Vector3f::UnitX());
    ASSERT_TRUE(match.valid) << match.candidates << ' ' << match.supported << ' ' << match.unique;
    EXPECT_NEAR(match.shift, motion, 0.025f);
  }
}

TEST(SurfaceTexture, ConstantPeriodicAndAffineRampsCannotClaimUniqueMotion) {
  for (int pattern : {0, 2, 3}) {
    SCOPED_TRACE(pattern);
    const auto match = nano_gicp::matchSurfaceTexture(walls(0.12f, pattern), walls(0.f, pattern),
        Eigen::Isometry3f::Identity(), Eigen::Vector3f::UnitX());
    EXPECT_FALSE(match.valid);
  }
}

TEST(SurfaceTexture, MissingOverlapAndSearchBoundaryFailClosed) {
  const auto reference = walls(0.f);
  for (float motion : {0.4f, 20.f}) {
    SCOPED_TRACE(motion);
    const auto match = nano_gicp::matchSurfaceTexture(walls(motion), reference,
        Eigen::Isometry3f::Identity(), Eigen::Vector3f::UnitX());
    EXPECT_FALSE(match.valid);
  }
  EXPECT_FALSE(nano_gicp::matchSurfaceTexture(nullptr, reference,
      Eigen::Isometry3f::Identity(), Eigen::Vector3f::UnitX()).valid);
}

TEST(SurfaceTexture, OpposingPatchMotionsRejectTheMeasurement) {
  auto source = walls(0.16f);
  for (auto& p : *source) {
    if (p.y > 0.f) { p.x += 0.32f; }
  }
  const auto match = nano_gicp::matchSurfaceTexture(source, walls(0.f),
      Eigen::Isometry3f::Identity(), Eigen::Vector3f::UnitX());
  EXPECT_FALSE(match.valid);
}

TEST(SurfaceTexture, CommonWorldRotationAndTranslationPreserveTheMeasurement) {
  auto source = walls(0.13f, 1, 1.f, 0.f, 0.013f), reference = walls(0.f);
  const Eigen::Matrix3f R = Eigen::AngleAxisf(0.6f, Eigen::Vector3f(1.f, 2.f, 3.f).normalized()).toRotationMatrix();
  const Eigen::Vector3f translation(35.f, -21.f, 8.f);
  for (auto cloud : {source, reference}) {
    for (auto& p : *cloud) { p.getVector3fMap() = R * p.getVector3fMap() + translation; }
  }
  const auto match = nano_gicp::matchSurfaceTexture(source, reference,
      Eigen::Isometry3f::Identity(), R.col(0));
  ASSERT_TRUE(match.valid);
  EXPECT_NEAR(match.shift, 0.13f, 0.015f);
}

TEST(SurfaceTextureConstraint, LeftPerturbationJacobianMatchesAcceptanceCost) {
  nano_gicp::SurfaceTextureConstraint constraint;
  constraint.axis = Eigen::Vector3f(1.f, 2.f, -1.f).normalized();
  constraint.center = Eigen::Vector3f(3.f, -4.f, 2.f);
  constraint.position = -0.8;
  constraint.information = 4.;
  Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
  pose.rotate(Eigen::AngleAxisf(0.2f, Eigen::Vector3f::UnitY()));
  pose.translation() = Eigen::Vector3f(0.2f, -0.1f, 0.3f);
  const auto evaluate = [&](const Eigen::Isometry3f& T, Eigen::Matrix<double, 6, 1>* gradient = nullptr) {
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
    double cost = 0.;
    constraint.accumulate(T, &H, &b, &cost);
    if (gradient) { *gradient = b; }
    return cost;
  };
  Eigen::Matrix<double, 6, 1> b;
  evaluate(pose, &b);
  constexpr float eps = 0.001f;
  for (int axis = 0; axis < 6; ++axis) {
    Eigen::Isometry3f plus = pose, minus = pose;
    if (axis < 3) {
      plus.prerotate(Eigen::AngleAxisf(eps, Eigen::Vector3f::Unit(axis)));
      minus.prerotate(Eigen::AngleAxisf(-eps, Eigen::Vector3f::Unit(axis)));
    } else {
      plus.pretranslate(eps * Eigen::Vector3f::Unit(axis - 3));
      minus.pretranslate(-eps * Eigen::Vector3f::Unit(axis - 3));
    }
    EXPECT_NEAR((evaluate(plus) - evaluate(minus)) / (2.f * eps), 2. * b(axis), 0.01);
  }
}

TEST(SurfaceTextureRegistration, MeasurementCorrectsWeakTranslationInThePoseSolver) {
  const auto reference = movingHead(0.f, -0.4f);
  auto geometry = [](const TextureCloud::Ptr& cloud) {
    auto out = std::make_shared<pcl::PointCloud<dlio::Point>>();
    for (size_t i = 0; i < cloud->size(); i += 12) {
      dlio::Point p;
      p.getVector3fMap() = (*cloud)[i].getVector3fMap();
      p.intensity = (*cloud)[i].intensity;
      out->push_back(p);
    }
    return out;
  };
  for (float motion : {-0.12f, 0.f, 0.12f}) {
    SCOPED_TRACE(motion);
    const auto source = movingHead(motion, 0.3f);
    nano_gicp::NanoGICP<dlio::Point, dlio::Point> g;
    g.setNumThreads(1);
    g.setPhotometricWeight(0.f);
    g.setRegularizationMethod(nano_gicp::RegularizationMethod::PLANE);
    g.setMaximumIterations(30);
    g.setTransformationEpsilon(1e-5f);
    g.setRotationEpsilon(1e-5f);
    g.setMaxCorrespondenceDistance(0.5f);
    g.setInputSource(geometry(source));
    g.setInputTarget(geometry(reference));
    g.setSurfaceTextureConfig({}, 100.f, 0.05f);
    g.setSurfaceTextureFrames(source, reference, Eigen::Vector3f::Zero());
    pcl::PointCloud<dlio::Point> aligned;
    Eigen::Matrix4f prior = Eigen::Matrix4f::Identity();
    prior(0, 3) = 0.03f;
    g.align(aligned, prior);
    ASSERT_TRUE(g.lastSurfaceTextureMatch().valid);
    EXPECT_NEAR(g.getFinalTransformation()(0, 3), motion, 0.025f);
    EXPECT_LT((g.getFinalTransformation().block<2, 1>(1, 3).norm()), 0.01f);
    // Fully observed geometry (ratio threshold zero) must get no constraint;
    // a previous successful scan must not leave a stale measurement behind.
    g.setSurfaceTextureConfig({}, 100.f, 0.f);
    g.align(aligned, prior);
    EXPECT_FALSE(g.lastSurfaceTextureMatch().valid);
    g.setSurfaceTextureConfig({}, 0.f, 0.05f);
    g.align(aligned, prior);
    EXPECT_FALSE(g.lastSurfaceTextureMatch().valid);
  }
}
