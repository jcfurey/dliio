// Unit tests for the continuous-time IMU integration kernel
// (OdomNode::integrateImuInternal) -- the analytic constant-jerk /
// constant-angular-acceleration deskew math from the DLIO paper -- against
// closed-form trajectories.

#include <gtest/gtest.h>

#include <cmath>

#include "dlio/odom.h"

namespace {

using ImuMeas = dlio::OdomNode::ImuMeas;

// IMU stream sampled at `rate` Hz from t=0 to t=duration with constant body
// angular velocity and constant (world == body for identity attitude tests)
// linear acceleration. Forward-time order (oldest at front) -- this matches the
// private, forward-time copy imuMeasFromTimeRange hands integrateImuInternal in
// production.
std::vector<ImuMeas> makeImuStream(double duration, double rate,
                                   const Eigen::Vector3f& ang_vel,
                                   const Eigen::Vector3f& lin_accel) {
  const double dt = 1.0 / rate;
  const int n = static_cast<int>(duration * rate) + 1;
  std::vector<ImuMeas> v;
  for (int k = 0; k < n; k++) {
    ImuMeas m;
    m.stamp = k * dt;
    m.dt = dt;
    m.ang_vel = ang_vel;
    m.lin_accel = lin_accel;
    v.push_back(m);
  }
  return v;
}

}  // namespace

TEST(ImuIntegration, ConstantAccelerationMatchesClosedForm) {
  // a = (1, 0, 0) m/s^2, zero rotation, zero gravity:
  // p(t) = 0.5 * a * t^2, exactly representable by the constant-jerk model.
  const Eigen::Vector3f a(1.f, 0.f, 0.f);
  auto buf = makeImuStream(1.0, 100.0, Eigen::Vector3f::Zero(), a);

  std::vector<double> stamps{0.25, 0.5, 0.75, 1.0};
  auto frames = dlio::OdomNode::integrateImuInternal(
      Eigen::Quaternionf::Identity(), Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero(),
      stamps, buf, /*gravity=*/0.0);

  ASSERT_EQ(frames.size(), stamps.size());
  for (size_t i = 0; i < stamps.size(); i++) {
    const double t = stamps[i];
    Eigen::Vector3f expected = 0.5f * a * static_cast<float>(t * t);
    Eigen::Vector3f got = frames[i].block<3, 1>(0, 3);
    EXPECT_NEAR((got - expected).norm(), 0.f, 2e-3)
        << "t=" << t << " got " << got.transpose()
        << " expected " << expected.transpose();
    // attitude must stay identity
    Eigen::AngleAxisf rot(Eigen::Matrix3f(frames[i].block<3, 3>(0, 0)));
    EXPECT_NEAR(rot.angle(), 0.f, 1e-4);
  }
}

TEST(ImuIntegration, ConstantAngularVelocityMatchesClosedForm) {
  // omega = (0, 0, 0.5) rad/s, no acceleration, zero gravity:
  // attitude(t) = Rz(omega * t), position stays at the origin.
  const double w = 0.5;
  auto buf = makeImuStream(1.0, 100.0, Eigen::Vector3f(0.f, 0.f, w),
                           Eigen::Vector3f::Zero());

  std::vector<double> stamps{0.2, 0.6, 1.0};
  auto frames = dlio::OdomNode::integrateImuInternal(
      Eigen::Quaternionf::Identity(), Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero(),
      stamps, buf, /*gravity=*/0.0);

  ASSERT_EQ(frames.size(), stamps.size());
  for (size_t i = 0; i < stamps.size(); i++) {
    const double t = stamps[i];
    Eigen::Matrix3f R = frames[i].block<3, 3>(0, 0);
    Eigen::Matrix3f R_expected =
        Eigen::AngleAxisf(static_cast<float>(w * t), Eigen::Vector3f::UnitZ()).toRotationMatrix();
    Eigen::AngleAxisf err(Eigen::Matrix3f(R.transpose() * R_expected));
    EXPECT_NEAR(err.angle(), 0.f, 2e-3) << "t=" << t;
    float drift = frames[i].block<3, 1>(0, 3).norm();
    EXPECT_NEAR(drift, 0.f, 1e-4);
  }
}

TEST(ImuIntegration, GravityIsSubtracted) {
  // Stationary IMU measuring +g on z (specific force), gravity passed in:
  // the integrated position must stay at the origin.
  const double g = 9.80665;
  auto buf = makeImuStream(1.0, 100.0, Eigen::Vector3f::Zero(),
                           Eigen::Vector3f(0.f, 0.f, static_cast<float>(g)));

  std::vector<double> stamps{0.5, 1.0};
  auto frames = dlio::OdomNode::integrateImuInternal(
      Eigen::Quaternionf::Identity(), Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero(),
      stamps, buf, g);

  ASSERT_EQ(frames.size(), stamps.size());
  for (const auto& T : frames) {
    float drift = T.block<3, 1>(0, 3).norm();
    EXPECT_NEAR(drift, 0.f, 1e-4);
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
