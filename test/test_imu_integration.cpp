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

// --- Spline-order verification (VERIFICATION_2026-06-27) ---
// The deskew kernel is a piecewise CUBIC in position (constant jerk per IMU
// interval; Chen, Nemiroff & Lopez, ICRA 2023) -- an analytic continuous-time
// trajectory, DLIO's counterpart to the B-spline trajectories of spline-based
// CT odometry (e.g. CT-ICP, RESPLE). The tests below pin the spline properties
// the suite above did not: the cubic term itself, knot continuity at IMU sample
// boundaries, and agreement with an independent numerical integrator.

namespace {

// Wiggly stream: per-sample lin_accel from a caller-supplied profile a(t).
template <typename AccelFn>
std::vector<ImuMeas> makeProfiledStream(double duration, double rate, AccelFn a_of_t) {
  const double dt = 1.0 / rate;
  const int n = static_cast<int>(duration * rate) + 1;
  std::vector<ImuMeas> v;
  for (int k = 0; k < n; ++k) {
    ImuMeas m;
    m.stamp = k * dt;
    m.dt = dt;
    m.ang_vel = Eigen::Vector3f::Zero();
    m.lin_accel = a_of_t(k * dt);
    v.push_back(m);
  }
  return v;
}

}  // namespace

// The CUBIC term: a linear acceleration ramp a(t) = j*t is EXACTLY representable
// by the constant-jerk model, so p(t) = j*t^3/6 must be reproduced.
TEST(ImuIntegration, ConstantJerkMatchesCubicClosedForm) {
  const Eigen::Vector3f j(0.6f, -0.3f, 0.2f);   // jerk [m/s^3]
  auto buf = makeProfiledStream(1.0, 100.0, [&](double t) {
    return Eigen::Vector3f(j * static_cast<float>(t));
  });

  std::vector<double> stamps{0.3, 0.55, 0.8, 1.0};
  auto frames = dlio::OdomNode::integrateImuInternal(
      Eigen::Quaternionf::Identity(), Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero(),
      stamps, buf, /*gravity=*/0.0);

  ASSERT_EQ(frames.size(), stamps.size());
  for (size_t i = 0; i < stamps.size(); ++i) {
    const float t = static_cast<float>(stamps[i]);
    const Eigen::Vector3f expected = j * (t * t * t) / 6.f;
    const Eigen::Vector3f got = frames[i].block<3, 1>(0, 3);
    EXPECT_NEAR((got - expected).norm(), 0.f, 1e-3)
        << "t=" << t << " got " << got.transpose()
        << " expected " << expected.transpose();
  }
}

// Knot (C0) continuity at IMU sample boundaries: the piecewise polynomial's
// segment-k endpoint must equal segment-(k+1)'s start, so deskew stamps
// straddling a boundary by +-eps must be O(eps*|v|) apart -- even on a wiggly
// profile where a segment-coefficient mismatch would produce a visible jump.
TEST(ImuIntegration, KnotContinuityAcrossImuBoundaries) {
  const double dt = 1.0 / 100.0;
  auto buf = makeProfiledStream(1.0, 100.0, [](double t) {
    return Eigen::Vector3f(std::sin(7.0 * t), std::cos(5.0 * t), 0.3 * t);
  });

  const double eps = 1e-5;
  std::vector<double> stamps;
  for (int b = 20; b <= 80; b += 20) {                 // boundaries at b*dt
    stamps.push_back(b * dt - eps);
    stamps.push_back(b * dt + eps);
  }
  auto frames = dlio::OdomNode::integrateImuInternal(
      Eigen::Quaternionf::Identity(), Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero(),
      stamps, buf, 0.0);
  ASSERT_EQ(frames.size(), stamps.size());
  for (size_t i = 0; i + 1 < frames.size(); i += 2) {
    const Eigen::Vector3f pa = frames[i].block<3, 1>(0, 3);
    const Eigen::Vector3f pb = frames[i + 1].block<3, 1>(0, 3);
    EXPECT_NEAR((pb - pa).norm(), 0.f, 1e-3)
        << "position discontinuity at IMU boundary pair " << i / 2;
  }
}

// Independent cross-check: fine-step numerical integration of the SAME
// piecewise-linear a(t) (zero rotation, so world == body) must agree with the
// analytic kernel -- two independent integrators, one trajectory.
TEST(ImuIntegration, MatchesNumericalIntegrationOnWigglyProfile) {
  const double rate = 100.0, duration = 1.0, dt = 1.0 / rate;
  auto profile = [](double t) {
    return Eigen::Vector3f(std::sin(9.0 * t), -0.8 * std::cos(4.0 * t), 0.5 * t);
  };
  auto buf = makeProfiledStream(duration, rate, profile);
  const int n = static_cast<int>(buf.size());

  std::vector<double> stamps{0.33, 0.61, 0.97};
  auto frames = dlio::OdomNode::integrateImuInternal(
      Eigen::Quaternionf::Identity(), Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero(),
      stamps, buf, 0.0);
  ASSERT_EQ(frames.size(), stamps.size());

  // Numeric reference: integrate the piecewise-LINEAR interpolation of the
  // sampled accel (exactly the model the kernel assumes) with small steps.
  auto accel_at = [&](double t) -> Eigen::Vector3f {
    int k = static_cast<int>(t / dt);
    if (k > n - 2) { k = n - 2; }
    const float s = static_cast<float>((t - k * dt) / dt);
    return buf[k].lin_accel * (1.f - s) + buf[k + 1].lin_accel * s;
  };
  const double h = 1e-4;
  Eigen::Vector3d p = Eigen::Vector3d::Zero(), v = Eigen::Vector3d::Zero();
  size_t si = 0;
  double t = 0.0;
  while (si < stamps.size()) {
    double step = h;
    bool at_stamp = false;
    if (t + h >= stamps[si]) { step = stamps[si] - t; at_stamp = true; }
    const Eigen::Vector3d a0 = accel_at(t).cast<double>();
    const Eigen::Vector3d a1 = accel_at(t + step).cast<double>();
    // exact update for linear-in-t accel over [t, t+step]
    p += v * step + (a0 / 2.0 + (a1 - a0) / 6.0) * step * step;
    v += 0.5 * (a0 + a1) * step;
    t += step;
    if (at_stamp) {
      const Eigen::Vector3f got = frames[si].block<3, 1>(0, 3);
      EXPECT_NEAR((got.cast<double>() - p).norm(), 0.0, 2e-3)
          << "stamp " << stamps[si] << " analytic " << got.transpose()
          << " numeric " << p.transpose();
      ++si;
    }
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
