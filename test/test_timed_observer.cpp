#include <gtest/gtest.h>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <random>
#include <vector>
#include "dlio/timed_observer.h"

namespace {
dlio::ObserverSnapshot initial() {
  dlio::ObserverSnapshot out;
  out.stamp = 0.;
  out.covariance.diagonal() << .0001, .0001, .0001, .000025, .000025, .000025,
      .0001, .0001, .0001, .000025, .000025, .000025, .000001, .000001, .000001;
  return out;
}
dlio::ObserverInput input(double stamp, double period = .01) {
  dlio::ObserverInput out;
  out.stamp = stamp; out.period = period;
  out.accel = Eigen::Vector3d(0., 0., 9.80665);
  out.gyro = Eigen::Vector3d(0., 0., .2);
  return out;
}
dlio::PoseCovariance noise() {
  dlio::PoseCovariance out = dlio::PoseCovariance::Zero();
  out.diagonal() << .0016, .0016, .0016, .0001, .0001, .0001;
  return out;
}
Eigen::Quaterniond yaw(double angle) {
  return Eigen::Quaterniond(Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()));
}
}

TEST(TimedObserver, ConstantAccelerationAndJointCovariance) {
  dlio::TimedObserver observer;
  observer.initialize(initial());
  for (int i = 1; i <= 200; ++i) {
    auto u = input(i*.01); u.gyro.setZero(); u.accel.x() = 1.;
    ASSERT_TRUE(observer.append(u));
  }
  const auto& result = observer.latest();
  EXPECT_NEAR(result.state.p.x(), 2., 1e-10);
  EXPECT_NEAR(result.state.v.x(), 2., 1e-10);
  EXPECT_NEAR(result.state.p.z(), 0., 1e-10);
  EXPECT_GT(std::abs(result.covariance(0, 6)), .0001);
  EXPECT_GT(std::abs(result.covariance(6, 4)), .0001);
  EXPECT_LT(result.covariance(6, 9), 0.);
  EXPECT_GT(Eigen::SelfAdjointEigenSolver<dlio::ObserverCovariance>(result.covariance).eigenvalues().minCoeff(), 0.);
}

TEST(TimedObserver, DelayedMeasurementsReplayToIdenticalStateAndCovariance) {
  dlio::TimedObserver timely, delayed;
  timely.initialize(initial()); delayed.initialize(initial());
  int a = 0, b = 0;
  const double scans[] = {.315, .715, 1.115, 1.515};
  for (int i = 1; i <= 200; ++i) {
    const auto u = input(i*.01);
    ASSERT_TRUE(timely.append(u)); ASSERT_TRUE(delayed.append(u));
    auto update = [&](dlio::TimedObserver& observer, int& k, double delay) {
      if (k < 4 && u.stamp >= scans[k]+delay) {
        ASSERT_TRUE(observer.correct(scans[k], Eigen::Vector3d(.1*scans[k], 0., 0.), yaw(.2*scans[k]), noise()));
        ++k;
      }
    };
    update(timely, a, 0.); update(delayed, b, .25);
  }
  ASSERT_EQ(a, 4); ASSERT_EQ(b, 4);
  EXPECT_LT(dlio::TimedObserver::difference(timely.latest().state, delayed.latest().state).norm(), 1e-10);
  EXPECT_TRUE(timely.latest().covariance.isApprox(delayed.latest().covariance, 1e-10));
}

TEST(TimedObserver, SplitImuSampleRetainsItsNoiseCorrelation) {
  dlio::ObserverSettings settings;
  settings.gravity = 0.; settings.accel_density = 1.; settings.gyro_density = 0.;
  settings.accel_bias_walk = settings.gyro_bias_walk = 0.;
  dlio::TimedObserver whole(settings), split(settings);
  auto start = initial(); start.covariance = 1e-12*dlio::ObserverCovariance::Identity();
  whole.initialize(start); split.initialize(start);
  dlio::ObserverInput u; u.stamp = 1.; u.period = 1.;
  ASSERT_TRUE(whole.append(u)); ASSERT_TRUE(split.append(u));
  ASSERT_TRUE(split.correct(.4, Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(), noise(), false));
  EXPECT_NEAR(split.latest().covariance(6, 6), 1., 1e-9);
  EXPECT_NEAR(split.latest().covariance(0, 0), .25, 1e-9);
  EXPECT_TRUE(split.latest().covariance.isApprox(whole.latest().covariance, 1e-8));
}

TEST(TimedObserver, NoiseAssumptionsDoNotChangeObserverMean) {
  dlio::ObserverSettings low, high;
  high.accel_density *= 10.; high.gyro_density *= 10.;
  dlio::TimedObserver a(low), b(high);
  a.initialize(initial()); b.initialize(initial());
  for (int i = 1; i <= 100; ++i) {
    ASSERT_TRUE(a.append(input(i*.01))); ASSERT_TRUE(b.append(input(i*.01)));
    if (i % 20 == 0) {
      Eigen::Vector3d p(.01*i, -.001*i, .002*i);
      ASSERT_TRUE(a.correct(i*.01, p, yaw(.01*i), noise()));
      ASSERT_TRUE(b.correct(i*.01, p, yaw(.01*i), 100.*noise()));
    }
  }
  EXPECT_EQ(dlio::TimedObserver::difference(a.latest().state, b.latest().state).norm(), 0.);
  EXPECT_GT(b.latest().covariance.trace(), a.latest().covariance.trace());
}

TEST(TimedObserver, InvalidInputsFailAndClippingInvalidatesCovariance) {
  dlio::ObserverSettings settings; settings.accel_bias_limit = .001;
  dlio::TimedObserver observer(settings);
  observer.initialize(initial());
  EXPECT_FALSE(observer.append(input(-1.)));
  ASSERT_TRUE(observer.append(input(.1)));
  EXPECT_FALSE(observer.append(input(.1)));
  dlio::ObserverSnapshot result;
  EXPECT_FALSE(observer.at(.2, result));
  EXPECT_FALSE(observer.correct(.1, Eigen::Vector3d::Ones(), yaw(0.), dlio::PoseCovariance::Zero()));
  EXPECT_FALSE(observer.latest().covariance_valid);  // rejected raw sample is latched
  observer.initialize(initial());
  ASSERT_TRUE(observer.append(input(.1)));
  ASSERT_TRUE(observer.latest().covariance_valid);
  ASSERT_TRUE(observer.correct(.1, Eigen::Vector3d::Ones(), yaw(0.), noise()));
  EXPECT_FALSE(observer.latest().covariance_valid);
}

TEST(TimedObserver, FullPoseAndTwistProjectionsMatchPerturbations) {
  auto s = initial();
  s.state.q = Eigen::Quaterniond(Eigen::AngleAxisd(.7, Eigen::Vector3d(1., 2., 3.).normalized()));
  s.state.v = Eigen::Vector3d(1., 2., -.3);
  Eigen::Matrix<double, 6, 15> pose, twist;
  constexpr double delta = 1e-6;
  for (int i = 0; i < 15; ++i) {
    dlio::ObserverVector direction = dlio::ObserverVector::Zero(); direction(i) = delta;
    auto hi = dlio::TimedObserver::plus(s.state, direction);
    auto lo = dlio::TimedObserver::plus(s.state, -direction);
    Eigen::Matrix<double, 6, 1> p_hi, p_lo, t_hi, t_lo;
    p_hi << hi.p, s.state.q*dlio::TimedObserver::difference(hi, s.state).segment<3>(3);
    p_lo << lo.p, s.state.q*dlio::TimedObserver::difference(lo, s.state).segment<3>(3);
    t_hi << hi.q.conjugate()*hi.v, -hi.bg;
    t_lo << lo.q.conjugate()*lo.v, -lo.bg;
    pose.col(i) = (p_hi-p_lo)/(2.*delta);
    twist.col(i) = (t_hi-t_lo)/(2.*delta);
  }
  EXPECT_TRUE(dlio::TimedObserver::poseCovariance(s).isApprox(pose*s.covariance*pose.transpose(), 1e-8));
  EXPECT_TRUE(dlio::TimedObserver::twistCovariance(s).isApprox(twist*s.covariance*twist.transpose(), 1e-8));
}

TEST(TimedObserver, MonteCarloMatchesDeclaredIndependentNoiseModel) {
  constexpr int count = 800;
  dlio::ObserverSettings settings; settings.accel_bias_walk = settings.gyro_bias_walk = 0.;
  dlio::TimedObserver model(settings);
  const auto start = initial();
  model.initialize(start);
  std::mt19937 generator(9017);
  std::normal_distribution<double> normal;
  std::vector<dlio::ObserverState> particles;
  for (int j = 0; j < count; ++j) {
    dlio::ObserverVector error;
    for (int k = 0; k < 15; ++k) error(k) = std::sqrt(start.covariance(k, k))*normal(generator);
    particles.push_back(dlio::TimedObserver::plus(start.state, error));
  }
  for (int i = 1; i <= 100; ++i) {
    const auto raw = input(i*.01);
    ASSERT_TRUE(model.append(raw));
    for (auto& particle : particles) {
      auto noisy = raw;
      for (int k = 0; k < 3; ++k) {
        noisy.accel(k) += settings.accel_density/std::sqrt(raw.period)*normal(generator);
        noisy.gyro(k) += settings.gyro_density/std::sqrt(raw.period)*normal(generator);
      }
      particle = dlio::TimedObserver::motion(particle, noisy, .01, settings.gravity);
    }
    if (i % 20 == 0) {
      const auto q = yaw(.2*raw.stamp);
      ASSERT_TRUE(model.correct(raw.stamp, Eigen::Vector3d::Zero(), q, noise()));
      for (auto& particle : particles) {
        Eigen::Vector3d p, rotation;
        for (int k = 0; k < 3; ++k) { p(k) = .04*normal(generator); rotation(k) = .01*normal(generator); }
        Eigen::Quaterniond measured(Eigen::AngleAxisd(rotation.norm(), rotation.normalized()));
        particle = dlio::TimedObserver::measurement(particle, p, measured*q, .2, settings);
      }
    }
  }
  dlio::ObserverCovariance empirical = dlio::ObserverCovariance::Zero();
  dlio::ObserverVector mean = dlio::ObserverVector::Zero();
  for (const auto& particle : particles) {
    const auto error = dlio::TimedObserver::difference(particle, model.latest().state);
    mean += error/count; empirical += error*error.transpose()/count;
  }
  empirical -= mean*mean.transpose();
  const auto& expected = model.latest().covariance;
  EXPECT_LT((empirical-expected).norm()/expected.norm(), .2);
  for (int i = 0; i < 15; ++i) EXPECT_NEAR(empirical(i, i)/expected(i, i), 1., .3) << i;
  const double average_nees = expected.ldlt().solve(empirical+mean*mean.transpose()).trace();
  EXPECT_GT(average_nees, 12.); EXPECT_LT(average_nees, 18.);
}
