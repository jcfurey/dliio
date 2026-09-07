#include "dlio/timed_observer.h"
#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace dlio {
namespace {
constexpr double step = 1e-5;
Eigen::Quaterniond exponential(const Eigen::Vector3d& angle) {
  const double norm = angle.norm();
  if (norm < 1e-14) return Eigen::Quaterniond(1., .5*angle.x(), .5*angle.y(), .5*angle.z()).normalized();
  return Eigen::Quaterniond(Eigen::AngleAxisd(norm, angle/norm));
}
Eigen::Vector3d logarithm(Eigen::Quaterniond q) {
  q.normalize();
  if (q.w() < 0.) q.coeffs() *= -1.;
  const double norm = q.vec().norm();
  return q.vec()*(norm < 1e-14 ? 2. : 2.*std::atan2(norm, q.w())/norm);
}
Eigen::Matrix3d skew(const Eigen::Vector3d& x) {
  Eigen::Matrix3d m;
  m << 0., -x.z(), x.y(), x.z(), 0., -x.x(), -x.y(), x.x(), 0.;
  return m;
}
bool finite(const ObserverState& s) {
  return s.p.allFinite() && s.q.coeffs().allFinite() && std::abs(s.q.norm()-1.) < 1e-6 &&
      s.v.allFinite() && s.ba.allFinite() && s.bg.allFinite();
}
bool validCovariance(const ObserverCovariance& covariance) {
  if (!covariance.allFinite()) return false;
  const auto factor = covariance.ldlt();
  return factor.info() == Eigen::Success && factor.isPositive();
}
}

TimedObserver::TimedObserver(ObserverSettings settings) : settings_(settings) {
  for (double value : {settings.gravity, settings.kp, settings.kv, settings.kq, settings.kab, settings.kgb,
      settings.accel_bias_limit, settings.gyro_bias_limit, settings.accel_density, settings.gyro_density,
      settings.accel_bias_walk, settings.gyro_bias_walk})
    if (!std::isfinite(value) || value < 0.) throw std::invalid_argument("Invalid observer setting");
}

ObserverState TimedObserver::plus(const ObserverState& state, const ObserverVector& error) {
  ObserverState out = state;
  out.p += error.segment<3>(0);
  out.q = (out.q*exponential(error.segment<3>(3))).normalized();
  out.v += error.segment<3>(6); out.ba += error.segment<3>(9); out.bg += error.segment<3>(12);
  return out;
}

ObserverVector TimedObserver::difference(const ObserverState& state, const ObserverState& reference) {
  ObserverVector out;
  out << state.p-reference.p, logarithm(reference.q.conjugate()*state.q),
      state.v-reference.v, state.ba-reference.ba, state.bg-reference.bg;
  return out;
}

ObserverState TimedObserver::motion(const ObserverState& state, const ObserverInput& input,
                                  double dt, double gravity) {
  ObserverState out = state;
  const Eigen::Vector3d omega = input.gyro-state.bg;
  Eigen::Vector3d acceleration = state.q*(input.accel-state.ba);
  acceleration.z() -= gravity;
  out.p += state.v*dt+.5*dt*dt*acceleration;
  out.v += dt*acceleration;
  out.q = (state.q*Eigen::Quaterniond(1., .5*dt*omega.x(), .5*dt*omega.y(), .5*dt*omega.z())).normalized();
  return out;
}

ObserverState TimedObserver::measurement(const ObserverState& state, const Eigen::Vector3d& position,
    const Eigen::Quaterniond& orientation, double dt, const ObserverSettings& settings, bool* clipped) {
  ObserverState out = state;
  const Eigen::Vector3d error = position-state.p;
  const Eigen::Quaterniond qe = state.q.conjugate()*orientation;
  Eigen::Quaterniond correction;
  correction.w() = 1.-std::abs(qe.w());
  correction.vec() = (qe.w() < 0. ? -1. : 1.)*qe.vec();
  correction = state.q*correction;
  out.ba -= dt*settings.kab*(state.q.conjugate()*error);
  out.bg -= dt*settings.kgb*qe.w()*qe.vec();
  const Eigen::Vector3d ba = out.ba, bg = out.bg;
  out.ba = out.ba.array().min(settings.accel_bias_limit).max(-settings.accel_bias_limit);
  out.bg = out.bg.array().min(settings.gyro_bias_limit).max(-settings.gyro_bias_limit);
  if (clipped) *clipped = !out.ba.isApprox(ba, 1e-14) || !out.bg.isApprox(bg, 1e-14);
  out.p += dt*settings.kp*error;
  out.v += dt*settings.kv*error;
  out.q.coeffs() += dt*settings.kq*correction.coeffs();
  out.q.normalize();
  return out;
}

void TimedObserver::initialize(const ObserverSnapshot& initial) {
  if (!std::isfinite(initial.stamp) || initial.stamp < 0. || !finite(initial.state) ||
      !initial.covariance.allFinite() || !initial.covariance.isApprox(initial.covariance.transpose(), 1e-10) ||
      initial.covariance.ldlt().info() != Eigen::Success || !initial.covariance.ldlt().isPositive())
    throw std::invalid_argument("Invalid initial observer state or covariance");
  checkpoint_ = latest_ = initial;
  pending_.clear(); initialized_ = true;
}

void TimedObserver::invalidate() {
  checkpoint_.covariance_valid = latest_.covariance_valid = false;
}

void TimedObserver::predict(ObserverSnapshot& snapshot, const ObserverInput& input, double until) const {
  const double dt = until-snapshot.stamp;
  if (dt <= 0.) return;
  const ObserverState state = snapshot.state;
  const ObserverState predicted = motion(state, input, dt, settings_.gravity);
  ObserverCovariance transition;
  Eigen::Matrix<double, 15, 6> sensitivity;
  for (int i = 0; i < 15; ++i) {
    ObserverVector delta = ObserverVector::Zero(); delta(i) = step;
    transition.col(i) = (difference(motion(plus(state, delta), input, dt, settings_.gravity), predicted)-
        difference(motion(plus(state, -delta), input, dt, settings_.gravity), predicted))/(2.*step);
  }
  for (int i = 0; i < 6; ++i) {
    ObserverInput high = input, low = input;
    if (i < 3) { high.accel(i) += step; low.accel(i) -= step; }
    else { high.gyro(i-3) += step; low.gyro(i-3) -= step; }
    sensitivity.col(i) = (difference(motion(state, high, dt, settings_.gravity), predicted)-
        difference(motion(state, low, dt, settings_.gravity), predicted))/(2.*step);
  }
  PoseCovariance noise = PoseCovariance::Zero();
  noise.diagonal().head<3>().setConstant(settings_.accel_density*settings_.accel_density/input.period);
  noise.diagonal().tail<3>().setConstant(settings_.gyro_density*settings_.gyro_density/input.period);
  if (snapshot.input_stamp != input.stamp) snapshot.input_cross.setZero();
  const Eigen::Matrix<double, 15, 6> propagated_cross = transition*snapshot.input_cross;
  ObserverCovariance covariance = transition*snapshot.covariance*transition.transpose()+
      sensitivity*noise*sensitivity.transpose()-propagated_cross*sensitivity.transpose()-
      sensitivity*propagated_cross.transpose();
  covariance.diagonal().segment<3>(9).array() += settings_.accel_bias_walk*settings_.accel_bias_walk*dt;
  covariance.diagonal().segment<3>(12).array() += settings_.gyro_bias_walk*settings_.gyro_bias_walk*dt;
  snapshot.input_cross = propagated_cross-sensitivity*noise;
  snapshot.input_noise = noise;
  snapshot.input_stamp = input.stamp;
  snapshot.measured_gyro = input.gyro;
  snapshot.covariance = .5*(covariance+covariance.transpose());
  snapshot.state = predicted;
  snapshot.stamp = until;
  snapshot.covariance_valid = snapshot.covariance_valid && finite(predicted) && validCovariance(snapshot.covariance);
}

bool TimedObserver::append(const ObserverInput& input) {
  if (!initialized_ || !std::isfinite(input.stamp) || input.stamp <= latest_.stamp ||
      !std::isfinite(input.period) || input.period <= 0. || !input.accel.allFinite() || !input.gyro.allFinite()) {
    if (initialized_) invalidate();
    return false;
  }
  if (pending_.size() >= 8192) { invalidate(); return false; }
  pending_.push_back(input);
  predict(latest_, input, input.stamp);
  return true;
}

bool TimedObserver::at(double stamp, ObserverSnapshot& result) const {
  if (!initialized_ || !std::isfinite(stamp) || stamp < checkpoint_.stamp || stamp > latest_.stamp) return false;
  result = checkpoint_;
  for (const auto& input : pending_) {
    if (result.stamp >= stamp) break;
    if (input.stamp > result.stamp) predict(result, input, std::min(input.stamp, stamp));
  }
  return std::abs(result.stamp-stamp) < 1e-8;
}

bool TimedObserver::correct(double stamp, const Eigen::Vector3d& position,
    const Eigen::Quaterniond& orientation, const PoseCovariance& noise, bool use_pose) {
  ObserverSnapshot updated;
  if (!at(stamp, updated) || stamp <= checkpoint_.stamp) return false;
  if (use_pose) {
    if (!position.allFinite() || !orientation.coeffs().allFinite() || std::abs(orientation.norm()-1.) > 1e-6 ||
        !noise.allFinite() || !noise.isApprox(noise.transpose(), 1e-10) || noise.llt().info() != Eigen::Success) return false;
    const double dt = stamp-checkpoint_.stamp;
    if (dt > 1.) return false;
    const ObserverState state = updated.state;
    bool clipped = false;
    const ObserverState corrected = measurement(state, position, orientation, dt, settings_, &clipped);
    ObserverCovariance transition;
    Eigen::Matrix<double, 15, 6> sensitivity;
    for (int i = 0; i < 15; ++i) {
      ObserverVector delta = ObserverVector::Zero(); delta(i) = step;
      transition.col(i) = (difference(measurement(plus(state, delta), position, orientation, dt, settings_), corrected)-
          difference(measurement(plus(state, -delta), position, orientation, dt, settings_), corrected))/(2.*step);
    }
    for (int i = 0; i < 6; ++i) {
      Eigen::Vector3d high_p = position, low_p = position;
      Eigen::Quaterniond high_q = orientation, low_q = orientation;
      if (i < 3) { high_p(i) += step; low_p(i) -= step; }
      else {
        Eigen::Vector3d delta = Eigen::Vector3d::Zero(); delta(i-3) = step;
        // ROS pose orientation uncertainty is a fixed/world-axis tangent.
        high_q = exponential(delta)*orientation; low_q = exponential(-delta)*orientation;
      }
      sensitivity.col(i) = (difference(measurement(state, high_p, high_q, dt, settings_), corrected)-
          difference(measurement(state, low_p, low_q, dt, settings_), corrected))/(2.*step);
    }
    const ObserverCovariance covariance = transition*updated.covariance*transition.transpose()+
        sensitivity*noise*sensitivity.transpose();
    updated.covariance = .5*(covariance+covariance.transpose());
    updated.input_cross = transition*updated.input_cross;
    updated.state = corrected;
    // Bias clipping is outside the local Gaussian error model. Do not report
    // the clipped Jacobian's artificially small variance as credible.
    updated.covariance_valid = updated.covariance_valid && !clipped && finite(corrected) && validCovariance(updated.covariance);
  }
  checkpoint_ = updated;
  latest_ = checkpoint_;
  while (!pending_.empty() && pending_.front().stamp <= stamp) pending_.pop_front();
  for (const auto& input : pending_) predict(latest_, input, input.stamp);
  return true;
}

PoseCovariance TimedObserver::poseCovariance(const ObserverSnapshot& snapshot) {
  Eigen::Matrix<double, 6, 15> projection = Eigen::Matrix<double, 6, 15>::Zero();
  projection.block<3, 3>(0, 0).setIdentity();
  projection.block<3, 3>(3, 3) = snapshot.state.q.toRotationMatrix();
  return projection*snapshot.covariance*projection.transpose();
}

PoseCovariance TimedObserver::twistCovariance(const ObserverSnapshot& snapshot) {
  Eigen::Matrix<double, 6, 15> projection = Eigen::Matrix<double, 6, 15>::Zero();
  projection.block<3, 3>(0, 6) = snapshot.state.q.toRotationMatrix().transpose();
  projection.block<3, 3>(0, 3) = skew(snapshot.state.q.conjugate()*snapshot.state.v);
  projection.block<3, 3>(3, 12) = -Eigen::Matrix3d::Identity();
  PoseCovariance direct = PoseCovariance::Zero();
  direct.block<3, 3>(3, 3) = -Eigen::Matrix3d::Identity();
  PoseCovariance covariance = projection*snapshot.covariance*projection.transpose()+
      direct*snapshot.input_noise*direct.transpose()+
      projection*snapshot.input_cross*direct.transpose()+direct*snapshot.input_cross.transpose()*projection.transpose();
  return .5*(covariance+covariance.transpose());
}

}  // namespace dlio
