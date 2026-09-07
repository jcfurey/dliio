#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <deque>

namespace dlio {

// Error coordinates: world position, right/body attitude tangent, world
// velocity, body accelerometer bias, body gyro bias. This is a conditional
// noise model: registration/map and sensor correlations require separate care.
using ObserverVector = Eigen::Matrix<double, 15, 1>;
using ObserverCovariance = Eigen::Matrix<double, 15, 15>;
using PoseCovariance = Eigen::Matrix<double, 6, 6>;

struct ObserverState {
  Eigen::Vector3d p = Eigen::Vector3d::Zero();
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
  Eigen::Vector3d v = Eigen::Vector3d::Zero();
  Eigen::Vector3d ba = Eigen::Vector3d::Zero(), bg = Eigen::Vector3d::Zero();
};

struct ObserverInput {
  double stamp = 0., period = 0.;
  Eigen::Vector3d accel = Eigen::Vector3d::Zero(), gyro = Eigen::Vector3d::Zero();
};

struct ObserverSettings {
  double gravity = 9.80665;
  double kp = 2.25, kv = 2.8125, kq = 2., kab = .28125, kgb = .25;
  double accel_bias_limit = 10., gyro_bias_limit = 1.;
  // Declared experiment assumptions. Densities per sqrt(Hz), bias random
  // walks per sqrt(s); these are not sensor calibration from the tunnel bag.
  double accel_density = .03, gyro_density = .002;
  double accel_bias_walk = .0005, gyro_bias_walk = .00005;
};

struct ObserverSnapshot {
  double stamp = -1.;
  ObserverState state;
  ObserverCovariance covariance = ObserverCovariance::Zero();
  // Correlation with the one IMU sample that may straddle a scan timestamp.
  // Retaining this avoids counting its noise as independent on both sides.
  Eigen::Matrix<double, 15, 6> input_cross = Eigen::Matrix<double, 15, 6>::Zero();
  PoseCovariance input_noise = PoseCovariance::Zero();
  double input_stamp = -1.;
  Eigen::Vector3d measured_gyro = Eigen::Vector3d::Zero();
  bool covariance_valid = true;
};

// The same geometric observer gains, applied at acquisition time. A scan
// update advances the preceding scan checkpoint, corrects there, then replays
// retained RAW IMU samples (with the new biases) to the latest publication time.
// P follows these actual operations; it never changes the observer gains.
class TimedObserver {
 public:
  explicit TimedObserver(ObserverSettings settings = {});
  void initialize(const ObserverSnapshot& initial);
  bool initialized() const { return initialized_; }
  void invalidate();  // latch until explicit reinitialization
  bool append(const ObserverInput& input);
  bool at(double stamp, ObserverSnapshot& result) const;
  bool correct(double stamp, const Eigen::Vector3d& position,
               const Eigen::Quaterniond& orientation, const PoseCovariance& noise,
               bool use_pose = true);
  const ObserverSnapshot& latest() const { return latest_; }
  const ObserverSnapshot& scan() const { return checkpoint_; }
  size_t retainedInputs() const { return pending_.size(); }
  static ObserverState motion(const ObserverState&, const ObserverInput&, double dt, double gravity);
  static ObserverState measurement(const ObserverState&, const Eigen::Vector3d&,
      const Eigen::Quaterniond&, double dt, const ObserverSettings&, bool* clipped = nullptr);
  static ObserverState plus(const ObserverState&, const ObserverVector&);
  static ObserverVector difference(const ObserverState&, const ObserverState& reference);
  static PoseCovariance poseCovariance(const ObserverSnapshot&);
  static PoseCovariance twistCovariance(const ObserverSnapshot&);
 private:
  void predict(ObserverSnapshot&, const ObserverInput&, double until) const;
  ObserverSettings settings_;
  bool initialized_ = false;
  ObserverSnapshot checkpoint_, latest_;
  std::deque<ObserverInput> pending_;
};

}  // namespace dlio
