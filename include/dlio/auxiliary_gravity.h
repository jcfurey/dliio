#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <mutex>

namespace dlio {

// World +Z expressed in the body frame. No navigation yaw is represented.
// This buffer only interpolates between received, nearby acquisition stamps.
class GravityBuffer {
 public:
  struct Sample { int64_t stamp_ns; Eigen::Vector3d up_body; };
  bool push(int64_t stamp, const Eigen::Vector3d& direction) {
    const double norm = direction.norm();
    if (stamp < 0 || !direction.allFinite() || std::abs(norm-1.) > .01) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    if (!samples_.empty() && stamp <= samples_.back().stamp_ns) return false;
    samples_.push_back({stamp, direction/norm});
    while (samples_.size() > 2000 || samples_.back().stamp_ns-samples_.front().stamp_ns > 5000000000)
      samples_.pop_front();
    return true;
  }
  bool at(int64_t stamp, int64_t max_gap_ns, Eigen::Vector3d& direction,
          int64_t& lower_ns, int64_t& upper_ns) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto upper = std::lower_bound(samples_.begin(), samples_.end(), stamp,
        [](const Sample& sample, int64_t query) { return sample.stamp_ns < query; });
    if (upper == samples_.end()) return false;
    if (upper->stamp_ns == stamp) {
      direction = upper->up_body; lower_ns = upper_ns = stamp; return true;
    }
    if (upper == samples_.begin()) return false;
    const auto lower = std::prev(upper);
    const int64_t gap = upper->stamp_ns-lower->stamp_ns;
    if (gap > max_gap_ns || gap <= 0) return false;
    const double fraction = double(stamp-lower->stamp_ns)/gap;
    direction = (1.-fraction)*lower->up_body+fraction*upper->up_body;
    if (direction.norm() < .5) return false;
    direction.normalize();
    lower_ns = lower->stamp_ns; upper_ns = upper->stamp_ns;
    return true;
  }
 private:
  mutable std::mutex mutex_;
  std::deque<Sample> samples_;
};

struct GravityCorrection {
  Eigen::Quaterniond orientation;
  double disagreement = 0., applied_angle = 0.;
  bool accepted = false;
};

// A bounded, deterministic tilt correction, not a calibrated Kalman update.
// The shortest left rotation maps the measured world-up direction toward +Z.
// Its instantaneous axis is horizontal: heading is not measured or fused.
inline GravityCorrection correctGravity(const Eigen::Quaterniond& orientation,
    const Eigen::Vector3d& up_body, double dt, double gain, double max_rate,
    double max_disagreement) {
  GravityCorrection result{orientation};
  if (!orientation.coeffs().allFinite() || std::abs(orientation.norm()-1.) > .01 ||
      !up_body.allFinite() || std::abs(up_body.norm()-1.) > .01 ||
      !std::isfinite(dt) || !std::isfinite(gain) || !std::isfinite(max_rate) ||
      !std::isfinite(max_disagreement) || dt <= 0. || dt > 1. || gain < 0. || max_rate < 0.) return result;
  const Eigen::Vector3d up_world = orientation.normalized()*up_body.normalized();
  const Eigen::Vector3d axis = up_world.cross(Eigen::Vector3d::UnitZ());
  const double sine = axis.norm();
  result.disagreement = std::atan2(sine, up_world.z());
  if (result.disagreement > max_disagreement || result.disagreement > 1.5707963267948966) return result;
  result.accepted = true;
  if (sine < 1e-12 || gain == 0. || max_rate == 0.) return result;
  result.applied_angle = std::min(-std::expm1(-gain*dt)*result.disagreement, max_rate*dt);
  result.orientation = (Eigen::Quaterniond(Eigen::AngleAxisd(result.applied_angle, axis/sine))*orientation).normalized();
  return result;
}

}  // namespace dlio
