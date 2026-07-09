// SLOSH GUARD — online detector for the corkscrew / back-and-forth oscillation.
//
// The operational failure this campaign chases is the literature's
// "degradation": small-to-large back-and-forth oscillation of the odometry
// along the weak (tunnel) axis — a limit cycle powered by the observer's
// velocity state (the flywheel), fed by gate chatter and map-lock aliasing
// (doc/FUSION_ARCHITECTURE.md). The offline harness detects it as trajectory
// reversals (analyze_traj.py `revs`); this moves that detector INTO the
// estimator so the node can respond while it is happening.
//
// Mechanism: feed the signed per-scan output step projected on the tracked
// (weak) axis. Steps below `deadband` are ignored (stationary noise). Over a
// sliding window of the last `window` ACTIVE samples, the fraction of
// sign flips between consecutive active samples is the oscillation score: a
// straight traverse scores ~0 (all steps same sign), a corkscrew scores ~1
// (alternating). Hysteresis: engage at >= engage_frac, disengage at <=
// disengage_frac (or when activity dies), so the response doesn't chatter.
//
// The response is wired in odom.cc (extra velocity damping along the tracked
// axis + keyframe veto + diagnostics); this class is the pure, unit-testable
// detector (test/test_slosh_guard.cpp), no ROS / node state.
#pragma once

#include <cstddef>
#include <vector>

namespace dlio {

class SloshGuard {
 public:
  // window: number of ACTIVE (|step| > deadband) samples considered.
  // engage_frac / disengage_frac: flip-fraction hysteresis bounds.
  // min_active: minimum active samples before engagement is possible (avoids
  // engaging off 2-3 samples at startup).
  SloshGuard(int window = 20, float deadband = 0.02f,
             float engage_frac = 0.5f, float disengage_frac = 0.25f,
             int min_active = 8)
      : window_(window < 2 ? 2 : window),
        deadband_(deadband < 0.f ? 0.f : deadband),
        engage_frac_(engage_frac),
        disengage_frac_(disengage_frac),
        min_active_(min_active < 2 ? 2 : min_active) {}

  // Feed one scan's signed step along the tracked axis. axis_valid = false
  // (no weak axis this scan / axis changed identity) contributes nothing but
  // DECAYS the window so a stale oscillation verdict doesn't outlive the
  // conditions that produced it.
  void update(float step_along_axis, bool axis_valid) {
    if (!axis_valid) {
      if (!signs_.empty()) { signs_.erase(signs_.begin()); }
    } else if (step_along_axis > deadband_ || step_along_axis < -deadband_) {
      signs_.push_back(step_along_axis > 0.f ? 1 : -1);
      if (static_cast<int>(signs_.size()) > window_) { signs_.erase(signs_.begin()); }
    }
    // (active samples below deadband leave the window unchanged: a pause in
    // motion neither builds nor clears oscillation evidence.)

    const float f = reversalFraction();
    const int active = static_cast<int>(signs_.size());
    if (!engaged_) {
      if (active >= min_active_ && f >= engage_frac_) {
        engaged_ = true;
        ++activations_;
      }
    } else {
      if (active < min_active_ || f <= disengage_frac_) { engaged_ = false; }
    }
  }

  // Fraction of sign flips between consecutive active samples, in [0,1].
  float reversalFraction() const {
    if (signs_.size() < 2) { return 0.f; }
    int flips = 0;
    for (size_t i = 1; i < signs_.size(); ++i) {
      if (signs_[i] != signs_[i - 1]) { ++flips; }
    }
    return static_cast<float>(flips) / static_cast<float>(signs_.size() - 1);
  }

  bool engaged() const { return engaged_; }
  long activations() const { return activations_; }
  int activeSamples() const { return static_cast<int>(signs_.size()); }

 private:
  int window_;
  float deadband_;
  float engage_frac_;
  float disengage_frac_;
  int min_active_;
  std::vector<int> signs_;   // +/-1 per active sample, oldest first
  bool engaged_ = false;
  long activations_ = 0;
};

}  // namespace dlio
