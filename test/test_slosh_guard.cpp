// Unit tests for the SLOSH GUARD (dlio::SloshGuard, doc/RUNTIME_GUARDS.md).
//
// The guard is the offline harness's reversal verdict (analyze_traj.py `revs`)
// moved into the estimator: a sliding-window sign-flip fraction on the signed
// per-scan output step along the weak axis, with engage/disengage hysteresis.
// These pin: (1) a straight traverse never engages, (2) a corkscrew engages,
// (3) the deadband ignores stationary noise, (4) hysteresis (no chatter at
// the boundary; disengages when the oscillation stops), (5) axis-invalid
// scans decay the window, (6) the activation counter.

#include <gtest/gtest.h>
#include "dlio/slosh_guard.h"

using dlio::SloshGuard;

TEST(SloshGuard, StraightTraverseNeverEngages) {
  SloshGuard g(20, 0.02f, 0.5f, 0.25f, 8);
  for (int i = 0; i < 100; ++i) {
    g.update(0.1f, true);                 // steady forward motion
    EXPECT_FALSE(g.engaged());
  }
  EXPECT_EQ(g.activations(), 0);
  EXPECT_FLOAT_EQ(g.reversalFraction(), 0.f);
}

TEST(SloshGuard, CorkscrewEngages) {
  SloshGuard g(20, 0.02f, 0.5f, 0.25f, 8);
  for (int i = 0; i < 12; ++i) {
    g.update((i % 2 == 0) ? 0.1f : -0.1f, true);   // alternating slosh
  }
  EXPECT_TRUE(g.engaged());
  EXPECT_GE(g.reversalFraction(), 0.9f);
  EXPECT_EQ(g.activations(), 1);
}

TEST(SloshGuard, DeadbandIgnoresStationaryNoise) {
  SloshGuard g(20, 0.02f, 0.5f, 0.25f, 8);
  for (int i = 0; i < 100; ++i) {
    g.update((i % 2 == 0) ? 0.005f : -0.005f, true);  // sub-deadband jitter
  }
  EXPECT_FALSE(g.engaged());
  EXPECT_EQ(g.activeSamples(), 0);        // nothing crossed the deadband
}

TEST(SloshGuard, MinActiveBlocksEarlyEngagement) {
  SloshGuard g(20, 0.02f, 0.5f, 0.25f, 8);
  for (int i = 0; i < 7; ++i) {           // 7 active samples < min_active 8
    g.update((i % 2 == 0) ? 0.1f : -0.1f, true);
    EXPECT_FALSE(g.engaged());            // pure alternation, but too few samples
  }
  g.update(-0.1f, true);                   // 8th active sample reaches min_active
  EXPECT_TRUE(g.engaged());
}

TEST(SloshGuard, DisengagesWhenOscillationStops) {
  SloshGuard g(10, 0.02f, 0.5f, 0.25f, 4);
  for (int i = 0; i < 10; ++i) { g.update((i % 2 == 0) ? 0.1f : -0.1f, true); }
  ASSERT_TRUE(g.engaged());
  // Straight motion refills the window with same-sign steps -> frac decays
  // below the disengage bar.
  for (int i = 0; i < 10; ++i) { g.update(0.1f, true); }
  EXPECT_FALSE(g.engaged());
  EXPECT_EQ(g.activations(), 1);           // one engage/disengage cycle
}

TEST(SloshGuard, HysteresisNoChatterAtBoundary) {
  // A flip fraction that sits BETWEEN disengage (0.25) and engage (0.5) must
  // preserve the current state, whichever it is. Blocks of three same-sign
  // steps (+ + + - - -) flip once per 3 samples -> fraction ~0.27-0.36 in a
  // 12-window: inside the hysteresis band.
  SloshGuard g(12, 0.02f, 0.5f, 0.25f, 4);
  auto feed_blocks = [&](int n) {
    for (int i = 0; i < n; ++i) { g.update(((i / 3) % 2 == 0) ? 0.1f : -0.1f, true); }
  };
  feed_blocks(24);
  EXPECT_FALSE(g.engaged());               // never crossed 0.5 -> stays off
  // Engage with a pure alternation, then feed the block pattern briefly:
  // stays ON (fraction still above the 0.25 disengage bar).
  for (int i = 0; i < 12; ++i) { g.update((i % 2 == 0) ? 0.1f : -0.1f, true); }
  ASSERT_TRUE(g.engaged());
  feed_blocks(6);
  EXPECT_TRUE(g.engaged());
}

TEST(SloshGuard, AxisInvalidDecaysWindow) {
  SloshGuard g(10, 0.02f, 0.5f, 0.25f, 4);
  for (int i = 0; i < 10; ++i) { g.update((i % 2 == 0) ? 0.1f : -0.1f, true); }
  ASSERT_TRUE(g.engaged());
  // No weak axis for a stretch: evidence decays until the verdict drops.
  for (int i = 0; i < 10; ++i) { g.update(0.f, false); }
  EXPECT_FALSE(g.engaged());
  EXPECT_EQ(g.activeSamples(), 0);
}

TEST(SloshGuard, ReengagementCountsAgain) {
  SloshGuard g(10, 0.02f, 0.5f, 0.25f, 4);
  auto slosh = [&](int n) {
    for (int i = 0; i < n; ++i) { g.update((i % 2 == 0) ? 0.1f : -0.1f, true); }
  };
  slosh(10);
  ASSERT_TRUE(g.engaged());
  for (int i = 0; i < 12; ++i) { g.update(0.1f, true); }   // straight -> disengage
  ASSERT_FALSE(g.engaged());
  slosh(11);
  EXPECT_TRUE(g.engaged());
  EXPECT_EQ(g.activations(), 2);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
