// Unit tests for the GenZ-ICP adaptive blend weight (nano_gicp::genzPlaneWeight).
//
// genzPlaneWeight maps a scan's geometric translation-block conditioning
// (lambda_min/lambda_max in (0,1]) to the point-to-plane weight alpha in
// [floor,1]; (1 - alpha) is the point-to-point share blended in. These pin:
// (1) the OFF / bit-identical contract (floor >= 1 -> alpha == 1 always),
// (2) healthy conditioning -> pure point-to-plane, (3) the linear ramp and the
// degenerate-axis floor, (4) input-safety (NaN / <=0 / bad knee).

#include <gtest/gtest.h>
#include "nano_gicp/genz_weight.h"

#include <cmath>
#include <limits>

using nano_gicp::genzPlaneWeight;

// (1) floor >= 1 -> feature OFF: alpha is 1 for ANY conditioning (bit-identical).
TEST(GenZWeight, FloorOneIsOff) {
  EXPECT_DOUBLE_EQ(genzPlaneWeight(1.0, 1.0, 0.1), 1.0);
  EXPECT_DOUBLE_EQ(genzPlaneWeight(0.5, 1.0, 0.1), 1.0);
  EXPECT_DOUBLE_EQ(genzPlaneWeight(1e-6, 1.0, 0.1), 1.0);   // even a collapsed axis
  EXPECT_DOUBLE_EQ(genzPlaneWeight(0.5, 2.0, 0.1), 1.0);    // floor > 1 guarded -> off
}

// (2) Conditioning at/above the knee -> pure point-to-plane (alpha == 1).
TEST(GenZWeight, HealthyIsPurePlane) {
  EXPECT_DOUBLE_EQ(genzPlaneWeight(0.10, 0.5, 0.10), 1.0);  // exactly at the knee
  EXPECT_DOUBLE_EQ(genzPlaneWeight(0.80, 0.5, 0.10), 1.0);  // well above
  EXPECT_DOUBLE_EQ(genzPlaneWeight(1.00, 0.5, 0.10), 1.0);  // isotropic
}

// (3) Below the knee, alpha ramps linearly from the floor (at ratio->0) to 1
// (at ratio == knee). Check the exact midpoint of the band.
TEST(GenZWeight, RampsLinearlyBelowKnee) {
  const double floor = 0.5, knee = 0.10;
  // ratio = knee/2 -> halfway up the ramp: alpha = floor + (1-floor)*0.5.
  EXPECT_NEAR(genzPlaneWeight(0.05, floor, knee), 0.5 + 0.5 * 0.5, 1e-12);
  // ratio = knee/4 -> a quarter up: alpha = floor + (1-floor)*0.25.
  EXPECT_NEAR(genzPlaneWeight(0.025, floor, knee), 0.5 + 0.5 * 0.25, 1e-12);
}

// (4) A fully collapsed / non-positive conditioning sits at the floor (maximally
// blended toward point-to-point), and the output never leaves [floor, 1].
TEST(GenZWeight, CollapsedAxisHitsFloor) {
  const double floor = 0.4, knee = 0.10;
  EXPECT_NEAR(genzPlaneWeight(1e-9, floor, knee), floor, 1e-6);
  EXPECT_DOUBLE_EQ(genzPlaneWeight(0.0, floor, knee), floor);
  EXPECT_DOUBLE_EQ(genzPlaneWeight(-1.0, floor, knee), floor);   // guard: ratio <= 0
}

// (5) Input safety: NaN ratio and a degenerate (<=0) knee fall back to the floor,
// never NaN, never out of range.
TEST(GenZWeight, InputSafety) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double a = genzPlaneWeight(nan, 0.5, 0.1);
  EXPECT_FALSE(std::isnan(a));
  EXPECT_DOUBLE_EQ(a, 0.5);                                  // NaN ratio -> floor
  EXPECT_DOUBLE_EQ(genzPlaneWeight(0.5, 0.5, 0.0), 0.5);     // knee <= 0 -> floor
  EXPECT_DOUBLE_EQ(genzPlaneWeight(0.5, 0.5, -1.0), 0.5);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
