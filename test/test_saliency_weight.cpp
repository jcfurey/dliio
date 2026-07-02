// Unit tests for the saliency kernel (nano_gicp::pointSaliency / saliencyMultiplier).
// Saliency classifies a point's local neighborhood (planar wall -> ~0, edge/corner
// -> ~1) from the covariance eigenvalues; the multiplier turns that into a GICP
// up-weight, off (boost <= 1) -> 1.0 / bit-identical.

#include <gtest/gtest.h>
#include "nano_gicp/saliency_weight.h"

#include <Eigen/Core>

using nano_gicp::pointSaliency;
using nano_gicp::saliencyMultiplier;

// A planar neighborhood (l0~0, l1~l2) is NOT salient (~0).
TEST(Saliency, PlaneIsNotSalient) {
  EXPECT_NEAR(pointSaliency(Eigen::Vector3f(1e-4f, 1.0f, 1.0f)), 0.0f, 1e-3f);
}

// An edge / line (l0~l1~0, l2 large) IS salient (~1) -- high linearity.
TEST(Saliency, EdgeIsSalient) {
  EXPECT_NEAR(pointSaliency(Eigen::Vector3f(1e-4f, 1e-4f, 1.0f)), 1.0f, 1e-3f);
}

// A scattered / isotropic neighborhood (l0~l1~l2, i.e. NOISE) is NOT salient:
// linearity ~0. The original 1-planarity formulation boosted scatter at full
// weight (up-weighting noise -- the sal8 stall, FINDINGS_2026-06-26); the
// linearity form leaves it at baseline (Weinmann et al. eigen-features).
TEST(Saliency, ScatterIsNotSalient) {
  EXPECT_NEAR(pointSaliency(Eigen::Vector3f(1.0f, 1.0f, 1.0f)), 0.0f, 1e-6f);
}

// Degenerate (l2 <= 0) -> 0, no div-by-zero.
TEST(Saliency, DegenerateIsZero) {
  EXPECT_EQ(pointSaliency(Eigen::Vector3f(0.f, 0.f, 0.f)), 0.0f);
  EXPECT_EQ(pointSaliency(Eigen::Vector3f(-1.f, -1.f, -1.f)), 0.0f);
}

// Output is always clamped to [0,1].
TEST(Saliency, ClampedRange) {
  const float s = pointSaliency(Eigen::Vector3f(0.3f, 0.4f, 1.0f));
  EXPECT_GE(s, 0.0f);
  EXPECT_LE(s, 1.0f);
}

// Multiplier OFF: boost <= 1 -> always 1.0 regardless of saliency (bit-identical).
TEST(Saliency, MultiplierOffIsOne) {
  EXPECT_FLOAT_EQ(saliencyMultiplier(1.0f, 1.0f), 1.0f);
  EXPECT_FLOAT_EQ(saliencyMultiplier(1.0f, 0.5f), 1.0f);
  EXPECT_FLOAT_EQ(saliencyMultiplier(0.0f, 4.0f), 1.0f);   // planar point: baseline even when on
}

// Multiplier ON: salient point gets the full boost, linear in between.
TEST(Saliency, MultiplierScalesWithSaliency) {
  EXPECT_FLOAT_EQ(saliencyMultiplier(1.0f, 4.0f), 4.0f);   // max salient -> boost
  EXPECT_FLOAT_EQ(saliencyMultiplier(0.5f, 4.0f), 2.5f);   // 1 + 3*0.5
  EXPECT_FLOAT_EQ(saliencyMultiplier(0.0f, 4.0f), 1.0f);   // planar -> 1
}

// Out-of-range saliency is clamped before scaling.
TEST(Saliency, MultiplierClampsSaliency) {
  EXPECT_FLOAT_EQ(saliencyMultiplier(2.0f, 3.0f), 3.0f);   // clamp 2->1
  EXPECT_FLOAT_EQ(saliencyMultiplier(-1.0f, 3.0f), 1.0f);  // clamp -1->0
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
