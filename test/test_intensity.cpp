// Unit tests for the radiometric intensity correction kernel
// (OdomNode::correctIntensity): range falloff + optional incidence-angle, the
// Kashani et al. model  I' = clamp(I * (r/r_ref)^alpha / max(cos, cos_min), 0, 255).

#include <gtest/gtest.h>

#include <cmath>

#include "dlio/odom.h"

namespace {
constexpr float kAlpha = 2.0f;   // inverse-square
constexpr float kRRef = 1.0f;    // 1 m anchor
constexpr float kCosMin = 0.2f;
}  // namespace

TEST(IntensityCorrection, RangeOnlyAtNormalIncidence) {
  // cos = 1 (normal incidence) -> pure range model I * (r/r_ref)^alpha.
  // I=10 at r=2 m -> 10 * 4 = 40.
  float got = dlio::OdomNode::correctIntensity(10.f, 2.f, 1.0f, kAlpha, kRRef, kCosMin);
  EXPECT_NEAR(got, 40.f, 1e-3);
  // at the reference range the range factor is 1.
  EXPECT_NEAR(dlio::OdomNode::correctIntensity(30.f, 1.f, 1.0f, kAlpha, kRRef, kCosMin),
              30.f, 1e-3);
}

TEST(IntensityCorrection, IncidenceBrightensGrazingReturns) {
  // Same range, more grazing (smaller cos) -> larger correction (divide by cos).
  const float r = 1.5f;
  float normal = dlio::OdomNode::correctIntensity(20.f, r, 1.0f, kAlpha, kRRef, kCosMin);
  float grazing = dlio::OdomNode::correctIntensity(20.f, r, 0.5f, kAlpha, kRRef, kCosMin);
  EXPECT_GT(grazing, normal);
  EXPECT_NEAR(grazing, normal / 0.5f, 1e-3);  // exactly 1/cos more (before clamping)
}

TEST(IntensityCorrection, CosMinFloorsGrazingBlowup) {
  // cos below the floor is clamped to cos_min (no division by ~0).
  float at_floor = dlio::OdomNode::correctIntensity(5.f, 1.f, kCosMin, kAlpha, kRRef, kCosMin);
  float below = dlio::OdomNode::correctIntensity(5.f, 1.f, 0.001f, kAlpha, kRRef, kCosMin);
  EXPECT_NEAR(below, at_floor, 1e-4);  // 0.001 clamped up to cos_min
  EXPECT_TRUE(std::isfinite(below));
}

TEST(IntensityCorrection, ClampsTo255AndGuardsBadInputs) {
  // result saturates at 255
  EXPECT_NEAR(dlio::OdomNode::correctIntensity(200.f, 4.f, 1.0f, kAlpha, kRRef, kCosMin),
              255.f, 1e-3);
  // non-positive range or r_ref returns the input unchanged (no inf/NaN)
  EXPECT_EQ(dlio::OdomNode::correctIntensity(17.f, 0.f, 1.0f, kAlpha, kRRef, kCosMin), 17.f);
  EXPECT_EQ(dlio::OdomNode::correctIntensity(17.f, 2.f, 1.0f, kAlpha, /*r_ref=*/0.f, kCosMin), 17.f);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
