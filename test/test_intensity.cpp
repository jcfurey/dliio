// Unit tests for the radiometric intensity correction kernel
// (OdomNode::correctIntensity): range falloff + optional incidence-angle, the
// Kashani et al. model  I' = I * (r/r_ref)^alpha / max(cos, cos_min), without display clipping.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>

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

TEST(IntensityCorrection, PreservesDynamicRangeAndGuardsBadInputs) {
  // Corrected signal may exceed both 8-bit values and the raw sensor scale.
  EXPECT_NEAR(dlio::OdomNode::correctIntensity(200.f, 4.f, 1.0f, kAlpha, kRRef, kCosMin),
              3200.f, 1e-3);
  // non-positive range or r_ref returns the input unchanged (no inf/NaN)
  EXPECT_EQ(dlio::OdomNode::correctIntensity(17.f, 0.f, 1.0f, kAlpha, kRRef, kCosMin), 17.f);
  EXPECT_EQ(dlio::OdomNode::correctIntensity(17.f, 2.f, 1.0f, kAlpha, /*r_ref=*/0.f, kCosMin), 17.f);
}

// --- denoiseOrganizedChannel: spatial box-blur of the per-point image channel ---
//
// Near-IR/ambient is shot-noise-dominated per pixel; a K x K box-blur over the
// organized grid averages the independent noise down while preserving structured
// texture. Restricted to valid-return pixels (finite x), in-place on .lidar_intensity.

namespace {
using DCloud = pcl::PointCloud<dlio::Point>;
constexpr float kNanX = std::numeric_limits<float>::quiet_NaN();

// Organized W x H cloud; points[row*W + col].lidar_intensity = refl, x = 1 (valid).
DCloud::Ptr makeOrganized(int W, int H) {
  auto c = std::make_shared<DCloud>();
  c->width = W; c->height = H; c->is_dense = false;
  c->points.resize(static_cast<size_t>(W) * H);
  for (auto& p : c->points) { p.x = 1.f; p.y = 0.f; p.z = 0.f; p.intensity = 0.f; p.lidar_intensity = 0.f; }
  return c;
}
}  // namespace

TEST(DenoiseOrganizedChannel, KernelLeOneIsNoOp) {
  const int W = 4, H = 4;
  auto c = makeOrganized(W, H);
  for (int i = 0; i < W * H; ++i) { c->points[i].lidar_intensity = 10.f + i; }  // distinct
  dlio::OdomNode::denoiseOrganizedChannel(c, W, H, 1);   // K=1 -> no-op
  for (int i = 0; i < W * H; ++i) { EXPECT_FLOAT_EQ(c->points[i].lidar_intensity, 10.f + i); }
  dlio::OdomNode::denoiseOrganizedChannel(c, W, H, 0);   // K=0 -> no-op
  for (int i = 0; i < W * H; ++i) { EXPECT_FLOAT_EQ(c->points[i].lidar_intensity, 10.f + i); }
}

TEST(DenoiseOrganizedChannel, ReducesHighFrequencyNoise) {
  // A checkerboard +/-amp around `base` is pure per-pixel high-frequency signal
  // (the shot-noise analogue). A 3x3 box-blur cancels it on interior pixels.
  const int W = 8, H = 8;
  const float base = 100.f, amp = 20.f;
  auto c = makeOrganized(W, H);
  for (int r = 0; r < H; ++r) {
    for (int col = 0; col < W; ++col) {
      c->points[r * W + col].lidar_intensity = base + (((r + col) % 2) ? -amp : amp);
    }
  }
  dlio::OdomNode::denoiseOrganizedChannel(c, W, H, 3);
  double sumsq = 0.0; int cnt = 0;
  for (int r = 1; r < H - 1; ++r) {           // interior only (full 3x3 support)
    for (int col = 1; col < W - 1; ++col) {
      const double d = c->points[r * W + col].lidar_intensity - base;
      sumsq += d * d; ++cnt;
    }
  }
  const double rms = std::sqrt(sumsq / cnt);
  EXPECT_LT(rms, amp / 4.0);                   // was amp=20; 3x3 checkerboard -> ~2.2
}

TEST(DenoiseOrganizedChannel, ExcludesInvalidNeighborsAndKeepsInvalid) {
  // 3x3, all valid=100 except the 4 edge-neighbors of the center, which are INVALID
  // (NaN x) with a poisoned value. The center average must ignore them, and an
  // invalid pixel's own slot must be left untouched.
  const int W = 3, H = 3;
  auto c = makeOrganized(W, H);
  for (auto& p : c->points) { p.lidar_intensity = 100.f; }
  auto poison = [&](int r, int col) {
    auto& p = c->points[r * W + col]; p.x = kNanX; p.lidar_intensity = 9999.f;
  };
  poison(0, 1); poison(2, 1); poison(1, 0); poison(1, 2);   // 4 edge neighbors
  dlio::OdomNode::denoiseOrganizedChannel(c, W, H, 3);
  // Center's valid support = itself + 4 corners, all 100 -> stays 100.
  EXPECT_FLOAT_EQ(c->points[1 * W + 1].lidar_intensity, 100.f);
  // An invalid pixel is skipped entirely -> left at its poisoned value.
  EXPECT_FLOAT_EQ(c->points[0 * W + 1].lidar_intensity, 9999.f);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
