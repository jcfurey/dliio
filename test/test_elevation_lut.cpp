// Unit tests for the per-row elevation LUT (nano_gicp/elevation_lut.h).
//
// This code was previously file-local in nano_gicp.cc with NO direct tests, and
// duplicated by hand in the node's keyframe-reference sampler. The 2026-07-26
// audit found the copies gated on DIFFERENT validity rules, so a LUT with a
// single NaN row split the two sides onto different projection models. These
// tests pin the validity predicate (the thing that was missing) and the
// inversion's contract, so the shared version cannot regress.
//
// Why a NaN row is realistic: the node builds the LUT once, from the first
// organized scan, filling only rows that had a return. A beam seeing sky/void
// leaves quiet_NaN in that row for the rest of the run.

#include <gtest/gtest.h>
#include "nano_gicp/elevation_lut.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using nano_gicp::elevationLutUsable;
using nano_gicp::rowFromElevationLut;

namespace {

// Increasing, non-uniformly spaced (an OS-series beam table in miniature).
std::vector<float> makeIncreasing() {
  return {-0.40f, -0.25f, -0.15f, -0.08f, 0.00f, 0.09f, 0.20f, 0.36f};
}

std::vector<float> makeDecreasing() {
  auto v = makeIncreasing();
  std::reverse(v.begin(), v.end());
  return v;
}

}  // namespace

// ---- validity predicate ------------------------------------------------

TEST(ElevationLut, AcceptsWellFormedTables) {
  const auto inc = makeIncreasing();
  const auto dec = makeDecreasing();
  EXPECT_TRUE(elevationLutUsable(inc, static_cast<int>(inc.size())));
  EXPECT_TRUE(elevationLutUsable(dec, static_cast<int>(dec.size())));
}

// THE regression this file exists for: one NaN row must make the whole table
// unusable, so every consumer falls back to the linear model together.
TEST(ElevationLut, RejectsAnyNaNRow) {
  auto lut = makeIncreasing();
  lut[3] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(elevationLutUsable(lut, static_cast<int>(lut.size())));

  // ... including the endpoints, which is what the orientation test reads.
  auto first = makeIncreasing();
  first[0] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(elevationLutUsable(first, static_cast<int>(first.size())));
  auto last = makeIncreasing();
  last.back() = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(elevationLutUsable(last, static_cast<int>(last.size())));
}

TEST(ElevationLut, RejectsInfiniteRow) {
  auto lut = makeIncreasing();
  lut[5] = std::numeric_limits<float>::infinity();
  EXPECT_FALSE(elevationLutUsable(lut, static_cast<int>(lut.size())));
}

TEST(ElevationLut, RejectsSizeMismatchAndTooSmall) {
  const auto lut = makeIncreasing();
  EXPECT_FALSE(elevationLutUsable(lut, static_cast<int>(lut.size()) + 1));
  EXPECT_FALSE(elevationLutUsable(lut, 0));
  EXPECT_FALSE(elevationLutUsable({}, 0));
  EXPECT_FALSE(elevationLutUsable({0.1f}, 1));           // needs >= 2 entries
}

// Flat and non-monotonic tables are rejected: the inversion picks its search
// orientation from the endpoints and scans intervals, which is unsound on a
// table that reverses direction (and a flat segment has no invertible slope).
TEST(ElevationLut, RejectsFlatAndNonMonotonic) {
  auto flat = makeIncreasing();
  flat[4] = flat[3];                                     // equal adjacent rows
  EXPECT_FALSE(elevationLutUsable(flat, static_cast<int>(flat.size())));

  auto zig = makeIncreasing();
  zig[5] = zig[3] - 0.01f;                               // dips back down
  EXPECT_FALSE(elevationLutUsable(zig, static_cast<int>(zig.size())));

  std::vector<float> constant(8, 0.1f);
  EXPECT_FALSE(elevationLutUsable(constant, 8));
}

// ---- inversion ---------------------------------------------------------

TEST(ElevationLut, InvertsExactlyAtKnots) {
  const auto lut = makeIncreasing();
  for (size_t k = 0; k < lut.size(); ++k) {
    float row = -1.f, slope = 0.f;
    ASSERT_TRUE(rowFromElevationLut(lut, lut[k], row, slope)) << "knot " << k;
    EXPECT_NEAR(row, static_cast<float>(k), 1e-4f) << "knot " << k;
  }
}

TEST(ElevationLut, InterpolatesMidSegmentWithLocalSlope) {
  const auto lut = makeIncreasing();
  const float mid = 0.5f * (lut[2] + lut[3]);            // between rows 2 and 3
  float row = -1.f, slope = 0.f;
  ASSERT_TRUE(rowFromElevationLut(lut, mid, row, slope));
  EXPECT_NEAR(row, 2.5f, 1e-4f);
  EXPECT_NEAR(slope, lut[3] - lut[2], 1e-6f);            // rad/row, this segment
  // Non-uniform beams: the local slope must differ between segments, which is
  // the whole reason the LUT exists rather than a single linear el(row).
  float row2 = 0.f, slope2 = 0.f;
  ASSERT_TRUE(rowFromElevationLut(lut, 0.5f * (lut[6] + lut[7]), row2, slope2));
  EXPECT_GT(std::abs(slope2 - slope), 1e-3f);
}

TEST(ElevationLut, HandlesDecreasingTables) {
  const auto lut = makeDecreasing();
  for (size_t k = 0; k < lut.size(); ++k) {
    float row = -1.f, slope = 0.f;
    ASSERT_TRUE(rowFromElevationLut(lut, lut[k], row, slope)) << "knot " << k;
    EXPECT_NEAR(row, static_cast<float>(k), 1e-4f) << "knot " << k;
  }
  float row = -1.f, slope = 0.f;
  const float mid = 0.5f * (lut[1] + lut[2]);
  ASSERT_TRUE(rowFromElevationLut(lut, mid, row, slope));
  EXPECT_NEAR(row, 1.5f, 1e-4f);
  EXPECT_LT(slope, 0.f);                                 // decreasing with row
}

// Out-of-coverage elevations must be REJECTED, not extrapolated: the caller
// drops the point rather than projecting it to a fabricated row.
TEST(ElevationLut, RejectsOutOfCoverage) {
  const auto lut = makeIncreasing();
  float row = -99.f, slope = 0.f;
  EXPECT_FALSE(rowFromElevationLut(lut, lut.front() - 0.05f, row, slope));
  EXPECT_FALSE(rowFromElevationLut(lut, lut.back() + 0.05f, row, slope));
  EXPECT_FLOAT_EQ(row, -99.f);                           // outputs untouched
}

TEST(ElevationLut, NaNElevationIsRejectedNotMatched) {
  const auto lut = makeIncreasing();
  float row = -99.f, slope = 0.f;
  EXPECT_FALSE(rowFromElevationLut(lut, std::numeric_limits<float>::quiet_NaN(),
                                   row, slope));
  EXPECT_FALSE(rowFromElevationLut(lut, std::numeric_limits<float>::infinity(),
                                   row, slope));
  EXPECT_FLOAT_EQ(row, -99.f);
}

TEST(ElevationLut, DegenerateTableIsSafe) {
  float row = 0.f, slope = 0.f;
  EXPECT_FALSE(rowFromElevationLut({}, 0.1f, row, slope));
  EXPECT_FALSE(rowFromElevationLut({0.1f}, 0.1f, row, slope));
}

// A usable table must always invert its own knots -- the property the node
// relies on when it gates every consumer on elevationLutUsable().
TEST(ElevationLut, UsableImpliesAllKnotsInvertible) {
  for (const auto& lut : {makeIncreasing(), makeDecreasing()}) {
    ASSERT_TRUE(elevationLutUsable(lut, static_cast<int>(lut.size())));
    for (size_t k = 0; k < lut.size(); ++k) {
      float row = -1.f, slope = 0.f;
      EXPECT_TRUE(rowFromElevationLut(lut, lut[k], row, slope));
      EXPECT_GT(std::abs(slope), 0.f);                   // invertible slope
    }
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
