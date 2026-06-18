// Unit tests for OdomNode::detectSensorType -- the field-name -> SensorType
// mapping that drives the per-point time accessor in deskew. The Hesai/Livox
// split is the subtle bit: both carry a 'timestamp' field, disambiguated by
// magnitude (Hesai = absolute seconds < 1e14; Livox = absolute nanoseconds
// > 1e14), so it is worth pinning down explicitly.

#include <gtest/gtest.h>

#include <vector>
#include <string>

#include <sensor_msgs/msg/point_field.hpp>

#include "dlio/odom.h"

namespace {

using dlio::SensorType;
using PF = sensor_msgs::msg::PointField;

PF field(const std::string& name, uint8_t datatype = PF::FLOAT32) {
  PF f; f.name = name; f.datatype = datatype; f.offset = 0; f.count = 1;
  return f;
}

// A typical xyz + intensity prefix shared by every cloud.
std::vector<PF> base() {
  return { field("x"), field("y"), field("z"), field("intensity") };
}

std::vector<PF> with(const std::string& time_field) {
  auto v = base();
  v.push_back(field(time_field));
  return v;
}

}  // namespace

TEST(SensorDetection, OusterFromTField) {
  EXPECT_EQ(dlio::OdomNode::detectSensorType(with("t"), true, 0.0), SensorType::OUSTER);
}

TEST(SensorDetection, VelodyneFromTimeField) {
  EXPECT_EQ(dlio::OdomNode::detectSensorType(with("time"), true, 0.0), SensorType::VELODYNE);
}

TEST(SensorDetection, HesaiFromAbsoluteSecondsTimestamp) {
  // Absolute Unix seconds (~1.7e9) is well below the 1e14 split.
  EXPECT_EQ(dlio::OdomNode::detectSensorType(with("timestamp"), true, 1.7e9),
            SensorType::HESAI);
}

TEST(SensorDetection, LivoxFromAbsoluteNanosecondTimestamp) {
  // Absolute nanoseconds (~1.7e18) is above the 1e14 split.
  EXPECT_EQ(dlio::OdomNode::detectSensorType(with("timestamp"), true, 1.7e18),
            SensorType::LIVOX);
}

TEST(SensorDetection, TimestampWithoutPointsIsUnknown) {
  // No points -> the magnitude can't be inspected, so neither Hesai nor Livox.
  EXPECT_EQ(dlio::OdomNode::detectSensorType(with("timestamp"), false, 0.0),
            SensorType::UNKNOWN);
}

TEST(SensorDetection, TimestampExactlyAtThresholdIsUnknown) {
  // Boundary: strictly-less / strictly-greater, so exactly 1e14 matches neither.
  EXPECT_EQ(dlio::OdomNode::detectSensorType(with("timestamp"), true, 1e14),
            SensorType::UNKNOWN);
}

TEST(SensorDetection, PlainXyziIsUnknown) {
  EXPECT_EQ(dlio::OdomNode::detectSensorType(base(), true, 0.0), SensorType::UNKNOWN);
}

TEST(SensorDetection, EarlierFieldWins) {
  // Field order is the tie-break: a cloud carrying both 't' and 'timestamp'
  // resolves to Ouster because 't' appears first.
  std::vector<PF> v = base();
  v.push_back(field("t"));
  v.push_back(field("timestamp"));
  EXPECT_EQ(dlio::OdomNode::detectSensorType(v, true, 1.7e18), SensorType::OUSTER);
}

// --- intensity <-> reflectivity fallback (resolvePhotometricChannel) ----------
namespace {
// Returns {use_reflectivity, photometric_active} after resolution.
std::pair<bool, bool> resolve(bool has_refl, bool has_int, bool want_refl, bool active = true) {
  bool use_refl = want_refl, act = active;
  dlio::OdomNode::resolvePhotometricChannel(has_refl, has_int, use_refl, act);
  return {use_refl, act};
}
}  // namespace

TEST(PhotometricChannel, RequestedChannelPresentIsUnchanged) {
  EXPECT_EQ(resolve(/*refl*/true,  /*int*/true,  /*want_refl*/true ), std::make_pair(true,  true));
  EXPECT_EQ(resolve(/*refl*/true,  /*int*/true,  /*want_refl*/false), std::make_pair(false, true));
  EXPECT_EQ(resolve(/*refl*/true,  /*int*/false, /*want_refl*/true ), std::make_pair(true,  true));
  EXPECT_EQ(resolve(/*refl*/false, /*int*/true,  /*want_refl*/false), std::make_pair(false, true));
}

TEST(PhotometricChannel, ReflectivityFallsBackToIntensity) {
  // reflectivity requested, absent, but intensity present -> intensity, still on.
  EXPECT_EQ(resolve(/*refl*/false, /*int*/true, /*want_refl*/true), std::make_pair(false, true));
}

TEST(PhotometricChannel, IntensityFallsBackToReflectivity) {
  // intensity requested, absent, but reflectivity present -> reflectivity, still on.
  EXPECT_EQ(resolve(/*refl*/true, /*int*/false, /*want_refl*/false), std::make_pair(true, true));
}

TEST(PhotometricChannel, NeitherFieldDisablesTheTerm) {
  EXPECT_EQ(resolve(/*refl*/false, /*int*/false, /*want_refl*/true ), std::make_pair(true,  false));
  EXPECT_EQ(resolve(/*refl*/false, /*int*/false, /*want_refl*/false), std::make_pair(false, false));
}

TEST(PhotometricChannel, InactiveTermIsLeftUntouched) {
  // Term off (weight 0): never touched, regardless of fields.
  EXPECT_EQ(resolve(/*refl*/false, /*int*/false, /*want_refl*/true, /*active*/false),
            std::make_pair(true, false));
}

// --- Sub-floor reject keep-mask (specular ghost removal) ---
// Helper: append `n` points at (x,y,z) jittered within a cell into the arrays.
namespace {
void addCluster(std::vector<float>& xs, std::vector<float>& ys, std::vector<float>& zs,
                float x, float y, float z, int n) {
  for (int i = 0; i < n; ++i) { xs.push_back(x); ys.push_back(y); zs.push_back(z); }
}
}  // namespace

TEST(SubFloorReject, DropsGhostsBelowFloorKeepsFloor) {
  std::vector<float> xs, ys, zs;
  addCluster(xs, ys, zs, 0.5f, 0.5f, 0.0f, 20);   // dense floor at z=0 (cell 0,0)
  addCluster(xs, ys, zs, 0.5f, 0.5f, -2.0f, 3);   // sparse ghosts 2 m below
  auto keep = dlio::OdomNode::subFloorKeepMask(xs, ys, zs, 0.f, 0.f,
      /*radius*/20.f, /*cell*/1.f, /*z_bin*/0.1f, /*min_bin*/8, /*margin*/0.3f, /*maxFrac*/0.5f);
  for (int i = 0; i < 20; ++i) { EXPECT_EQ(keep[i], 1) << "floor point " << i; }
  for (int i = 20; i < 23; ++i) { EXPECT_EQ(keep[i], 0) << "ghost point " << i; }
}

TEST(SubFloorReject, FloorEstimateNotDraggedDownBySparseGhosts) {
  // Even with ghosts present, the floor is the dense bin, so floor points at z=0
  // are never rejected (the ghosts can't pull the cell floor to -2).
  std::vector<float> xs, ys, zs;
  addCluster(xs, ys, zs, 0.5f, 0.5f, 0.0f, 30);
  addCluster(xs, ys, zs, 0.5f, 0.5f, -1.5f, 5);   // still < min_bin (8)
  auto keep = dlio::OdomNode::subFloorKeepMask(xs, ys, zs, 0.f, 0.f,
      20.f, 1.f, 0.1f, 8, 0.3f, 0.5f);
  for (int i = 0; i < 30; ++i) { EXPECT_EQ(keep[i], 1); }
  for (int i = 30; i < 35; ++i) { EXPECT_EQ(keep[i], 0); }
}

TEST(SubFloorReject, NoDenseFloorKeepsEverything) {
  // Sparse scene: no cell reaches min_bin -> no floor estimated -> keep all,
  // including a low point that is NOT a confirmed ghost.
  std::vector<float> xs, ys, zs;
  for (int i = 0; i < 5; ++i) { addCluster(xs, ys, zs, 0.5f + i, 0.5f, 0.0f, 1); }
  addCluster(xs, ys, zs, 0.5f, 0.5f, -3.0f, 1);
  auto keep = dlio::OdomNode::subFloorKeepMask(xs, ys, zs, 0.f, 0.f,
      20.f, 1.f, 0.1f, 8, 0.3f, 0.5f);
  for (size_t i = 0; i < keep.size(); ++i) { EXPECT_EQ(keep[i], 1); }
}

TEST(SubFloorReject, SafetyCapDisablesOverlyAggressiveReject) {
  // If the reject would exceed maxRejectFrac, drop nothing (floor estimate is
  // suspect, not the scan).
  std::vector<float> xs, ys, zs;
  addCluster(xs, ys, zs, 0.5f, 0.5f, 0.0f, 20);
  addCluster(xs, ys, zs, 0.5f, 0.5f, -2.0f, 6);   // 6/26 ~ 0.23
  auto keep = dlio::OdomNode::subFloorKeepMask(xs, ys, zs, 0.f, 0.f,
      20.f, 1.f, 0.1f, 8, 0.3f, /*maxFrac*/0.10f);   // 0.23 > 0.10 -> no-op
  for (size_t i = 0; i < keep.size(); ++i) { EXPECT_EQ(keep[i], 1); }
}

TEST(SubFloorReject, PointsOutsideRadiusAreKept) {
  std::vector<float> xs, ys, zs;
  addCluster(xs, ys, zs, 0.5f, 0.5f, 0.0f, 20);     // floor near origin
  addCluster(xs, ys, zs, 100.f, 100.f, -5.0f, 1);   // far ghost, outside radius
  auto keep = dlio::OdomNode::subFloorKeepMask(xs, ys, zs, 0.f, 0.f,
      /*radius*/8.f, 1.f, 0.1f, 8, 0.3f, 0.5f);
  EXPECT_EQ(keep.back(), 1);   // far point kept (not analyzed)
}

TEST(SubFloorReject, MarginProtectsPointsJustBelowFloor) {
  std::vector<float> xs, ys, zs;
  addCluster(xs, ys, zs, 0.5f, 0.5f, 0.0f, 20);
  addCluster(xs, ys, zs, 0.5f, 0.5f, -0.20f, 1);    // within margin 0.3 -> kept
  addCluster(xs, ys, zs, 0.5f, 0.5f, -0.50f, 1);    // beyond margin     -> dropped
  auto keep = dlio::OdomNode::subFloorKeepMask(xs, ys, zs, 0.f, 0.f,
      20.f, 1.f, 0.1f, 8, 0.3f, 0.5f);
  EXPECT_EQ(keep[20], 1);   // -0.20 m
  EXPECT_EQ(keep[21], 0);   // -0.50 m
}

TEST(SubFloorReject, EmptyInputIsSafe) {
  std::vector<float> xs, ys, zs;
  auto keep = dlio::OdomNode::subFloorKeepMask(xs, ys, zs, 0.f, 0.f, 8.f, 1.f, 0.1f, 8, 0.3f, 0.5f);
  EXPECT_TRUE(keep.empty());
}
