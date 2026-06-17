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
