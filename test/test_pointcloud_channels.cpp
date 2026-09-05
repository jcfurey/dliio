// Exercise the real intake/deskew/voxel path with wire-format PointCloud2 data,
// rather than round-tripping dlio::Point (which hides sensor datatype mismatches).
#include <gtest/gtest.h>

#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <numeric>

#include "dlio/map.h"
#include "dlio/odom.h"
#include "dlio/pointcloud_fields.h"

namespace dlio {
struct MapNodeTestAccess {
  static void ingest(MapNode& node, const pcl::PointCloud<PointType>& cloud) {
    auto msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
    pcl::toROSMsg(cloud, *msg);
    node.callbackKeyframe(msg);
  }
  static const pcl::PointCloud<PointType>& cloud(const MapNode& node) {
    return *node.dlio_map;
  }
  static bool save(MapNode& node, const std::string& directory, float leaf_size) {
    using Service = direct_lidar_inertial_odometry::srv::SavePCD;
    auto request = std::make_shared<Service::Request>();
    auto response = std::make_shared<Service::Response>();
    request->save_path = directory;
    request->leaf_size = leaf_size;
    node.savePCD(request, response);
    return response->success;
  }
};

struct OdomNodeTestAccess {
  static void ingest(OdomNode& node, const sensor_msgs::msg::PointCloud2& msg) {
    node.getScanFromROS(std::make_shared<sensor_msgs::msg::PointCloud2>(msg));
  }
  static const pcl::PointCloud<PointType>& scan(const OdomNode& node) {
    return *node.original_scan;
  }
  static const cv::Mat& image(const OdomNode& node) { return node.lidar_refl_img_; }
  static bool useReflectivity(const OdomNode& node) { return node.use_reflectivity_; }
  static bool photometricActive(const OdomNode& node) { return node.photometric_active_; }
  static void applyLiveParams(OdomNode& node) { node.applyLiveParams(); }
  static const pcl::PointCloud<PointType>& voxelize(OdomNode& node) {
    node.imu_buffer.push_back({0.0, 0.01, Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero()});
    node.preprocessPoints();
    return *node.current_scan;
  }
};
}  // namespace dlio

namespace {
using Access = dlio::OdomNodeTestAccess;
using PF = sensor_msgs::msg::PointField;

template<typename T>
void put(sensor_msgs::msg::PointCloud2& msg, size_t i, size_t offset, T value) {
  const size_t base = (i / msg.width) * msg.row_step + (i % msg.width) * msg.point_step;
  std::memcpy(msg.data.data() + base + offset, &value, sizeof(T));
}

sensor_msgs::msg::PointCloud2 ousterScan(bool native = false, bool row_padding = false) {
  sensor_msgs::msg::PointCloud2 msg;
  msg.header.stamp.sec = 10;
  msg.header.frame_id = "lidar";
  msg.width = 4;
  msg.height = 4;
  msg.point_step = 32;
  msg.row_step = msg.width * msg.point_step + (row_padding ? 8 : 0);
  msg.is_dense = true;
  auto field = [&](const char* name, int offset, int datatype) {
    PF f;
    f.name = name; f.offset = offset; f.datatype = datatype; f.count = 1;
    msg.fields.push_back(f);
  };
  field("x", 0, PF::FLOAT32); field("y", 4, PF::FLOAT32); field("z", 8, PF::FLOAT32);
  field(native ? "signal" : "intensity", 12, native ? PF::UINT16 : PF::FLOAT32);
  field("t", 16, PF::UINT32);
  field("reflectivity", 20, native ? PF::UINT8 : PF::UINT16);
  field(native ? "near_ir" : "ambient", 22, PF::UINT16);
  msg.data.resize(msg.height * msg.row_step, 0);
  for (size_t i = 0; i < msg.width * msg.height; ++i) {
    put(msg, i, 0, 4.01f + 0.01f * (i % msg.width));
    put(msg, i, 4, 2.01f + 0.01f * (i / msg.width));
    put(msg, i, 8, 1.01f);
    if (native) { put(msg, i, 12, static_cast<uint16_t>(1000 + i)); }
    else { put(msg, i, 12, 1000.f + i); }
    put(msg, i, 16, static_cast<uint32_t>((15 - i) * 1000)); // reverse deskew order
    if (native) { put(msg, i, 20, static_cast<uint8_t>(20 + i)); }
    else { put(msg, i, 20, static_cast<uint16_t>(20 + i)); }
    put(msg, i, 22, static_cast<uint16_t>(2000 + i));
  }
  return msg;
}

rclcpp::NodeOptions options(double weight = 0.0, const std::string& channel = "reflectivity") {
  rclcpp::NodeOptions opts;
  opts.append_parameter_override("odom/debug/dashboard", false);
  opts.append_parameter_override("odom/gicp/photometricWeight", weight);
  opts.append_parameter_override("odom/gicp/photometricChannel", channel);
  opts.append_parameter_override("odom/preprocessing/intensityAlpha", 0.0);
  opts.append_parameter_override("odom/gicp/photometricScale", 65535.0);
  return opts;
}

class PointCloudChannels : public ::testing::Test {
protected:
  static void SetUpTestSuite() { rclcpp::init(0, nullptr); }
  static void TearDownTestSuite() { rclcpp::shutdown(); }
};

pcl::PointCloud<PointType> mapChannelCloud() {
  pcl::PointCloud<PointType> cloud;
  for (int i = 0; i < 4; ++i) {
    PointType p;
    p.x = i < 2 ? 0.1f + 0.1f * i : 0.6f + 0.1f * (i - 2);
    p.y = p.z = 0.1f;
    p.intensity = 1000.f + 2000.f * i;
    p.reflectivity = 10.f + 20.f * i;
    p.intensity_corrected = 10000.f + 20000.f * i;
    p.lidar_intensity = 3000.f + 6000.f * i;
    p.timestamp = 0.0;
    cloud.push_back(p);
  }
  return cloud;
}

TEST_F(PointCloudChannels, AccumulatedMapPreservesRawAndDerivedChannels) {
  using MapAccess = dlio::MapNodeTestAccess;
  dlio::MapNode node;
  MapAccess::ingest(node, mapChannelCloud());
  const auto& cloud = MapAccess::cloud(node);
  ASSERT_EQ(cloud.size(), 2u);
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_FLOAT_EQ(cloud[i].intensity, 2000.f + 4000.f * i);
    EXPECT_FLOAT_EQ(cloud[i].reflectivity, 20.f + 40.f * i);
    EXPECT_FLOAT_EQ(cloud[i].intensity_corrected, 20000.f + 40000.f * i);
    EXPECT_FLOAT_EQ(cloud[i].lidar_intensity, 6000.f + 12000.f * i);
  }
}

TEST_F(PointCloudChannels, SavedPcdPreservesChannelsAfterCoarserVoxelization) {
  using MapAccess = dlio::MapNodeTestAccess;
  dlio::MapNode node;
  MapAccess::ingest(node, mapChannelCloud());
  char directory[] = "/tmp/dliio-map-channels-XXXXXX";
  ASSERT_NE(mkdtemp(directory), nullptr);
  const auto path = std::string(directory) + "/dlio_map.pcd";
  const bool saved = MapAccess::save(node, directory, 1.0f);
  pcl::PointCloud<PointType> reloaded;
  const int loaded = saved ? pcl::io::loadPCDFile(path, reloaded) : -1;
  std::filesystem::remove_all(directory);
  ASSERT_TRUE(saved);
  ASSERT_EQ(loaded, 0);
  ASSERT_EQ(reloaded.size(), 1u);
  EXPECT_FLOAT_EQ(reloaded[0].intensity, 4000.f);
  EXPECT_FLOAT_EQ(reloaded[0].reflectivity, 40.f);
  EXPECT_FLOAT_EQ(reloaded[0].intensity_corrected, 40000.f);
  EXPECT_FLOAT_EQ(reloaded[0].lidar_intensity, 12000.f);
}

TEST_F(PointCloudChannels, OriginalOusterPreservesBothFieldsWithTermsOff) {
  dlio::OdomNode node(options());
  Access::ingest(node, ousterScan());
  const auto& scan = Access::scan(node);
  ASSERT_EQ(scan.size(), 16u);
  for (size_t i = 0; i < scan.size(); ++i) {
    EXPECT_FLOAT_EQ(scan[i].intensity, 1000.f + i);
    EXPECT_FLOAT_EQ(scan[i].reflectivity, 20.f + i);
  }
  const auto& voxels = Access::voxelize(node);
  ASSERT_EQ(voxels.size(), 1u);
  EXPECT_FLOAT_EQ(voxels[0].intensity, 1007.5f);
  EXPECT_FLOAT_EQ(voxels[0].reflectivity, 27.5f);
}

TEST_F(PointCloudChannels, NativeOusterSignalAndReflectivityAreDecoded) {
  dlio::OdomNode node(options(0.3));
  Access::ingest(node, ousterScan(true));
  const auto& scan = Access::scan(node);
  ASSERT_EQ(scan.size(), 16u);
  for (size_t i = 0; i < scan.size(); ++i) {
    EXPECT_FLOAT_EQ(scan[i].intensity, 1000.f + i);
    EXPECT_FLOAT_EQ(scan[i].reflectivity, 20.f + i);
  }
}

TEST_F(PointCloudChannels, NativeLegacyUint32SignalIsDecoded) {
  auto msg = ousterScan(true);
  msg.fields[3].datatype = PF::UINT32;
  for (size_t i = 0; i < 16; ++i) { put(msg, i, 12, static_cast<uint32_t>(70000 + i)); }
  dlio::OdomNode node(options(0.3, "signal"));
  Access::ingest(node, msg);
  for (size_t i = 0; i < 16; ++i) {
    EXPECT_FLOAT_EQ(Access::scan(node)[i].intensity, 70000.f + i);
    EXPECT_FLOAT_EQ(Access::scan(node)[i].intensity_corrected, 70000.f + i);
  }
}

TEST_F(PointCloudChannels, IntensityCorrectionRespectsConfiguredSixteenBitScale) {
  dlio::OdomNode node(options(0.3, "intensity"));
  Access::ingest(node, ousterScan());
  const auto& scan = Access::scan(node);
  ASSERT_EQ(scan.size(), 16u);
  for (size_t i = 0; i < scan.size(); ++i) {
    EXPECT_FLOAT_EQ(scan[i].intensity, 1000.f + i); // alpha=0: identity correction
    EXPECT_FLOAT_EQ(scan[i].intensity_corrected, 1000.f + i);
  }
}

TEST_F(PointCloudChannels, ImageChannelIsIndependentOfPhotometricReflectivity) {
  auto opts = options(0.3);
  opts.append_parameter_override("odom/lidar_image/enabled", true);
  opts.append_parameter_override("odom/lidar_image/channel", std::string("ambient"));
  opts.append_parameter_override("odom/lidar_image/scale", 4096.0);
  dlio::OdomNode node(opts);
  Access::ingest(node, ousterScan());
  const auto& img = Access::image(node);
  ASSERT_EQ(img.rows, 4); ASSERT_EQ(img.cols, 4);
  for (int i = 0; i < 16; ++i) {
    EXPECT_FLOAT_EQ(Access::scan(node)[i].reflectivity, 20.f + i);
    EXPECT_FLOAT_EQ(img.at<float>(i / 4, i % 4), (2000.f + i) / 4096.f);
  }
}

TEST_F(PointCloudChannels, FlowOnlyUsesSelectedChannelAndPreservesReflectivity) {
  auto opts = options();
  opts.append_parameter_override("odom/lidar_image/flow/enabled", true);
  opts.append_parameter_override("odom/lidar_image/channel", std::string("ambient"));
  opts.append_parameter_override("odom/lidar_image/scale", 4096.0);
  dlio::OdomNode node(opts);
  Access::ingest(node, ousterScan(true));
  const auto& img = Access::image(node);
  ASSERT_EQ(img.rows, 4); ASSERT_EQ(img.cols, 4);
  for (int i = 0; i < 16; ++i) {
    EXPECT_FLOAT_EQ(Access::scan(node)[i].reflectivity, 20.f + i);
    EXPECT_FLOAT_EQ(img.at<float>(i / 4, i % 4), (2000.f + i) / 4096.f);
  }
  const auto& voxels = Access::voxelize(node);
  ASSERT_EQ(voxels.size(), 1u);
  EXPECT_FLOAT_EQ(voxels[0].reflectivity, 27.5f);
  EXPECT_FLOAT_EQ(voxels[0].lidar_intensity, 2007.5f);
}

TEST_F(PointCloudChannels, RangeCorrectionLeavesSensorFieldsUnmodified) {
  auto opts = options(0.0, "intensity");
  opts.append_parameter_override("odom/preprocessing/intensityAlpha", 2.0);
  dlio::OdomNode node(opts);
  Access::ingest(node, ousterScan());
  const auto& scan = Access::scan(node);
  ASSERT_EQ(scan.size(), 16u);
  for (size_t i = 0; i < scan.size(); ++i) {
    EXPECT_FLOAT_EQ(scan[i].intensity, 1000.f + i);
    EXPECT_FLOAT_EQ(scan[i].reflectivity, 20.f + i);
    EXPECT_NEAR(scan[i].intensity_corrected,
                scan[i].intensity * scan[i].getVector3fMap().squaredNorm(), 0.01f);
  }
}

TEST_F(PointCloudChannels, ChannelDecodeHonorsOrganizedRowPadding) {
  dlio::OdomNode node(options(0.3));
  Access::ingest(node, ousterScan(false, true));
  const auto& scan = Access::scan(node);
  ASSERT_EQ(scan.size(), 16u);
  for (size_t i = 0; i < scan.size(); ++i) {
    EXPECT_FLOAT_EQ(scan[i].reflectivity, 20.f + i);
  }
}

TEST_F(PointCloudChannels, FlowDenoiseDoesNotChangeSensorMeasurements) {
  auto opts = options();
  opts.append_parameter_override("odom/lidar_image/flow/enabled", true);
  opts.append_parameter_override("odom/lidar_image/channel", std::string("ambient"));
  opts.append_parameter_override("odom/lidar_image/scale", 4096.0);
  opts.append_parameter_override("odom/lidar_image/denoiseKernel", 3);
  dlio::OdomNode node(opts);
  Access::ingest(node, ousterScan());
  EXPECT_FLOAT_EQ(Access::image(node).at<float>(0, 0), 2002.5f / 4096.f);
  for (size_t i = 0; i < Access::scan(node).size(); ++i) {
    EXPECT_FLOAT_EQ(Access::scan(node)[i].reflectivity, 20.f + i);
    EXPECT_FLOAT_EQ(Access::scan(node)[i].intensity, 1000.f + i);
  }
}

TEST_F(PointCloudChannels, LiveEnableUsesChannelResolvedWhileDisabled) {
  auto msg = ousterScan();
  msg.fields.erase(msg.fields.begin() + 3); // reflectivity-only profile
  dlio::OdomNode node(options(0.0, "intensity"));
  Access::ingest(node, msg);
  EXPECT_TRUE(Access::useReflectivity(node));
  ASSERT_TRUE(node.set_parameter(rclcpp::Parameter("odom/gicp/photometricWeight", 0.3)).successful);
  Access::applyLiveParams(node);
  EXPECT_TRUE(Access::photometricActive(node));
}

TEST_F(PointCloudChannels, MissingChannelsCannotBeLiveEnabledOrCreateAnImage) {
  auto opts = options(0.0);
  opts.append_parameter_override("odom/lidar_image/enabled", true);
  auto msg = ousterScan();
  msg.fields.erase(msg.fields.begin() + 3, msg.fields.end()); // XYZ only
  dlio::OdomNode node(opts);
  Access::ingest(node, msg);
  EXPECT_TRUE(Access::image(node).empty());
  ASSERT_TRUE(node.set_parameter(rclcpp::Parameter("odom/gicp/photometricWeight", 0.3)).successful);
  Access::applyLiveParams(node);
  EXPECT_FALSE(Access::photometricActive(node));
}

TEST(PointCloudScalarField, ReadsUnalignedBigEndianValues) {
  sensor_msgs::msg::PointCloud2 msg;
  msg.width = msg.height = 1;
  msg.point_step = msg.row_step = 3;
  msg.is_bigendian = true;
  PF f; f.name = "reflectivity"; f.offset = 1; f.datatype = PF::UINT16; f.count = 1;
  msg.fields.push_back(f);
  msg.data = {0, 0x12, 0x34};
  const dlio::PointCloudScalarField field(msg, {"reflectivity"});
  ASSERT_TRUE(field);
  EXPECT_FLOAT_EQ(field[0], 4660.f);
}

TEST(PointCloudScalarField, RejectsTruncatedOrUnsupportedFields) {
  auto msg = ousterScan();
  msg.data.pop_back();
  EXPECT_FALSE(dlio::PointCloudScalarField(msg, {"reflectivity"}));
  msg = ousterScan();
  msg.fields[5].offset = msg.point_step - 1; // UINT16 runs past the point
  EXPECT_FALSE(dlio::PointCloudScalarField(msg, {"reflectivity"}));
  msg.fields[5].offset = 20;
  msg.fields[5].count = 2; // array, not a scalar channel
  EXPECT_FALSE(dlio::PointCloudScalarField(msg, {"reflectivity"}));
}

TEST_F(PointCloudChannels, InvalidLayoutIsRejectedBeforePclConversion) {
  auto msg = ousterScan();
  msg.data.resize(1);
  dlio::OdomNode node(options(0.3));
  Access::ingest(node, msg);
  EXPECT_TRUE(Access::scan(node).empty());
  EXPECT_TRUE(Access::image(node).empty());
  msg = ousterScan();
  msg.fields[5].offset = msg.point_step - 1;
  Access::ingest(node, msg);
  EXPECT_TRUE(Access::scan(node).empty());
}

TEST_F(PointCloudChannels, VoxelOverflowPreservesPassThroughChannels) {
  auto msg = ousterScan();
  // PCL refuses grids with more than INT32_MAX cells and returns the input;
  // no saved leaf layout is available for custom-channel averaging in that case.
  for (size_t i = 0; i < 16; ++i) {
    put(msg, i, 0, i % 2 ? 1000.f : -1000.f);
    put(msg, i, 4, i % 4 ? 1000.f : -1000.f);
    put(msg, i, 8, i % 8 ? 1000.f : -1000.f);
  }
  dlio::OdomNode node(options());
  Access::ingest(node, msg);
  const auto& cloud = Access::voxelize(node);
  ASSERT_EQ(cloud.size(), 16u);
  for (const auto& p : cloud) {
    const float i = 15.f - static_cast<float>(p.t / 1000);
    EXPECT_FLOAT_EQ(p.intensity, 1000.f + i);
    EXPECT_FLOAT_EQ(p.reflectivity, 20.f + i);
  }
}
}  // namespace
