// Regression coverage for consumers that build maps from registered scans.
#include <gtest/gtest.h>

#include <chrono>
#include <cstring>
#include <future>
#include <limits>
#include <thread>
#include <tf2_msgs/msg/tf_message.hpp>

#include "dlio/map.h"
#include "dlio/odom.h"
#include "dlio/pointcloud_fields.h"

namespace dlio {
struct OdomNodeTestAccess {
  using Output = OdomNode::ScanOutput;
  using KeyframeOutput = OdomNode::KeyframeOutput;
  static Output capture(OdomNode& n, pcl::PointCloud<PointType>::ConstPtr cloud) {
    n.scan_stamp = 10.05;
    n.scan_header_stamp = rclcpp::Time(10, 0, RCL_ROS_TIME);
    n.lidarPose.p = Eigen::Vector3f(3, 4, 5);
    n.lidarPose.q.setIdentity();
    n.T_corr.setIdentity(); n.T_corr(0, 3) = 2.f;
    n.wait_until_move_ = true; n.length_traversed = 1.;
    return n.snapshotScanOutput(cloud);
  }
  static void advance(OdomNode& n) {
    n.scan_stamp = 20.05;
    n.scan_header_stamp = rclcpp::Time(20, 0, RCL_ROS_TIME);
    n.lidarPose.p.setConstant(100.f); n.T_corr(0, 3) = 50.f;
    n.state.p.setConstant(200.f); n.imu_stamp = rclcpp::Time(21, 0, RCL_ROS_TIME);
    n.length_traversed = 0.;
    n.T_prior(0, 3) = 100.f;
  }
  static bool scanConnected(OdomNode& n) {
    return n.deskewed_pub->get_subscription_count() && n.scan_pose_pub->get_subscription_count();
  }
  static bool keyframeConnected(OdomNode& n) {
    return n.kf_cloud_pub->get_subscription_count() && n.kf_stamped_pose_pub->get_subscription_count();
  }
  static void publish(OdomNode& n, Output out) { n.publishToROS(std::move(out)); }
  static const nav_msgs::msg::Path& path(OdomNode& n) { return n.path_ros; }
  static void keyframe(OdomNode& n, pcl::PointCloud<PointType>::ConstPtr cloud) {
    n.scan_stamp = 11.05;
    n.lidarPose.p = Eigen::Vector3f(7, 8, 9); n.lidarPose.q.setIdentity();
    n.T_corr.setIdentity();
    n.publishKeyframe({n.snapshotScanOutput(cloud), Eigen::Vector2f::Zero(), false});
  }
  static KeyframeOutput captureKeyframe(OdomNode& n, pcl::PointCloud<PointType>::ConstPtr dense,
                                       pcl::PointCloud<PointType>::ConstPtr filtered, bool use_filtered) {
    capture(n, filtered);
    n.current_scan = filtered; n.deskewed_scan = dense; n.keyframe_filtered_ = use_filtered;
    n.T_prior.setIdentity();
    return n.snapshotKeyframeOutput();
  }
  static void publishKeyframe(OdomNode& n, KeyframeOutput out) { n.publishKeyframe(std::move(out)); }
  static bool mappingDue(OdomNode& n, double stamp, float x, float yaw = 0.f, bool veto = false) {
    n.scan_stamp = stamp;
    n.lidarPose.p = Eigen::Vector3f(x, 0, 0);
    n.lidarPose.q = Eigen::Quaternionf(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()));
    n.fuse_tripped_scan_ = veto;
    return n.mappingObservationDue();
  }
  static bool mappingConnected(OdomNode& n) {
    return n.mapping_cloud_pub->get_subscription_count() && n.mapping_pose_pub->get_subscription_count();
  }
  static void publishMapping(OdomNode& n, KeyframeOutput out) { n.publishMapping(std::move(out)); }
  static bool observationConnected(OdomNode& n) { return n.mapping_observation_pub->get_subscription_count() > 0; }
  static void queueMapping(OdomNode& n, pcl::PointCloud<PointType>::ConstPtr cloud, double stamp) {
    capture(n, cloud);
    n.scan_stamp = stamp;
    n.lidarPose.p.x() = static_cast<float>(stamp);
    n.deskewed_scan = cloud;
    n.queueMappingPublish();
    if (n.publish_mapping_thread.joinable()) { n.publish_mapping_thread.join(); }
  }
  static bool alignMapping(OdomNode& n, bool overlap = true) {
    auto target = std::make_shared<pcl::PointCloud<PointType>>();
    for (int a = 1; a <= 6; ++a) {
      for (int b = 1; b <= 6; ++b) {
        PointType point{};
        point.x = .1f*a; point.y = .1f*b; point.z = 0.f; target->push_back(point);
        point.x = 0.f; point.y = .1f*a; point.z = .1f*b; target->push_back(point);
        point.x = .1f*a; point.y = 0.f; point.z = .1f*b; target->push_back(point);
      }
    }
    auto source = std::make_shared<pcl::PointCloud<PointType>>(*target);
    if (!overlap) { for (auto& point : *source) { point.x += 5.f; } }
    n.gicp.setNumThreads(1);
    n.gicp.setCorrespondenceRandomness(10);
    n.gicp.setMaxCorrespondenceDistance(.5);
    n.gicp.setInputTarget(target);
    n.gicp.setInputSource(source);
    pcl::PointCloud<PointType> aligned;
    n.gicp.align(aligned);
    const bool converged = n.gicp.hasConverged();
    n.gicp_hasConverged.store(!converged); // deliberately stale diagnostic status
    return converged;
  }
  static void initializeTarget(OdomNode& n) { n.initializeInputTarget(); }
  static size_t registrationPoints(OdomNode& n) { return n.keyframes.back().second->size(); }
  static void floorFilter(OdomNode& n) {
    n.subfloor_reject_enabled_ = true; n.gravity_align_ = true;
  }
  static void observer(OdomNode& n) {
    n.geo.first_opt_done = true;
    n.imu_stamp = rclcpp::Time(12, 300000000, RCL_ROS_TIME);
    n.state.p = Eigen::Vector3f(4, 5, 6);
    n.state.q = Eigen::Quaternionf(Eigen::AngleAxisf(M_PI_2, Eigen::Vector3f::UnitZ()));
    n.state.v.lin.w = Eigen::Vector3f(1, 2, 3);
    n.state.v.lin.b.setConstant(99.f); // deliberately stale cached body velocity
    n.state.v.ang.b = Eigen::Vector3f(.1f, .2f, .3f);
    n.publishPose();
  }
  static void scan(OdomNode& n, sensor_msgs::msg::PointCloud2::SharedPtr msg) { n.callbackPointCloud(msg); }
  static void running(OdomNode& n, bool value) { n.setMainLoopRunning(value); }
  static bool running(OdomNode& n) {
    std::lock_guard<std::mutex> lock(n.main_loop_running_mutex);
    return n.main_loop_running;
  }
  static void waitSubmap(OdomNode& n) { n.pauseSubmapBuildIfNeeded(); }
  static int64_t lastScan(OdomNode& n) { return n.last_scan_input_ns_; }
  static void imu(OdomNode& n, int64_t ns) {
    auto msg = std::make_shared<sensor_msgs::msg::Imu>();
    msg->header.stamp = rclcpp::Time(ns, RCL_ROS_TIME);
    msg->linear_acceleration.z = 9.80665;
    n.callbackImu(msg);
  }
  static size_t imuCount(OdomNode& n) { return n.imu_buffer.size(); }
  static double imuDt(OdomNode& n) { return n.imu_buffer.front().dt; }
  static double imuTime(OdomNode& n) { return n.imu_buffer.front().stamp; }
  static void select(OdomNode& n, float lidar_x, float observer_x) {
    n.lidarPose.p = Eigen::Vector3f(lidar_x, 0, 0);
    n.state.p = Eigen::Vector3f(observer_x, 0, 0);
    n.scan_stamp = 10.05;
    n.keyframe_thresh_dist_ = 1.; n.keyframe_thresh_rot_ = 45.;
    if (n.keyframes.empty()) {
      n.keyframes.push_back({{Eigen::Vector3f::Zero(), Eigen::Quaternionf::Identity()}, n.current_scan});
    }
    n.updateKeyframes();
  }
  static size_t keyframeCount(OdomNode& n) { return n.keyframes.size(); }
  static double keyframeTime(OdomNode& n) { return n.keyframe_timestamps.back().seconds(); }
};
struct MapNodeTestAccess {
  static void ingest(MapNode& n, sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) { n.callbackKeyframe(msg); }
  static sensor_msgs::msg::PointCloud2::ConstSharedPtr message(MapNode& n) { return n.mapMessage(); }
  static bool save(MapNode& n, float leaf) {
    using Service = direct_lidar_inertial_odometry::srv::SavePCD;
    auto req = std::make_shared<Service::Request>();
    auto res = std::make_shared<Service::Response>();
    req->leaf_size = leaf; n.savePCD(req, res); return res->success;
  }
};
} // namespace dlio

namespace {
using Access = dlio::OdomNodeTestAccess;
using MapAccess = dlio::MapNodeTestAccess;
using Cloud = sensor_msgs::msg::PointCloud2;
using Pose = geometry_msgs::msg::PoseStamped;
using namespace std::chrono_literals;

rclcpp::NodeOptions options() {
  rclcpp::NodeOptions opts;
  opts.append_parameter_override("odom/debug/dashboard", false);
  opts.append_parameter_override("imu/calibration", false);
  opts.append_parameter_override("pointcloud/deskew", false);
  return opts;
}
pcl::PointCloud<PointType>::Ptr cloud() {
  auto out = std::make_shared<pcl::PointCloud<PointType>>();
  PointType p;
  p.x = 1.f; p.y = 2.f; p.z = 3.f;
  p.intensity = 1000.f; p.reflectivity = 42.f;
  p.intensity_corrected = 1500.f; p.lidar_intensity = 900.f;
  p.timestamp = 123.; out->push_back(p);
  return out;
}
Cloud::SharedPtr message(int sec = 10) {
  auto out = std::make_shared<Cloud>();
  pcl::toROSMsg(*cloud(), *out);
  out->header.frame_id = "odom"; out->header.stamp.sec = sec;
  return out;
}
bool waitFor(rclcpp::Executor& exec, const std::function<bool()>& done) {
  const auto end = std::chrono::steady_clock::now() + 3s;
  while (!done() && std::chrono::steady_clock::now() < end) {
    exec.spin_some(); std::this_thread::sleep_for(2ms);
  }
  return done();
}
void checkChannels(const Cloud& msg) {
  for (const auto& field : msg.fields) {
    EXPECT_NE(field.name, "t"); EXPECT_NE(field.name, "time"); EXPECT_NE(field.name, "timestamp");
  }
  EXPECT_FLOAT_EQ(dlio::PointCloudScalarField(msg, {"intensity"})[0], 1000.f);
  EXPECT_FLOAT_EQ(dlio::PointCloudScalarField(msg, {"reflectivity"})[0], 42.f);
  EXPECT_FLOAT_EQ(dlio::PointCloudScalarField(msg, {"intensity_corrected"})[0], 1500.f);
  EXPECT_FLOAT_EQ(dlio::PointCloudScalarField(msg, {"lidar_intensity"})[0], 900.f);
}
class OutputContract : public ::testing::Test {
protected:
  static void SetUpTestSuite() { rclcpp::init(0, nullptr); }
  static void TearDownTestSuite() { rclcpp::shutdown(); }
};

TEST_F(OutputContract, ScanCloudPoseAndPathRemainPairedAfterNextScanAdvances) {
  dlio::OdomNode node(options());
  auto sink = std::make_shared<rclcpp::Node>("scan_output_sink");
  Cloud::ConstSharedPtr received;
  Pose::ConstSharedPtr pose;
  auto sub = sink->create_subscription<Cloud>("deskewed", 10, [&](Cloud::ConstSharedPtr m) { received = m; });
  auto psub = sink->create_subscription<Pose>("scan_pose", 10, [&](Pose::ConstSharedPtr m) { pose = m; });
  rclcpp::executors::SingleThreadedExecutor exec; exec.add_node(sink);
  ASSERT_TRUE(waitFor(exec, [&] { return Access::scanConnected(node); }));
  auto output = Access::capture(node, cloud());
  Access::advance(node);
  Access::publish(node, output);
  ASSERT_TRUE(waitFor(exec, [&] { return received && pose; }));
  EXPECT_EQ(received->header, pose->header);
  EXPECT_EQ(received->header.frame_id, "odom");
  EXPECT_EQ(received->header.stamp.sec, 10);
  EXPECT_EQ(received->header.stamp.nanosec, 50000000u);
  EXPECT_DOUBLE_EQ(pose->pose.position.x, 3.);
  EXPECT_FLOAT_EQ(dlio::PointCloudScalarField(*received, {"x"})[0], 3.f);
  ASSERT_EQ(Access::path(node).poses.size(), 1u);
  EXPECT_EQ(Access::path(node).poses.front(), *pose);
  checkChannels(*received);
}

TEST_F(OutputContract, KeyframeCloudHasItsOwnStampedPoseAndPreservesChannels) {
  dlio::OdomNode node(options());
  auto sink = std::make_shared<rclcpp::Node>("keyframe_output_sink");
  Cloud::ConstSharedPtr received;
  Pose::ConstSharedPtr pose;
  auto sub = sink->create_subscription<Cloud>("kf_cloud", 10, [&](Cloud::ConstSharedPtr m) { received = m; });
  auto psub = sink->create_subscription<Pose>("kf_pose_stamped", 10, [&](Pose::ConstSharedPtr m) { pose = m; });
  rclcpp::executors::SingleThreadedExecutor exec; exec.add_node(sink);
  ASSERT_TRUE(waitFor(exec, [&] { return Access::keyframeConnected(node); }));
  Access::keyframe(node, cloud());
  ASSERT_TRUE(waitFor(exec, [&] { return received && pose; }));
  EXPECT_EQ(received->header, pose->header);
  EXPECT_EQ(pose->header.stamp.sec, 11);
  EXPECT_EQ(pose->header.stamp.nanosec, 50000000u);
  EXPECT_DOUBLE_EQ(pose->pose.position.x, 7.);
  checkChannels(*received);
}

TEST_F(OutputContract, DenseKeyframeKeepsNearbyReturnsAndFrozenRegistration) {
  dlio::OdomNode node(options());
  auto sink = std::make_shared<rclcpp::Node>("dense_keyframe_sink");
  Cloud::ConstSharedPtr received;
  Pose::ConstSharedPtr pose;
  auto sub = sink->create_subscription<Cloud>("kf_cloud", 10, [&](Cloud::ConstSharedPtr m) { received = m; });
  auto psub = sink->create_subscription<Pose>("kf_pose_stamped", 10, [&](Pose::ConstSharedPtr m) { pose = m; });
  rclcpp::executors::SingleThreadedExecutor exec; exec.add_node(sink);
  ASSERT_TRUE(waitFor(exec, [&] { return Access::keyframeConnected(node); }));
  auto dense = cloud();
  auto second = dense->front(); second.x += .001f; second.reflectivity = 84.f;
  dense->push_back(second);
  auto filtered = cloud();
  auto output = Access::captureKeyframe(node, dense, filtered, false);
  auto sparse = Access::captureKeyframe(node, dense, filtered, true);
  EXPECT_EQ(output.scan.cloud, dense);
  EXPECT_EQ(sparse.scan.cloud, filtered);
  Access::advance(node);
  Access::publishKeyframe(node, output);
  ASSERT_TRUE(waitFor(exec, [&] { return received && pose; }));
  EXPECT_EQ(received->width * received->height, 2u);
  EXPECT_EQ(received->header, pose->header);
  EXPECT_EQ(pose->header.stamp.sec, 10);
  EXPECT_EQ(pose->header.stamp.nanosec, 50000000u);
  EXPECT_DOUBLE_EQ(pose->pose.position.x, 3.);
  EXPECT_FLOAT_EQ(dlio::PointCloudScalarField(*received, {"x"})[0], 3.f);
  EXPECT_FLOAT_EQ(dlio::PointCloudScalarField(*received, {"x"})[1], second.x + 2.f);
  EXPECT_FLOAT_EQ(dlio::PointCloudScalarField(*received, {"reflectivity"})[1], 84.f);
  EXPECT_FLOAT_EQ(dense->front().x, 1.f); // correction never mutates either source
  EXPECT_FLOAT_EQ(filtered->front().x, 1.f);
  checkChannels(*received);
}

TEST_F(OutputContract, InitialTargetPublishesDenseCloudButRetainsSparseRegistrationHistory) {
  using Observation = direct_lidar_inertial_odometry::msg::MappingObservation;
  auto opts = options(); opts.append_parameter_override("map/observation/enabled", true);
  dlio::OdomNode node(opts);
  auto sink = std::make_shared<rclcpp::Node>("initial_dense_keyframe_sink");
  Cloud::ConstSharedPtr received;
  Pose::ConstSharedPtr pose;
  std::vector<Observation::ConstSharedPtr> observations;
  auto sub = sink->create_subscription<Cloud>("kf_cloud", 10, [&](Cloud::ConstSharedPtr m) { received = m; });
  auto psub = sink->create_subscription<Pose>("kf_pose_stamped", 10, [&](Pose::ConstSharedPtr m) { pose = m; });
  auto osub = sink->create_subscription<Observation>("mapping_observation", 8,
      [&](Observation::ConstSharedPtr m) { observations.push_back(m); });
  rclcpp::executors::SingleThreadedExecutor exec; exec.add_node(sink);
  ASSERT_TRUE(waitFor(exec, [&] { return Access::keyframeConnected(node) && Access::observationConnected(node); }));
  auto dense = cloud(); dense->push_back(dense->front());
  Access::captureKeyframe(node, dense, cloud(), false);
  Access::initializeTarget(node);
  ASSERT_TRUE(waitFor(exec, [&] { return received && pose; }));
  EXPECT_EQ(received->width * received->height, 2u);
  EXPECT_EQ(received->header, pose->header);
  EXPECT_EQ(Access::registrationPoints(node), 1u);
  EXPECT_EQ(Access::keyframeCount(node), 1u);
  ASSERT_TRUE(Access::alignMapping(node));
  Access::queueMapping(node, dense, 11.05);
  ASSERT_TRUE(waitFor(exec, [&] { return !observations.empty(); }));
  ASSERT_EQ(observations.size(), 1u);
  EXPECT_EQ(observations.front()->observation_id, 0u);
  EXPECT_EQ(observations.front()->header.stamp.sec, 11);
  EXPECT_TRUE(observations.front()->registration_converged);
}

TEST_F(OutputContract, DenseKeyframeHonorsOptionalGhostFilterWithCapturedCenter) {
  dlio::OdomNode node(options());
  auto sink = std::make_shared<rclcpp::Node>("dense_floor_keyframe_sink");
  Cloud::ConstSharedPtr received;
  auto sub = sink->create_subscription<Cloud>("kf_cloud", 10, [&](Cloud::ConstSharedPtr m) { received = m; });
  auto psub = sink->create_subscription<Pose>("kf_pose_stamped", 10, [](Pose::ConstSharedPtr) {});
  rclcpp::executors::SingleThreadedExecutor exec; exec.add_node(sink);
  ASSERT_TRUE(waitFor(exec, [&] { return Access::keyframeConnected(node); }));
  auto dense = cloud();
  for (int i = 0; i < 19; ++i) { dense->push_back(dense->front()); }
  auto ghost = dense->front(); ghost.z -= 1.f; dense->push_back(ghost);
  Access::floorFilter(node);
  auto output = Access::captureKeyframe(node, dense, cloud(), false);
  Access::advance(node);
  Access::publishKeyframe(node, output);
  ASSERT_TRUE(waitFor(exec, [&] { return bool(received); }));
  EXPECT_EQ(received->width * received->height, 20u);
  EXPECT_EQ(dense->size(), 21u);
  checkChannels(*received);
}

TEST_F(OutputContract, MappingObservationsCoverRevisitsWithoutChangingRegistrationHistory) {
  auto opts = options(); opts.append_parameter_override("map/observation/enabled", true);
  dlio::OdomNode node(opts);
  EXPECT_TRUE(Access::mappingDue(node, 1., 0.f));
  EXPECT_FALSE(Access::mappingDue(node, 1.1, 1.f)); // bounded publication rate
  EXPECT_FALSE(Access::mappingDue(node, 1.3, .1f));
  EXPECT_TRUE(Access::mappingDue(node, 1.6, .21f));
  EXPECT_TRUE(Access::mappingDue(node, 1.9, 0.f)); // revisit the origin
  EXPECT_FALSE(Access::mappingDue(node, 2.2, 0.f)); // stationary scans do not accumulate
  EXPECT_TRUE(Access::mappingDue(node, 2.5, 0.f, .1f)); // pure rotation
  EXPECT_FALSE(Access::mappingDue(node, 2.8, 1.f, .1f, true));
  EXPECT_TRUE(Access::mappingDue(node, 3.1, 1.f, .1f)); // veto did not advance selection
  EXPECT_EQ(Access::keyframeCount(node), 0u);
}

TEST_F(OutputContract, MappingOutputFreezesPoseCorrectionAndAllScalarChannels) {
  dlio::OdomNode node(options());
  auto sink = std::make_shared<rclcpp::Node>("mapping_output_sink");
  Cloud::ConstSharedPtr received; Pose::ConstSharedPtr pose;
  auto sub = sink->create_subscription<Cloud>("mapping_cloud", 8, [&](Cloud::ConstSharedPtr m) { received = m; });
  auto psub = sink->create_subscription<Pose>("mapping_pose", 8, [&](Pose::ConstSharedPtr m) { pose = m; });
  rclcpp::executors::SingleThreadedExecutor exec; exec.add_node(sink);
  ASSERT_TRUE(waitFor(exec, [&] { return Access::mappingConnected(node); }));
  auto output = Access::captureKeyframe(node, cloud(), cloud(), false);
  Access::advance(node);
  Access::publishMapping(node, output);
  ASSERT_TRUE(waitFor(exec, [&] { return received && pose; }));
  EXPECT_EQ(received->header, output.scan.pose.header);
  EXPECT_EQ(*pose, output.scan.pose);
  EXPECT_FLOAT_EQ(dlio::PointCloudScalarField(*received, {"x"})[0], 3.f);
  checkChannels(*received);
}

TEST_F(OutputContract, AtomicMappingOutputFreezesPoseQualityAndReportsUnknownCovariance) {
  using Observation = direct_lidar_inertial_odometry::msg::MappingObservation;
  dlio::OdomNode node(options());
  auto sink = std::make_shared<rclcpp::Node>("atomic_mapping_sink");
  Observation::ConstSharedPtr received;
  auto sub = sink->create_subscription<Observation>("mapping_observation", 8,
      [&](Observation::ConstSharedPtr m) { received = m; });
  rclcpp::executors::SingleThreadedExecutor exec; exec.add_node(sink);
  ASSERT_TRUE(waitFor(exec, [&] { return Access::observationConnected(node); }));
  auto output = Access::captureKeyframe(node, cloud(), cloud(), false);
  output.observation_id = 17;
  output.registration_converged = true;
  output.degenerate_translation_modes = 1;
  Access::advance(node);
  Access::publishMapping(node, output);
  ASSERT_TRUE(waitFor(exec, [&] { return bool(received); }));
  EXPECT_EQ(received->header, output.scan.pose.header);
  EXPECT_EQ(received->cloud.header, received->header);
  EXPECT_EQ(received->registered_pose, output.scan.pose.pose);
  EXPECT_EQ(received->observation_id, 17u);
  EXPECT_TRUE(received->registration_converged);
  EXPECT_EQ(received->degenerate_translation_modes, 1u);
  EXPECT_EQ(received->covariance_kind, Observation::UNKNOWN);
  for (double value : received->pose_covariance) { EXPECT_DOUBLE_EQ(value, 0.); }
  EXPECT_FALSE(received->covariance_model.empty());
  EXPECT_EQ(received->source_session_id.size(), 36u);
  EXPECT_FALSE(received->base_frame_id.empty());
  EXPECT_FLOAT_EQ(dlio::PointCloudScalarField(received->cloud, {"x"})[0], 3.f);
  checkChannels(received->cloud);
}

TEST_F(OutputContract, AtomicMappingSequencesUseCurrentRegistrationInOneSession) {
  using Observation = direct_lidar_inertial_odometry::msg::MappingObservation;
  auto opts = options(); opts.append_parameter_override("map/observation/enabled", true);
  dlio::OdomNode node(opts);
  auto sink = std::make_shared<rclcpp::Node>("mapping_sequence_sink");
  std::vector<Observation::ConstSharedPtr> received;
  auto sub = sink->create_subscription<Observation>("mapping_observation", 8,
      [&](Observation::ConstSharedPtr m) { received.push_back(m); });
  rclcpp::executors::SingleThreadedExecutor exec; exec.add_node(sink);
  ASSERT_TRUE(waitFor(exec, [&] { return Access::observationConnected(node); }));
  ASSERT_TRUE(Access::alignMapping(node));
  Access::queueMapping(node, cloud(), 10.05);
  Access::queueMapping(node, cloud(), 11.05);
  ASSERT_FALSE(Access::alignMapping(node, false));
  Access::queueMapping(node, cloud(), 12.05);
  ASSERT_TRUE(waitFor(exec, [&] { return received.size() == 3; }));
  EXPECT_EQ(received[0]->observation_id, 0u);
  EXPECT_EQ(received[1]->observation_id, 1u);
  EXPECT_EQ(received[0]->source_session_id, received[1]->source_session_id);
  EXPECT_TRUE(received[0]->registration_converged);
  EXPECT_TRUE(received[1]->registration_converged);
  EXPECT_FALSE(received[2]->registration_converged);
  EXPECT_EQ(received[2]->observation_id, 2u);
  EXPECT_EQ(received[0]->source_session_id, received[2]->source_session_id);
  EXPECT_EQ(received[0]->header.stamp.sec, 10);
  EXPECT_EQ(received[1]->header.stamp.sec, 11);
}

TEST_F(OutputContract, OdometryTwistIsBodyFrameAndTfMatchesObserverSnapshot) {
  dlio::OdomNode node(options());
  auto sink = std::make_shared<rclcpp::Node>("observer_output_sink");
  nav_msgs::msg::Odometry::ConstSharedPtr odom;
  tf2_msgs::msg::TFMessage::ConstSharedPtr tf;
  auto sub = sink->create_subscription<nav_msgs::msg::Odometry>("odom", 10,
      [&](nav_msgs::msg::Odometry::ConstSharedPtr m) { odom = m; });
  auto tsub = sink->create_subscription<tf2_msgs::msg::TFMessage>("/tf", 100,
      [&](tf2_msgs::msg::TFMessage::ConstSharedPtr m) { tf = m; });
  rclcpp::executors::SingleThreadedExecutor exec; exec.add_node(sink);
  ASSERT_TRUE(waitFor(exec, [&] { Access::observer(node); return odom && tf; }));
  EXPECT_NEAR(odom->twist.twist.linear.x, 2., 1e-6);
  EXPECT_NEAR(odom->twist.twist.linear.y, -1., 1e-6);
  EXPECT_NEAR(odom->twist.twist.linear.z, 3., 1e-6);
  EXPECT_NEAR(odom->twist.twist.angular.z, .3, 1e-6);
  ASSERT_EQ(tf->transforms.size(), 1u);
  const auto& transform = tf->transforms.front();
  EXPECT_EQ(transform.header, odom->header);
  EXPECT_EQ(transform.child_frame_id, odom->child_frame_id);
  EXPECT_DOUBLE_EQ(transform.transform.translation.x, odom->pose.pose.position.x);
  EXPECT_EQ(transform.transform.rotation, odom->pose.pose.orientation);
}

TEST_F(OutputContract, KeyframeSelectionUsesRegisteredPoseInsteadOfLaterObserver) {
  dlio::OdomNode node(options());
  Access::select(node, .1f, 10.f);
  EXPECT_EQ(Access::keyframeCount(node), 1u);
  Access::select(node, 2.f, .1f);
  EXPECT_EQ(Access::keyframeCount(node), 2u);
  EXPECT_NEAR(Access::keyframeTime(node), 10.05, 1e-9);
}

TEST_F(OutputContract, SkippedScanReleasesSubmapAndRejectsTimeReversal) {
  dlio::OdomNode node(options());
  Access::running(node, true);
  auto worker = std::async(std::launch::async, [&] { Access::waitSubmap(node); });
  Access::scan(node, message()); // no IMU; returns before creating a target
  EXPECT_FALSE(Access::running(node));
  const bool released = worker.wait_for(1s) == std::future_status::ready;
  Access::running(node, false); // keep test teardown safe on regression
  EXPECT_TRUE(released);
  worker.get();
  Access::scan(node, message(9));
  Access::scan(node, message(10));
  EXPECT_EQ(Access::lastScan(node), 10000000000LL);
  Access::scan(node, message(11));
  EXPECT_EQ(Access::lastScan(node), 11000000000LL);
}

TEST_F(OutputContract, ImuStartupDuplicatesAndTimeReversalDoNotInventIntegrationTime) {
  dlio::OdomNode node(options());
  Access::imu(node, 10000000000LL);
  EXPECT_EQ(Access::imuCount(node), 0u);
  Access::imu(node, 10010000000LL);
  ASSERT_EQ(Access::imuCount(node), 1u);
  EXPECT_NEAR(Access::imuDt(node), .01, 1e-9);
  Access::imu(node, 10010000000LL);
  Access::imu(node, 10000000000LL);
  EXPECT_EQ(Access::imuCount(node), 1u);
  Access::imu(node, 10020000000LL);
  EXPECT_EQ(Access::imuCount(node), 2u);
  EXPECT_NEAR(Access::imuDt(node), .01, 1e-9);
  EXPECT_NEAR(Access::imuTime(node), 10.02, 1e-9);
}

TEST_F(OutputContract, MapRejectsWrongFramesMalformedCloudsAndInvalidTimes) {
  dlio::MapNode node;
  auto wrong = message(); wrong->header.frame_id = "lidar";
  MapAccess::ingest(node, wrong);
  auto short_data = message(); short_data->data.pop_back();
  MapAccess::ingest(node, short_data);
  auto no_x = message(); no_x->fields.erase(no_x->fields.begin());
  MapAccess::ingest(node, no_x);
  auto bad_time = message(); bad_time->header.stamp.nanosec = 1000000000;
  EXPECT_NO_THROW(MapAccess::ingest(node, bad_time));
  EXPECT_FALSE(MapAccess::message(node));
  MapAccess::ingest(node, message());
  ASSERT_TRUE(MapAccess::message(node));
  checkChannels(*MapAccess::message(node));
}

TEST_F(OutputContract, ComposedMapAcceptsIntraProcessInputWithLatchedDdsOutput) {
  rclcpp::NodeOptions opts;
  opts.use_intra_process_comms(true);
  dlio::MapNode node(opts);
  MapAccess::ingest(node, message());
  const auto output = MapAccess::message(node);
  ASSERT_TRUE(output);
  EXPECT_EQ(output->width, 1u);
  checkChannels(*output);
}

TEST_F(OutputContract, MapSnapshotsKeepMeasurementTimeAndRejectDuplicateKeyframes) {
  dlio::MapNode node;
  MapAccess::ingest(node, message());
  const auto first = MapAccess::message(node);
  ASSERT_TRUE(first);
  const auto old_data = first->data;
  EXPECT_EQ(first, MapAccess::message(node));
  MapAccess::ingest(node, message(9));
  MapAccess::ingest(node, message(10));
  EXPECT_EQ(first, MapAccess::message(node));
  MapAccess::ingest(node, message(11));
  const auto second = MapAccess::message(node);
  ASSERT_TRUE(second);
  EXPECT_EQ(second->width, 2u);
  EXPECT_EQ(second->header.stamp.sec, 11);
  EXPECT_EQ(first->header.stamp.sec, 10);
  EXPECT_EQ(first->data, old_data);
  EXPECT_EQ(first->width, 1u);
}

TEST_F(OutputContract, MapDecodesBigEndianPaddedRowsAndDropsNonfiniteGeometry) {
  auto msg = message();
  // Remove the overlapping time union before byte-swapping scalar fields.
  dlio::stripAcquisitionTimeFields(msg->fields);
  msg->height = 2; msg->row_step += 8;
  const auto first = msg->data;
  msg->data.resize(2 * msg->row_step, 0);
  std::copy(first.begin(), first.end(), msg->data.begin() + msg->row_step);
  const float invalid = std::numeric_limits<float>::quiet_NaN();
  std::memcpy(msg->data.data() + msg->row_step, &invalid, sizeof(float));
  for (size_t row = 0; row < 2; ++row) {
    for (const auto& field : msg->fields) {
      auto begin = msg->data.begin() + row * msg->row_step + field.offset;
      std::reverse(begin, begin + sizeof(float));
    }
  }
  msg->is_bigendian = true;
  dlio::MapNode node; MapAccess::ingest(node, msg);
  auto output = MapAccess::message(node);
  ASSERT_TRUE(output);
  EXPECT_EQ(output->width, 1u);
  EXPECT_FLOAT_EQ(dlio::PointCloudScalarField(*output, {"x"})[0], 1.f);
  checkChannels(*output);
}

TEST_F(OutputContract, MapRejectsInvalidVoxelSizesAtStartupAndExport) {
  for (double leaf : {0., -1., std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::quiet_NaN(), 1e-45}) {
    rclcpp::NodeOptions opts; opts.append_parameter_override("map/sparse/leafSize", leaf);
    EXPECT_THROW(dlio::MapNode node(opts), std::invalid_argument);
  }
  dlio::MapNode node;
  EXPECT_FALSE(MapAccess::save(node, 0.f));
  EXPECT_FALSE(MapAccess::save(node, std::numeric_limits<float>::quiet_NaN()));
}
} // namespace
