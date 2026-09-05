/***********************************************************
 *                                                         *
 * Copyright (c)                                           *
 *                                                         *
 * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
 * University of California, Los Angeles                   *
 *                                                         *
 * Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez   *
 * Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu         *
 *                                                         *
 ***********************************************************/

#include "dlio/map.h"
#include "dlio/utils.h"
#include "dlio/pointcloud_fields.h"
#include "rclcpp/create_timer.hpp"
#include "rclcpp/version.h"

#include <filesystem>
#include <cmath>
#include <stdexcept>
#include <system_error>

namespace {
bool validLeafSize(double leaf) {
  const float value = static_cast<float>(leaf);
  return value > 0.f && std::isfinite(value) && std::isfinite(1.f / value);
}
// The typed VoxelGrid centroid only accumulates PCL's built-in fields. Use
// the field-aware blob filter so custom raw/derived channels survive both
// the published map and its final PCD export.
pcl::PointCloud<PointType>::Ptr voxelizeMap(
    const pcl::PointCloud<PointType>::ConstPtr& input, float leaf_size) {
  auto blob = std::make_shared<pcl::PCLPointCloud2>();
  pcl::toPCLPointCloud2(*input, *blob);
  // Per-scan timestamps have no meaning once keyframes are accumulated.
  // Their overlapping uint32/float/double union is also unsuitable for PCL's
  // generic scalar averaging; pass only XYZ and the four float channels.
  dlio::stripAcquisitionTimeFields(blob->fields);
  pcl::VoxelGrid<pcl::PCLPointCloud2> filter;
  filter.setLeafSize(leaf_size, leaf_size, leaf_size);
  filter.setDownsampleAllData(true);
  filter.setInputCloud(blob);
  pcl::PCLPointCloud2 filtered;
  filter.filter(filtered);
  auto output = std::make_shared<pcl::PointCloud<PointType>>();
  pcl::fromPCLPointCloud2(filtered, *output);
  return output;
}
}  // namespace

dlio::MapNode::MapNode(const rclcpp::NodeOptions& options)
    : Node("dlio_map_node", options) {

  this->getParams();

  this->keyframe_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  auto keyframe_sub_opt = rclcpp::SubscriptionOptions();
  keyframe_sub_opt.callback_group = this->keyframe_cb_group;
  this->keyframe_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>("keyframes", 10,
      std::bind(&dlio::MapNode::callbackKeyframe, this, std::placeholders::_1), keyframe_sub_opt);

  // Humble's intra-process manager does not support transient-local publishers.
  // Keep keyframe delivery intra-process; the latched preview is a DDS output.
  rclcpp::PublisherOptions map_options;
  map_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Disable;
  this->map_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "map", rclcpp::QoS(1).transient_local(), map_options);

  this->save_pcd_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  this->save_pcd_srv = this->create_service<direct_lidar_inertial_odometry::srv::SavePCD>("save_pcd",
      std::bind(&dlio::MapNode::savePCD, this, std::placeholders::_1, std::placeholders::_2),
#if RCLCPP_VERSION_MAJOR >= 28
      rclcpp::ServicesQoS(),
#else
      rmw_qos_profile_services_default,
#endif
      this->save_pcd_cb_group);

  this->dlio_map = std::make_shared<pcl::PointCloud<PointType>>();

  // Republish the (growing) map on a low-rate timer instead of on every
  // keyframe: republishing the whole accumulated corridor map per keyframe is
  // the most expensive output on a long run. Rate is configurable; the publish
  // is also skipped when nobody is subscribed.
  double map_pub_rate;
  this->declare_parameter<double>("map/publishRate", 1.0);
  this->get_parameter("map/publishRate", map_pub_rate);
  if (!std::isfinite(map_pub_rate) || map_pub_rate < 0.0) {
    throw std::invalid_argument("map/publishRate must be finite and nonnegative");
  }
  if (map_pub_rate > 0.0) {
    // Node-clock timer (respects use_sim_time): the map republish rate tracks
    // sim time under bag replay rather than free-running on wall time.
    this->map_pub_timer = rclcpp::create_timer(this, this->get_clock(),
        rclcpp::Duration::from_seconds(1.0 / map_pub_rate),
        std::bind(&dlio::MapNode::publishMap, this));
  }

  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);

}

dlio::MapNode::~MapNode() {}

void dlio::MapNode::getParams() {

  // same key the odometry node uses, so the map and odometry frames stay in sync
  this->declare_parameter<std::string>("frames/odom", "odom");
  this->declare_parameter<double>("map/sparse/leafSize", 0.5);

  this->get_parameter("frames/odom", this->odom_frame);
  this->get_parameter("map/sparse/leafSize", this->leaf_size_);
  if (!validLeafSize(this->leaf_size_)) {
    throw std::invalid_argument("map/sparse/leafSize must be finite and positive");
  }
}

void dlio::MapNode::start() {
}

void dlio::MapNode::callbackKeyframe(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& keyframe) {

  // This preview map accepts already registered odom-frame keyframes. Relabeling
  // a sensor-frame cloud as odom silently corrupts the map; do not guess a TF.
  if (keyframe->header.frame_id != this->odom_frame ||
      keyframe->header.stamp.sec < 0 || keyframe->header.stamp.nanosec >= 1000000000u ||
      !dlio::PointCloudScalarField::validLayout(*keyframe)) {
    RCLCPP_WARN(this->get_logger(), "Rejecting keyframe with wrong frame or invalid layout");
    return;
  }
  const rclcpp::Time stamp(keyframe->header.stamp, RCL_ROS_TIME);
  {
    std::lock_guard<std::mutex> lock(this->map_mutex);
    if (this->have_keyframe_stamp_ && stamp <= this->last_keyframe_stamp_) {
      RCLCPP_WARN(this->get_logger(), "Rejecting duplicate or out-of-order keyframe; restart the map for a new session");
      return;
    }
  }
  const dlio::PointCloudScalarField x(*keyframe, {"x"}), y(*keyframe, {"y"}), z(*keyframe, {"z"});
  if (!x || !y || !z) { return; }
  const dlio::PointCloudScalarField intensity(*keyframe, {"intensity"});
  const dlio::PointCloudScalarField reflectivity(*keyframe, {"reflectivity"});
  const dlio::PointCloudScalarField corrected(*keyframe, {"intensity_corrected"});
  const dlio::PointCloudScalarField image(*keyframe, {"lidar_intensity"});
  auto keyframe_pcl = std::make_shared<pcl::PointCloud<PointType>>();
  const size_t count = static_cast<size_t>(keyframe->width) * keyframe->height;
  keyframe_pcl->reserve(count);
  for (size_t i = 0; i < count; ++i) {
    PointType p;
    p.x = x[i]; p.y = y[i]; p.z = z[i];
    if (!p.getVector3fMap().allFinite()) { continue; }
    p.intensity = intensity ? intensity[i] : 0.f;
    p.reflectivity = reflectivity ? reflectivity[i] : 0.f;
    p.intensity_corrected = corrected ? corrected[i] : p.intensity;
    p.lidar_intensity = image ? image[i] : p.reflectivity;
    p.timestamp = 0.0;
    keyframe_pcl->push_back(p);
  }
  if (keyframe_pcl->empty()) { return; }

  // voxel filter
  keyframe_pcl = voxelizeMap(keyframe_pcl, this->leaf_size_);

  // Accumulate into the map; publishing happens on the low-rate timer below.
  // (lock: savePCD and publishMap run in other callback groups and may read.)
  std::lock_guard<std::mutex> lock(this->map_mutex);
  // Keyframe callbacks are mutually exclusive, so ordering remains valid.
  *this->dlio_map += *keyframe_pcl;
  this->last_keyframe_stamp_ = stamp;
  this->have_keyframe_stamp_ = true;
  ++this->map_revision_;
}

void dlio::MapNode::publishMap() {

  // Skip the whole-map serialize when nobody is listening.
  if (this->map_pub->get_subscription_count() == 0) { return; }

  const auto message = this->mapMessage();
  if (message) { this->map_pub->publish(*message); }
}

sensor_msgs::msg::PointCloud2::ConstSharedPtr dlio::MapNode::mapMessage() {
  // Serialize only when map contents change. Periodic republishing still serves
  // volatile/late subscribers, without repeated whole-map copy and conversion.
  std::lock_guard<std::mutex> cache_lock(this->publish_cache_mutex_);
  pcl::PointCloud<PointType>::Ptr snapshot;
  rclcpp::Time stamp(0, 0, RCL_ROS_TIME);
  uint64_t revision;
  {
    std::lock_guard<std::mutex> lock(this->map_mutex);
    if (this->dlio_map->empty()) { return nullptr; }
    if (this->cached_map_ && this->cached_revision_ == this->map_revision_) { return this->cached_map_; }
    snapshot = std::make_shared<pcl::PointCloud<PointType>>(*this->dlio_map);
    stamp = this->last_keyframe_stamp_;
    revision = this->map_revision_;
  }
  auto message = std::make_shared<sensor_msgs::msg::PointCloud2>();
  pcl::toROSMsg(*snapshot, *message);
  dlio::stripAcquisitionTimeFields(message->fields);
  message->header.stamp = stamp;
  message->header.frame_id = this->odom_frame;
  this->cached_revision_ = revision;
  this->cached_map_ = message;
  return message;
}

void dlio::MapNode::savePCD(std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Request> req,
                            std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Response> res) {

  if (!validLeafSize(req->leaf_size)) {
    RCLCPP_ERROR(this->get_logger(), "save_pcd: leaf size must be finite and positive");
    res->success = false;
    return;
  }
  std::unique_lock<std::mutex> lock(this->map_mutex);
  pcl::PointCloud<PointType>::Ptr m = std::make_shared<pcl::PointCloud<PointType>>(*this->dlio_map);
  lock.unlock();

  float leaf_size = req->leaf_size;
  std::string p = req->save_path;

  std::error_code directory_error;
  const bool is_directory = std::filesystem::is_directory(p, directory_error);
  if (directory_error) {
    RCLCPP_ERROR(this->get_logger(), "save_pcd: could not inspect directory %s: %s",
                 p.c_str(), directory_error.message().c_str());
    res->success = false;
    return;
  }

  if (!is_directory) {
    RCLCPP_ERROR(this->get_logger(), "save_pcd: could not find directory %s", p.c_str());
    res->success = false;
    return;
  }

  if (m->empty()) {
    RCLCPP_WARN(this->get_logger(), "save_pcd: map is empty, nothing to save");
    res->success = false;
    return;
  }

  RCLCPP_INFO(this->get_logger(), "save_pcd: saving map (%zu points) to %s/dlio_map.pcd with leaf size %.2f",
              m->size(), p.c_str(), leaf_size);

  // voxelize map
  m = voxelizeMap(m, leaf_size);

  // save map
  pcl::PCLPointCloud2 blob;
  pcl::toPCLPointCloud2(*m, blob);
  dlio::stripAcquisitionTimeFields(blob.fields);
  int ret = pcl::io::savePCDFile(p + "/dlio_map.pcd", blob,
      Eigen::Vector4f::Zero(), Eigen::Quaternionf::Identity(), true);
  res->success = ret == 0;

  if (res->success) {
    RCLCPP_INFO(this->get_logger(), "save_pcd: done (%zu points after voxelization)", m->size());
  } else {
    RCLCPP_ERROR(this->get_logger(), "save_pcd: failed to write %s/dlio_map.pcd", p.c_str());
  }
}
