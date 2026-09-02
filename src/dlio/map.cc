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
#include "rclcpp/create_timer.hpp"

#include <filesystem>
#include <system_error>

dlio::MapNode::MapNode(const rclcpp::NodeOptions& options)
    : Node("dlio_map_node", options) {

  this->getParams();

  this->keyframe_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  auto keyframe_sub_opt = rclcpp::SubscriptionOptions();
  keyframe_sub_opt.callback_group = this->keyframe_cb_group;
  this->keyframe_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>("keyframes", 10,
      std::bind(&dlio::MapNode::callbackKeyframe, this, std::placeholders::_1), keyframe_sub_opt);

  this->map_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "map", rclcpp::QoS(1).transient_local());

  this->save_pcd_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  this->save_pcd_srv = this->create_service<direct_lidar_inertial_odometry::srv::SavePCD>("save_pcd",
      std::bind(&dlio::MapNode::savePCD, this, std::placeholders::_1, std::placeholders::_2), rclcpp::ServicesQoS(), this->save_pcd_cb_group);

  this->dlio_map = std::make_shared<pcl::PointCloud<PointType>>();

  // Republish the (growing) map on a low-rate timer instead of on every
  // keyframe: republishing the whole accumulated corridor map per keyframe is
  // the most expensive output on a long run. Rate is configurable; the publish
  // is also skipped when nobody is subscribed.
  double map_pub_rate;
  this->declare_parameter<double>("map/publishRate", 1.0);
  this->get_parameter("map/publishRate", map_pub_rate);
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
}

void dlio::MapNode::start() {
}

void dlio::MapNode::callbackKeyframe(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& keyframe) {

  // convert scan to pcl format
  pcl::PointCloud<PointType>::Ptr keyframe_pcl = std::make_shared<pcl::PointCloud<PointType>>();
  pcl::fromROSMsg(*keyframe, *keyframe_pcl);

  // voxel filter
  this->voxelgrid.setLeafSize(this->leaf_size_, this->leaf_size_, this->leaf_size_);
  this->voxelgrid.setInputCloud(keyframe_pcl);
  this->voxelgrid.filter(*keyframe_pcl);

  // Accumulate into the map; publishing happens on the low-rate timer below.
  // (lock: savePCD and publishMap run in other callback groups and may read.)
  std::lock_guard<std::mutex> lock(this->map_mutex);
  *this->dlio_map += *keyframe_pcl;
}

void dlio::MapNode::publishMap() {

  // Skip the whole-map serialize when nobody is listening.
  if (this->map_pub->get_subscription_count() == 0) { return; }

  // Copy the cloud under the lock (brief), serialize outside it so keyframe
  // accumulation isn't stalled by the toROSMsg of a large map.
  pcl::PointCloud<PointType>::Ptr snapshot;
  {
    std::lock_guard<std::mutex> lock(this->map_mutex);
    if (this->dlio_map->empty()) { return; }
    snapshot = std::make_shared<pcl::PointCloud<PointType>>(*this->dlio_map);
  }
  sensor_msgs::msg::PointCloud2 map_ros;
  pcl::toROSMsg(*snapshot, map_ros);
  map_ros.header.stamp = this->now();
  map_ros.header.frame_id = this->odom_frame;
  this->map_pub->publish(map_ros);
}

void dlio::MapNode::savePCD(std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Request> req,
                            std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Response> res) {

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
  pcl::VoxelGrid<PointType> vg;
  vg.setLeafSize(leaf_size, leaf_size, leaf_size);
  vg.setInputCloud(m);
  vg.filter(*m);

  // save map
  int ret = pcl::io::savePCDFileBinary(p + "/dlio_map.pcd", *m);
  res->success = ret == 0;

  if (res->success) {
    RCLCPP_INFO(this->get_logger(), "save_pcd: done (%zu points after voxelization)", m->size());
  } else {
    RCLCPP_ERROR(this->get_logger(), "save_pcd: failed to write %s/dlio_map.pcd", p.c_str());
  }
}
