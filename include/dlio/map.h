#pragma once

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

#include "dlio/dlio.h"

// ROS
#include "rclcpp/rclcpp.hpp"
#include "direct_lidar_inertial_odometry/srv/save_pcd.hpp"
#include <sensor_msgs/msg/point_cloud2.hpp>

// PCL
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>

#include <mutex>

class dlio::MapNode: public rclcpp::Node {

public:

  explicit MapNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
  ~MapNode();

  void start();

private:

  friend struct MapNodeTestAccess;

  void getParams();

  void callbackKeyframe(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& keyframe);
  void publishMap();
  sensor_msgs::msg::PointCloud2::ConstSharedPtr mapMessage();

  void savePCD(std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Request> req,
               std::shared_ptr<direct_lidar_inertial_odometry::srv::SavePCD::Response> res);

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr keyframe_sub;
  rclcpp::CallbackGroup::SharedPtr keyframe_cb_group, save_pcd_cb_group;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_pub;
  rclcpp::TimerBase::SharedPtr map_pub_timer;

  rclcpp::Service<direct_lidar_inertial_odometry::srv::SavePCD>::SharedPtr save_pcd_srv;

  pcl::PointCloud<PointType>::Ptr dlio_map;
  std::mutex map_mutex;  // guards dlio_map across callbackKeyframe / publishMap / savePCD
  rclcpp::Time last_keyframe_stamp_{0, 0, RCL_ROS_TIME};
  bool have_keyframe_stamp_ = false;
  uint64_t map_revision_ = 0;
  std::mutex publish_cache_mutex_;
  uint64_t cached_revision_ = 0;
  sensor_msgs::msg::PointCloud2::ConstSharedPtr cached_map_;

  std::string odom_frame;

  double leaf_size_;

};
