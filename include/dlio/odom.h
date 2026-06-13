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
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <deque>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>

// BOOST
#include <boost/format.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/adaptor/adjacent_filtered.hpp>

// PCL
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/surface/convex_hull.h>
#include <pcl_conversions/pcl_conversions.h>

class dlio::OdomNode: public rclcpp::Node {

public:

  explicit OdomNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
  ~OdomNode();

  void start();

  // IMU measurement sample. Public (with the static integration entry point
  // below) so the continuous-time integration math is unit-testable.
  struct ImuMeas {
    double stamp;
    double dt; // defined as the difference between the current and the previous measurement
    Eigen::Vector3f ang_vel;
    Eigen::Vector3f lin_accel;
  };

  // Pure constant-jerk / constant-angular-acceleration integration between
  // IMU samples, evaluated at sorted_timestamps (the DLIO paper's analytic
  // deskew kernel). Static and side-effect-free.
  static std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
    integrateImuInternal(Eigen::Quaternionf q_init, Eigen::Vector3f p_init, Eigen::Vector3f v_init,
                         const std::vector<double>& sorted_timestamps,
                         boost::circular_buffer<ImuMeas>::reverse_iterator begin_imu_it,
                         boost::circular_buffer<ImuMeas>::reverse_iterator end_imu_it,
                         double gravity);

private:

  struct State;

  void getParams();

  void callbackPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr pc);
  void callbackImu(const sensor_msgs::msg::Imu::SharedPtr imu);
  void callbackImage(const sensor_msgs::msg::Image::SharedPtr img);
  // Set up the visual term on `gicp` for this scan (picks the image nearest
  // scan_stamp, computes the camera transforms). Returns true if the visual
  // term is active this scan. Stash-then-store of the previous frame is handled
  // by the caller (getNextPose) after align().
  bool setupVisualForScan();

  void publishPose();
  void publishStaticTransforms();

  void publishToROS(pcl::PointCloud<PointType>::ConstPtr published_cloud, Eigen::Matrix4f T_cloud);
  void publishCloud(pcl::PointCloud<PointType>::ConstPtr published_cloud, Eigen::Matrix4f T_cloud);
  void publishKeyframe(std::pair<std::pair<Eigen::Vector3f, Eigen::Quaternionf>,
                       pcl::PointCloud<PointType>::ConstPtr> kf, rclcpp::Time timestamp);

  void getScanFromROS(const sensor_msgs::msg::PointCloud2::SharedPtr& pc);
  void preprocessPoints();
  void deskewPointcloud();
  void initializeInputTarget();
  void setInputSource();

  void initializeDLIO();

  void getNextPose();
  bool imuMeasFromTimeRange(double start_time, double end_time,
                            boost::circular_buffer<ImuMeas>::reverse_iterator& begin_imu_it,
                            boost::circular_buffer<ImuMeas>::reverse_iterator& end_imu_it);
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
    integrateImu(double start_time, Eigen::Quaternionf q_init, Eigen::Vector3f p_init, Eigen::Vector3f v_init,
                 const std::vector<double>& sorted_timestamps);
  void propagateGICP();

  void propagateState();
  void updateState();

  void setAdaptiveParams();

  void computeMetrics();
  void computeSpaciousness();
  void computeDensity();

  sensor_msgs::msg::Imu::SharedPtr transformImu(const sensor_msgs::msg::Imu::SharedPtr& imu);

  void updateKeyframes();
  void pruneKeyframes();
  void computeConvexHull();
  void computeConcaveHull();
  void pushSubmapIndices(std::vector<float> dists, int k, std::vector<int> frames);
  void buildSubmap(State vehicle_state);
  void buildKeyframesAndSubmap(State vehicle_state);
  void pauseSubmapBuildIfNeeded();

  void debug();

  // Publish a diagnostic_msgs/DiagnosticArray on /diagnostics once per scan.
  // Mirrors RESPLE's /diagnostics so the same capture/plot tooling works;
  // published independently of the ANSI dashboard (debug() is gated on
  // dashboard_, diagnostics should flow regardless).
  void publishDiagnostics();

  rclcpp::TimerBase::SharedPtr publish_timer;

  // Subscribers
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub;
  rclcpp::CallbackGroup::SharedPtr lidar_cb_group, imu_cb_group, image_cb_group;

  // Publishers
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr kf_pose_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr kf_cloud_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr deskewed_pub;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub;

  // TF
  std::shared_ptr<tf2_ros::TransformBroadcaster> br;
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_br;

  // ROS Msgs
  nav_msgs::msg::Odometry odom_ros;
  geometry_msgs::msg::PoseStamped pose_ros;
  nav_msgs::msg::Path path_ros;
  geometry_msgs::msg::PoseArray kf_pose_ros;

  // Flags
  std::atomic<bool> dlio_initialized;
  std::atomic<bool> first_valid_scan;
  std::atomic<bool> first_imu_received;
  std::atomic<bool> imu_calibrated;
  std::atomic<bool> submap_hasChanged;
  std::atomic<bool> gicp_hasConverged;
  std::atomic<bool> deskew_status;
  std::atomic<int> deskew_size;

  // Threads
  std::thread publish_thread;
  std::thread publish_keyframe_thread;
  std::thread debug_thread;

  // Distance traveled (maintained incrementally in callbackPointCloud)
  double length_traversed;
  Eigen::Vector3f length_prev_p;

  // Keyframes
  std::vector<std::pair<std::pair<Eigen::Vector3f, Eigen::Quaternionf>,
                        pcl::PointCloud<PointType>::ConstPtr>> keyframes;
  std::vector<rclcpp::Time> keyframe_timestamps;
  std::vector<std::shared_ptr<const nano_gicp::CovarianceList>> keyframe_normals;
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> keyframe_transformations;
  std::mutex keyframes_mutex;

  // Sensor Type
  dlio::SensorType sensor;

  // Frames
  std::string odom_frame;
  std::string baselink_frame;
  std::string lidar_frame;
  std::string imu_frame;

  // Preprocessing
  pcl::CropBox<PointType> crop;
  pcl::VoxelGrid<PointType> voxel;

  // Point Clouds
  pcl::PointCloud<PointType>::ConstPtr original_scan;
  pcl::PointCloud<PointType>::ConstPtr deskewed_scan;
  pcl::PointCloud<PointType>::ConstPtr current_scan;

  int num_processed_keyframes;

  pcl::ConvexHull<PointType> convex_hull;
  pcl::ConcaveHull<PointType> concave_hull;
  std::vector<int> keyframe_convex;
  std::vector<int> keyframe_concave;

  // Submap
  pcl::PointCloud<PointType>::ConstPtr submap_cloud;
  std::shared_ptr<const nano_gicp::CovarianceList> submap_normals;

  std::vector<int> submap_kf_idx_curr;
  std::vector<int> submap_kf_idx_prev;

  bool new_submap_is_ready;
  std::future<void> submap_future;
  std::condition_variable submap_build_cv;
  bool main_loop_running;
  std::mutex main_loop_running_mutex;

  // Timestamps
  rclcpp::Time scan_header_stamp;
  double scan_stamp;
  double prev_scan_stamp;
  std::vector<double> comp_times;
  std::vector<double> imu_rates;
  std::vector<double> lidar_rates;

  double first_scan_stamp;
  double elapsed_time;

  // GICP
  nano_gicp::NanoGICP<PointType, PointType> gicp;
  nano_gicp::NanoGICP<PointType, PointType> gicp_temp;

  // Transformations
  Eigen::Matrix4f T, T_prior, T_corr;

  Eigen::Vector3f origin;

  struct Extrinsics {
    struct SE3 {
      Eigen::Vector3f t;
      Eigen::Matrix3f R;
    };
    SE3 baselink2imu;
    SE3 baselink2lidar;
    Eigen::Matrix4f baselink2imu_T;
    Eigen::Matrix4f baselink2lidar_T;
  }; Extrinsics extrinsics;

  // IMU
  rclcpp::Time imu_stamp;
  double first_imu_stamp;
  double prev_imu_stamp;

  ImuMeas imu_meas;

  boost::circular_buffer<ImuMeas> imu_buffer;
  std::mutex mtx_imu;
  std::condition_variable cv_imu_stamp;


  // Geometric Observer
  struct Geo {
    bool first_opt_done;
    std::mutex mtx;
    double dp;
    double dq_deg;
    Eigen::Vector3f prev_p;
    Eigen::Quaternionf prev_q;
    Eigen::Vector3f prev_vel;
  }; Geo geo;

  // State Vector
  struct ImuBias {
    Eigen::Vector3f gyro;
    Eigen::Vector3f accel;
  };

  struct Frames {
    Eigen::Vector3f b;
    Eigen::Vector3f w;
  };

  struct Velocity {
    Frames lin;
    Frames ang;
  };

  struct State {
    Eigen::Vector3f p; // position in world frame
    Eigen::Quaternionf q; // orientation in world frame
    Velocity v;
    ImuBias b; // imu biases in body frame
  }; State state;

  struct Pose {
    Eigen::Vector3f p; // position in world frame
    Eigen::Quaternionf q; // orientation in world frame
  };
  Pose lidarPose;

  // Metrics
  struct Metrics {
    std::vector<float> spaciousness;
    std::vector<float> density;
  }; Metrics metrics;

  std::string cpu_type;
  std::vector<double> cpu_percents;
  clock_t lastCPU, lastSysCPU, lastUserCPU;
  int numProcessors;

  // Degeneracy-gate activity, written only on the scan thread in getNextPose()
  // and read by publishDiagnostics(): directions held to the IMU prior this
  // scan, and the cumulative count of scans where the gate fired.
  int loc_gate_axes_current_ = 0;
  uint64_t loc_gate_updates_cumulative_ = 0;

  // Separate CPU-time baseline for publishDiagnostics() so its utilization
  // delta is independent of debug()'s (each maintains its own since-last-call
  // window); -1 sentinel until the first sample.
  clock_t lastCPU_diag_ = -1, lastSysCPU_diag_ = 0, lastUserCPU_diag_ = 0;
  std::vector<double> cpu_percents_diag_;

  // Parameters
  std::string version_;
  int num_threads_;

  bool deskew_;

  double gravity_;

  bool time_offset_;

  bool adaptive_params_;


  double keyframe_thresh_dist_;
  double keyframe_thresh_rot_;

  int max_keyframes_;
  bool dashboard_;

  int submap_knn_;
  int submap_kcv_;
  int submap_kcc_;

  bool densemap_filtered_;
  bool wait_until_move_;

  double crop_size_;

  bool vf_use_;
  double vf_res_;

  bool imu_calibrate_;
  bool calibrate_gyro_;
  bool calibrate_accel_;
  bool gravity_align_;
  double imu_calib_time_;
  int imu_buffer_size_;
  Eigen::Matrix3f imu_accel_sm_;

  int gicp_min_num_points_;
  int gicp_k_correspondences_;
  double gicp_max_corr_dist_;
  int gicp_max_iter_;
  double gicp_transformation_ep_;
  double gicp_rotation_ep_;
  double gicp_init_lambda_factor_;

  std::vector<double> pose_cov_;
  std::vector<double> twist_cov_;

  double geo_Kp_;
  double geo_Kv_;
  double geo_Kq_;
  double geo_Kab_;
  double geo_Kgb_;
  double geo_abias_max_;
  double geo_gbias_max_;
  // Intensity range correction
  double intensity_alpha_;
  double intensity_r_ref_;
  // Photometric channel: false = intensity, true = reflectivity
  bool use_reflectivity_;
  // photometric term enabled (weight > 0); gates the intensity range correction
  bool photometric_active_;

  // --- Direct visual (camera) photometric term (off by default) ---
  // Frame-to-frame direct image alignment that constrains the LiDAR-degenerate
  // tunnel axis (see nano_gicp accumulateVisualResidual). All images are kept
  // undistorted, single-channel, normalized CV_32F.
  bool visual_enabled_;
  double visual_weight_;
  double visual_huber_delta_;
  double visual_max_dt_;                 // max |image_stamp - scan_stamp| [s]
  double visual_gate_max_trans_;         // gate safety floor: max step [m]
  double visual_gate_max_rot_;           // gate safety floor: max step [rad]
  std::vector<double> camera_intrinsics_;  // fx, fy, cx, cy
  std::vector<double> camera_distortion_;  // plumb_bob k1,k2,p1,p2,k3
  Eigen::Matrix4f cam2lidar_T_;            // T_lidar_cam (maps cam point -> lidar)

  std::mutex image_mtx_;
  std::deque<std::pair<double, cv::Mat>> image_buffer_;  // (stamp, undistorted gray f32)
  cv::Mat vis_map1_, vis_map2_;          // undistort rectify maps (built lazily)
  std::atomic<bool> visual_maps_ready_;

  // Previous frame carried scan-to-scan (the warp target for the next scan).
  cv::Mat visual_prev_img_;
  Eigen::Isometry3f visual_T_cw_prev_;
  bool visual_has_prev_;
  // Stash of this scan's frame, promoted to "previous" after align().
  cv::Mat visual_cur_pending_;
  bool visual_cur_pending_valid_;

  // --- Frame-to-MAP camera term (absolute anchor to map landmarks) ---
  bool visual_map_enabled_;
  double visual_map_weight_;
  double visual_map_gate_max_trans_;   // [m]  per-scan rescue budget (absolute anchor)
  double visual_map_gate_max_rot_;     // [rad]
  double visual_map_view_angle_;       // [rad] max viewing-ray deviation before a ref is dropped
  // Per-keyframe reference brightness/ray (sampled at creation, index-aligned
  // with `keyframes`); concatenated into submap_visual_refs in buildSubmap.
  std::vector<std::shared_ptr<const nano_gicp::VisualRefList>> keyframe_visual_refs;
  std::shared_ptr<const nano_gicp::VisualRefList> submap_visual_refs;

  // Sample per-point camera reference brightness for a keyframe cloud (world
  // points at the prior) given that keyframe's world->camera transform + image.
  std::shared_ptr<const nano_gicp::VisualRefList>
  sampleKeyframeVisualRefs(const pcl::PointCloud<PointType>::ConstPtr& cloud,
                           const Eigen::Isometry3f& T_cw, const cv::Mat& img);

  // --- COIN-LIO LiDAR intensity-image term ---
  bool lidar_image_enabled_;
  double lidar_image_weight_;
  cv::Mat lidar_refl_img_;             // current scan reflectivity image (/scale), CV_32FC1
  cv::Mat lidar_range_img_;            // current scan range image [m], CV_32FC1 (occlusion check)
  float lidar_az_a_, lidar_az_b_, lidar_el_a_, lidar_el_b_;  // self-calibrated spherical model
  std::vector<float> lidar_el_lut_;    // per-row mean elevation [rad] (non-uniform beams)
  double lidar_range_abs_tol_, lidar_range_rel_tol_;  // occlusion tolerance [m], fraction
  bool lidar_proj_ready_;
  bool lidar_img_ready_;               // a usable image was built for this scan
  // Build the reflectivity image + spherical projection model from an organized
  // scan (called in getScanFromROS before NaN removal); no-op if not organized.
  void buildLidarIntensityImage(const pcl::PointCloud<PointType>::ConstPtr& organized,
                                int width, int height);

};
