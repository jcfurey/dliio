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

#include "dlio/odom.h"
#include "dlio/utils.h"
#include <set>

#include <queue>

#include "rclcpp/qos.hpp"
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#ifdef HAVE_LIVOX_ROS_DRIVER2
#include <livox_ros_driver2/msg/custom_msg.hpp>

namespace {
// Convert a Livox custom message to a PointType cloud and serialize it as a
// PointCloud2. Each point's timestamp is stored as ABSOLUTE nanoseconds (a
// double = base_ns + per-point offset_time) so getScanFromROS classifies the
// cloud as LIVOX (timestamp > 1e14) and deskew recovers absolute seconds via
// the existing LIVOX branch -- no separate point struct needed.
sensor_msgs::msg::PointCloud2 livoxToPointCloud2(
    const livox_ros_driver2::msg::CustomMsg& livox, const std::string& frame_id) {
  const double base_ns = static_cast<double>(livox.header.stamp.sec) * 1e9
                       + static_cast<double>(livox.header.stamp.nanosec);

  pcl::PointCloud<dlio::Point> cloud;
  cloud.points.reserve(livox.point_num);
  for (std::uint32_t i = 0; i < livox.point_num; ++i) {
    const auto& src = livox.points[i];
    dlio::Point p;
    p.x = src.x;
    p.y = src.y;
    p.z = src.z;
    p.intensity = static_cast<float>(src.reflectivity);
    p.reflectivity = static_cast<float>(src.reflectivity);
    p.timestamp = base_ns + static_cast<double>(src.offset_time);
    cloud.points.push_back(p);
  }
  cloud.width = static_cast<std::uint32_t>(cloud.points.size());
  cloud.height = 1;
  cloud.is_dense = true;

  sensor_msgs::msg::PointCloud2 out;
  pcl::toROSMsg(cloud, out);
  out.header.stamp = livox.header.stamp;
  out.header.frame_id = frame_id;
  return out;
}
}  // namespace
#endif

// Keep statistics history vectors bounded: drop the oldest half once they
// exceed max_size, so long runs don't grow memory (and debug() stays O(window)).
template <typename T>
static void cap_history(std::vector<T>& v, size_t max_size = 1000) {
  if (v.size() > max_size) {
    v.erase(v.begin(), v.begin() + v.size()/2);
  }
}

// Bilinear sample of a single-channel CV_32F image (caller guarantees the 2x2
// neighborhood is in bounds). Shared by the keyframe visual-ref sampler.
static inline float bilinearF(const cv::Mat& img, float u, float v) {
  const int x0 = static_cast<int>(std::floor(u));
  const int y0 = static_cast<int>(std::floor(v));
  const float ax = u - x0, ay = v - y0;
  const float* r0 = img.ptr<float>(y0);
  const float* r1 = img.ptr<float>(y0 + 1);
  const float top = r0[x0] * (1.f - ax) + r0[x0 + 1] * ax;
  const float bot = r1[x0] * (1.f - ax) + r1[x0 + 1] * ax;
  return top * (1.f - ay) + bot * ay;
}

dlio::OdomNode::OdomNode(const rclcpp::NodeOptions& options)
    : Node("dlio_odom_node", options) {

  this->getParams();

  this->num_threads_ = omp_get_max_threads();

  this->dlio_initialized = false;
  this->first_valid_scan = false;
  this->first_imu_received = false;
  if (this->imu_calibrate_) {this->imu_calibrated = false;}
  else {this->imu_calibrated = true;}
  this->deskew_status = false;
  this->deskew_size = 0;

  // NOTE: declare with slash-separated names via dlio::declare_param so these match
  // the keys in cfg/params.yaml. Dot-separated names do NOT match the YAML keys and
  // silently fall back to the defaults (which left the photometric term disabled).
  double photometricWeight;
  dlio::declare_param(this, "odom/gicp/photometricWeight", photometricWeight, 0.0,
      "Weight of the photometric GICP residual relative to the geometric term (live-tunable)", 0.0, 10.0);

  // Intensity range correction parameters
  dlio::declare_param(this, "odom/preprocessing/intensityAlpha", this->intensity_alpha_, 2.0,
      "Intensity range-correction falloff exponent (2.0 = inverse-square)");
  dlio::declare_param(this, "odom/preprocessing/intensityRRef", this->intensity_r_ref_, 1.0,
      "Intensity range-correction reference range [m]");
  dlio::declare_param(this, "odom/preprocessing/intensityIncidence", this->intensity_incidence_, false,
      "Also divide intensity by cos(incidence angle) (Kashani radiometric model); organized scans only");
  dlio::declare_param(this, "odom/preprocessing/intensityCosMin", this->intensity_cos_min_, 0.2,
      "Floor on cos(incidence) to avoid grazing-angle blow-up (0.2 ~= 78 deg)");
  int gradientKNeighbors;
  dlio::declare_param(this, "odom/gicp/gradientKNeighbors", gradientKNeighbors, 10,
      "Neighbors used to estimate the spatial photometric gradient on the submap");

  // Photometric channel: "intensity" (range-dependent; pairs with the range
  // correction below) or "reflectivity" (e.g. Ouster calibrated reflectivity,
  // already range-normalized -> the range correction is skipped for it).
  std::string photometricChannel;
  dlio::declare_param(this, "odom/gicp/photometricChannel", photometricChannel, std::string("intensity"),
      "Point field feeding the photometric term: 'intensity' or 'reflectivity'");
  this->use_reflectivity_ = (photometricChannel == "reflectivity");
  this->photometric_active_ = (photometricWeight > 0.0);
  if (photometricChannel != "intensity" && photometricChannel != "reflectivity") {
    RCLCPP_WARN(this->get_logger(),
        "Unknown odom/gicp/photometricChannel '%s'; defaulting to 'intensity'.",
        photometricChannel.c_str());
  }

  this->gicp.setPhotometricWeight(photometricWeight);
  this->gicp.setGradientKNeighbors(gradientKNeighbors);
  this->gicp.setPhotometricChannel(this->use_reflectivity_);

  // Full-scale of the photometric channel; the channel is normalized by this
  // so photometricWeight is sensor-independent (255 covers 8-bit intensity and
  // Ouster calibrated reflectivity; use 65535 for raw 16-bit channels).
  double photometricScale;
  dlio::declare_param(this, "odom/gicp/photometricScale", photometricScale, 255.0,
      "Full-scale of the photometric channel (channel is divided by this; 255 for 8-bit / Ouster reflectivity)");
  this->gicp.setPhotometricScale(static_cast<float>(photometricScale));

  // Huber threshold on the normalized photometric residual; residuals beyond
  // it are downweighted so specular/wet-surface outliers can't shove the
  // pose at full weight. <= 0 disables.
  double photometricHuberDelta;
  dlio::declare_param(this, "odom/gicp/photometricHuberDelta", photometricHuberDelta, 0.05,
      "Huber threshold on the normalized photometric residual (live-tunable; <=0 disables)", 0.0, 1.0);
  this->gicp.setPhotometricHuberDelta(static_cast<float>(photometricHuberDelta));

  // GICP covariance regularization:
  //   min_eig (default)    - clamp small singular values (this fork's historical
  //                          behavior, previously mislabeled "plane")
  //   plane                - true GICP plane-to-plane: fixed (1, 1, 1e-3) discs
  //   normalized_min_eig   - scale-normalized clamp
  //   frobenius | none
  std::string regularizationMethod;
  dlio::declare_param(this, "odom/gicp/regularizationMethod", regularizationMethod, std::string("min_eig"),
      "GICP covariance regularization: min_eig | plane | normalized_min_eig | frobenius | none");
  nano_gicp::RegularizationMethod reg_method = nano_gicp::RegularizationMethod::MIN_EIG;
  if (regularizationMethod == "plane") { reg_method = nano_gicp::RegularizationMethod::PLANE; }
  else if (regularizationMethod == "normalized_min_eig") { reg_method = nano_gicp::RegularizationMethod::NORMALIZED_MIN_EIG; }
  else if (regularizationMethod == "frobenius") { reg_method = nano_gicp::RegularizationMethod::FROBENIUS; }
  else if (regularizationMethod == "none") { reg_method = nano_gicp::RegularizationMethod::NONE; }
  else if (regularizationMethod != "min_eig") {
    RCLCPP_WARN(this->get_logger(),
        "Unknown odom/gicp/regularizationMethod '%s'; using 'min_eig'.",
        regularizationMethod.c_str());
  }
  this->gicp.setRegularizationMethod(reg_method);

  // Degeneracy gate for geometrically self-similar environments (featureless
  // tunnels/conduits): rotation/translation Hessian blocks are eigen-analyzed
  // separately and the GICP update is projected off directions below
  // ratio * block_lambda_max, holding the IMU prior there. 0 disables.
  double degeneracyThreshRatio;
  dlio::declare_param(this, "odom/gicp/degeneracyThreshRatio", degeneracyThreshRatio, 0.005,
      "Degeneracy gate ratio (live-tunable; 0 disables)", 0.0, 1.0);
  this->gicp.setDegeneracyThreshRatio(static_cast<float>(degeneracyThreshRatio));

  // gicp_temp prepares the submap target (kd-tree + photometric gradients) in
  // the background thread, so it needs the same photometric configuration.
  this->gicp_temp.setPhotometricWeight(photometricWeight);
  this->gicp_temp.setGradientKNeighbors(gradientKNeighbors);
  this->gicp_temp.setPhotometricChannel(this->use_reflectivity_);
  this->gicp_temp.setPhotometricScale(static_cast<float>(photometricScale));
  this->gicp_temp.setRegularizationMethod(reg_method);

  this->lidar_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  auto lidar_sub_opt = rclcpp::SubscriptionOptions();
  lidar_sub_opt.callback_group = this->lidar_cb_group;
  // SensorDataQoS (best-effort): LiDAR drivers commonly publish sensor data
  // best-effort; a reliable subscription would be QoS-incompatible and
  // silently receive nothing.
  this->lidar_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>("pointcloud",
      rclcpp::SensorDataQoS().keep_last(1),
      std::bind(&dlio::OdomNode::callbackPointCloud, this, std::placeholders::_1), lidar_sub_opt);

  this->imu_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  auto imu_sub_opt = rclcpp::SubscriptionOptions();
  imu_sub_opt.callback_group = this->imu_cb_group;
  this->imu_sub = this->create_subscription<sensor_msgs::msg::Imu>("imu", rclcpp::SensorDataQoS(),
      std::bind(&dlio::OdomNode::callbackImu, this, std::placeholders::_1), imu_sub_opt);

  // Camera image for the optional direct visual term (off by default). Only
  // subscribed when enabled, so the LIO path is untouched otherwise.
  if (this->visual_enabled_) {
    this->image_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    auto image_sub_opt = rclcpp::SubscriptionOptions();
    image_sub_opt.callback_group = this->image_cb_group;
    this->image_sub = this->create_subscription<sensor_msgs::msg::Image>("camera",
        rclcpp::SensorDataQoS().keep_last(10),
        std::bind(&dlio::OdomNode::callbackImage, this, std::placeholders::_1), image_sub_opt);
    RCLCPP_INFO(this->get_logger(),
        "Direct visual term ENABLED (weight %.3f); subscribing to 'camera'.",
        this->visual_weight_);
  }

  this->odom_pub     = this->create_publisher<nav_msgs::msg::Odometry>("odom", 1);
  this->pose_pub     = this->create_publisher<geometry_msgs::msg::PoseStamped>("pose", 1);
  this->path_pub     = this->create_publisher<nav_msgs::msg::Path>("path", 1);
  this->kf_pose_pub  = this->create_publisher<geometry_msgs::msg::PoseArray>("kf_pose", 1);
  this->kf_cloud_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("kf_cloud", 1);
  this->deskewed_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("deskewed", 1);

#ifdef HAVE_LIVOX_ROS_DRIVER2
  // Optional raw Livox ingestion: subscribe to a livox_ros_driver2 CustomMsg on
  // 'livox', republish it as PointCloud2 on 'livox2dlio'. To use a raw Livox
  // stream, remap the cloud input to it (pointcloud:=livox2dlio). Decoupled from
  // the estimator threading -- it's purely a format shim in its own callback group.
  this->livox_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  auto livox_sub_opt = rclcpp::SubscriptionOptions();
  livox_sub_opt.callback_group = this->livox_cb_group;
  this->livox_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "livox2dlio", rclcpp::SensorDataQoS().keep_last(1));
  this->livox_sub = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(
      "livox", rclcpp::SensorDataQoS().keep_last(1),
      [this](const livox_ros_driver2::msg::CustomMsg::SharedPtr msg) {
        this->livox_pub->publish(livoxToPointCloud2(*msg, this->lidar_frame));
      }, livox_sub_opt);
  RCLCPP_INFO(this->get_logger(),
      "Livox CustomMsg ingestion ENABLED: converting 'livox' -> 'livox2dlio' "
      "(remap pointcloud:=livox2dlio to use it).");
#endif
  // Absolute /diagnostics (unlike the relative pubs above, this is NOT remapped
  // by the launch file) so the standard diagnostics topic always lands at /diagnostics.
  this->diag_pub = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
      "/diagnostics", rclcpp::QoS(10).best_effort());

  // Live parameter tuning (ros2 param set during a run). Registered after all
  // params are declared so initialization doesn't trip it.
  this->on_set_handle_ = this->add_on_set_parameters_callback(
      std::bind(&dlio::OdomNode::onSetParams, this, std::placeholders::_1));

  this->br = std::make_shared<tf2_ros::TransformBroadcaster>(*this);
  this->static_br = std::make_shared<tf2_ros::StaticTransformBroadcaster>(*this);
  if (this->extrinsics_source_ == "tf") {
    // robot_state_publisher owns base_link->sensor TF from the URDF; resolve the
    // extrinsics from tf2 instead of publishing our own static transforms.
    // Gate processing until they're available (the YAML values remain as a
    // fallback if the lookups never succeed).
    this->extrinsics_ready_.store(false);
    this->tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    this->tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*this->tf_buffer_);
    this->extrinsics_timer_ = this->create_wall_timer(
        std::chrono::duration<double>(0.5),
        std::bind(&dlio::OdomNode::resolveExtrinsicsFromTf, this));
    RCLCPP_INFO(this->get_logger(),
        "extrinsics/source=tf: waiting for base_link->{imu,lidar} from tf2...");
  } else {
    this->publishStaticTransforms();
  }

  // constant diagonal covariance on the published odometry (set once;
  // the message object is reused by publishPose)
  for (int i = 0; i < 6; i++) {
    this->odom_ros.pose.covariance[i*7] = this->pose_cov_[i];
    this->odom_ros.twist.covariance[i*7] = this->twist_cov_[i];
  }
  // Frame ids are constant too -- set them once here rather than re-assigning
  // these std::string members on every 100 Hz publishPose tick (the reused
  // message objects are only touched by the pose timer).
  this->odom_ros.header.frame_id = this->odom_frame;
  this->odom_ros.child_frame_id = this->baselink_frame;
  this->pose_ros.header.frame_id = this->odom_frame;

  this->publish_timer = this->create_wall_timer(std::chrono::duration<double>(0.01), 
      std::bind(&dlio::OdomNode::publishPose, this));

  this->T = Eigen::Matrix4f::Identity();
  this->T_prior = Eigen::Matrix4f::Identity();
  this->T_corr = Eigen::Matrix4f::Identity();

  this->origin = Eigen::Vector3f(0., 0., 0.);
  this->state.p = Eigen::Vector3f(0., 0., 0.);
  this->state.q = Eigen::Quaternionf(1., 0., 0., 0.);
  this->state.v.lin.b = Eigen::Vector3f(0., 0., 0.);
  this->state.v.lin.w = Eigen::Vector3f(0., 0., 0.);
  this->state.v.ang.b = Eigen::Vector3f(0., 0., 0.);
  this->state.v.ang.w = Eigen::Vector3f(0., 0., 0.);

  this->lidarPose.p = Eigen::Vector3f(0., 0., 0.);
  this->lidarPose.q = Eigen::Quaternionf(1., 0., 0., 0.);

  this->imu_meas.stamp = 0.;
  this->imu_meas.ang_vel[0] = 0.;
  this->imu_meas.ang_vel[1] = 0.;
  this->imu_meas.ang_vel[2] = 0.;
  this->imu_meas.lin_accel[0] = 0.;
  this->imu_meas.lin_accel[1] = 0.;
  this->imu_meas.lin_accel[2] = 0.;

  this->imu_buffer.set_capacity(this->imu_buffer_size_);
  this->first_imu_stamp = 0.;
  this->prev_imu_stamp = 0.;

  this->original_scan = std::make_shared<const pcl::PointCloud<PointType>>();
  this->deskewed_scan = std::make_shared<const pcl::PointCloud<PointType>>();
  this->current_scan = std::make_shared<const pcl::PointCloud<PointType>>();
  this->submap_cloud = std::make_shared<const pcl::PointCloud<PointType>>();

  // Visual term runtime state
  this->visual_maps_ready_ = false;
  this->visual_has_prev_ = false;
  this->visual_cur_pending_valid_ = false;
  this->visual_T_cw_prev_ = Eigen::Isometry3f::Identity();
  this->lidar_proj_ready_ = false;
  this->lidar_img_ready_ = false;
  this->lidar_az_a_ = 1.f; this->lidar_az_b_ = 0.f;
  this->lidar_el_a_ = 1.f; this->lidar_el_b_ = 0.f;

  this->num_processed_keyframes = 0;

  this->submap_hasChanged = true;
  this->submap_kf_idx_prev.clear();

  this->first_scan_stamp = 0.;
  this->elapsed_time = 0.;
  this->length_traversed = 0.;
  this->length_prev_p = Eigen::Vector3f(0., 0., 0.);

  this->convex_hull.setDimension(3);
  this->concave_hull.setDimension(3);
  this->concave_hull.setAlpha(this->keyframe_thresh_dist_);
  this->concave_hull.setKeepInformation(true);

  this->gicp.setCorrespondenceRandomness(this->gicp_k_correspondences_);
  this->gicp.setMaxCorrespondenceDistance(this->gicp_max_corr_dist_);
  this->gicp.setMaximumIterations(this->gicp_max_iter_);
  this->gicp.setTransformationEpsilon(this->gicp_transformation_ep_);
  this->gicp.setRotationEpsilon(this->gicp_rotation_ep_);
  this->gicp.setInitialLambdaFactor(this->gicp_init_lambda_factor_);

  this->gicp_temp.setCorrespondenceRandomness(this->gicp_k_correspondences_);
  this->gicp_temp.setMaxCorrespondenceDistance(this->gicp_max_corr_dist_);
  this->gicp_temp.setMaximumIterations(this->gicp_max_iter_);
  this->gicp_temp.setTransformationEpsilon(this->gicp_transformation_ep_);
  this->gicp_temp.setRotationEpsilon(this->gicp_rotation_ep_);
  this->gicp_temp.setInitialLambdaFactor(this->gicp_init_lambda_factor_);

  pcl::Registration<PointType, PointType>::KdTreeReciprocalPtr temp;
  this->gicp.setSearchMethodSource(temp, true);
  this->gicp.setSearchMethodTarget(temp, true);
  this->gicp_temp.setSearchMethodSource(temp, true);
  this->gicp_temp.setSearchMethodTarget(temp, true);

  this->geo.first_opt_done = false;
  this->geo.prev_vel = Eigen::Vector3f(0., 0., 0.);

  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);

  this->crop.setNegative(true);
  this->crop.setMin(Eigen::Vector4f(-this->crop_size_, -this->crop_size_, -this->crop_size_, 1.0));
  this->crop.setMax(Eigen::Vector4f(this->crop_size_, this->crop_size_, this->crop_size_, 1.0));

  this->voxel.setLeafSize(this->vf_res_, this->vf_res_, this->vf_res_);

  this->metrics.spaciousness.push_back(0.);
  this->metrics.density.push_back(this->gicp_max_corr_dist_);

  // CPU Specs
  char CPUBrandString[0x40];
  memset(CPUBrandString, 0, sizeof(CPUBrandString));

  this->cpu_type = "";

  #ifdef HAS_CPUID
  unsigned int CPUInfo[4] = {0,0,0,0};
  __cpuid(0x80000000, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
  unsigned int nExIds = CPUInfo[0];
  for (unsigned int i = 0x80000000; i <= nExIds; ++i) {
    __cpuid(i, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
    if (i == 0x80000002)
      memcpy(CPUBrandString, CPUInfo, sizeof(CPUInfo));
    else if (i == 0x80000003)
      memcpy(CPUBrandString + 16, CPUInfo, sizeof(CPUInfo));
    else if (i == 0x80000004)
      memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
  }
  this->cpu_type = CPUBrandString;
  boost::trim(this->cpu_type);
  #endif

  FILE* file;
  struct tms timeSample;
  char line[128];

  this->lastCPU = times(&timeSample);
  this->lastSysCPU = timeSample.tms_stime;
  this->lastUserCPU = timeSample.tms_utime;

  file = fopen("/proc/cpuinfo", "r");
  this->numProcessors = 0;
  while(fgets(line, 128, file) != nullptr) {
      if (strncmp(line, "processor", 9) == 0) this->numProcessors++;
  }
  fclose(file);

}

dlio::OdomNode::~OdomNode() {

  // Unblock the background submap thread (it may be paused waiting on
  // main_loop_running) and wait for it; then join the worker threads.
  // Threads are joined (not detached) so a component unload or shutdown
  // cannot leave them running against a destroyed node.
  {
    std::lock_guard<std::mutex> lk(this->main_loop_running_mutex);
    this->main_loop_running = false;
  }
  this->submap_build_cv.notify_all();
  if (this->submap_future.valid()) { this->submap_future.wait(); }

  if (this->publish_thread.joinable()) { this->publish_thread.join(); }
  if (this->publish_keyframe_thread.joinable()) { this->publish_keyframe_thread.join(); }
  if (this->debug_thread.joinable()) { this->debug_thread.join(); }

}

void dlio::OdomNode::getParams() {

  // Version
  dlio::declare_param(this, "version", this->version_, "0.0.0");

  // Frames
  dlio::declare_param(this, "frames/odom", this->odom_frame, "odom");
  dlio::declare_param(this, "frames/baselink", this->baselink_frame, "base_link");
  dlio::declare_param(this, "frames/lidar", this->lidar_frame, "lidar");
  dlio::declare_param(this, "frames/imu", this->imu_frame, "imu");
  dlio::declare_param(this, "frames/camera", this->camera_frame_, "camera");
  // Extrinsics source: "yaml" (the extrinsics/* params below) or "tf" (look them
  // up from tf2, e.g. published by robot_state_publisher from the robot URDF).
  dlio::declare_param(this, "extrinsics/source", this->extrinsics_source_, std::string("yaml"));

  // Deskew Flag
  dlio::declare_param(this, "pointcloud/deskew", this->deskew_, true);

  // Gravity
  dlio::declare_param(this, "odom/gravity", this->gravity_, 9.80665);

  // Compute time offset between lidar and imu
  dlio::declare_param(this, "odom/computeTimeOffset", this->time_offset_, false);

  // Keyframe Threshold
  dlio::declare_param(this, "odom/keyframe/threshD", this->keyframe_thresh_dist_, 0.1,
      "Keyframe translation threshold [m] (live-tunable)", 0.0, 10.0);
  dlio::declare_param(this, "odom/keyframe/threshR", this->keyframe_thresh_rot_, 1.0,
      "Keyframe rotation threshold [deg] (live-tunable)", 0.0, 180.0);

  // Bound on the keyframe map (0 = unlimited). When exceeded, the most
  // spatially redundant processed keyframe is removed.
  dlio::declare_param(this, "odom/keyframe/maxKeyframes", this->max_keyframes_, 0,
      "Bound on the keyframe map; most redundant keyframe pruned when exceeded (0 = unlimited)");

  // Terminal status dashboard (ANSI clear-screen); disable when logs are
  // multiplexed (ros2 launch, containers, systemd).
  dlio::declare_param(this, "odom/debug/dashboard", this->dashboard_, true,
      "Terminal ANSI status dashboard; disable under multiplexed logging");

  // Submap
  dlio::declare_param(this, "odom/submap/keyframe/knn", this->submap_knn_, 10);
  dlio::declare_param(this, "odom/submap/keyframe/kcv", this->submap_kcv_, 10);
  dlio::declare_param(this, "odom/submap/keyframe/kcc", this->submap_kcc_, 10);

  // Dense map resolution
  dlio::declare_param(this, "map/dense/filtered", this->densemap_filtered_, true);

  // Wait until movement to publish map
  dlio::declare_param(this, "map/waitUntilMove", this->wait_until_move_, false);

  // Crop Box Filter
  dlio::declare_param(this, "odom/preprocessing/cropBoxFilter/size", this->crop_size_, 1.0);

  // Voxel Grid Filter
  dlio::declare_param(this, "pointcloud/voxelize", this->vf_use_, true);
  dlio::declare_param(this, "odom/preprocessing/voxelFilter/res", this->vf_res_, 0.05);

  // Adaptive Parameters
  dlio::declare_param(this, "adaptive", this->adaptive_params_, true);

  // Extrinsics
  std::vector<double> t_default{0., 0., 0.};
  std::vector<double> R_default{1., 0., 0., 0., 1., 0., 0., 0., 1.};

  // center of gravity to imu
  std::vector<double> baselink2imu_t, baselink2imu_R;
  dlio::declare_param(this, "extrinsics/baselink2imu/t", baselink2imu_t, t_default);
  dlio::declare_param(this, "extrinsics/baselink2imu/R", baselink2imu_R, R_default);
  this->extrinsics.baselink2imu.t =
    Eigen::Vector3f(baselink2imu_t[0], baselink2imu_t[1], baselink2imu_t[2]);
  this->extrinsics.baselink2imu.R =
    Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(std::vector<float>(baselink2imu_R.begin(), baselink2imu_R.end()).data(), 3, 3);
  this->extrinsics.baselink2imu_T = Eigen::Matrix4f::Identity();
  this->extrinsics.baselink2imu_T.block(0, 3, 3, 1) = this->extrinsics.baselink2imu.t;
  this->extrinsics.baselink2imu_T.block(0, 0, 3, 3) = this->extrinsics.baselink2imu.R;

  // center of gravity to lidar
  std::vector<double> baselink2lidar_t, baselink2lidar_R;
  dlio::declare_param(this, "extrinsics/baselink2lidar/t", baselink2lidar_t, t_default);
  dlio::declare_param(this, "extrinsics/baselink2lidar/R", baselink2lidar_R, R_default);

  this->extrinsics.baselink2lidar.t =
    Eigen::Vector3f(baselink2lidar_t[0], baselink2lidar_t[1], baselink2lidar_t[2]);
  this->extrinsics.baselink2lidar.R =
    Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(std::vector<float>(baselink2lidar_R.begin(), baselink2lidar_R.end()).data(), 3, 3);

  this->extrinsics.baselink2lidar_T = Eigen::Matrix4f::Identity();
  this->extrinsics.baselink2lidar_T.block(0, 3, 3, 1) = this->extrinsics.baselink2lidar.t;
  this->extrinsics.baselink2lidar_T.block(0, 0, 3, 3) = this->extrinsics.baselink2lidar.R;

  // camera -> lidar extrinsic (same storage convention as baselink2lidar: the
  // R,t block maps a camera-frame point into the lidar frame, i.e. T_lidar_cam)
  std::vector<double> cam2lidar_t, cam2lidar_R;
  dlio::declare_param(this, "extrinsics/cam2lidar/t", cam2lidar_t, t_default);
  dlio::declare_param(this, "extrinsics/cam2lidar/R", cam2lidar_R, R_default);
  this->cam2lidar_T_ = Eigen::Matrix4f::Identity();
  this->cam2lidar_T_.block(0, 3, 3, 1) =
      Eigen::Vector3f(cam2lidar_t[0], cam2lidar_t[1], cam2lidar_t[2]);
  this->cam2lidar_T_.block(0, 0, 3, 3) =
      Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(
          std::vector<float>(cam2lidar_R.begin(), cam2lidar_R.end()).data(), 3, 3);

  // Direct visual (camera) photometric term (off by default).
  dlio::declare_param(this, "odom/visual/enabled", this->visual_enabled_, false,
      "Enable the direct visual (camera) photometric residual (constrains the LiDAR-degenerate tunnel axis)");
  dlio::declare_param(this, "odom/visual/weight", this->visual_weight_, 0.0,
      "Weight of the visual photometric residual relative to the geometric GICP term");
  dlio::declare_param(this, "odom/visual/huberDelta", this->visual_huber_delta_, 0.05,
      "Huber threshold on the normalized visual residual (<= 0 disables robustification)");
  dlio::declare_param(this, "odom/visual/maxTimeDiff", this->visual_max_dt_, 0.05,
      "Max |image_stamp - scan_stamp| [s] to pair a camera frame with a scan");
  // Degeneracy-gate safety floor: when the visual term rescues a LiDAR-
  // degenerate axis, its total deviation from the IMU prior along that axis is
  // bounded to these PER-SCAN budgets (summed across LM iterations), so a
  // wrong/biased visual constraint can neither diverge nor inflate the path.
  dlio::declare_param(this, "odom/visual/gateMaxStepTrans", this->visual_gate_max_trans_, 0.3,
      "Visual gate safety floor: max per-scan translation correction on a rescued degenerate axis [m]");
  dlio::declare_param(this, "odom/visual/gateMaxStepRot", this->visual_gate_max_rot_, 0.05,
      "Visual gate safety floor: max per-scan rotation correction on a rescued degenerate axis [rad]");

  // Frame-to-MAP camera term: anchors absolute position to map landmarks (wall
  // texture/graffiti), breaking the geometric self-similarity that drags the
  // pose back in a tunnel. Off by default; requires odom/visual/enabled too.
  dlio::declare_param(this, "odom/visual/map/enabled", this->visual_map_enabled_, false,
      "Enable the frame-to-MAP camera photometric term (absolute anchor to keyframe landmarks)");
  dlio::declare_param(this, "odom/visual/map/weight", this->visual_map_weight_, 0.0,
      "Weight of the frame-to-map camera residual relative to the geometric GICP term");
  dlio::declare_param(this, "odom/visual/map/gateMaxStepTrans", this->visual_map_gate_max_trans_, 1.0,
      "Per-scan rescue budget for the absolute map anchor on a degenerate translation axis [m]");
  dlio::declare_param(this, "odom/visual/map/gateMaxStepRot", this->visual_map_gate_max_rot_, 0.1,
      "Per-scan rescue budget for the absolute map anchor on a degenerate rotation axis [rad]");
  dlio::declare_param(this, "odom/visual/map/viewAngleMax", this->visual_map_view_angle_, 0.6,
      "Max viewing-ray deviation [rad] between a map ref and the current view before it is dropped");

  // COIN-LIO LiDAR intensity-image term: anchors absolute position to wall
  // texture in the LiDAR reflectivity image (360deg coverage; better suited to
  // this rig than the narrow camera). Reuses the absolute-anchor gate budget
  // (odom/visual/map/gateMaxStep*). OFF by default; needs an organized scan.
  dlio::declare_param(this, "odom/lidar_image/enabled", this->lidar_image_enabled_, false,
      "Enable the COIN-LIO LiDAR intensity-image frame-to-map term (organized scan + reflectivity)");
  dlio::declare_param(this, "odom/lidar_image/weight", this->lidar_image_weight_, 0.0,
      "Weight of the LiDAR intensity-image residual relative to the geometric GICP term");
  dlio::declare_param(this, "odom/lidar_image/rangeAbsTol", this->lidar_range_abs_tol_, 0.5,
      "Occlusion check: absolute range tolerance [m] for accepting a projected map point");
  dlio::declare_param(this, "odom/lidar_image/rangeRelTol", this->lidar_range_rel_tol_, 0.1,
      "Occlusion check: relative range tolerance (fraction of pixel range)");
  // Camera intrinsics (fx, fy, cx, cy) and plumb_bob distortion (k1,k2,p1,p2,k3).
  // Defaults are the 06042026 bag's embedded /lucid_camera_1 camera_info.
  std::vector<double> intr_default{1094.19, 1092.23, 969.58, 721.31};
  std::vector<double> dist_default{-0.04409, 0.05337, -0.00124, 0.00002, 0.0};
  dlio::declare_param(this, "camera/intrinsics", this->camera_intrinsics_, intr_default);
  dlio::declare_param(this, "camera/distortion", this->camera_distortion_, dist_default);
  if (this->camera_intrinsics_.size() != 4) { this->camera_intrinsics_ = intr_default; }
  if (this->camera_distortion_.size() < 4)  { this->camera_distortion_ = dist_default; }

  // IMU
  dlio::declare_param(this, "odom/imu/calibration/accel", this->calibrate_accel_, true);
  dlio::declare_param(this, "odom/imu/calibration/gyro", this->calibrate_gyro_, true);
  dlio::declare_param(this, "odom/imu/calibration/time", this->imu_calib_time_, 3.0);
  dlio::declare_param(this, "odom/imu/bufferSize", this->imu_buffer_size_, 2000);

  std::vector<double> accel_default{0., 0., 0.}; std::vector<double> prior_accel_bias;
  std::vector<double> gyro_default{0., 0., 0.}; std::vector<double> prior_gyro_bias;

  dlio::declare_param(this, "odom/imu/approximateGravity", this->gravity_align_, true);
  dlio::declare_param(this, "imu/calibration", this->imu_calibrate_, true);
  // Scale incoming accel by gravity (for IMUs that report it in units of g,
  // e.g. Livox built-in IMUs); default off keeps the m/s^2 convention.
  dlio::declare_param(this, "imu/normalized", this->imu_normalized_, false);
  dlio::declare_param(this, "imu/intrinsics/accel/bias", prior_accel_bias, accel_default);
  dlio::declare_param(this, "imu/intrinsics/gyro/bias", prior_gyro_bias, gyro_default);

  // scale-misalignment matrix
  std::vector<double> imu_sm_default{1., 0., 0., 0., 1., 0., 0., 0., 1.};
  std::vector<double> imu_sm;

  dlio::declare_param(this, "imu/intrinsics/accel/sm", imu_sm, imu_sm_default);

  if (!this->imu_calibrate_) {
    this->state.b.accel[0] = prior_accel_bias[0];
    this->state.b.accel[1] = prior_accel_bias[1];
    this->state.b.accel[2] = prior_accel_bias[2];
    this->state.b.gyro[0] = prior_gyro_bias[0];
    this->state.b.gyro[1] = prior_gyro_bias[1];
    this->state.b.gyro[2] = prior_gyro_bias[2];
    this->imu_accel_sm_ = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(std::vector<float>(imu_sm.begin(), imu_sm.end()).data(), 3, 3);
  } else {
    this->state.b.accel = Eigen::Vector3f(0., 0., 0.);
    this->state.b.gyro = Eigen::Vector3f(0., 0., 0.);
    this->imu_accel_sm_ = Eigen::Matrix3f::Identity();
  }

  // GICP
  dlio::declare_param(this, "odom/gicp/minNumPoints", this->gicp_min_num_points_, 100);
  dlio::declare_param(this, "odom/gicp/kCorrespondences", this->gicp_k_correspondences_, 20);
  dlio::declare_param(this, "odom/gicp/maxCorrespondenceDistance", this->gicp_max_corr_dist_,
      std::sqrt(std::numeric_limits<double>::max()));
  dlio::declare_param(this, "odom/gicp/maxIterations", this->gicp_max_iter_, 64);
  dlio::declare_param(this, "odom/gicp/transformationEpsilon", this->gicp_transformation_ep_, 0.0005);
  dlio::declare_param(this, "odom/gicp/rotationEpsilon", this->gicp_rotation_ep_, 0.0005);
  dlio::declare_param(this, "odom/gicp/initLambdaFactor", this->gicp_init_lambda_factor_, 1e-9);

  // Published odometry covariance (diagonal: x y z roll pitch yaw).
  // All-zero covariance makes the odometry unusable for downstream fusion
  // (robot_localization etc. either reject it or trust it infinitely).
  std::vector<double> cov_default{0.01, 0.01, 0.01, 0.0025, 0.0025, 0.0025};
  dlio::declare_param(this, "odom/covariance/pose", this->pose_cov_, cov_default);
  dlio::declare_param(this, "odom/covariance/twist", this->twist_cov_, cov_default);
  if (this->pose_cov_.size() != 6) { this->pose_cov_ = cov_default; }
  if (this->twist_cov_.size() != 6) { this->twist_cov_ = cov_default; }

  // Geometric Observer
  dlio::declare_param(this, "odom/geo/Kp", this->geo_Kp_, 1.0, "Observer position gain (live-tunable)", 0.0, 100.0);
  dlio::declare_param(this, "odom/geo/Kv", this->geo_Kv_, 1.0, "Observer velocity gain (live-tunable)", 0.0, 100.0);
  dlio::declare_param(this, "odom/geo/Kq", this->geo_Kq_, 1.0, "Observer orientation gain (live-tunable)", 0.0, 100.0);
  dlio::declare_param(this, "odom/geo/Kab", this->geo_Kab_, 1.0, "Observer accel-bias gain (live-tunable)", 0.0, 100.0);
  dlio::declare_param(this, "odom/geo/Kgb", this->geo_Kgb_, 1.0, "Observer gyro-bias gain (live-tunable)", 0.0, 100.0);
  dlio::declare_param(this, "odom/geo/abias_max", this->geo_abias_max_, 1.0, "Accel-bias clamp [m/s^2] (live-tunable)", 0.0, 50.0);
  dlio::declare_param(this, "odom/geo/gbias_max", this->geo_gbias_max_, 1.0, "Gyro-bias clamp [rad/s] (live-tunable)", 0.0, 10.0);
}

void dlio::OdomNode::start() {

  printf("\033[2J\033[1;1H");
  std::cout << std::endl
            << "+-------------------------------------------------------------------+" << std::endl;
  std::cout << "|               Direct LiDAR-Inertial Odometry v" << this->version_  << "               |"
            << std::endl;
  std::cout << "+-------------------------------------------------------------------+" << std::endl;

}

void dlio::OdomNode::publishPose() {

  // This timer runs on its own thread; snapshot the state + stamp under geo.mtx
  // (the IMU thread writes them via propagateState/updateState under the same
  // lock) so the published pose is internally consistent, not a torn read.
  State st;
  rclcpp::Time stamp;
  {
    std::lock_guard<std::mutex> lock(this->geo.mtx);
    st = this->state;
    stamp = this->imu_stamp;
  }

  // nav_msgs::msg::Odometry  (frame_id / child_frame_id set once in the ctor)
  this->odom_ros.header.stamp = stamp;

  this->odom_ros.pose.pose.position.x = st.p[0];
  this->odom_ros.pose.pose.position.y = st.p[1];
  this->odom_ros.pose.pose.position.z = st.p[2];

  this->odom_ros.pose.pose.orientation.w = st.q.w();
  this->odom_ros.pose.pose.orientation.x = st.q.x();
  this->odom_ros.pose.pose.orientation.y = st.q.y();
  this->odom_ros.pose.pose.orientation.z = st.q.z();

  this->odom_ros.twist.twist.linear.x = st.v.lin.w[0];
  this->odom_ros.twist.twist.linear.y = st.v.lin.w[1];
  this->odom_ros.twist.twist.linear.z = st.v.lin.w[2];

  this->odom_ros.twist.twist.angular.x = st.v.ang.b[0];
  this->odom_ros.twist.twist.angular.y = st.v.ang.b[1];
  this->odom_ros.twist.twist.angular.z = st.v.ang.b[2];

  this->odom_pub->publish(this->odom_ros);

  // geometry_msgs::msg::PoseStamped  (frame_id set once in the ctor)
  this->pose_ros.header.stamp = stamp;

  this->pose_ros.pose.position.x = st.p[0];
  this->pose_ros.pose.position.y = st.p[1];
  this->pose_ros.pose.position.z = st.p[2];

  this->pose_ros.pose.orientation.w = st.q.w();
  this->pose_ros.pose.orientation.x = st.q.x();
  this->pose_ros.pose.orientation.y = st.q.y();
  this->pose_ros.pose.orientation.z = st.q.z();

  this->pose_pub->publish(this->pose_ros);

}

void dlio::OdomNode::publishToROS(pcl::PointCloud<PointType>::ConstPtr published_cloud, Eigen::Matrix4f T_cloud) {
  this->publishCloud(published_cloud, T_cloud);

  // nav_msgs::msg::Path
  this->path_ros.header.stamp = this->imu_stamp;
  this->path_ros.header.frame_id = this->odom_frame;

  geometry_msgs::msg::PoseStamped p;
  p.header.stamp = this->imu_stamp;
  p.header.frame_id = this->odom_frame;
  p.pose.position.x = this->state.p[0];
  p.pose.position.y = this->state.p[1];
  p.pose.position.z = this->state.p[2];
  p.pose.orientation.w = this->state.q.w();
  p.pose.orientation.x = this->state.q.x();
  p.pose.orientation.y = this->state.q.y();
  p.pose.orientation.z = this->state.q.z();

  this->path_ros.poses.push_back(p);
  // bound the path message: at scan rate an unbounded Path grows quadratically
  // in publish bandwidth over long runs
  if (this->path_ros.poses.size() > 10000) {
    this->path_ros.poses.erase(this->path_ros.poses.begin(),
                               this->path_ros.poses.begin() + 1000);
  }
  this->path_pub->publish(this->path_ros);

  // transform: odom to baselink
  geometry_msgs::msg::TransformStamped transformStamped;

  transformStamped.header.stamp = this->imu_stamp;
  transformStamped.header.frame_id = this->odom_frame;
  transformStamped.child_frame_id = this->baselink_frame;

  transformStamped.transform.translation.x = this->state.p[0];
  transformStamped.transform.translation.y = this->state.p[1];
  transformStamped.transform.translation.z = this->state.p[2];

  transformStamped.transform.rotation.w = this->state.q.w();
  transformStamped.transform.rotation.x = this->state.q.x();
  transformStamped.transform.rotation.y = this->state.q.y();
  transformStamped.transform.rotation.z = this->state.q.z();

  br->sendTransform(transformStamped);

  // baselink->imu and baselink->lidar are fixed extrinsics, so they are NOT
  // re-sent here: in extrinsics/source=yaml they are published once (latched) by
  // the static broadcaster; in =tf robot_state_publisher owns them (see ctor).

}

void dlio::OdomNode::publishStaticTransforms() {

  // transform: baselink to imu
  geometry_msgs::msg::TransformStamped transformStamped;
  transformStamped.header.stamp = this->now();
  transformStamped.header.frame_id = this->baselink_frame;
  transformStamped.child_frame_id = this->imu_frame;

  transformStamped.transform.translation.x = this->extrinsics.baselink2imu.t[0];
  transformStamped.transform.translation.y = this->extrinsics.baselink2imu.t[1];
  transformStamped.transform.translation.z = this->extrinsics.baselink2imu.t[2];

  Eigen::Quaternionf q(this->extrinsics.baselink2imu.R);
  transformStamped.transform.rotation.w = q.w();
  transformStamped.transform.rotation.x = q.x();
  transformStamped.transform.rotation.y = q.y();
  transformStamped.transform.rotation.z = q.z();

  this->static_br->sendTransform(transformStamped);

  // transform: baselink to lidar
  transformStamped.header.frame_id = this->baselink_frame;
  transformStamped.child_frame_id = this->lidar_frame;

  transformStamped.transform.translation.x = this->extrinsics.baselink2lidar.t[0];
  transformStamped.transform.translation.y = this->extrinsics.baselink2lidar.t[1];
  transformStamped.transform.translation.z = this->extrinsics.baselink2lidar.t[2];

  Eigen::Quaternionf qq(this->extrinsics.baselink2lidar.R);
  transformStamped.transform.rotation.w = qq.w();
  transformStamped.transform.rotation.x = qq.x();
  transformStamped.transform.rotation.y = qq.y();
  transformStamped.transform.rotation.z = qq.z();

  this->static_br->sendTransform(transformStamped);

}

void dlio::OdomNode::resolveExtrinsicsFromTf() {
  // Look up the latest available transforms. base_link<-imu and base_link<-lidar
  // are required; lidar<-camera only when the visual term is enabled.
  auto toMatrix = [](const geometry_msgs::msg::TransformStamped& tf) {
    const auto& q = tf.transform.rotation;
    const auto& t = tf.transform.translation;
    Eigen::Matrix4f M = Eigen::Matrix4f::Identity();
    M.block<3, 3>(0, 0) = Eigen::Quaternionf(static_cast<float>(q.w), static_cast<float>(q.x),
                                             static_cast<float>(q.y), static_cast<float>(q.z))
                              .normalized().toRotationMatrix();
    M.block<3, 1>(0, 3) = Eigen::Vector3f(static_cast<float>(t.x), static_cast<float>(t.y),
                                          static_cast<float>(t.z));
    return M;
  };

  try {
    if (!this->tf_buffer_->canTransform(this->baselink_frame, this->imu_frame, tf2::TimePointZero) ||
        !this->tf_buffer_->canTransform(this->baselink_frame, this->lidar_frame, tf2::TimePointZero)) {
      throw tf2::TransformException("pending");
    }
    const auto T_bi = toMatrix(
        this->tf_buffer_->lookupTransform(this->baselink_frame, this->imu_frame, tf2::TimePointZero));
    const auto T_bl = toMatrix(
        this->tf_buffer_->lookupTransform(this->baselink_frame, this->lidar_frame, tf2::TimePointZero));
    Eigen::Matrix4f T_lc = this->cam2lidar_T_;  // keep YAML cam2lidar unless tf has it
    if (this->visual_enabled_ &&
        this->tf_buffer_->canTransform(this->lidar_frame, this->camera_frame_, tf2::TimePointZero)) {
      T_lc = toMatrix(
          this->tf_buffer_->lookupTransform(this->lidar_frame, this->camera_frame_, tf2::TimePointZero));
    }

    // Commit the extrinsics, then release via the atomic store so the scan/imu
    // threads (which acquire-load extrinsics_ready_ before reading) see them.
    this->extrinsics.baselink2imu_T = T_bi;
    this->extrinsics.baselink2imu.t = T_bi.block<3, 1>(0, 3);
    this->extrinsics.baselink2imu.R = T_bi.block<3, 3>(0, 0);
    this->extrinsics.baselink2lidar_T = T_bl;
    this->extrinsics.baselink2lidar.t = T_bl.block<3, 1>(0, 3);
    this->extrinsics.baselink2lidar.R = T_bl.block<3, 3>(0, 0);
    this->cam2lidar_T_ = T_lc;

    this->extrinsics_ready_.store(true);
    this->extrinsics_timer_->cancel();
    RCLCPP_INFO(this->get_logger(), "extrinsics resolved from tf2 "
        "(base_link->imu t=[%.4f %.4f %.4f], base_link->lidar t=[%.4f %.4f %.4f]).",
        this->extrinsics.baselink2imu.t[0], this->extrinsics.baselink2imu.t[1],
        this->extrinsics.baselink2imu.t[2], this->extrinsics.baselink2lidar.t[0],
        this->extrinsics.baselink2lidar.t[1], this->extrinsics.baselink2lidar.t[2]);
  } catch (const tf2::TransformException& e) {
    if (++this->extrinsics_attempts_ >= 20) {  // ~10 s
      RCLCPP_WARN(this->get_logger(),
          "extrinsics/source=tf: transforms unavailable after %d attempts (%s); "
          "falling back to the YAML extrinsics.", this->extrinsics_attempts_, e.what());
      this->extrinsics_ready_.store(true);
      this->extrinsics_timer_->cancel();
    } else {
      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
          "extrinsics/source=tf: still waiting for base_link->{imu,lidar} (%s)...", e.what());
    }
  }
}

void dlio::OdomNode::publishCloud(pcl::PointCloud<PointType>::ConstPtr published_cloud, Eigen::Matrix4f T_cloud) {

  // Skip the (per-scan) dense transform + serialize when nobody is listening,
  // so RViz/Foxglove not subscribed to the deskewed cloud costs nothing -- the
  // single biggest viz-vs-estimator CPU contention source.
  if (this->deskewed_pub->get_subscription_count() == 0) { return; }

  if (this->wait_until_move_) {
    if (this->length_traversed < 0.1) { return; }
  }

  pcl::PointCloud<PointType>::Ptr deskewed_scan_t_ = std::make_shared<pcl::PointCloud<PointType>>();

  pcl::transformPointCloud (*published_cloud, *deskewed_scan_t_, T_cloud);

  // published deskewed cloud
  sensor_msgs::msg::PointCloud2 deskewed_ros;
  pcl::toROSMsg(*deskewed_scan_t_, deskewed_ros);
  deskewed_ros.header.stamp = this->scan_header_stamp;
  deskewed_ros.header.frame_id = this->odom_frame;
  this->deskewed_pub->publish(deskewed_ros);

}

void dlio::OdomNode::publishKeyframe(std::pair<std::pair<Eigen::Vector3f, Eigen::Quaternionf>, pcl::PointCloud<PointType>::ConstPtr> kf, rclcpp::Time timestamp) {

  // Push back
  geometry_msgs::msg::Pose p;
  p.position.x = kf.first.first[0];
  p.position.y = kf.first.first[1];
  p.position.z = kf.first.first[2];
  p.orientation.w = kf.first.second.w();
  p.orientation.x = kf.first.second.x();
  p.orientation.y = kf.first.second.y();
  p.orientation.z = kf.first.second.z();
  this->kf_pose_ros.poses.push_back(p);

  // Publish
  this->kf_pose_ros.header.stamp = timestamp;
  this->kf_pose_ros.header.frame_id = this->odom_frame;
  this->kf_pose_pub->publish(this->kf_pose_ros);

  // publish keyframe scan for map (only when a consumer -- the map node and/or a
  // viz client -- is subscribed; the toROSMsg of a full keyframe cloud is not free)
  if (this->kf_cloud_pub->get_subscription_count() > 0 &&
      kf.second->points.size() == kf.second->width * kf.second->height) {
    sensor_msgs::msg::PointCloud2 keyframe_cloud_ros;
    pcl::toROSMsg(*kf.second, keyframe_cloud_ros);
    keyframe_cloud_ros.header.stamp = timestamp;
    keyframe_cloud_ros.header.frame_id = this->odom_frame;
    this->kf_cloud_pub->publish(keyframe_cloud_ros);
  }

}

float dlio::OdomNode::correctIntensity(float intensity, float range, float cos_incidence,
                                       float alpha, float r_ref, float cos_min) {
  if (!(range > 0.f) || r_ref <= 0.f) { return intensity; }
  const float c = std::max(cos_incidence, cos_min);  // c>0 (cos_min should be >0)
  return std::clamp(intensity * std::pow(range / r_ref, alpha) / c, 0.f, 255.f);
}

dlio::SensorType dlio::OdomNode::detectSensorType(
    const std::vector<sensor_msgs::msg::PointField>& fields,
    bool has_points, double first_timestamp) {
  for (const auto& field : fields) {
    if (field.name == "t") {
      return dlio::SensorType::OUSTER;
    } else if (field.name == "time") {
      return dlio::SensorType::VELODYNE;
    } else if (field.name == "timestamp" && has_points && first_timestamp < 1e14) {
      return dlio::SensorType::HESAI;
    } else if (field.name == "timestamp" && has_points && first_timestamp > 1e14) {
      return dlio::SensorType::LIVOX;
    }
  }
  return dlio::SensorType::UNKNOWN;
}

rcl_interfaces::msg::SetParametersResult
dlio::OdomNode::onSetParams(const std::vector<rclcpp::Parameter>& params) {
  // Parameters that may be retuned live (the drift-hunt knobs). Others still
  // "set" at the ROS level but have no effect until restart.
  static const std::set<std::string> kLive = {
    "odom/gicp/photometricWeight", "odom/gicp/photometricHuberDelta",
    "odom/gicp/photometricScale", "odom/gicp/degeneracyThreshRatio",
    "odom/gicp/maxCorrespondenceDistance",
    "odom/keyframe/threshD", "odom/keyframe/threshR",
    "odom/geo/Kp", "odom/geo/Kv", "odom/geo/Kq", "odom/geo/Kab",
    "odom/geo/Kgb", "odom/geo/abias_max", "odom/geo/gbias_max"
  };
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;  // ranges already validated by rclcpp before this
  std::lock_guard<std::mutex> lock(this->live_mtx_);
  for (const auto& p : params) {
    if (kLive.count(p.get_name())) {
      this->live_pending_[p.get_name()] = p.as_double();
      this->live_dirty_.store(true);
    }
  }
  return result;
}

void dlio::OdomNode::applyLiveParams() {
  if (!this->live_dirty_.load()) { return; }
  std::map<std::string, double> pending;
  {
    std::lock_guard<std::mutex> lock(this->live_mtx_);
    pending.swap(this->live_pending_);
    this->live_dirty_.store(false);
  }
  for (const auto& kv : pending) {
    const std::string& name = kv.first;
    const double v = kv.second;
    if (name == "odom/gicp/photometricWeight") {
      this->gicp.setPhotometricWeight(static_cast<float>(v));
      this->gicp_temp.setPhotometricWeight(static_cast<float>(v));
      this->photometric_active_ = (v > 0.0);
    } else if (name == "odom/gicp/photometricHuberDelta") {
      this->gicp.setPhotometricHuberDelta(static_cast<float>(v));
      this->gicp_temp.setPhotometricHuberDelta(static_cast<float>(v));
    } else if (name == "odom/gicp/photometricScale") {
      this->gicp.setPhotometricScale(static_cast<float>(v));
      this->gicp_temp.setPhotometricScale(static_cast<float>(v));
    } else if (name == "odom/gicp/degeneracyThreshRatio") {
      this->gicp.setDegeneracyThreshRatio(static_cast<float>(v));
    } else if (name == "odom/gicp/maxCorrespondenceDistance") {
      this->gicp_max_corr_dist_ = v;
      if (!this->adaptive_params_) {  // adaptive recomputes this each scan
        this->gicp.setMaxCorrespondenceDistance(static_cast<float>(v));
        this->gicp_temp.setMaxCorrespondenceDistance(static_cast<float>(v));
      }
    } else if (name == "odom/keyframe/threshD") { this->keyframe_thresh_dist_ = v;
    } else if (name == "odom/keyframe/threshR") { this->keyframe_thresh_rot_ = v;
    } else if (name == "odom/geo/Kp")  { this->geo_Kp_ = v;
    } else if (name == "odom/geo/Kv")  { this->geo_Kv_ = v;
    } else if (name == "odom/geo/Kq")  { this->geo_Kq_ = v;
    } else if (name == "odom/geo/Kab") { this->geo_Kab_ = v;
    } else if (name == "odom/geo/Kgb") { this->geo_Kgb_ = v;
    } else if (name == "odom/geo/abias_max") { this->geo_abias_max_ = v;
    } else if (name == "odom/geo/gbias_max") { this->geo_gbias_max_ = v;
    }
    RCLCPP_INFO(this->get_logger(), "live param: %s = %.6g", name.c_str(), v);
  }
}

void dlio::OdomNode::getScanFromROS(const sensor_msgs::msg::PointCloud2::SharedPtr& pc) {

  pcl::PointCloud<PointType>::Ptr original_scan_ = std::make_shared<pcl::PointCloud<PointType>>();
  pcl::fromROSMsg(*pc, *original_scan_);

  // Populate the reflectivity channel from the raw message. pcl::fromROSMsg only
  // copies fields whose datatype matches our struct (reflectivity is a float here,
  // but sensors publish it as uint8/uint16), so copy it explicitly with conversion.
  // Done before NaN removal so indices still line up 1:1 with the message.
  // Also needed by the COIN-LIO LiDAR intensity-image term (independent of the
  // 3D-spatial reflectivity photometric term).
  if ((this->use_reflectivity_ && this->photometric_active_) || this->lidar_image_enabled_) {
    auto rfield = std::find_if(pc->fields.begin(), pc->fields.end(),
        [](const sensor_msgs::msg::PointField& f){ return f.name == "reflectivity"; });
    if (rfield != pc->fields.end()) {
      const size_t n = original_scan_->points.size();
      auto fill = [&](auto it) {
        for (size_t i = 0; i < n; ++i, ++it) {
          original_scan_->points[i].reflectivity = static_cast<float>(*it);
        }
      };
      using PF = sensor_msgs::msg::PointField;
      switch (rfield->datatype) {
        case PF::UINT8:   fill(sensor_msgs::PointCloud2ConstIterator<uint8_t >(*pc, "reflectivity")); break;
        case PF::UINT16:  fill(sensor_msgs::PointCloud2ConstIterator<uint16_t>(*pc, "reflectivity")); break;
        case PF::UINT32:  fill(sensor_msgs::PointCloud2ConstIterator<uint32_t>(*pc, "reflectivity")); break;
        case PF::FLOAT32: fill(sensor_msgs::PointCloud2ConstIterator<float   >(*pc, "reflectivity")); break;
        default:
          RCLCPP_WARN_ONCE(this->get_logger(),
              "reflectivity field has unsupported datatype %u; channel will be zero.",
              rfield->datatype);
          break;
      }
    } else {
      RCLCPP_WARN_ONCE(this->get_logger(),
          "photometricChannel=reflectivity but the cloud has no 'reflectivity' field.");
    }
  }

  // COIN-LIO LiDAR intensity image: build from the ORGANIZED grid before NaN
  // removal flattens it. No-op if the cloud isn't organized.
  this->lidar_img_ready_ = false;
  if (this->lidar_image_enabled_ && pc->height > 1 &&
      original_scan_->height == pc->height && original_scan_->width == pc->width) {
    this->buildLidarIntensityImage(original_scan_, pc->width, pc->height);
  }

  // Radiometric intensity correction (raw-intensity photometric path only):
  // range falloff and, optionally, incidence angle (Kashani et al.). Applied
  // BEFORE NaN removal so the organized grid is available for cheap per-point
  // normals. Skipped for reflectivity (sensor-calibrated) and when the
  // photometric term is off, so downstream consumers only see modified
  // intensities when the feature is actually in use.
  if (!this->use_reflectivity_ && this->photometric_active_ && this->intensity_r_ref_ > 0.0) {
    const float alpha   = static_cast<float>(this->intensity_alpha_);
    const float r_ref   = static_cast<float>(this->intensity_r_ref_);
    const float cos_min = static_cast<float>(this->intensity_cos_min_);
    const bool organized = (original_scan_->height > 1 && original_scan_->width > 1);
    const bool do_incidence = this->intensity_incidence_ && organized;

    if (do_incidence) {
      const int W = static_cast<int>(original_scan_->width);
      const int H = static_cast<int>(original_scan_->height);
      for (int row = 0; row < H; ++row) {
        for (int col = 0; col < W; ++col) {
          auto& pt = original_scan_->at(col, row);
          const float r = std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
          if (!(r > 0.f)) { continue; }
          // Surface normal from organized neighbors (right & down); falls back
          // to range-only (cos=1) at borders / where a neighbor is invalid.
          float cos_a = 1.0f;
          if (col + 1 < W && row + 1 < H) {
            const auto& pr = original_scan_->at(col + 1, row);
            const auto& pd = original_scan_->at(col, row + 1);
            if (std::isfinite(pr.x) && std::isfinite(pd.x)) {
              const Eigen::Vector3f c(pt.x, pt.y, pt.z);
              Eigen::Vector3f n = (Eigen::Vector3f(pr.x, pr.y, pr.z) - c)
                                  .cross(Eigen::Vector3f(pd.x, pd.y, pd.z) - c);
              const float nn = n.norm();
              if (nn > 1e-6f) { cos_a = std::abs((c / r).dot(n / nn)); }
            }
          }
          pt.intensity = correctIntensity(pt.intensity, r, cos_a, alpha, r_ref, cos_min);
        }
      }
    } else {
      for (auto& pt : original_scan_->points) {
        const float r = std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
        pt.intensity = correctIntensity(pt.intensity, r, 1.0f, alpha, r_ref, cos_min);
      }
    }
  }

  // Remove NaNs
  std::vector<int> idx;
  original_scan_->is_dense = false;
  pcl::removeNaNFromPointCloud(*original_scan_, *original_scan_, idx);

  // Crop Box Filter
  this->crop.setInputCloud(original_scan_);
  this->crop.filter(*original_scan_);

  // automatically detect sensor type
  const bool has_points = !original_scan_->points.empty();
  this->sensor = detectSensorType(pc->fields, has_points,
      has_points ? original_scan_->points[0].timestamp : 0.0);

  if (this->sensor == dlio::SensorType::UNKNOWN) {
    this->deskew_ = false;
  }

  this->scan_header_stamp = pc->header.stamp;
  this->original_scan = original_scan_;

}

void dlio::OdomNode::preprocessPoints() {

  // Deskew the original dlio-type scan
  if (this->deskew_) {

    this->deskewPointcloud();

    if (!this->first_valid_scan) {
      return;
    }

  } else {

    this->scan_stamp = rclcpp::Time(this->scan_header_stamp).seconds();

    // don't process scans until IMU data is present
    if (!this->first_valid_scan) {

      if (this->imu_buffer.empty() || this->scan_stamp <= this->imu_buffer.back().stamp) {
        return;
      }

      this->first_valid_scan = true;
      this->T_prior = this->T; // assume no motion for the first scan

    } else {

      // IMU prior for second scan onwards
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> frames;
      frames = this->integrateImu(this->prev_scan_stamp, this->lidarPose.q, this->lidarPose.p,
                                this->geo.prev_vel.cast<float>(), {this->scan_stamp});

    if (frames.size() > 0) {
      this->T_prior = frames.back();
    } else {
      this->T_prior = this->T;
    }

    }

    pcl::PointCloud<PointType>::Ptr deskewed_scan_ = std::make_shared<pcl::PointCloud<PointType>>();
    pcl::transformPointCloud (*this->original_scan, *deskewed_scan_,
                              this->T_prior * this->extrinsics.baselink2lidar_T);
    this->deskewed_scan = deskewed_scan_;
    this->deskew_status = false;
  }

  // Voxel Grid Filter
  if (this->vf_use_) {
    // Filter straight from deskewed_scan into a fresh output cloud: VoxelGrid
    // reads its input and writes a separate output, so there is no need to
    // deep-copy the (full-resolution) deskewed cloud first. deskewed_scan is
    // left untouched for the dense-map publish path.
    pcl::PointCloud<PointType>::Ptr current_scan_ = std::make_shared<pcl::PointCloud<PointType>>();
    this->voxel.setInputCloud(this->deskewed_scan);
    this->voxel.filter(*current_scan_);
    this->current_scan = current_scan_;
  } else {
    this->current_scan = this->deskewed_scan;
  }

}

void dlio::OdomNode::deskewPointcloud() {

  // an empty scan (e.g. fully cropped) would otherwise crash on the
  // first-point/median-timestamp lookups below
  if (this->original_scan->points.empty()) {
    this->scan_stamp = rclcpp::Time(this->scan_header_stamp).seconds();
    this->deskewed_scan = this->original_scan;
    this->deskew_status = false;
    return;
  }

  // pcl::PointCloud(width, height): N points wide, 1 row tall (unorganized)
  pcl::PointCloud<PointType>::Ptr deskewed_scan_ =
      std::make_shared<pcl::PointCloud<PointType>>(this->original_scan->points.size(), 1);
  // individual point timestamps should be relative to this time
  double sweep_ref_time = rclcpp::Time(this->scan_header_stamp).seconds();

  // sort points by timestamp and build list of timestamps
  std::function<bool(const PointType&, const PointType&)> point_time_cmp;
  std::function<bool(boost::range::index_value<PointType&, long>,
                     boost::range::index_value<PointType&, long>)> point_time_neq;
  std::function<double(boost::range::index_value<PointType&, long>)> extract_point_time;

  if (this->sensor == dlio::SensorType::OUSTER) {

    point_time_cmp = [](const PointType& p1, const PointType& p2)
      { return p1.t < p2.t; };
    point_time_neq = [](boost::range::index_value<PointType&, long> p1,
                        boost::range::index_value<PointType&, long> p2)
      { return p1.value().t != p2.value().t; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType&, long> pt)
      { return sweep_ref_time + pt.value().t * 1e-9f; };

  } else if (this->sensor == dlio::SensorType::VELODYNE) {

    point_time_cmp = [](const PointType& p1, const PointType& p2)
      { return p1.time < p2.time; };
    point_time_neq = [](boost::range::index_value<PointType&, long> p1,
                        boost::range::index_value<PointType&, long> p2)
      { return p1.value().time != p2.value().time; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType&, long> pt)
      { return sweep_ref_time + pt.value().time; };

  } else if (this->sensor == dlio::SensorType::HESAI) {

    point_time_cmp = [](const PointType& p1, const PointType& p2)
      { return p1.timestamp < p2.timestamp; };
    point_time_neq = [](boost::range::index_value<PointType&, long> p1,
                        boost::range::index_value<PointType&, long> p2)
      { return p1.value().timestamp != p2.value().timestamp; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType&, long> pt)
      { return pt.value().timestamp; };
  } else if (this->sensor == dlio::SensorType::LIVOX) {
    point_time_cmp = [](const PointType& p1, const PointType& p2)
      { return p1.timestamp < p2.timestamp; };
    point_time_neq = [](boost::range::index_value<PointType&, long> p1,
                        boost::range::index_value<PointType&, long> p2)
      { return p1.value().timestamp != p2.value().timestamp; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType&, long> pt)
      { return pt.value().timestamp * 1e-9f; };
  }

  // copy points into deskewed_scan_ in order of timestamp
  std::partial_sort_copy(this->original_scan->points.begin(), this->original_scan->points.end(),
                         deskewed_scan_->points.begin(), deskewed_scan_->points.end(), point_time_cmp);

  // filter unique timestamps
  auto points_unique_timestamps = deskewed_scan_->points
                                  | boost::adaptors::indexed()
                                  | boost::adaptors::adjacent_filtered(point_time_neq);

  // extract timestamps from points and put them in their own list
  std::vector<double> timestamps;
  std::vector<int> unique_time_indices;

  // compute offset between sweep reference time and first point timestamp
  double offset = 0.0;
  if (this->time_offset_) {
    offset = sweep_ref_time - extract_point_time(*points_unique_timestamps.begin());
  }

  // build list of unique timestamps and indices of first point with each timestamp
  for (auto it = points_unique_timestamps.begin(); it != points_unique_timestamps.end(); it++) {
    timestamps.push_back(extract_point_time(*it) + offset);
    unique_time_indices.push_back(it->index());
  }
  unique_time_indices.push_back(deskewed_scan_->points.size());

  int median_pt_index = timestamps.size() / 2;
  this->scan_stamp = timestamps[median_pt_index]; // set this->scan_stamp to the timestamp of the median point

  // don't process scans until IMU data is present
  if (!this->first_valid_scan) {
    if (this->imu_buffer.empty() || this->scan_stamp <= this->imu_buffer.back().stamp) {
      return;
    }

    this->first_valid_scan = true;
    this->T_prior = this->T; // assume no motion for the first scan
    pcl::transformPointCloud (*deskewed_scan_, *deskewed_scan_, this->T_prior * this->extrinsics.baselink2lidar_T);
    this->deskewed_scan = deskewed_scan_;
    this->deskew_status = true;
    return;
  }

  // IMU prior & deskewing for second scan onwards
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> frames;
  frames = this->integrateImu(this->prev_scan_stamp, this->lidarPose.q, this->lidarPose.p,
                              this->geo.prev_vel.cast<float>(), timestamps);
  this->deskew_size = frames.size(); // if integration successful, equal to timestamps.size()

  // if there are no frames between the start and end of the sweep
  // that probably means that there's a sync issue
  if (frames.size() != timestamps.size()) {
    // not fatal: gracefully degrades to a rigid (non-deskewed) transform below
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
        "IMU data does not cover the scan period (time sync / dropout?); "
        "skipping motion correction for this scan");

    this->T_prior = this->T;
    pcl::transformPointCloud (*deskewed_scan_, *deskewed_scan_, this->T_prior * this->extrinsics.baselink2lidar_T);
    this->deskewed_scan = deskewed_scan_;
    this->deskew_status = false;
    return;
  }

  // update prior to be the estimated pose at the median time of the scan (corresponds to this->scan_stamp)
  this->T_prior = frames[median_pt_index];

#pragma omp parallel for num_threads(this->num_threads_)
  for (int i = 0; i < timestamps.size(); i++) {

    Eigen::Matrix4f T = frames[i] * this->extrinsics.baselink2lidar_T;

    // transform point to world frame
    for (int k = unique_time_indices[i]; k < unique_time_indices[i+1]; k++) {
      auto &pt = deskewed_scan_->points[k];
      pt.getVector4fMap()[3] = 1.;
      pt.getVector4fMap() = T * pt.getVector4fMap();
    }
  }

  this->deskewed_scan = deskewed_scan_;
  this->deskew_status = true;

}

void dlio::OdomNode::initializeInputTarget() {

  this->prev_scan_stamp = this->scan_stamp;

  // keep history of keyframes
  this->keyframes.push_back(std::make_pair(std::make_pair(this->lidarPose.p, this->lidarPose.q), this->current_scan));
  this->keyframe_timestamps.push_back(this->scan_header_stamp);
  this->keyframe_normals.push_back(std::make_shared<const nano_gicp::CovarianceList>(this->gicp.getSourceCovariances()));
  this->keyframe_transformations.push_back(this->T_corr);

  // Sample per-point camera reference brightness for the frame-to-map term
  // (index-aligned with keyframes; p_kf_cam is camera-frame so it survives the
  // later world re-transform in buildKeyframesAndSubmap untouched).
  if (this->visual_map_enabled_) {
    Eigen::Matrix4f T_wc = this->T_prior * this->extrinsics.baselink2lidar_T * this->cam2lidar_T_;
    Eigen::Isometry3f T_cw; T_cw.matrix() = T_wc.inverse();
    cv::Mat img = this->visual_cur_pending_valid_ ? this->visual_cur_pending_ : cv::Mat();
    this->keyframe_visual_refs.push_back(this->sampleKeyframeVisualRefs(this->current_scan, T_cw, img));
  }

}

void dlio::OdomNode::setInputSource() {
  this->gicp.setInputSource(this->current_scan);
}

void dlio::OdomNode::initializeDLIO() {

  // Wait for IMU
  if (!this->first_imu_received || !this->imu_calibrated) {
    return;
  }

  this->dlio_initialized = true;
  std::cout << std::endl << " DLIO initialized!" << std::endl;

}

void dlio::OdomNode::callbackPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr pc) {

  if (!this->extrinsics_ready_.load()) { return; }  // tf extrinsics not resolved yet

  std::unique_lock<decltype(this->main_loop_running_mutex)> lock(main_loop_running_mutex);
  this->main_loop_running = true;
  lock.unlock();

  // Join the previous scan's dashboard thread before this scan mutates any of
  // the stats vectors it reads (comp_times / lidar_rates / metrics, all
  // push_back + cap_history). Doing it here -- before computeMetrics below --
  // closes the iterator-invalidation race for the scan-thread-written stats.
  // (imu_rates is written by the IMU thread and is guarded by mtx_imu instead.)
  if (this->debug_thread.joinable()) { this->debug_thread.join(); }

  // Commit any parameters retuned via `ros2 param set` since the last scan
  // (staged by onSetParams on the executor thread; applied here on the scan
  // thread so gicp/observer state is only ever mutated from one thread).
  this->applyLiveParams();

  double then = this->now().seconds();

  if (this->first_scan_stamp == 0.) {
    this->first_scan_stamp = rclcpp::Time(pc->header.stamp).seconds();
  }

  // DLIO Initialization procedures (IMU calib, gravity align)
  if (!this->dlio_initialized) {
    this->initializeDLIO();
  }

  // Convert incoming scan into DLIO format
  this->getScanFromROS(pc);

  // Preprocess points
  this->preprocessPoints();

  if (!this->first_valid_scan) {
    return;
  }

  if (this->current_scan->points.size() <= this->gicp_min_num_points_) {
    // not fatal: this scan is skipped; odometry continues on IMU propagation
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
        "Low number of points in the cloud (%zu <= %d); skipping scan",
        this->current_scan->points.size(), this->gicp_min_num_points_);
    return;
  }

  // Compute Metrics (inline: cheap relative to registration, removes the
  // data race on original_scan/metrics vectors the old worker thread had,
  // and setAdaptiveParams below now uses THIS scan's metrics, not the
  // previous scan's)
  this->computeMetrics();

  // Set Adaptive Parameters
  if (this->adaptive_params_) {
    this->setAdaptiveParams();
  }

  // Set new frame as input source
  this->setInputSource();

  // Set initial frame as first keyframe
  if (this->keyframes.size() == 0) {
    this->initializeInputTarget();
    this->main_loop_running = false;
    this->submap_future =
      std::async( std::launch::async, &dlio::OdomNode::buildKeyframesAndSubmap, this, this->state );
    this->submap_future.wait(); // wait until completion
    return;
  }

  // Get the next pose via IMU + S2M + GEO
  this->getNextPose();

  // Update current keyframe poses and map
  this->updateKeyframes();

  // Optionally bound the keyframe map. Only safe while the background submap
  // thread is idle (new_submap_is_ready), since pruning re-indexes the
  // keyframe vectors that buildKeyframesAndSubmap iterates.
  if (this->max_keyframes_ > 0 && this->new_submap_is_ready
      && (int)this->keyframes.size() > this->max_keyframes_) {
    this->pruneKeyframes();
  }

  // Build keyframe normals and submap if needed (and if we're not already waiting)
  if (this->new_submap_is_ready) {
    this->main_loop_running = false;
    this->submap_future =
      std::async( std::launch::async, &dlio::OdomNode::buildKeyframesAndSubmap, this, this->state );
  } else {
    lock.lock();
    this->main_loop_running = false;
    lock.unlock();
    this->submap_build_cv.notify_one();
  }

  // Update distance traveled incrementally (replaces the unbounded trajectory
  // vector that debug() used to re-integrate from scratch every scan)
  double l = (this->state.p - this->length_prev_p).norm();
  if (l >= 0.1) {
    this->length_traversed += l;
    this->length_prev_p = this->state.p;
  }

  // Update time stamps. Capture the inter-scan period BEFORE overwriting
  // prev_scan_stamp -- the CPU-starvation block below needs it (computing it
  // after the overwrite made it always 0, silently killing those diagnostics).
  const double scan_period = this->scan_stamp - this->prev_scan_stamp;
  this->lidar_rates.push_back( 1. / scan_period );
  cap_history(this->lidar_rates);
  this->prev_scan_stamp = this->scan_stamp;
  this->elapsed_time = this->scan_stamp - this->first_scan_stamp;

  // Publish stuff to ROS
  pcl::PointCloud<PointType>::ConstPtr published_cloud;
  if (this->densemap_filtered_) {
    published_cloud = this->current_scan;
  } else {
    published_cloud = this->deskewed_scan;
  }
  if (this->publish_thread.joinable()) { this->publish_thread.join(); }
  this->publish_thread = std::thread( &dlio::OdomNode::publishToROS, this, published_cloud, this->T_corr );

  // Update some statistics
  const double comp_time = this->now().seconds() - then;
  this->comp_times.push_back(comp_time);
  cap_history(this->comp_times);
  this->gicp_hasConverged = this->gicp.hasConverged();

  // CPU-starvation indicator: a scan whose processing took longer than the scan
  // period means the node cannot keep real time and WILL fall behind / drop
  // scans under load -- the documented #1 cause of tunnel-run failures (and NOT
  // an algorithm fault). Surface it in /diagnostics so a starved run is
  // distinguishable from a genuine divergence at a glance. (scan_period was
  // captured above, before prev_scan_stamp was advanced.)
  this->last_realtime_factor_ = (scan_period > 0.0) ? (comp_time / scan_period) : 0.0;
  if (scan_period > 0.0 && comp_time > scan_period) { ++this->compute_overruns_; }
  // Estimate scans dropped by the transport (best-effort) when the inter-scan
  // gap is well over the nominal period.
  if (scan_period > 0.0 && this->prev_scan_period_ > 0.0
      && scan_period > 1.8 * this->prev_scan_period_) {
    this->scans_dropped_est_ += static_cast<long>(scan_period / this->prev_scan_period_) - 1;
  }
  if (scan_period > 0.0) { this->prev_scan_period_ = scan_period; }

  // Publish /diagnostics every scan (independent of the dashboard toggle below).
  this->publishDiagnostics();

  // Terminal dashboard (odom/debug/dashboard); disable under launch files,
  // containers, or logging setups where the ANSI clear-screen output garbles
  // multiplexed logs.
  if (this->dashboard_) {
    // (previous dashboard thread already joined at the top of this callback)
    this->debug_thread = std::thread( &dlio::OdomNode::debug, this );
  }

  this->geo.first_opt_done = true;

}

void dlio::OdomNode::callbackImu(const sensor_msgs::msg::Imu::SharedPtr imu_raw) {

  // transformImu uses the baselink<-imu extrinsic; wait until it's resolved
  // (tf mode). Drops the brief startup window before tf is available.
  if (!this->extrinsics_ready_.load()) { return; }

  this->first_imu_received = true;

  sensor_msgs::msg::Imu::SharedPtr imu = this->transformImu( imu_raw );
  {
    // imu_stamp is read by publishPose (timer thread) under geo.mtx; guard the write.
    std::lock_guard<std::mutex> lock(this->geo.mtx);
    this->imu_stamp = imu->header.stamp;
  }
  double imu_stamp_secs = rclcpp::Time(imu->header.stamp).seconds();

  Eigen::Vector3f lin_accel;
  Eigen::Vector3f ang_vel;

  // Get IMU samples
  ang_vel[0] = imu->angular_velocity.x;
  ang_vel[1] = imu->angular_velocity.y;
  ang_vel[2] = imu->angular_velocity.z;

  // Livox-style IMUs report acceleration in units of g; scale to m/s^2.
  const float accel_scale = this->imu_normalized_ ? static_cast<float>(this->gravity_) : 1.0f;
  lin_accel[0] = imu->linear_acceleration.x * accel_scale;
  lin_accel[1] = imu->linear_acceleration.y * accel_scale;
  lin_accel[2] = imu->linear_acceleration.z * accel_scale;

  if (this->first_imu_stamp == 0.) {
    this->first_imu_stamp = imu_stamp_secs;
  }

  // IMU calibration procedure - do for three seconds
  if (!this->imu_calibrated) {

    static int num_samples = 0;
    static Eigen::Vector3f gyro_avg (0., 0., 0.);
    static Eigen::Vector3f accel_avg (0., 0., 0.);
    static bool print = true;

    if ((imu_stamp_secs - this->first_imu_stamp) < this->imu_calib_time_) {

      num_samples++;

      gyro_avg[0] += ang_vel[0];
      gyro_avg[1] += ang_vel[1];
      gyro_avg[2] += ang_vel[2];

      accel_avg[0] += lin_accel[0];
      accel_avg[1] += lin_accel[1];
      accel_avg[2] += lin_accel[2];

      if(print) {
        std::cout << std::endl << " Calibrating IMU for " << this->imu_calib_time_ << " seconds... ";
        std::cout.flush();
        print = false;
      }

    } else {

      std::cout << "done" << std::endl << std::endl;

      gyro_avg /= num_samples;
      accel_avg /= num_samples;

      Eigen::Vector3f grav_vec (0., 0., this->gravity_);

      if (this->gravity_align_) {

        // Estimate gravity vector - Only approximate if biases have not been pre-calibrated
        grav_vec = (accel_avg - this->state.b.accel).normalized() * abs(this->gravity_);
        Eigen::Quaternionf grav_q = Eigen::Quaternionf::FromTwoVectors(grav_vec, Eigen::Vector3f(0., 0., this->gravity_));

        // set gravity aligned orientation
        this->state.q = grav_q;
        this->T.block(0,0,3,3) = this->state.q.toRotationMatrix();
        this->lidarPose.q = this->state.q;

        // rpy
        auto euler = grav_q.toRotationMatrix().eulerAngles(2, 1, 0);
        double yaw = euler[0] * (180.0/M_PI);
        double pitch = euler[1] * (180.0/M_PI);
        double roll = euler[2] * (180.0/M_PI);

        // use alternate representation if the yaw is smaller
        if (abs(remainder(yaw + 180.0, 360.0)) < abs(yaw)) {
          yaw   = remainder(yaw + 180.0,   360.0);
          pitch = remainder(180.0 - pitch, 360.0);
          roll  = remainder(roll + 180.0,  360.0);
        }
        std::cout << " Estimated initial attitude:" << std::endl;
        std::cout << "   Roll  [deg]: " << to_string_with_precision(roll, 4) << std::endl;
        std::cout << "   Pitch [deg]: " << to_string_with_precision(pitch, 4) << std::endl;
        std::cout << "   Yaw   [deg]: " << to_string_with_precision(yaw, 4) << std::endl;
        std::cout << std::endl;
      }

      if (this->calibrate_accel_) {

        // subtract gravity from avg accel to get bias
        this->state.b.accel = accel_avg - grav_vec;

        std::cout << " Accel biases [xyz]: " << to_string_with_precision(this->state.b.accel[0], 8) << ", "
                                             << to_string_with_precision(this->state.b.accel[1], 8) << ", "
                                             << to_string_with_precision(this->state.b.accel[2], 8) << std::endl;
      }

      if (this->calibrate_gyro_) {

        this->state.b.gyro = gyro_avg;

        std::cout << " Gyro biases  [xyz]: " << to_string_with_precision(this->state.b.gyro[0], 8) << ", "
                                             << to_string_with_precision(this->state.b.gyro[1], 8) << ", "
                                             << to_string_with_precision(this->state.b.gyro[2], 8) << std::endl;
      }

      this->imu_calibrated = true;

      // Anchor dt for the first post-calibration measurement; otherwise the
      // first dt is computed against prev_imu_stamp = 0 (a ~1e9 s step) and
      // poisons the IMU buffer / state propagation, causing startup drift.
      this->prev_imu_stamp = imu_stamp_secs;

    }

  } else {

    double dt = imu_stamp_secs - this->prev_imu_stamp;
    if (dt == 0) { dt = 1.0/200.0; }
    {
      // imu_rates is read by publishDiagnostics (scan thread, every scan) and
      // the dashboard thread; guard the push+cap_history so the erase can't
      // invalidate a concurrent reader's iterators.
      std::lock_guard<decltype(this->mtx_imu)> rlk(this->mtx_imu);
      this->imu_rates.push_back( 1./dt );
      cap_history(this->imu_rates);
    }

    // Apply the calibrated bias to the new IMU measurements
    this->imu_meas.stamp = imu_stamp_secs;
    this->imu_meas.dt = dt;
    this->prev_imu_stamp = this->imu_meas.stamp;

    Eigen::Vector3f lin_accel_corrected = (this->imu_accel_sm_ * lin_accel) - this->state.b.accel;
    Eigen::Vector3f ang_vel_corrected = ang_vel - this->state.b.gyro;

    this->imu_meas.lin_accel = lin_accel_corrected;
    this->imu_meas.ang_vel = ang_vel_corrected;

    // Store calibrated IMU measurements into imu buffer for manual integration later.
    this->mtx_imu.lock();
    this->imu_buffer.push_front(this->imu_meas);
    this->mtx_imu.unlock();

    // Notify the callbackPointCloud thread that IMU data exists for this time
    this->cv_imu_stamp.notify_one();

    if (this->geo.first_opt_done) {
      // Geometric Observer: Propagate State
      this->propagateState();
    }

  }

}

void dlio::OdomNode::callbackImage(const sensor_msgs::msg::Image::SharedPtr img) {

  if (!this->visual_enabled_) { return; }

  // Convert to single-channel 8-bit (handles mono/rgb/bgr/bayer encodings).
  cv_bridge::CvImageConstPtr cvp;
  try {
    cvp = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
  } catch (const std::exception& e) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
        "cv_bridge could not convert image to mono8: %s", e.what());
    return;
  }
  if (cvp->image.empty()) { return; }

  // Build undistort maps once, from the configured intrinsics/distortion and
  // the first image's size.
  if (!this->visual_maps_ready_.load()) {
    const auto& I = this->camera_intrinsics_;
    cv::Mat K = (cv::Mat_<double>(3, 3) << I[0], 0.0, I[2], 0.0, I[1], I[3], 0.0, 0.0, 1.0);
    cv::Mat D(static_cast<int>(this->camera_distortion_.size()), 1, CV_64F);
    for (size_t i = 0; i < this->camera_distortion_.size(); ++i) { D.at<double>(static_cast<int>(i)) = this->camera_distortion_[i]; }
    cv::initUndistortRectifyMap(K, D, cv::Mat(), K, cvp->image.size(), CV_16SC2,
                               this->vis_map1_, this->vis_map2_);
    this->visual_maps_ready_ = true;
  }

  cv::Mat undistorted;
  cv::remap(cvp->image, undistorted, this->vis_map1_, this->vis_map2_, cv::INTER_LINEAR);

  // Buffer the UNDISTORTED MONO8 image (1 byte/px). The single frame matched to
  // a scan is converted to normalized float in setupVisualForScan -- storing 30
  // full-res CV_32FC1 frames here would be ~4x the memory (hundreds of MB at 2MP).
  const double stamp = rclcpp::Time(img->header.stamp).seconds();
  {
    std::lock_guard<std::mutex> lock(this->image_mtx_);
    this->image_buffer_.emplace_back(stamp, undistorted);
    while (this->image_buffer_.size() > 30) { this->image_buffer_.pop_front(); }
  }
}

bool dlio::OdomNode::setupVisualForScan() {

  // Default everything off for this scan; (re)enable below per term.
  this->visual_cur_pending_valid_ = false;
  this->gicp.setVisualEnabled(false);
  this->gicp.setVisualMapWeight(0.f);

  const bool want_f2f = this->visual_enabled_ && this->visual_weight_ > 0.0;
  const bool want_f2m = this->visual_enabled_ && this->visual_map_enabled_ && this->visual_map_weight_ > 0.0;
  if (!want_f2f && !want_f2m) { return false; }

  // Pick the buffered image nearest this scan's stamp (within tolerance).
  cv::Mat cur_mono;  // mono8 (the buffer stores 8-bit to save memory)
  double best_dt = this->visual_max_dt_;
  {
    std::lock_guard<std::mutex> lock(this->image_mtx_);
    for (const auto& kv : this->image_buffer_) {
      const double dt = std::abs(kv.first - this->scan_stamp);
      if (dt <= best_dt) { best_dt = dt; cur_mono = kv.second; }
    }
  }
  if (cur_mono.empty()) { return false; }  // graceful LiDAR-only this scan
  // Convert the single matched frame to normalized float (the residual unit).
  cv::Mat cur_img;
  cur_mono.convertTo(cur_img, CV_32FC1, 1.0 / 255.0);

  // world -> current camera, from the prior (predicted) pose. Shared by both the
  // frame-to-frame (raw source points) and frame-to-map (fixed map points) terms.
  Eigen::Matrix4f T_wc_cur = this->T_prior * this->extrinsics.baselink2lidar_T * this->cam2lidar_T_;
  Eigen::Isometry3f T_cw_cur;
  T_cw_cur.matrix() = T_wc_cur.inverse();

  this->gicp.setVisualIntrinsics(
      static_cast<float>(this->camera_intrinsics_[0]), static_cast<float>(this->camera_intrinsics_[1]),
      static_cast<float>(this->camera_intrinsics_[2]), static_cast<float>(this->camera_intrinsics_[3]));
  this->gicp.setVisualHuberDelta(static_cast<float>(this->visual_huber_delta_));
  this->gicp.setVisualCurrentFrame(cur_img, T_cw_cur);

  // Stash for promotion to "previous" after align() and for keyframe sampling.
  this->visual_cur_pending_ = cur_img;
  this->visual_cur_pending_valid_ = true;

  // Frame-to-MAP term: anchors to map landmarks; works on the first frame too.
  if (want_f2m) {
    this->gicp.setVisualMapWeight(static_cast<float>(this->visual_map_weight_));
    this->gicp.setVisualMapGateMaxStep(static_cast<float>(this->visual_map_gate_max_trans_),
                                       static_cast<float>(this->visual_map_gate_max_rot_));
    this->gicp.setVisualMapViewAngleMax(static_cast<float>(this->visual_map_view_angle_));
  }

  // Frame-to-frame term: needs a previous frame to warp against.
  if (want_f2f) {
    this->gicp.setVisualWeight(static_cast<float>(this->visual_weight_));
    this->gicp.setVisualGateMaxStep(static_cast<float>(this->visual_gate_max_trans_),
                                    static_cast<float>(this->visual_gate_max_rot_));
    if (this->visual_has_prev_) { this->gicp.setVisualEnabled(true); }
    this->gicp.setVisualPreviousFrame(this->visual_prev_img_, this->visual_T_cw_prev_);
  }

  return want_f2m || (want_f2f && this->visual_has_prev_);
}

std::shared_ptr<const nano_gicp::VisualRefList>
dlio::OdomNode::sampleKeyframeVisualRefs(const pcl::PointCloud<PointType>::ConstPtr& cloud,
                                         const Eigen::Isometry3f& T_cw, const cv::Mat& img) {
  // Always returns a list sized cloud->size() (all-invalid if no usable image),
  // so it stays index-aligned with the keyframe cloud through submap assembly.
  auto refs = std::make_shared<nano_gicp::VisualRefList>(cloud->size());
  if (img.empty() || img.type() != CV_32FC1 || this->camera_intrinsics_.size() != 4) {
    return refs;
  }
  const float fx = this->camera_intrinsics_[0], fy = this->camera_intrinsics_[1];
  const float cx = this->camera_intrinsics_[2], cy = this->camera_intrinsics_[3];
  const Eigen::Matrix3f R = T_cw.linear();
  const Eigen::Vector3f t = T_cw.translation();
  const float bw = 2.f;
  const float umax = static_cast<float>(img.cols) - 1.f - bw;
  const float vmax = static_cast<float>(img.rows) - 1.f - bw;
  for (size_t i = 0; i < cloud->size(); ++i) {
    const auto& p = cloud->at(i);
    const Eigen::Vector3f Pc = R * Eigen::Vector3f(p.x, p.y, p.z) + t;
    nano_gicp::VisualRef vr;
    vr.valid = 0;
    if (Pc.z() > 1e-3f) {
      const float u = fx * Pc.x() / Pc.z() + cx;
      const float v = fy * Pc.y() / Pc.z() + cy;
      if (u >= bw && u <= umax && v >= bw && v <= vmax) {
        const float gu = 0.5f * (bilinearF(img, u + 1.f, v) - bilinearF(img, u - 1.f, v));
        const float gv = 0.5f * (bilinearF(img, u, v + 1.f) - bilinearF(img, u, v - 1.f));
        if (std::abs(gu) > 1e-6f || std::abs(gv) > 1e-6f) {  // gradient-bearing only
          vr.ref = bilinearF(img, u, v);
          vr.p_kf_cam = Pc;
          vr.valid = 1;
        }
      }
    }
    (*refs)[i] = vr;
  }
  return refs;
}

void dlio::OdomNode::buildLidarIntensityImage(const pcl::PointCloud<PointType>::ConstPtr& organized,
                                              int width, int height) {
  // Reflectivity image in native (row=ring, col=azimuth) order, normalized to
  // the same /scale units as the residual reference (point.reflectivity/scale).
  const float inv_scale = 1.f / 255.0f;
  cv::Mat img(height, width, CV_32FC1, cv::Scalar(0.f));
  cv::Mat rng(height, width, CV_32FC1, cv::Scalar(0.f));  // range [m]; 0 = no return
  for (int row = 0; row < height; ++row) {
    float* dst = img.ptr<float>(row);
    float* drng = rng.ptr<float>(row);
    for (int col = 0; col < width; ++col) {
      const auto& p = organized->at(col, row);   // organized access (col, row)
      if (std::isfinite(p.x) && std::isfinite(p.reflectivity)) {
        dst[col] = p.reflectivity * inv_scale;
        drng[col] = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
      }
    }
  }
  this->lidar_refl_img_ = img;
  this->lidar_range_img_ = rng;
  this->lidar_img_ready_ = true;

  // Self-calibrate the spherical model once (sensor geometry is fixed):
  //   el(row) ~ el_a*row + el_b   (least-squares over per-row mean elevation)
  //   az(col) ~ az_a*col + az_b   (least-squares over one well-populated row, unwrapped)
  if (this->lidar_proj_ready_) { return; }

  // Elevation vs row (also captured as a per-row LUT for the non-uniform beams).
  this->lidar_el_lut_.assign(height, std::numeric_limits<float>::quiet_NaN());
  double sr = 0, se = 0, sre = 0, srr = 0; int ne = 0;
  for (int row = 0; row < height; ++row) {
    double accum = 0; int cnt = 0;
    for (int col = 0; col < width; ++col) {
      const auto& p = organized->at(col, row);
      const float rxy = std::sqrt(p.x * p.x + p.y * p.y);
      if (std::isfinite(p.x) && rxy > 1e-3f) { accum += std::atan2(p.z, rxy); ++cnt; }
    }
    if (cnt > 0) {
      const double el = accum / cnt;
      sr += row; se += el; sre += row * el; srr += (double)row * row; ++ne;
      this->lidar_el_lut_[row] = static_cast<float>(el);
    }
  }
  // Azimuth vs col on the most-populated row.
  int best_row = height / 2, best_cnt = -1;
  for (int row = 0; row < height; ++row) {
    int cnt = 0;
    for (int col = 0; col < width; ++col) {
      const auto& p = organized->at(col, row);
      if (std::isfinite(p.x) && (p.x * p.x + p.y * p.y) > 1e-6f) { ++cnt; }
    }
    if (cnt > best_cnt) { best_cnt = cnt; best_row = row; }
  }
  double sc = 0, sa = 0, sca = 0, scc = 0; int na = 0; double prev_az = 0, unwrap = 0;
  for (int col = 0; col < width; ++col) {
    const auto& p = organized->at(col, best_row);
    if (!std::isfinite(p.x) || (p.x * p.x + p.y * p.y) < 1e-6f) { continue; }
    double az = std::atan2(p.y, p.x);
    if (na > 0) {  // unwrap to keep the fit linear across the +/-pi seam
      while (az - prev_az > M_PI)  az -= 2.0 * M_PI;
      while (az - prev_az < -M_PI) az += 2.0 * M_PI;
    }
    prev_az = az;
    sc += col; sa += az; sca += col * az; scc += (double)col * col; ++na;
    (void)unwrap;
  }

  if (ne >= 2 && na >= 2) {
    const double el_den = ne * srr - sr * sr;
    const double az_den = na * scc - sc * sc;
    if (std::abs(el_den) > 1e-9 && std::abs(az_den) > 1e-9) {
      this->lidar_el_a_ = static_cast<float>((ne * sre - sr * se) / el_den);
      this->lidar_el_b_ = static_cast<float>((se - this->lidar_el_a_ * sr) / ne);
      this->lidar_az_a_ = static_cast<float>((na * sca - sc * sa) / az_den);
      this->lidar_az_b_ = static_cast<float>((sa - this->lidar_az_a_ * sc) / na);
      if (std::abs(this->lidar_el_a_) > 1e-9f && std::abs(this->lidar_az_a_) > 1e-9f) {
        this->lidar_proj_ready_ = true;
        RCLCPP_INFO(this->get_logger(),
            "LiDAR intensity image %dx%d; spherical model el=%.5f*row%+.4f, az=%.6f*col%+.4f",
            width, height, this->lidar_el_a_, this->lidar_el_b_, this->lidar_az_a_, this->lidar_az_b_);
      }
    }
  }
}

void dlio::OdomNode::getNextPose() {

  // Check if the new submap is ready to be used
  this->new_submap_is_ready = (this->submap_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready);

  if (this->new_submap_is_ready && this->submap_hasChanged) {

    // Adopt the submap target prepared in the background by buildSubmap():
    // cloud + kd-tree + photometric gradients from gicp_temp, plus the
    // keyframe-derived covariances assembled alongside the submap. Nothing
    // expensive is recomputed here on the registration hot path.
    this->gicp.shareTargetDataFrom(this->gicp_temp);
    this->gicp.setTargetCovariances(this->submap_normals);
    if (this->visual_map_enabled_) {
      this->gicp.setTargetVisualRefs(this->submap_visual_refs);
    }

    this->submap_hasChanged = false;
  }

  // Configure the optional direct visual term for this scan (picks the camera
  // frame nearest scan_stamp; no-op / LiDAR-only if disabled or no image).
  this->setupVisualForScan();

  // COIN-LIO LiDAR intensity-image term: project map points into this scan's
  // reflectivity image (world->lidar from the prior pose). No-op unless enabled
  // and an organized image + spherical model are ready.
  if (this->lidar_image_enabled_ && this->lidar_image_weight_ > 0.0 &&
      this->lidar_img_ready_ && this->lidar_proj_ready_) {
    Eigen::Matrix4f T_wl = this->T_prior * this->extrinsics.baselink2lidar_T;
    Eigen::Isometry3f T_lw;
    T_lw.matrix() = T_wl.inverse();
    this->gicp.setLidarImage(this->lidar_refl_img_);
    this->gicp.setLidarProjection(this->lidar_az_a_, this->lidar_az_b_, this->lidar_el_a_, this->lidar_el_b_);
    // Per-row elevation LUT (non-uniform OS beams): pass only if fully finite
    // and strictly monotonic, else fall back to the linear el model.
    bool lut_ok = this->lidar_el_lut_.size() == (size_t)this->lidar_refl_img_.rows && !this->lidar_el_lut_.empty();
    for (size_t k = 1; k < this->lidar_el_lut_.size() && lut_ok; ++k) {
      const float a = this->lidar_el_lut_[k - 1], b = this->lidar_el_lut_[k];
      if (!std::isfinite(a) || !std::isfinite(b) || a == b) { lut_ok = false; }
    }
    this->gicp.setLidarElevationLut(lut_ok ? this->lidar_el_lut_ : std::vector<float>{});
    // Range image + tolerance for the occlusion / wrong-surface rejection.
    this->gicp.setLidarRangeImage(this->lidar_range_img_);
    this->gicp.setLidarRangeConsistency(static_cast<float>(this->lidar_range_abs_tol_),
                                        static_cast<float>(this->lidar_range_rel_tol_));
    this->gicp.setLidarFrame(T_lw);
    this->gicp.setLidarMapWeight(static_cast<float>(this->lidar_image_weight_));
  } else {
    this->gicp.setLidarMapWeight(0.f);
  }

  // Align with current submap with global IMU transformation as initial guess
  pcl::PointCloud<PointType>::Ptr aligned = std::make_shared<pcl::PointCloud<PointType>>();
  this->gicp.align(*aligned);

  // Surface degeneracy (e.g. featureless tunnel): the solver held the IMU
  // prior along the unobservable directions; warn so the operator knows the
  // estimate is dead-reckoning in those directions.
  int degenerate_dirs = this->gicp.lastDegenerateDirections();
  // Snapshot for /diagnostics (publishDiagnostics reads these on the same
  // scan thread): directions held this scan + cumulative scans the gate fired.
  this->loc_gate_axes_current_ = degenerate_dirs;
  if (degenerate_dirs > 0) {
    ++this->loc_gate_updates_cumulative_;
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
        "Scan-to-map registration is degenerate along %d direction(s); "
        "holding IMU prior there (geometrically self-similar environment?)",
        degenerate_dirs);
  }

  // Get final transformation in global frame
  this->T_corr = this->gicp.getFinalTransformation(); // "correction" transformation
  this->T = this->T_corr * this->T_prior;

  // Promote this scan's camera frame to "previous" for the next scan's warp,
  // using the corrected pose (world -> camera at this->T).
  if (this->visual_cur_pending_valid_) {
    Eigen::Matrix4f T_wc = this->T * this->extrinsics.baselink2lidar_T * this->cam2lidar_T_;
    this->visual_T_cw_prev_.matrix() = T_wc.inverse();
    this->visual_prev_img_ = this->visual_cur_pending_;
    this->visual_has_prev_ = true;
    // NOTE: keep visual_cur_pending_valid_ true here so updateKeyframes() can
    // sample this scan's image for keyframe visual refs; it is reset at the top
    // of the next setupVisualForScan().
  }

  // Update next global pose
  // Both source and target clouds are in the global frame now, so tranformation is global
  this->propagateGICP();

  // Geometric observer update
  this->updateState();

}

bool dlio::OdomNode::imuMeasFromTimeRange(double start_time, double end_time,
                                          std::vector<ImuMeas>& imu_range) {

  // Hold mtx_imu for the WHOLE operation -- the wait, the range scan, AND the
  // copy -- so the concurrent callbackImu push_front (which can overwrite the
  // circular buffer's oldest slot) cannot mutate the elements while we read
  // them. The integration then runs on the private copy, lock-free.
  std::unique_lock<decltype(this->mtx_imu)> lock(this->mtx_imu);

  if (this->imu_buffer.empty() || this->imu_buffer.front().stamp < end_time) {
    bool imu_arrived = this->cv_imu_stamp.wait_for(lock, std::chrono::seconds(1),
        [this, &end_time]{ return !this->imu_buffer.empty()
                                  && this->imu_buffer.front().stamp >= end_time; });
    if (!imu_arrived) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
          "Timed out waiting for IMU data covering the scan (IMU dropout?); "
          "skipping IMU integration for this range");
      return false;
    }
  }

  auto imu_it = this->imu_buffer.begin();

  auto last_imu_it = imu_it;
  imu_it++;
  while (imu_it != this->imu_buffer.end() && imu_it->stamp >= end_time) {
    last_imu_it = imu_it;
    imu_it++;
  }

  while (imu_it != this->imu_buffer.end() && imu_it->stamp >= start_time) {
    imu_it++;
  }

  if (imu_it == this->imu_buffer.end()) {
    return false;
  }
  imu_it++;

  // Copy the range in forward-time order (reverse iterators over the
  // newest-at-front buffer) into the caller's vector, under the lock.
  boost::circular_buffer<ImuMeas>::reverse_iterator e(last_imu_it);
  boost::circular_buffer<ImuMeas>::reverse_iterator b(imu_it);
  imu_range.assign(b, e);

  return imu_range.size() >= 2;  // need >=2 samples for the back-integration
}

std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
dlio::OdomNode::integrateImu(double start_time, Eigen::Quaternionf q_init, Eigen::Vector3f p_init,
                             Eigen::Vector3f v_init, const std::vector<double>& sorted_timestamps) {

  const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> empty;

  if (sorted_timestamps.empty() || start_time > sorted_timestamps.front()) {
    // invalid input, return empty vector
    return empty;
  }

  std::vector<ImuMeas> imu_range;
  if (this->imuMeasFromTimeRange(start_time, sorted_timestamps.back(), imu_range) == false) {
    // not enough IMU measurements, return empty vector
    return empty;
  }

  // Backwards integration to find pose at first IMU sample
  const ImuMeas& f1 = imu_range[0];
  const ImuMeas& f2 = imu_range[1];

  // Time between first two IMU samples
  double dt = f2.dt;

  // Time between first IMU sample and start_time
  double idt = start_time - f1.stamp;

  // Angular acceleration between first two IMU samples
  Eigen::Vector3f alpha_dt = f2.ang_vel - f1.ang_vel;
  Eigen::Vector3f alpha = alpha_dt / dt;

  // Average angular velocity (reversed) between first IMU sample and start_time
  Eigen::Vector3f omega_i = -(f1.ang_vel + 0.5*alpha*idt);

  // Set q_init to orientation at first IMU sample
  q_init = Eigen::Quaternionf (
    q_init.w() - 0.5*( q_init.x()*omega_i[0] + q_init.y()*omega_i[1] + q_init.z()*omega_i[2] ) * idt,
    q_init.x() + 0.5*( q_init.w()*omega_i[0] - q_init.z()*omega_i[1] + q_init.y()*omega_i[2] ) * idt,
    q_init.y() + 0.5*( q_init.z()*omega_i[0] + q_init.w()*omega_i[1] - q_init.x()*omega_i[2] ) * idt,
    q_init.z() + 0.5*( q_init.x()*omega_i[1] - q_init.y()*omega_i[0] + q_init.w()*omega_i[2] ) * idt
  );
  q_init.normalize();

  // Average angular velocity between first two IMU samples
  Eigen::Vector3f omega = f1.ang_vel + 0.5*alpha_dt;

  // Orientation at second IMU sample
  Eigen::Quaternionf q2 (
    q_init.w() - 0.5*( q_init.x()*omega[0] + q_init.y()*omega[1] + q_init.z()*omega[2] ) * dt,
    q_init.x() + 0.5*( q_init.w()*omega[0] - q_init.z()*omega[1] + q_init.y()*omega[2] ) * dt,
    q_init.y() + 0.5*( q_init.z()*omega[0] + q_init.w()*omega[1] - q_init.x()*omega[2] ) * dt,
    q_init.z() + 0.5*( q_init.x()*omega[1] - q_init.y()*omega[0] + q_init.w()*omega[2] ) * dt
  );
  q2.normalize();

  // Acceleration at first IMU sample
  Eigen::Vector3f a1 = q_init._transformVector(f1.lin_accel);
  a1[2] -= this->gravity_;

  // Acceleration at second IMU sample
  Eigen::Vector3f a2 = q2._transformVector(f2.lin_accel);
  a2[2] -= this->gravity_;

  // Jerk between first two IMU samples
  Eigen::Vector3f j = (a2 - a1) / dt;

  // Set v_init to velocity at first IMU sample (go backwards from start_time)
  v_init -= a1*idt + 0.5*j*idt*idt;

  // Set p_init to position at first IMU sample (go backwards from start_time)
  p_init -= v_init*idt + 0.5*a1*idt*idt + (1/6.)*j*idt*idt*idt;

  return integrateImuInternal(q_init, p_init, v_init, sorted_timestamps, imu_range, this->gravity_);
}

std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
dlio::OdomNode::integrateImuInternal(Eigen::Quaternionf q_init, Eigen::Vector3f p_init, Eigen::Vector3f v_init,
                                     const std::vector<double>& sorted_timestamps,
                                     const std::vector<ImuMeas>& imu,
                                     double gravity) {

  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> imu_se3;

  // Initialization
  Eigen::Quaternionf q = q_init;
  Eigen::Vector3f p = p_init;
  Eigen::Vector3f v = v_init;
  Eigen::Vector3f a = q._transformVector(imu[0].lin_accel);
  a[2] -= gravity;

  // Iterate over IMU measurements (forward in time) and timestamps
  auto stamp_it = sorted_timestamps.begin();

  for (size_t k = 1; k < imu.size(); ++k) {

    const ImuMeas& f0 = imu[k-1];
    const ImuMeas& f = imu[k];

    // Time between IMU samples
    double dt = f.dt;

    // Angular acceleration
    Eigen::Vector3f alpha_dt = f.ang_vel - f0.ang_vel;
    Eigen::Vector3f alpha = alpha_dt / dt;

    // Average angular velocity
    Eigen::Vector3f omega = f0.ang_vel + 0.5*alpha_dt;

    // Orientation at f0: the interpolation below integrates forward from
    // here with idt measured from f0.stamp, matching how position is
    // interpolated. Using the already-advanced q double-counted one IMU
    // sample of rotation (a constant omega*dt attitude offset on every
    // deskewed point). Caught by test_imu_integration.
    Eigen::Quaternionf q0 = q;

    // Orientation at f
    q = Eigen::Quaternionf (
      q.w() - 0.5*( q.x()*omega[0] + q.y()*omega[1] + q.z()*omega[2] ) * dt,
      q.x() + 0.5*( q.w()*omega[0] - q.z()*omega[1] + q.y()*omega[2] ) * dt,
      q.y() + 0.5*( q.z()*omega[0] + q.w()*omega[1] - q.x()*omega[2] ) * dt,
      q.z() + 0.5*( q.x()*omega[1] - q.y()*omega[0] + q.w()*omega[2] ) * dt
    );
    q.normalize();

    // Acceleration
    Eigen::Vector3f a0 = a;
    a = q._transformVector(f.lin_accel);
    a[2] -= gravity;

    // Jerk
    Eigen::Vector3f j_dt = a - a0;
    Eigen::Vector3f j = j_dt / dt;

    // Interpolate for given timestamps
    while (stamp_it != sorted_timestamps.end() && *stamp_it <= f.stamp) {
      // Time between previous IMU sample and given timestamp
      double idt = *stamp_it - f0.stamp;

      // Average angular velocity
      Eigen::Vector3f omega_i = f0.ang_vel + 0.5*alpha*idt;

      // Orientation (integrated forward from f0, like the position below)
      Eigen::Quaternionf q_i (
        q0.w() - 0.5*( q0.x()*omega_i[0] + q0.y()*omega_i[1] + q0.z()*omega_i[2] ) * idt,
        q0.x() + 0.5*( q0.w()*omega_i[0] - q0.z()*omega_i[1] + q0.y()*omega_i[2] ) * idt,
        q0.y() + 0.5*( q0.z()*omega_i[0] + q0.w()*omega_i[1] - q0.x()*omega_i[2] ) * idt,
        q0.z() + 0.5*( q0.x()*omega_i[1] - q0.y()*omega_i[0] + q0.w()*omega_i[2] ) * idt
      );
      q_i.normalize();

      // Position
      Eigen::Vector3f p_i = p + v*idt + 0.5*a0*idt*idt + (1/6.)*j*idt*idt*idt;

      // Transformation
      Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
      T.block(0, 0, 3, 3) = q_i.toRotationMatrix();
      T.block(0, 3, 3, 1) = p_i;

      imu_se3.push_back(T);

      stamp_it++;
    }

    // Position
    p += v*dt + 0.5*a0*dt*dt + (1/6.)*j_dt*dt*dt;

    // Velocity
    v += a0*dt + 0.5*j_dt*dt;

  }

  return imu_se3;

}

void dlio::OdomNode::propagateGICP() {

  this->lidarPose.p << this->T(0,3), this->T(1,3), this->T(2,3);

  Eigen::Matrix3f rotSO3;
  rotSO3 << this->T(0,0), this->T(0,1), this->T(0,2),
            this->T(1,0), this->T(1,1), this->T(1,2),
            this->T(2,0), this->T(2,1), this->T(2,2);

  Eigen::Quaternionf q(rotSO3);

  // Normalize quaternion
  double norm = sqrt(q.w()*q.w() + q.x()*q.x() + q.y()*q.y() + q.z()*q.z());
  q.w() /= norm; q.x() /= norm; q.y() /= norm; q.z() /= norm;
  this->lidarPose.q = q;

}

void dlio::OdomNode::propagateState() {

  // Lock thread to prevent state from being accessed by UpdateState
  std::lock_guard<std::mutex> lock( this->geo.mtx );

  double dt = this->imu_meas.dt;

  Eigen::Quaternionf qhat = this->state.q, omega;
  Eigen::Vector3f world_accel;

  // Transform accel from body to world frame
  world_accel = qhat._transformVector(this->imu_meas.lin_accel);

  // Accel propogation
  this->state.p[0] += this->state.v.lin.w[0]*dt + 0.5*dt*dt*world_accel[0];
  this->state.p[1] += this->state.v.lin.w[1]*dt + 0.5*dt*dt*world_accel[1];
  this->state.p[2] += this->state.v.lin.w[2]*dt + 0.5*dt*dt*(world_accel[2] - this->gravity_);

  this->state.v.lin.w[0] += world_accel[0]*dt;
  this->state.v.lin.w[1] += world_accel[1]*dt;
  this->state.v.lin.w[2] += (world_accel[2] - this->gravity_)*dt;
  this->state.v.lin.b = this->state.q.toRotationMatrix().inverse() * this->state.v.lin.w;

  // Gyro propogation
  omega.w() = 0;
  omega.vec() = this->imu_meas.ang_vel;
  Eigen::Quaternionf tmp = qhat * omega;
  this->state.q.w() += 0.5 * dt * tmp.w();
  this->state.q.vec() += 0.5 * dt * tmp.vec();

  // Ensure quaternion is properly normalized
  this->state.q.normalize();

  this->state.v.ang.b = this->imu_meas.ang_vel;
  this->state.v.ang.w = this->state.q.toRotationMatrix() * this->state.v.ang.b;

}

void dlio::OdomNode::updateState() {

  // Lock thread to prevent state from being accessed by PropagateState
  std::lock_guard<std::mutex> lock( this->geo.mtx );

  Eigen::Vector3f pin = this->lidarPose.p;
  Eigen::Quaternionf qin = this->lidarPose.q;
  double dt = this->scan_stamp - this->prev_scan_stamp;

  Eigen::Quaternionf qe, qhat, qcorr;
  qhat = this->state.q;

  // Constuct error quaternion
  qe = qhat.conjugate()*qin;

  double sgn = 1.;
  if (qe.w() < 0) {
    sgn = -1;
  }

  // Construct quaternion correction
  qcorr.w() = 1 - abs(qe.w());
  qcorr.vec() = sgn*qe.vec();
  qcorr = qhat * qcorr;

  Eigen::Vector3f err = pin - this->state.p;
  Eigen::Vector3f err_body;

  err_body = qhat.conjugate()._transformVector(err);

  double abias_max = this->geo_abias_max_;
  double gbias_max = this->geo_gbias_max_;

  // Update accel bias
  this->state.b.accel -= dt * this->geo_Kab_ * err_body;
  this->state.b.accel = this->state.b.accel.array().min(abias_max).max(-abias_max);

  // Update gyro bias
  this->state.b.gyro[0] -= dt * this->geo_Kgb_ * qe.w() * qe.x();
  this->state.b.gyro[1] -= dt * this->geo_Kgb_ * qe.w() * qe.y();
  this->state.b.gyro[2] -= dt * this->geo_Kgb_ * qe.w() * qe.z();
  this->state.b.gyro = this->state.b.gyro.array().min(gbias_max).max(-gbias_max);

  // Update state
  this->state.p += dt * this->geo_Kp_ * err;
  this->state.v.lin.w += dt * this->geo_Kv_ * err;

  this->state.q.w() += dt * this->geo_Kq_ * qcorr.w();
  this->state.q.x() += dt * this->geo_Kq_ * qcorr.x();
  this->state.q.y() += dt * this->geo_Kq_ * qcorr.y();
  this->state.q.z() += dt * this->geo_Kq_ * qcorr.z();
  this->state.q.normalize();

  // store previous pose, orientation, and velocity
  this->geo.prev_p = this->state.p;
  this->geo.prev_q = this->state.q;
  this->geo.prev_vel = this->state.v.lin.w;

}

sensor_msgs::msg::Imu::SharedPtr dlio::OdomNode::transformImu(const sensor_msgs::msg::Imu::SharedPtr& imu_raw) {

  auto imu = std::make_shared<sensor_msgs::msg::Imu>();

  // Copy header
  imu->header = imu_raw->header;

  double imu_stamp_secs = rclcpp::Time(imu->header.stamp).seconds();
  static double prev_stamp = imu_stamp_secs;
  double dt = imu_stamp_secs - prev_stamp;
  prev_stamp = imu_stamp_secs;
  
  if (dt == 0) { dt = 1.0/200.0; }

  // Transform angular velocity (will be the same on a rigid body, so just rotate to ROS convention)
  Eigen::Vector3f ang_vel(imu_raw->angular_velocity.x,
                          imu_raw->angular_velocity.y,
                          imu_raw->angular_velocity.z);

  Eigen::Vector3f ang_vel_cg = this->extrinsics.baselink2imu.R * ang_vel;

  imu->angular_velocity.x = ang_vel_cg[0];
  imu->angular_velocity.y = ang_vel_cg[1];
  imu->angular_velocity.z = ang_vel_cg[2];

  static Eigen::Vector3f ang_vel_cg_prev = ang_vel_cg;

  // Transform linear acceleration (need to account for component due to translational difference)
  Eigen::Vector3f lin_accel(imu_raw->linear_acceleration.x,
                            imu_raw->linear_acceleration.y,
                            imu_raw->linear_acceleration.z);

  Eigen::Vector3f lin_accel_cg = this->extrinsics.baselink2imu.R * lin_accel;

  lin_accel_cg = lin_accel_cg
                 + ((ang_vel_cg - ang_vel_cg_prev) / dt).cross(-this->extrinsics.baselink2imu.t)
                 + ang_vel_cg.cross(ang_vel_cg.cross(-this->extrinsics.baselink2imu.t));

  ang_vel_cg_prev = ang_vel_cg;

  imu->linear_acceleration.x = lin_accel_cg[0];
  imu->linear_acceleration.y = lin_accel_cg[1];
  imu->linear_acceleration.z = lin_accel_cg[2];

  return imu;

}

void dlio::OdomNode::computeMetrics() {
  this->computeSpaciousness();
  this->computeDensity();
}

void dlio::OdomNode::computeSpaciousness() {

  // compute range of points (work with SQUARED range to skip a per-point sqrt:
  // sqrt is monotonic, so median(sqrt(.)) == sqrt(median(.)) -- one sqrt total
  // instead of one per point on the full-resolution scan).
  std::vector<float> ds;
  ds.reserve(this->original_scan->points.size());

  for (const auto& pt : this->original_scan->points) {
    ds.push_back(pt.x * pt.x + pt.y * pt.y);
  }

  // median
  std::nth_element(ds.begin(), ds.begin() + ds.size()/2, ds.end());
  float median_curr = std::sqrt(ds[ds.size()/2]);
  static float median_prev = median_curr;
  float median_lpf = 0.95*median_prev + 0.05*median_curr;
  median_prev = median_lpf;

  // push
  this->metrics.spaciousness.push_back( median_lpf );
  cap_history(this->metrics.spaciousness);

}

void dlio::OdomNode::computeDensity() {

  float density;

  if (!this->geo.first_opt_done) {
    density = 0.;
  } else {
    density = this->gicp.source_density_;
  }

  static float density_prev = density;
  float density_lpf = 0.95*density_prev + 0.05*density;
  density_prev = density_lpf;

  this->metrics.density.push_back( density_lpf );
  cap_history(this->metrics.density);

}

void dlio::OdomNode::computeConvexHull() {

  // at least 4 keyframes for convex hull
  if (this->num_processed_keyframes < 4) {
    return;
  }

  // create a pointcloud with points at keyframes
  pcl::PointCloud<PointType>::Ptr cloud = std::make_shared<pcl::PointCloud<PointType>>();

  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
  for (int i = 0; i < this->num_processed_keyframes; i++) {
    PointType pt;
    pt.x = this->keyframes[i].first.first[0];
    pt.y = this->keyframes[i].first.first[1];
    pt.z = this->keyframes[i].first.first[2];
    cloud->push_back(pt);
  }
  lock.unlock();

  // calculate the convex hull of the point cloud
  this->convex_hull.setInputCloud(cloud);

  // get the indices of the keyframes on the convex hull
  pcl::PointCloud<PointType>::Ptr convex_points = std::make_shared<pcl::PointCloud<PointType>>();
  this->convex_hull.reconstruct(*convex_points);

  pcl::PointIndices::Ptr convex_hull_point_idx = std::make_shared<pcl::PointIndices>();
  this->convex_hull.getHullPointIndices(*convex_hull_point_idx);

  this->keyframe_convex.clear();
  for (int i=0; i<convex_hull_point_idx->indices.size(); ++i) {
    this->keyframe_convex.push_back(convex_hull_point_idx->indices[i]);
  }

}

void dlio::OdomNode::computeConcaveHull() {

  // at least 5 keyframes for concave hull
  if (this->num_processed_keyframes < 5) {
    return;
  }

  // create a pointcloud with points at keyframes
  auto cloud = std::make_shared<pcl::PointCloud<PointType>>();

  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
  for (int i = 0; i < this->num_processed_keyframes; i++) {
    PointType pt;
    pt.x = this->keyframes[i].first.first[0];
    pt.y = this->keyframes[i].first.first[1];
    pt.z = this->keyframes[i].first.first[2];
    cloud->push_back(pt);
  }
  lock.unlock();

  // calculate the concave hull of the point cloud
  this->concave_hull.setInputCloud(cloud);

  // get the indices of the keyframes on the concave hull
  pcl::PointCloud<PointType>::Ptr concave_points = std::make_shared<pcl::PointCloud<PointType>>();
  this->concave_hull.reconstruct(*concave_points);

  pcl::PointIndices::Ptr concave_hull_point_idx = std::make_shared<pcl::PointIndices>();
  this->concave_hull.getHullPointIndices(*concave_hull_point_idx);

  this->keyframe_concave.clear();
  for (int i=0; i<concave_hull_point_idx->indices.size(); ++i) {
    this->keyframe_concave.push_back(concave_hull_point_idx->indices[i]);
  }

}

void dlio::OdomNode::updateKeyframes() {

  // calculate difference in pose and rotation to all poses in trajectory
  float closest_d = std::numeric_limits<float>::infinity();
  int closest_idx = 0;
  int keyframes_idx = 0;

  int num_nearby = 0;

  for (const auto& k : this->keyframes) {

    // calculate distance between current pose and pose in keyframes
    const float kdx = this->state.p[0] - k.first.first[0];
    const float kdy = this->state.p[1] - k.first.first[1];
    const float kdz = this->state.p[2] - k.first.first[2];
    float delta_d = std::sqrt(kdx*kdx + kdy*kdy + kdz*kdz);

    // count the number nearby current pose
    if (delta_d <= this->keyframe_thresh_dist_ * 1.5){
      ++num_nearby;
    }

    // store into variable
    if (delta_d < closest_d) {
      closest_d = delta_d;
      closest_idx = keyframes_idx;
    }

    keyframes_idx++;

  }

  // get closest pose and corresponding rotation
  Eigen::Vector3f closest_pose = this->keyframes[closest_idx].first.first;
  Eigen::Quaternionf closest_pose_r = this->keyframes[closest_idx].first.second;

  // calculate distance between current pose and closest pose from above
  const float cdx = this->state.p[0] - closest_pose[0];
  const float cdy = this->state.p[1] - closest_pose[1];
  const float cdz = this->state.p[2] - closest_pose[2];
  float dd = std::sqrt(cdx*cdx + cdy*cdy + cdz*cdz);

  // calculate difference in orientation using SLERP
  Eigen::Quaternionf dq;

  if (this->state.q.dot(closest_pose_r) < 0.) {
    Eigen::Quaternionf lq = closest_pose_r;
    lq.w() *= -1.; lq.x() *= -1.; lq.y() *= -1.; lq.z() *= -1.;
    dq = this->state.q * lq.inverse();
  } else {
    dq = this->state.q * closest_pose_r.inverse();
  }

  double theta_rad = 2. * atan2(sqrt( pow(dq.x(), 2) + pow(dq.y(), 2) + pow(dq.z(), 2) ), dq.w());
  double theta_deg = theta_rad * (180.0/M_PI);

  // update keyframes
  // Decision table of the cascade below (D = distance > threshD,
  // R = rotation > threshR, N = num_nearby <= 1):
  //   D                    -> new keyframe (regardless of R)
  //   !D &&  R &&  N       -> new keyframe (pure rotation in an uncrowded spot)
  //   !D &&  R && !N       -> no  (rotation trigger suppressed near other kfs)
  //   !D && !R             -> no
  // i.e. the rotation criterion only applies inside the num_nearby carve-out.
  bool newKeyframe = false;

  if (abs(dd) > this->keyframe_thresh_dist_ || abs(theta_deg) > this->keyframe_thresh_rot_) {
    newKeyframe = true;
  }

  if (abs(dd) <= this->keyframe_thresh_dist_) {
    newKeyframe = false;
  }

  if (abs(dd) <= this->keyframe_thresh_dist_ && abs(theta_deg) > this->keyframe_thresh_rot_ && num_nearby <= 1) {
    newKeyframe = true;
  }

  if (newKeyframe) {

    // update keyframe vector
    std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
    this->keyframes.push_back(std::make_pair(std::make_pair(this->lidarPose.p, this->lidarPose.q), this->current_scan));
    this->keyframe_timestamps.push_back(this->scan_header_stamp);
    this->keyframe_normals.push_back(std::make_shared<const nano_gicp::CovarianceList>(this->gicp.getSourceCovariances()));
    this->keyframe_transformations.push_back(this->T_corr);
    if (this->visual_map_enabled_) {
      Eigen::Matrix4f T_wc = this->T_prior * this->extrinsics.baselink2lidar_T * this->cam2lidar_T_;
      Eigen::Isometry3f T_cw; T_cw.matrix() = T_wc.inverse();
      cv::Mat img = this->visual_cur_pending_valid_ ? this->visual_cur_pending_ : cv::Mat();
      this->keyframe_visual_refs.push_back(this->sampleKeyframeVisualRefs(this->current_scan, T_cw, img));
    }
    lock.unlock();

  }

}

void dlio::OdomNode::pruneKeyframes() {

  // Bound the keyframe map by removing the most spatially redundant keyframe:
  // the one with the smallest distance to its nearest neighbor. The first
  // keyframe (origin anchor) and the most recent ones (likely in the current
  // submap neighborhood) are never removed. Caller guarantees the background
  // submap thread is idle -- pruning re-indexes the vectors it iterates.
  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);

  const int keep_recent = 10;

  while ((int)this->keyframes.size() > this->max_keyframes_) {

    // candidates: processed keyframes only, excluding the anchor and the tail
    int last_candidate = std::min(this->num_processed_keyframes,
                                  (int)this->keyframes.size() - keep_recent);
    if (last_candidate <= 1) { break; }

    int prune_idx = -1;
    float min_nn_dist = std::numeric_limits<float>::max();
    for (int i = 1; i < last_candidate; i++) {
      float nn = std::numeric_limits<float>::max();
      for (int j = 0; j < (int)this->keyframes.size(); j++) {
        if (j == i) { continue; }
        float d = (this->keyframes[i].first.first - this->keyframes[j].first.first).norm();
        nn = std::min(nn, d);
      }
      if (nn < min_nn_dist) {
        min_nn_dist = nn;
        prune_idx = i;
      }
    }
    if (prune_idx < 0) { break; }

    this->keyframes.erase(this->keyframes.begin() + prune_idx);
    this->keyframe_timestamps.erase(this->keyframe_timestamps.begin() + prune_idx);
    this->keyframe_normals.erase(this->keyframe_normals.begin() + prune_idx);
    this->keyframe_transformations.erase(this->keyframe_transformations.begin() + prune_idx);
    if (this->visual_map_enabled_ && prune_idx < (int)this->keyframe_visual_refs.size()) {
      this->keyframe_visual_refs.erase(this->keyframe_visual_refs.begin() + prune_idx);
    }
    --this->num_processed_keyframes;
  }

  // indices into the keyframe vectors are no longer valid; force the next
  // submap build to start fresh
  this->submap_kf_idx_prev.clear();

}

void dlio::OdomNode::setAdaptiveParams() {

  // Spaciousness
  float sp = this->metrics.spaciousness.back();

  if (sp < 0.5) { sp = 0.5; }
  if (sp > 5.0) { sp = 5.0; }

  this->keyframe_thresh_dist_ = sp;

  // Density
  float den = this->metrics.density.back();

  if (den < 0.5*this->gicp_max_corr_dist_) { den = 0.5*this->gicp_max_corr_dist_; }
  if (den > 2.0*this->gicp_max_corr_dist_) { den = 2.0*this->gicp_max_corr_dist_; }

  if (sp < 5.0) { den = 0.5*this->gicp_max_corr_dist_; };
  if (sp > 5.0) { den = 2.0*this->gicp_max_corr_dist_; };

  this->gicp.setMaxCorrespondenceDistance(den);

  // Concave hull alpha
  this->concave_hull.setAlpha(this->keyframe_thresh_dist_);

}

void dlio::OdomNode::pushSubmapIndices(std::vector<float> dists, int k, std::vector<int> frames) {

  // make sure dists is not empty
  if (!dists.size()) { return; }

  // maintain max heap of at most k elements
  std::priority_queue<float> pq;

  for (auto d : dists) {
    if (pq.size() >= k && pq.top() > d) {
      pq.push(d);
      pq.pop();
    } else if (pq.size() < k) {
      pq.push(d);
    }
  }

  // get the kth smallest element, which should be at the top of the heap
  float kth_element = pq.top();

  // get all elements smaller or equal to the kth smallest element
  for (int i = 0; i < dists.size(); ++i) {
    if (dists[i] <= kth_element)
      this->submap_kf_idx_curr.push_back(frames[i]);
  }

}

void dlio::OdomNode::buildSubmap(State vehicle_state) {

  // clear vector of keyframe indices to use for submap
  this->submap_kf_idx_curr.clear();

  // calculate distance between current pose and poses in keyframe set
  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
  std::vector<float> ds;
  std::vector<int> keyframe_nn;
  ds.reserve(this->num_processed_keyframes);
  keyframe_nn.reserve(this->num_processed_keyframes);
  for (int i = 0; i < this->num_processed_keyframes; i++) {
    const float dx = vehicle_state.p[0] - this->keyframes[i].first.first[0];
    const float dy = vehicle_state.p[1] - this->keyframes[i].first.first[1];
    const float dz = vehicle_state.p[2] - this->keyframes[i].first.first[2];
    ds.push_back(std::sqrt(dx*dx + dy*dy + dz*dz));
    keyframe_nn.push_back(i);
  }
  lock.unlock();

  // get indices for top K nearest neighbor keyframe poses
  this->pushSubmapIndices(ds, this->submap_knn_, keyframe_nn);

  // get convex hull indices
  this->computeConvexHull();

  // get distances for each keyframe on convex hull
  std::vector<float> convex_ds;
  for (const auto& c : this->keyframe_convex) {
    convex_ds.push_back(ds[c]);
  }

  // get indices for top kNN for convex hull
  this->pushSubmapIndices(convex_ds, this->submap_kcv_, this->keyframe_convex);

  // get concave hull indices
  this->computeConcaveHull();

  // get distances for each keyframe on concave hull
  std::vector<float> concave_ds;
  for (const auto& c : this->keyframe_concave) {
    concave_ds.push_back(ds[c]);
  }

  // get indices for top kNN for concave hull
  this->pushSubmapIndices(concave_ds, this->submap_kcc_, this->keyframe_concave);

  // sort current and previous submap kf list of indices
  std::sort(this->submap_kf_idx_curr.begin(), this->submap_kf_idx_curr.end());
  std::sort(this->submap_kf_idx_prev.begin(), this->submap_kf_idx_prev.end());

  // remove duplicate indices
  auto last = std::unique(this->submap_kf_idx_curr.begin(), this->submap_kf_idx_curr.end());
  this->submap_kf_idx_curr.erase(last, this->submap_kf_idx_curr.end());

  // check if submap has changed from previous iteration
  if (this->submap_kf_idx_curr != this->submap_kf_idx_prev){

    this->submap_hasChanged = true;

    // Pause to prevent stealing resources from the main loop if it is running.
    this->pauseSubmapBuildIfNeeded();

    // reinitialize submap cloud and normals
    pcl::PointCloud<PointType>::Ptr submap_cloud_ = std::make_shared<pcl::PointCloud<PointType>>();
    std::shared_ptr<nano_gicp::CovarianceList> submap_normals_ (std::make_shared<nano_gicp::CovarianceList>());
    // Concatenate the per-point visual refs in the SAME order as the cloud so
    // indices line up with the target points (only when the map term is on and
    // the refs are index-aligned with the keyframes).
    const bool build_visual_refs = this->visual_map_enabled_
        && this->keyframe_visual_refs.size() == this->keyframes.size();
    std::shared_ptr<nano_gicp::VisualRefList> submap_visual_refs_ =
        build_visual_refs ? std::make_shared<nano_gicp::VisualRefList>() : nullptr;

    for (auto k : this->submap_kf_idx_curr) {

      // Copy the cloud + per-keyframe shared_ptrs (normals, visual refs) under
      // the lock: the main thread's updateKeyframes() push_back can REALLOCATE
      // these vectors, so reading element [k] unlocked races a concurrent
      // reallocation (use-after-free). shared_ptr copies are cheap; the heavy
      // insert() work is done after unlocking.
      lock.lock();
      pcl::PointCloud<PointType>::ConstPtr kf_cloud = this->keyframes[k].second;
      std::shared_ptr<const nano_gicp::CovarianceList> kf_normals = this->keyframe_normals[k];
      std::shared_ptr<const nano_gicp::VisualRefList> kf_refs =
          build_visual_refs ? this->keyframe_visual_refs[k] : nullptr;
      lock.unlock();

      *submap_cloud_ += *kf_cloud;
      submap_normals_->insert( std::end(*submap_normals_),
          std::begin(*kf_normals), std::end(*kf_normals) );
      if (build_visual_refs && kf_refs) {
        submap_visual_refs_->insert( std::end(*submap_visual_refs_),
            std::begin(*kf_refs), std::end(*kf_refs) );
      }
    }

    this->submap_cloud = submap_cloud_;
    this->submap_normals = submap_normals_;
    this->submap_visual_refs = submap_visual_refs_;

    // Pause to prevent stealing resources from the main loop if it is running.
    this->pauseSubmapBuildIfNeeded();

    // Prepare the new target in the background: builds the kd-tree and the
    // photometric gradients without blocking the main loop. Covariances come
    // from the keyframe normals assembled above (submap_normals), shared with
    // the registration instance in getNextPose().
    this->gicp_temp.registerInputTarget(this->submap_cloud);

    this->submap_kf_idx_prev = this->submap_kf_idx_curr;
  }
}

void dlio::OdomNode::buildKeyframesAndSubmap(State vehicle_state) {

  // transform the new keyframe(s) and associated covariance list(s)
    std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);

  for (int i = this->num_processed_keyframes; i < this->keyframes.size(); i++) {
    pcl::PointCloud<PointType>::ConstPtr raw_keyframe = this->keyframes[i].second;
    std::shared_ptr<const nano_gicp::CovarianceList> raw_covariances = this->keyframe_normals[i];
    Eigen::Matrix4f T = this->keyframe_transformations[i];
    lock.unlock();

  Eigen::Matrix4f Tf = T; // T is already a float

  pcl::PointCloud<PointType>::Ptr transformed_keyframe = std::make_shared<pcl::PointCloud<PointType>>();
  pcl::transformPointCloud (*raw_keyframe, *transformed_keyframe, Tf);

  std::shared_ptr<nano_gicp::CovarianceList> transformed_covariances (std::make_shared<nano_gicp::CovarianceList>(raw_covariances->size()));

  std::transform(raw_covariances->begin(), raw_covariances->end(), transformed_covariances->begin(),
               [&Tf](const Eigen::Matrix4f& cov) { return Tf * cov * Tf.transpose(); });

    
    ++this->num_processed_keyframes;

    lock.lock();
    this->keyframes[i].second = transformed_keyframe;
    this->keyframe_normals[i] = transformed_covariances;

    if (this->publish_keyframe_thread.joinable()) { this->publish_keyframe_thread.join(); }
    this->publish_keyframe_thread = std::thread( &dlio::OdomNode::publishKeyframe, this, this->keyframes[i], this->keyframe_timestamps[i] );
  }

  lock.unlock();

  // Pause to prevent stealing resources from the main loop if it is running.
  this->pauseSubmapBuildIfNeeded();

  this->buildSubmap(vehicle_state);
}

void dlio::OdomNode::pauseSubmapBuildIfNeeded() {
  std::unique_lock<decltype(this->main_loop_running_mutex)> lock(this->main_loop_running_mutex);
  this->submap_build_cv.wait(lock, [this]{ return !this->main_loop_running; });
}

void dlio::OdomNode::publishDiagnostics() {

  // Computation time (ms): last / avg / max over the retained window.
  double comp_last = 0.0, comp_avg = 0.0, comp_max = 0.0;
  if (!this->comp_times.empty()) {
    comp_last = this->comp_times.back() * 1000.;
    comp_avg = std::accumulate(this->comp_times.begin(), this->comp_times.end(), 0.0)
               / this->comp_times.size() * 1000.;
    comp_max = *std::max_element(this->comp_times.begin(), this->comp_times.end()) * 1000.;
  }

  // Sensor rates (Hz), averaged over the most recent window (mirrors debug()).
  const int win = 100;
  auto avg_tail = [win](const std::vector<double>& v) -> double {
    if (v.empty()) return 0.0;
    if ((int)v.size() < win) return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    return std::accumulate(v.end() - win, v.end(), 0.0) / win;
  };
  double imu_rate;
  { std::lock_guard<decltype(this->mtx_imu)> rlk(this->mtx_imu); imu_rate = avg_tail(this->imu_rates); }
  double lidar_rate = avg_tail(this->lidar_rates);

  // RAM (resident set, MB) from /proc/self/stat.
  double resident_mb = 0.0;
  {
    std::ifstream stat_stream("/proc/self/stat", std::ios_base::in);
    std::string ignore;
    unsigned long vsize = 0; long rss = 0;
    // fields: pid comm state ppid pgrp session tty_nr tpgid flags minflt
    //         cminflt majflt cmajflt utime stime cutime cstime priority
    //         nice num_threads itrealvalue starttime vsize rss
    for (int i = 0; i < 22; ++i) stat_stream >> ignore;
    stat_stream >> vsize >> rss;
    long page_kb = sysconf(_SC_PAGE_SIZE) / 1024;
    resident_mb = (rss * page_kb) / 1000.;
  }

  // CPU utilization since the previous diagnostics publish (own baseline so it
  // does not consume debug()'s since-last-call window).
  double cpu_percent = 0.0, cores = 0.0;
  {
    struct tms t; clock_t now = times(&t);
    if (this->lastCPU_diag_ != (clock_t)-1 && now > this->lastCPU_diag_ &&
        t.tms_stime >= this->lastSysCPU_diag_ && t.tms_utime >= this->lastUserCPU_diag_) {
      cpu_percent = (double)((t.tms_stime - this->lastSysCPU_diag_) +
                             (t.tms_utime - this->lastUserCPU_diag_));
      cpu_percent /= (now - this->lastCPU_diag_);
      cpu_percent /= this->numProcessors;
      cpu_percent *= 100.;
      cores = (cpu_percent / 100.) * this->numProcessors;
    }
    this->lastCPU_diag_ = now;
    this->lastSysCPU_diag_ = t.tms_stime;
    this->lastUserCPU_diag_ = t.tms_utime;
  }

  diagnostic_msgs::msg::DiagnosticArray arr;
  arr.header.stamp = this->scan_header_stamp;
  arr.header.frame_id = this->odom_frame;

  diagnostic_msgs::msg::DiagnosticStatus st;
  st.name = "DLIO Odometry";          // capture tooling filters on substring "DLIO"
  st.hardware_id = "DLIO";

  auto kv = [&st](const std::string& key, const std::string& value) {
    diagnostic_msgs::msg::KeyValue pair;
    pair.key = key; pair.value = value;
    st.values.push_back(pair);
  };
  auto fnum = [](double v, int prec = 3) {
    std::ostringstream os; os << std::fixed << std::setprecision(prec) << v; return os.str();
  };

  // Health: WARN while the degeneracy gate is active or the per-scan budget
  // (the LiDAR period) is blown; ERROR if GICP failed to converge.
  double scan_period_ms = (lidar_rate > 0.0) ? 1000.0 / lidar_rate : 0.0;
  if (!this->gicp_hasConverged.load()) {
    st.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
    st.message = "GICP did not converge";
  } else if (this->loc_gate_axes_current_ > 0) {
    st.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
    st.message = "Degenerate along " + std::to_string(this->loc_gate_axes_current_) +
                 " direction(s); holding IMU prior";
  } else if (scan_period_ms > 0.0 && comp_last > scan_period_ms) {
    st.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
    st.message = "Computation time exceeds scan period";
  } else {
    st.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
    st.message = "OK";
  }

  kv("Computation Time (ms)", fnum(comp_last, 2));
  kv("Avg Computation Time (ms)", fnum(comp_avg, 2));
  kv("Max Computation Time (ms)", fnum(comp_max, 2));
  kv("CPU Load (%)", fnum(cpu_percent, 1));
  kv("Cores Utilized", fnum(cores, 2));
  kv("RAM Allocation (MB)", fnum(resident_mb, 1));
  kv("LiDAR Rate (Hz)", fnum(lidar_rate, 2));
  kv("IMU Rate (Hz)", fnum(imu_rate, 2));
  kv("Distance Traveled (m)", fnum(this->length_traversed, 3));
  kv("Keyframes", std::to_string(this->keyframes.size()));
  kv("Deskewed Points", std::to_string(this->deskew_size.load()));
  kv("GICP Converged", this->gicp_hasConverged.load() ? "1" : "0");
  // CPU starvation: realtime factor (>1 = slower than real time), cumulative
  // compute overruns, and an estimate of transport-dropped scans.
  kv("Realtime Factor", fnum(this->last_realtime_factor_, 2));
  kv("CPU Starved", (this->last_realtime_factor_ > 1.0) ? "1" : "0");
  kv("Compute Overruns (cumulative)", std::to_string(this->compute_overruns_.load()));
  kv("Scans Dropped est (cumulative)", std::to_string(this->scans_dropped_est_.load()));
  kv("Degenerate Directions (current)", std::to_string(this->loc_gate_axes_current_));
  kv("Loc Gate Updates (cumulative)", std::to_string(this->loc_gate_updates_cumulative_));
  kv("Photometric Active", this->photometric_active_ ? "1" : "0");
  kv("Photometric Channel", this->use_reflectivity_ ? "reflectivity" : "intensity");
  kv("Visual Active", (this->visual_enabled_ && this->gicp.lastVisualCount() > 0) ? "1" : "0");
  kv("Visual Points", std::to_string(this->gicp.lastVisualCount()));
  kv("Visual Residual RMS", fnum(this->gicp.lastVisualRms(), 4));
  kv("Visual Rescued Axes", std::to_string(this->gicp.lastVisualRescuedDirections()));
  kv("Visual Map Active", (this->visual_map_enabled_ && this->gicp.lastVisualMapCount() > 0) ? "1" : "0");
  kv("Visual Map Points", std::to_string(this->gicp.lastVisualMapCount()));
  kv("Visual Map RMS", fnum(this->gicp.lastVisualMapRms(), 4));
  kv("Lidar Map Active", (this->lidar_image_enabled_ && this->gicp.lastLidarMapCount() > 0) ? "1" : "0");
  kv("Lidar Map Points", std::to_string(this->gicp.lastLidarMapCount()));
  kv("Lidar Map RMS", fnum(this->gicp.lastLidarMapRms(), 4));
  kv("Elapsed Time (s)", fnum(this->elapsed_time, 2));

  arr.status.push_back(st);
  this->diag_pub->publish(arr);
}

void dlio::OdomNode::debug() {

  // Snapshot the geo-protected state once under the lock; the IMU thread writes
  // it via propagateState/updateState under geo.mtx. (dashboard read only.)
  State dbg_state;
  { std::lock_guard<std::mutex> lock(this->geo.mtx); dbg_state = this->state; }

  // Total length traversed is maintained incrementally in callbackPointCloud
  double length_traversed = this->length_traversed;

  // Average computation time
  double avg_comp_time =
    std::accumulate(this->comp_times.begin(), this->comp_times.end(), 0.0) / this->comp_times.size();

  // Average sensor rates
  int win_size = 100;
  double avg_imu_rate;
  double avg_lidar_rate;
  {
    std::lock_guard<decltype(this->mtx_imu)> rlk(this->mtx_imu);
    if ((int)this->imu_rates.size() < win_size) {
      avg_imu_rate = this->imu_rates.empty() ? 0.0 :
        std::accumulate(this->imu_rates.begin(), this->imu_rates.end(), 0.0) / this->imu_rates.size();
    } else {
      avg_imu_rate =
        std::accumulate(this->imu_rates.end()-win_size, this->imu_rates.end(), 0.0) / win_size;
    }
  }
  if (this->lidar_rates.size() < win_size) {
    avg_lidar_rate =
      std::accumulate(this->lidar_rates.begin(), this->lidar_rates.end(), 0.0) / this->lidar_rates.size();
  } else {
    avg_lidar_rate =
      std::accumulate(this->lidar_rates.end()-win_size, this->lidar_rates.end(), 0.0) / win_size;
  }

  // RAM Usage
  double vm_usage = 0.0;
  double resident_set = 0.0;
  std::ifstream stat_stream("/proc/self/stat", std::ios_base::in); //get info from proc directory
  std::string pid, comm, state, ppid, pgrp, session, tty_nr;
  std::string tpgid, flags, minflt, cminflt, majflt, cmajflt;
  std::string utime, stime, cutime, cstime, priority, nice;
  std::string num_threads, itrealvalue, starttime;
  unsigned long vsize;
  long rss;
  stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
              >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
              >> utime >> stime >> cutime >> cstime >> priority >> nice
              >> num_threads >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest
  stat_stream.close();
  long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // for x86-64 is configured to use 2MB pages
  vm_usage = vsize / 1024.0;
  resident_set = rss * page_size_kb;

  // CPU Usage
  struct tms timeSample;
  clock_t now;
  double cpu_percent;
  now = times(&timeSample);
  if (now <= this->lastCPU || timeSample.tms_stime < this->lastSysCPU ||
      timeSample.tms_utime < this->lastUserCPU) {
      cpu_percent = -1.0;
  } else {
      cpu_percent = (timeSample.tms_stime - this->lastSysCPU) + (timeSample.tms_utime - this->lastUserCPU);
      cpu_percent /= (now - this->lastCPU);
      cpu_percent /= this->numProcessors;
      cpu_percent *= 100.;
  }
  this->lastCPU = now;
  this->lastSysCPU = timeSample.tms_stime;
  this->lastUserCPU = timeSample.tms_utime;
  this->cpu_percents.push_back(cpu_percent);
  cap_history(this->cpu_percents);
  double avg_cpu_usage =
    std::accumulate(this->cpu_percents.begin(), this->cpu_percents.end(), 0.0) / this->cpu_percents.size();

  // Print to terminal
  printf("\033[2J\033[1;1H");

  std::cout << std::endl
            << "+-------------------------------------------------------------------+" << std::endl;
  std::cout << "|               Direct LiDAR-Inertial Odometry v" << this->version_  << "               |"
            << std::endl;
  std::cout << "+-------------------------------------------------------------------+" << std::endl;

  std::time_t curr_time = this->scan_stamp;
  std::string asc_time = std::asctime(std::localtime(&curr_time)); asc_time.pop_back();
  std::cout << "| " << std::left << asc_time;
  std::cout << std::right << std::setfill(' ') << std::setw(42)
    << "Elapsed Time: " + to_string_with_precision(this->elapsed_time, 2) + " seconds "
    << "|" << std::endl;

  if ( !this->cpu_type.empty() ) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << this->cpu_type + " x " + std::to_string(this->numProcessors)
      << "|" << std::endl;
  }

  if (this->sensor == dlio::SensorType::OUSTER) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << "Sensor Rates: Ouster @ " + to_string_with_precision(avg_lidar_rate, 2)
                                   + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
      << "|" << std::endl;
  } else if (this->sensor == dlio::SensorType::VELODYNE) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << "Sensor Rates: Velodyne @ " + to_string_with_precision(avg_lidar_rate, 2)
                                     + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
      << "|" << std::endl;
  } else if (this->sensor == dlio::SensorType::HESAI) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << "Sensor Rates: Hesai @ " + to_string_with_precision(avg_lidar_rate, 2)
                                  + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
      << "|" << std::endl;
  } else if (this->sensor == dlio::SensorType::LIVOX) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << "Sensor Rates: Livox @ " + to_string_with_precision(avg_lidar_rate, 2)
                                  + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
      << "|" << std::endl;
  } else {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << "Sensor Rates: Unknown LiDAR @ " + to_string_with_precision(avg_lidar_rate, 2)
                                          + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
      << "|" << std::endl;
  }

  std::cout << "|===================================================================|" << std::endl;

  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Position     {W}  [xyz] :: " + to_string_with_precision(dbg_state.p[0], 4) + " "
                                + to_string_with_precision(dbg_state.p[1], 4) + " "
                                + to_string_with_precision(dbg_state.p[2], 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Orientation  {W} [wxyz] :: " + to_string_with_precision(dbg_state.q.w(), 4) + " "
                                + to_string_with_precision(dbg_state.q.x(), 4) + " "
                                + to_string_with_precision(dbg_state.q.y(), 4) + " "
                                + to_string_with_precision(dbg_state.q.z(), 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Lin Velocity {B}  [xyz] :: " + to_string_with_precision(dbg_state.v.lin.b[0], 4) + " "
                                + to_string_with_precision(dbg_state.v.lin.b[1], 4) + " "
                                + to_string_with_precision(dbg_state.v.lin.b[2], 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Ang Velocity {B}  [xyz] :: " + to_string_with_precision(dbg_state.v.ang.b[0], 4) + " "
                                + to_string_with_precision(dbg_state.v.ang.b[1], 4) + " "
                                + to_string_with_precision(dbg_state.v.ang.b[2], 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Accel Bias        [xyz] :: " + to_string_with_precision(dbg_state.b.accel[0], 8) + " "
                                + to_string_with_precision(dbg_state.b.accel[1], 8) + " "
                                + to_string_with_precision(dbg_state.b.accel[2], 8)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Gyro Bias         [xyz] :: " + to_string_with_precision(dbg_state.b.gyro[0], 8) + " "
                                + to_string_with_precision(dbg_state.b.gyro[1], 8) + " "
                                + to_string_with_precision(dbg_state.b.gyro[2], 8)
    << "|" << std::endl;

  std::cout << "|                                                                   |" << std::endl;

  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Distance Traveled  :: " + to_string_with_precision(length_traversed, 4) + " meters"
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Distance to Origin :: "
      + to_string_with_precision( sqrt(pow(dbg_state.p[0]-this->origin[0],2) +
                                       pow(dbg_state.p[1]-this->origin[1],2) +
                                       pow(dbg_state.p[2]-this->origin[2],2)), 4) + " meters"
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Registration       :: keyframes: " + std::to_string(this->keyframes.size()) + ", "
                               + "deskewed points: " + std::to_string(this->deskew_size)
    << "|" << std::endl;
  std::cout << "|                                                                   |" << std::endl;

  std::cout << std::right << std::setprecision(2) << std::fixed;
  std::cout << "| Computation Time :: "
    << std::setfill(' ') << std::setw(6) << this->comp_times.back()*1000. << " ms    // Avg: "
    << std::setw(6) << avg_comp_time*1000. << " / Max: "
    << std::setw(6) << *std::max_element(this->comp_times.begin(), this->comp_times.end())*1000.
    << "     |" << std::endl;
  std::cout << "| Cores Utilized   :: "
    << std::setfill(' ') << std::setw(6) << (cpu_percent/100.) * this->numProcessors << " cores // Avg: "
    << std::setw(6) << (avg_cpu_usage/100.) * this->numProcessors << " / Max: "
    << std::setw(6) << (*std::max_element(this->cpu_percents.begin(), this->cpu_percents.end()) / 100.)
                       * this->numProcessors
    << "     |" << std::endl;
  std::cout << "| CPU Load         :: "
    << std::setfill(' ') << std::setw(6) << cpu_percent << " %     // Avg: "
    << std::setw(6) << avg_cpu_usage << " / Max: "
    << std::setw(6) << *std::max_element(this->cpu_percents.begin(), this->cpu_percents.end())
    << "     |" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "RAM Allocation   :: " + to_string_with_precision(resident_set/1000., 2) + " MB"
    << "|" << std::endl;

  std::cout << "+-------------------------------------------------------------------+" << std::endl;

}
