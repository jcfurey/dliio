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
#include "dlio/slosh_guard.h"

#include <array>

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
#include <map>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

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
                         const std::vector<ImuMeas>& imu,
                         double gravity);

  // Radiometric intensity correction (range + optional incidence angle), the
  // Kashani et al. model  I' = I * (r/r_ref)^alpha / max(cos_incidence, cos_min)
  // clamped to [0,255]. cos_incidence = |beam . surface_normal| in [0,1]
  // (1 = normal incidence; pass 1 to apply range-only). Static and
  // side-effect-free for unit testing.
  static float correctIntensity(float intensity, float range, float cos_incidence,
                                 float alpha, float r_ref, float cos_min);

  // Map a cloud's point fields to the source sensor (for the per-point time
  // accessor used in deskew): 't' = Ouster, 'time' = Velodyne, 'timestamp' =
  // Hesai (absolute seconds, < 1e14) or Livox (absolute nanoseconds, > 1e14).
  // first_timestamp is the first point's `timestamp` value (only consulted when
  // a 'timestamp' field is present and has_points is true). Static for testing.
  static SensorType detectSensorType(const std::vector<sensor_msgs::msg::PointField>& fields,
                                     bool has_points, double first_timestamp);

  // Intensity<->reflectivity fallback: resolve the configured photometric channel
  // against the fields the cloud actually carries. In/out: use_reflectivity and
  // photometric_active are updated to the effective values -- reflectivity falls
  // back to intensity (and vice versa) when its field is absent, or the term is
  // disabled if neither field is present. No-op when the term is off or the
  // requested channel is available. Static + side-effect-free for unit testing.
  static void resolvePhotometricChannel(bool has_reflectivity, bool has_intensity,
                                        bool& use_reflectivity, bool& photometric_active);

  // Spatial box-blur the per-point image channel (the .reflectivity slot) over
  // the organized K x K neighbourhood, averaging per-pixel shot noise down
  // ~sqrt(valid neighbours) while preserving structured wall texture (near-IR is
  // shot-noise-dominated). Restricted to valid-return pixels (finite x), in-place,
  // and applied before the image is built so the current image and the keyframe
  // references denoise consistently. No-op for K <= 1. Static for unit testing.
  static void denoiseOrganizedChannel(const pcl::PointCloud<PointType>::Ptr& organized,
                                      int width, int height, int kernel);

  // Sub-floor reject keep-mask (specular water/mirror "ghost" removal). Points
  // are flat x/y/z arrays in a GRAVITY-ALIGNED frame (z = up). The floor is
  // estimated PER 2D CELL as the lowest z-bin holding >= min_bin_count points
  // (robust to the sparse sub-floor ghosts themselves, and to slope/curvature);
  // a point is dropped (mask 0) only if it lies within `radius` of (cx,cy) AND
  // more than `margin` below its cell's floor. Cells with no dense floor bin and
  // points outside the radius are kept. If the drop fraction would exceed
  // max_reject_frac, NOTHING is dropped (returns all-1) -- a safety no-op.
  // Static + side-effect-free for unit testing.
  static std::vector<uint8_t> subFloorKeepMask(
      const std::vector<float>& xs, const std::vector<float>& ys,
      const std::vector<float>& zs, float cx, float cy,
      float radius, float cell, float z_bin, int min_bin_count,
      float margin, float max_reject_frac);

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
  // Resolve base_link->{imu,lidar} (and lidar->camera) extrinsics from tf2
  // (populated by robot_state_publisher from URDF) when extrinsics/source==tf.
  // Runs on a timer until the transforms are available (or attempts exhausted,
  // then falls back to the YAML values). Sets extrinsics_ready_ when done.
  void resolveExtrinsicsFromTf();

  void publishToROS(pcl::PointCloud<PointType>::ConstPtr published_cloud, Eigen::Matrix4f T_cloud);
  void publishCloud(pcl::PointCloud<PointType>::ConstPtr published_cloud, Eigen::Matrix4f T_cloud);
  void publishKeyframe(std::pair<std::pair<Eigen::Vector3f, Eigen::Quaternionf>,
                       pcl::PointCloud<PointType>::ConstPtr> kf, rclcpp::Time timestamp);

  void getScanFromROS(const sensor_msgs::msg::PointCloud2::SharedPtr& pc);
  void preprocessPoints();
  // Apply subFloorKeepMask to this->current_scan in place (no-op unless enabled
  // and the world frame is gravity-aligned). Updates last_subfloor_rejected_.
  void rejectSubFloor();
  void deskewPointcloud();
  void initializeInputTarget();
  void setInputSource();

  void initializeDLIO();

  void getNextPose();
  bool imuMeasFromTimeRange(double start_time, double end_time, std::vector<ImuMeas>& imu_range);
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

  // Live parameter tuning: on_set callback stages new values under live_mtx_;
  // applyLiveParams() (called on the scan thread, top of callbackPointCloud)
  // commits them to the members / gicp -- so all estimator state stays mutated
  // on one thread, race-free, while ros2 param set can retune during a run.
  rcl_interfaces::msg::SetParametersResult onSetParams(const std::vector<rclcpp::Parameter>& params);
  void applyLiveParams();

  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr on_set_handle_;
  std::mutex live_mtx_;
  std::map<std::string, double> live_pending_;
  std::atomic<bool> live_dirty_{false};

  rclcpp::TimerBase::SharedPtr publish_timer;

  // Subscribers
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub;
  rclcpp::CallbackGroup::SharedPtr lidar_cb_group, imu_cb_group, image_cb_group;
  // Optional raw Livox CustomMsg ingestion (built only with livox_ros_driver2).
  // Type-erased so odom.h carries no livox_ros_driver2 dependency and the class
  // layout is identical with or without it; wired up in the constructor (.cc).
  rclcpp::SubscriptionBase::SharedPtr livox_sub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr livox_pub;
  rclcpp::CallbackGroup::SharedPtr livox_cb_group;

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

  // Extrinsics source: "yaml" (default) or "tf" (URDF via robot_state_publisher).
  std::string extrinsics_source_;
  std::string camera_frame_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::TimerBase::SharedPtr extrinsics_timer_;
  int extrinsics_attempts_ = 0;
  // Gates scan/imu processing until the extrinsics are resolved (true for the
  // yaml path; set false until tf resolves so the extrinsics writes happen-
  // before any reader -- the atomic provides the synchronization).
  std::atomic<bool> extrinsics_ready_{true};

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

  // Per-instance running state that used to live in function-local statics --
  // those alias across OdomNode instances composed into a single process.
  // transformImu centripetal-correction history:
  bool transform_imu_init_ = false;
  double transform_prev_stamp_ = 0.0;
  Eigen::Vector3f ang_vel_cg_prev_ = Eigen::Vector3f::Zero();
  // startup IMU-calibration accumulators:
  int calib_num_samples_ = 0;
  Eigen::Vector3f calib_gyro_avg_ = Eigen::Vector3f::Zero();
  Eigen::Vector3f calib_accel_avg_ = Eigen::Vector3f::Zero();
  bool calib_print_ = true;
  // spaciousness / density metric low-pass state:
  bool spaciousness_init_ = false;
  float spaciousness_prev_ = 0.f;
  float density_prev_ = 0.f;

  ImuMeas imu_meas;

  boost::circular_buffer<ImuMeas> imu_buffer;
  std::mutex mtx_imu;
  std::condition_variable cv_imu_stamp;


  // Geometric Observer
  struct Geo {
    std::atomic<bool> first_opt_done;
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

  // Degeneracy GOVERNOR (fail-safe): on the eigen-directions the gate held to
  // the IMU prior, that prior dead-reckons and can run away (the km-scale
  // tunnel divergence). The governor caps per-scan output motion along those
  // world-frame directions to a physical envelope; bounding the pose feeds the
  // velocity back down through updateState()'s observer (err = lidarPose - state
  // -> Kv). 0 caps disable each axis (bit-identical). Plus covariance inflation
  // so a downstream consumer de-weights the held axes. All written/read on the
  // scan thread except the cov, snapshotted under geo.mtx for publishPose().
  bool degen_gov_enabled_ = false;
  float degen_gov_max_step_trans_ = 0.f;  // [m]   max per-scan motion along a held trans axis; 0 = off
  float degen_gov_max_step_rot_ = 0.f;    // [rad] max per-scan motion about a held rot axis;   0 = off
  // PHYSICS FUSE (physics_fuse.h): clamp the TOTAL per-scan output step (any
  // direction -- catches the runaways the governor's held-axis scope misses,
  // including a diverging IMU prior). 0 = off (bit-identical). Scan thread only.
  double fuse_max_step_trans_ = 0.0;      // [m]   max per-scan output translation; 0 = off
  double fuse_max_step_rot_ = 0.0;        // [rad] max per-scan output rotation;    0 = off
  bool fuse_tripped_scan_ = false;        // this scan clamped (vetoes keyframing)
  long fuse_trips_ = 0;                   // cumulative trips (diagnostics)
  // SLOSH GUARD (slosh_guard.h): online corkscrew detector on the output step
  // along the weak axis; when engaged, extra velocity damping + keyframe veto.
  // All accessed on the scan thread (getNextPose -> updateState -> updateKeyframes).
  bool slosh_enabled_ = false;
  double slosh_deadband_ = 0.02;          // [m] ignore steps below this (stationary noise)
  double slosh_vel_damp_ = 0.5;           // per-scan velocity damping along the axis while engaged
  dlio::SloshGuard slosh_guard_;          // detector (window/fracs set from params at startup)
  Eigen::Vector3f slosh_axis_ = Eigen::Vector3f::Zero();  // tracked weak axis (persists across scans)
  bool slosh_axis_valid_ = false;
  int slosh_axis_hold_ = 20;              // scans the tracked axis survives without the gate flagging one
  int slosh_axis_stale_ = 0;              // consecutive scans with no weak axis from the gate
  double degen_gov_cov_pos_var_ = 0.0;    // [m^2]   variance added along a held position axis; 0 = none
  double degen_gov_cov_rot_var_ = 0.0;    // [rad^2] variance added along a held rotation axis; 0 = none
  // Extra pose covariance from the inflation, world frame, written on the scan
  // thread under geo.mtx and read by publishPose() under the same lock.
  std::array<double, 36> degen_cov_extra_{};  // row-major 6x6, added onto the base pose cov

  // CPU-starvation indicators (see /diagnostics): scans whose compute time
  // exceeded the scan period, and an estimate of transport-dropped scans.
  std::atomic<long> compute_overruns_{0};
  std::atomic<long> scans_dropped_est_{0};
  double last_realtime_factor_ = 0.0;
  double prev_scan_period_ = 0.0;

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
  bool keyframe_degen_gate_ = false;   // veto keyframes while degenerate axes are held (anti map-contamination)
  bool publish_tf_ = true;             // broadcast odom->baselink (off when a fusion EKF owns the TF)

  int max_keyframes_;
  bool dashboard_;

  int submap_knn_;
  int submap_kcv_;
  int submap_kcc_;

  bool densemap_filtered_;
  bool wait_until_move_;

  // Sub-floor reject (specular ghost removal); off by default -> bit-identical.
  bool subfloor_reject_enabled_ = false;
  double subfloor_margin_ = 0.30;       // [m] reject this far below the cell floor
  double subfloor_radius_ = 8.0;        // [m] horizontal window analyzed
  double subfloor_cell_ = 1.0;          // [m] 2D cell size for per-cell floor
  double subfloor_zbin_ = 0.10;         // [m] vertical bin for the floor histogram
  int    subfloor_min_bin_ = 8;         // min points in a z-bin to count as floor
  double subfloor_max_frac_ = 0.15;     // safety: skip if >this fraction would drop
  int    last_subfloor_rejected_ = 0;   // diagnostic: points dropped last scan

  double crop_size_;

  bool vf_use_;
  double vf_res_;

  bool imu_calibrate_;
  bool calibrate_gyro_;
  bool calibrate_accel_;
  bool gravity_align_;
  // Some IMUs (e.g. Livox built-in) report linear acceleration in units of g
  // rather than m/s^2; when true the accel is scaled by gravity on intake.
  bool imu_normalized_;
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
  double geo_degen_obs_gain_ = 1.0;  // LODESTAR-flavored observer gain on held-degenerate axes; 1 = off
  double geo_degen_vel_damp_ = 0.0;  // per-scan velocity damping along held axes (flywheel kill); 0 = off
  double geo_abias_max_;
  double geo_gbias_max_;
  // Intensity range correction
  double intensity_alpha_;
  double intensity_r_ref_;
  // Incidence-angle correction (organized scans only): divide by
  // max(|beam.normal|, cos_min). Off by default.
  bool intensity_incidence_;
  double intensity_cos_min_;
  // Photometric channel: false = intensity, true = reflectivity
  bool use_reflectivity_;
  // photometric term enabled (weight > 0); gates the intensity range correction
  bool photometric_active_;
  // One-time intensity<->reflectivity fallback resolution against the actual
  // cloud fields (set on the first scan; see getScanFromROS).
  bool channel_resolved_ = false;

  // --- Direct visual (camera) photometric term (off by default) ---
  // Frame-to-frame direct image alignment that constrains the LiDAR-degenerate
  // tunnel axis (see nano_gicp accumulateVisualResidual). All images are kept
  // undistorted, single-channel, normalized CV_32F.
  bool visual_enabled_;
  double visual_weight_;
  double visual_huber_delta_;
  double visual_max_dt_;                 // max |image_stamp - scan_stamp| [s]
  bool visual_dense_source_;             // feed the f2f term the dense deskewed cloud (not voxelised input_)
  int visual_dense_max_;                 // stride the dense f2f source down to <= this many points
  double last_visual_match_dt_ = -1.0;   // diagnostics: |dt| of the matched frame, -1 = none
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
  // Keyframe-image references for the map term (INTENSITY_AUDIT_2026-07-09):
  // per-point brightness sampled from each keyframe's FULL-RES reflectivity
  // image at creation (index-aligned with `keyframes`, /scale, < 0 invalid);
  // concatenated into submap_lidar_refs in buildSubmap and handed to the gicp
  // so the map term's reference carries pre-voxel texture.
  bool lidar_image_refs_enabled_ = false;
  std::vector<std::shared_ptr<const std::vector<float>>> keyframe_lidar_refs;
  std::shared_ptr<const std::vector<float>> submap_lidar_refs;
  std::shared_ptr<const std::vector<float>>
  sampleKeyframeLidarRefs(const pcl::PointCloud<PointType>::ConstPtr& cloud,
                          const Eigen::Isometry3f& T_lw, const cv::Mat& img);
  cv::Mat lidar_refl_img_;             // current scan reflectivity image (/scale), CV_32FC1
  cv::Mat lidar_range_img_;            // current scan range image [m], CV_32FC1 (occlusion check)
  float lidar_img_az_grad_energy_ = 0.f;  // mean |dI/dcol| over valid pairs (/scale units): full-res texture present?
  float lidar_az_a_, lidar_az_b_, lidar_el_a_, lidar_el_b_;  // self-calibrated spherical model
  std::vector<float> lidar_el_lut_;    // per-row mean elevation [rad] (non-uniform beams)
  double lidar_range_abs_tol_, lidar_range_rel_tol_;  // occlusion tolerance [m], fraction
  bool lidar_cond_scale_enabled_;      // direction-scale the lidar-map term along weak geom axes
  double lidar_cs_power_, lidar_cs_cap_;  // cond-scale exponent + per-direction boost cap
  bool lidar_dir_separated_enabled_ = false;  // restrict the lidar term to the weak subspace (LOFF)
  bool lidar_flow_enabled_ = false;    // frame-to-frame LiDAR flow term (EXPLORATION #2)
  double lidar_flow_weight_ = 0.0;     // flow term weight (count-normalized); 0 = off
  bool lidar_flow_image_ref_ = true;   // reference = current full-res image (true) vs voxel-averaged field (false)
  int lidar_flow_patch_ = 0;           // patch half-width [0,3]; 0 = single pixel (INTENSITY_AUDIT_2026-07-09)
  cv::Mat lidar_flow_prev_img_;        // previous scan's image, stashed for the flow term
  Eigen::Isometry3f lidar_flow_T_lw_prev_ = Eigen::Isometry3f::Identity();  // prev scan corrected world->lidar
  bool lidar_flow_prev_valid_ = false; // a previous image has been stashed
  double lidar_ds_ratio_ = 0.05;       // weak-subspace bar (fraction of lambda_max)
  bool genz_enabled_ = false;          // GenZ-ICP point-to-plane/point-to-point blend; off = bit-identical
  double genz_floor_ = 1.0;            // min point-to-plane weight alpha (1 = off)
  double genz_knee_ = 0.1;             // trans-block lambda_min/lambda_max at which blending starts
  double genz_point_weight_ = 1.0;     // isotropic point-to-point metric weight [1/m^2]
  bool xicp_ternary_enabled_ = false;  // X-ICP ternary localizability gate; off = existing gate
  double xicp_full_ratio_ = 0.05;      // upper (localizable) bar as a fraction of lambda_max
  double xicp_partial_budget_trans_ = 0.0;  // per-scan partial-band admission cap [m]; 0 = unbudgeted
  double xicp_partial_budget_rot_ = 0.0;    // per-scan partial-band admission cap [rad]; 0 = unbudgeted
  bool saliency_enabled_ = false;      // anti-dilution saliency weighting of the geometric term
  double saliency_boost_ = 1.0;        // weight of a maximally-salient source point (1 = off)
  std::string lidar_image_channel_;    // cloud field feeding the image slot: reflectivity|intensity|ambient
  double lidar_image_scale_;           // image full-scale normalization (per-channel; default 255)
  int lidar_image_denoise_kernel_;     // K x K spatial box-blur of the organized channel (<=1 = off; tames near-IR shot noise)
  bool lidar_proj_ready_;
  bool lidar_img_ready_;               // a usable image was built for this scan
  // Build the reflectivity image + spherical projection model from an organized
  // scan (called in getScanFromROS before NaN removal); no-op if not organized.
  void buildLidarIntensityImage(const pcl::PointCloud<PointType>::ConstPtr& organized,
                                int width, int height);

};
