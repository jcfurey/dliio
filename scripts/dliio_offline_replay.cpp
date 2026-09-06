// Estimator-only replay with fixed input order for Ouster or Exyn caches.
// Compile against the exact branch under test; do not mix OdomNode layouts.
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <rcl/time.h>
#include <rclcpp/serialization.hpp>
#include "dlio/odom.h"

namespace dlio {
struct OdomNodeTestAccess {
  static void imu(OdomNode& n, const sensor_msgs::msg::Imu::SharedPtr& m) { n.callbackImu(m); }
  static void cloud(OdomNode& n, const sensor_msgs::msg::PointCloud2::SharedPtr& m) { n.callbackPointCloud(m); }
  static void finish(OdomNode& n) {
    if (n.submap_future.valid()) { n.submap_future.wait(); }
    if (n.publish_thread.joinable()) { n.publish_thread.join(); }
  }
  static bool diverged(const OdomNode& n) {
    return !n.T.allFinite() || n.T.block<3, 1>(0, 3).norm() > 1000.f;
  }
  static void sample(OdomNode& n, std::ostream& out, double stamp, double ms, double wait_ms) {
    // Diagnose the first gross registration correction without paying for a
    // second target kd-tree on ordinary scans. Count actual spatial overlap
    // before and after the correction; disappearing matches are not a fit.
    static bool reported_jump = false;
    if (!reported_jump && (n.T.block<3, 1>(0, 3) - n.T_prior.block<3, 1>(0, 3)).norm() > 5.f) {
      const auto target = n.gicp.getInputTarget();
      if (target && !target->empty()) {
        nanoflann::KdTreeFLANN<PointType> tree(false);
        tree.setInputCloud(target);
        int before = 0, after = 0;
        std::vector<int> ids(1);
        std::vector<float> distances(1);
        const float max_sq = n.gicp_max_corr_dist_ * n.gicp_max_corr_dist_;
        for (const auto& point : *n.current_scan) {
          if (tree.nearestKSearch(point, 1, ids, distances) == 1 && distances[0] < max_sq) { ++before; }
          PointType corrected = point;
          corrected.getVector3fMap() = (n.T_corr * point.getVector4fMap()).head<3>();
          if (tree.nearestKSearch(corrected, 1, ids, distances) == 1 && distances[0] < max_sq) { ++after; }
        }
        std::cerr << "gross correction at scan " << std::setprecision(17) << n.scan_stamp
                  << ": overlap " << before << " -> " << after << " / " << n.current_scan->size()
                  << ", converged=" << n.gicp.hasConverged() << '\n';
        reported_jump = true;
      }
    }
    const Eigen::Quaternionf q(n.T.block<3, 3>(0, 0));
    out << stamp << ',' << n.T(0, 3) << ',' << n.T(1, 3) << ',' << n.T(2, 3)
        << ',' << q.x() << ',' << q.y() << ',' << q.z() << ',' << q.w()
        << ',' << n.state.p.x() << ',' << n.state.p.y() << ',' << n.state.p.z()
        << ',' << n.state.v.lin.w.x() << ',' << n.state.v.lin.w.y() << ',' << n.state.v.lin.w.z()
        << ',' << ms << ',' << wait_ms << ',' << n.current_scan->size()
        << ',' << n.gicp.lastLidarFlowCount() << ',' << n.gicp.lastLidarFlowRms()
        << ',' << n.gicp.lastLidarMapCount() << ',' << n.gicp.lastLidarMapRms()
        << ',' << n.gicp.lastVisualRescuedDirections() << ',' << n.gicp.lastDegenerateDirections()
        << ',' << n.gicp.lastPhotometricCount() << ',' << n.gicp.lastPhotometricRms()
        << ',' << n.deskew_status << ',' << n.keyframes.size()
        << ',' << n.scan_stamp << ',' << n.observer_time_aligned_
        << ',' << n.observer_lag_seconds_ << ',' << n.gicp.hasConverged()
        << ',' << n.state.b.accel.x() << ',' << n.state.b.accel.y() << ',' << n.state.b.accel.z()
        << ',' << n.gicp.lastGeoRotMargin() << ',' << n.gicp.lastGeoTransMargin()
        << ',' << n.gicp.lastDegenTransDirs().size()
        << ',' << n.gicp.lastSurfaceTextureWeakRatio()
        << ',' << n.gicp.lastSurfaceTextureMatch().valid
        << ',' << n.gicp.lastSurfaceTextureMatch().candidates
        << ',' << n.gicp.lastSurfaceTextureMatch().supported
        << ',' << n.gicp.lastSurfaceTextureMatch().unique
        << ',' << n.gicp.lastSurfaceTextureMatch().inliers
        << ',' << n.gicp.lastSurfaceTextureMatch().shift
        << ',' << n.gicp.lastSurfaceTextureMatch().sigma
        << ',' << n.gicp.lastSurfaceTextureMatch().correlation;
    const auto& texture = n.gicp.lastSurfaceTextureMatch();
    const auto& rejected = texture.rejected;
    out << ',' << static_cast<int>(texture.status)
        << ',' << rejected.examined << ',' << rejected.nonfinite << ',' << rejected.spacing
        << ',' << rejected.neighborhood << ',' << rejected.nonplanar << ',' << rejected.axis_normal
        << ',' << rejected.reference_support << ',' << rejected.reference_contrast
        << ',' << rejected.repeated << ',' << rejected.source_support << ',' << rejected.source_contrast
        << ',' << rejected.boundary << ',' << rejected.low_correlation << ',' << rejected.ambiguous
        << ',' << rejected.flat_peak;
    const auto& geometry = n.gicp.lastGeometryDiagnostics();
    out << ',' << geometry.valid << ',' << geometry.length_scale;
    for (int i = 0; i < 6; ++i) { out << ',' << geometry.eigenvalues(i); }
    for (int i = 0; i < 6; ++i) { out << ',' << geometry.weakest(i); }
    for (int i = 0; i < 6; ++i) { out << ',' << geometry.correction(i); }
    out << ',' << geometry.rotation_ratio << ',' << geometry.translation_ratio
        << ',' << geometry.schur_ratio << ',' << geometry.schur_retained
        << ',' << geometry.half_length_ratio << ',' << geometry.double_length_ratio
        << ',' << geometry.correction_projection << ',' << geometry.texture_projection;
    const Eigen::Quaternionf prior_q(n.T_prior.block<3, 3>(0, 0));
    out << ',' << n.T_prior(0, 3) << ',' << n.T_prior(1, 3) << ',' << n.T_prior(2, 3)
        << ',' << prior_q.x() << ',' << prior_q.y() << ',' << prior_q.z() << ',' << prior_q.w()
        << ',' << n.state.b.gyro.x() << ',' << n.state.b.gyro.y() << ',' << n.state.b.gyro.z();
    for (int i = 0; i < 6; ++i) {
      for (int j = i; j < 6; ++j) { out << ',' << geometry.hessian(i, j); }
    }
    out << '\n';
  }
};
}

struct ImuRecord { uint64_t stamp; double a[3], w[3]; };
static_assert(sizeof(ImuRecord) == 56);
int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: replay CACHE OUTPUT.csv SECONDS(0=all) --ros-args --params-file PARAMS\n";
    return 2;
  }
  const std::string cache = argv[1];
  const double seconds = std::stod(argv[3]);
  // clouds.cdr retains the adapted PointCloud2 schema, organization and signed
  // relative time offsets. Each record is a little-endian uint64 header stamp,
  // uint32 CDR length, then that many serialized bytes. Unlike an Ouster frame,
  // an Exyn sweep is unorganized and its header is at the END of acquisition.
  std::ifstream cdr_frames(cache + "/clouds.cdr", std::ios::binary);
  const bool use_cdr = cdr_frames.is_open();
  std::ifstream frames;
  if (use_cdr) { frames = std::move(cdr_frames); }
  else { frames.open(cache + "/frames.bin", std::ios::binary); }
  std::ifstream imu_file(cache + "/imu.bin", std::ios::binary);
  if (!frames || !imu_file) { throw std::runtime_error("cannot open cache"); }
  std::vector<ImuRecord> imus;
  ImuRecord rec;
  while (imu_file.read(reinterpret_cast<char*>(&rec), sizeof(rec))) { imus.push_back(rec); }
  std::ofstream out(argv[2]);
  if (!out) { throw std::runtime_error("cannot open output"); }
  // Preserve sub-millisecond differences even with Unix-epoch timestamps.
  out << std::setprecision(17);
  out << "stamp,x,y,z,qx,qy,qz,qw,state_x,state_y,state_z,vx,vy,vz,compute_ms,wait_ms,points,flow_count,flow_rms,map_count,map_rms,rescued,degenerate,photo_count,photo_rms,deskew,keyframes,scan_stamp,aligned,imu_lead,converged,bax,bay,baz,geo_rot,geo_trans,held_trans,texture_weak_ratio,texture_valid,texture_candidates,texture_supported,texture_unique,texture_inliers,texture_shift,texture_sigma,texture_correlation";
  out << ",texture_status,texture_examined,texture_rej_nonfinite,texture_rej_spacing"
         ",texture_rej_neighborhood,texture_rej_nonplanar,texture_rej_axis_normal"
         ",texture_rej_reference_support,texture_rej_reference_contrast,texture_rej_repeated"
         ",texture_rej_source_support,texture_rej_source_contrast,texture_rej_boundary"
         ",texture_rej_low_correlation,texture_rej_ambiguous,texture_rej_flat_peak"
         ",geometry_valid,geometry_length";
  for (int i = 0; i < 6; ++i) { out << ",geometry_eig_" << i; }
  for (int i = 0; i < 6; ++i) { out << ",geometry_mode_" << i; }
  for (int i = 0; i < 6; ++i) { out << ",geometry_correction_" << i; }
  out << ",geometry_rot_ratio,geometry_trans_ratio,geometry_schur_ratio,geometry_schur_retained"
         ",geometry_half_length_ratio,geometry_double_length_ratio"
         ",geometry_correction_projection,geometry_texture_projection"
         ",prior_x,prior_y,prior_z,prior_qx,prior_qy,prior_qz,prior_qw,bgx,bgy,bgz";
  for (int i = 0; i < 6; ++i) {
    for (int j = i; j < 6; ++j) { out << ",geometry_h_" << i << '_' << j; }
  }
  out << '\n';
  rclcpp::init(argc, argv);
  auto node = std::make_shared<dlio::OdomNode>();
  auto clock = node->get_clock()->get_clock_handle();
  if (rcl_enable_ros_time_override(clock) != RCL_RET_OK) { throw std::runtime_error("sim clock"); }
  auto set_clock = [&](uint64_t t) {
    if (rcl_set_ros_time_override(clock, t) != RCL_RET_OK) { throw std::runtime_error("set sim clock"); }
  };
  auto msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
  msg->height = 64; msg->width = 1024; msg->point_step = 32; msg->row_step = 1024 * 32;
  msg->is_dense = false; msg->is_bigendian = false; msg->header.frame_id = "os_lidar";
  auto field = [&](const std::string& name, uint32_t offset, uint8_t type) {
    sensor_msgs::msg::PointField f; f.name = name; f.offset = offset; f.datatype = type; f.count = 1;
    msg->fields.push_back(f);
  };
  field("x", 0, 7); field("y", 4, 7); field("z", 8, 7); field("intensity", 12, 7);
  field("t", 16, 6); field("reflectivity", 20, 4); field("ring", 22, 4);
  field("ambient", 24, 4); field("range", 28, 6);
  msg->data.resize(msg->row_step * msg->height);
  rclcpp::Serialization<sensor_msgs::msg::PointCloud2> serialization;
  size_t next_imu = 0, scans = 0;
  bool diverged = false;
  uint64_t stamp, rx, first = 0;
  const auto start = std::chrono::steady_clock::now();
  while (rclcpp::ok()) {
    if (!frames.read(reinterpret_cast<char*>(&stamp), 8)) {
      if (frames.gcount() != 0 || !frames.eof()) { throw std::runtime_error("partial cache stamp"); }
      break;
    }
    if (!first) { first = stamp; }
    if (seconds > 0 && (stamp - first) * 1e-9 > seconds) { break; }
    if (use_cdr) {
      uint32_t size = 0;
      if (!frames.read(reinterpret_cast<char*>(&size), 4) || size == 0 || size > 64 * 1024 * 1024) {
        throw std::runtime_error("invalid CDR cache length");
      }
      rclcpp::SerializedMessage serialized(size);
      auto& raw = serialized.get_rcl_serialized_message();
      raw.buffer_length = size;
      if (!frames.read(reinterpret_cast<char*>(raw.buffer), size)) { throw std::runtime_error("partial CDR record"); }
      serialization.deserialize_message(&serialized, msg.get());
      if (static_cast<uint64_t>(rclcpp::Time(msg->header.stamp).nanoseconds()) != stamp) {
        throw std::runtime_error("CDR header disagrees with cache stamp");
      }
    } else {
      if (!frames.read(reinterpret_cast<char*>(&rx), 8) ||
          !frames.read(reinterpret_cast<char*>(msg->data.data()), msg->data.size())) {
        throw std::runtime_error("partial cache record");
      }
    }
    // One sample beyond the end of the sweep supplies the interpolation bracket.
    while (next_imu < imus.size()) {
      const auto& v = imus[next_imu++];
      auto m = std::make_shared<sensor_msgs::msg::Imu>();
      m->header.stamp = rclcpp::Time(v.stamp, RCL_ROS_TIME);
      m->header.frame_id = use_cdr ? "imu" : "os_imu";
      m->linear_acceleration.x = v.a[0]; m->linear_acceleration.y = v.a[1]; m->linear_acceleration.z = v.a[2];
      m->angular_velocity.x = v.w[0]; m->angular_velocity.y = v.w[1]; m->angular_velocity.z = v.w[2];
      set_clock(v.stamp);
      dlio::OdomNodeTestAccess::imu(*node, m);
      if (v.stamp >= stamp + (use_cdr ? 0 : 100000000)) { break; }
    }
    msg->header.stamp = rclcpp::Time(stamp, RCL_ROS_TIME);
    const auto t0 = std::chrono::steady_clock::now();
    dlio::OdomNodeTestAccess::cloud(*node, msg);
    const auto t1 = std::chrono::steady_clock::now();
    // Consistent submap availability in every repetition. This measures the
    // estimator's math, not executor races or transport-induced scan loss.
    dlio::OdomNodeTestAccess::finish(*node);
    const auto t2 = std::chrono::steady_clock::now();
    dlio::OdomNodeTestAccess::sample(*node, out, stamp * 1e-9,
        std::chrono::duration<double, std::milli>(t1 - t0).count(),
        std::chrono::duration<double, std::milli>(t2 - t1).count());
    if (++scans % 500 == 0) { std::cout << "processed " << scans << " scans\n" << std::flush; out.flush(); }
    if (dlio::OdomNodeTestAccess::diverged(*node)) {
      std::cerr << "DIVERGED: nonfinite pose or >1 km from the start of the short tunnel recording\n";
      diverged = true;
      break;
    }
  }
  std::cout << "Processed " << scans << " scans in "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s\n";
  node.reset();
  rclcpp::shutdown();
  return diverged ? 3 : 0;
}
