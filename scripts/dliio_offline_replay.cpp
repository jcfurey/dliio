// Estimator-only replay with fixed input order for dliio_ouster_cache.py output.
// Compile against the exact branch under test; do not mix OdomNode layouts.
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <rcl/time.h>
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
        << ',' << n.deskew_status << ',' << n.keyframes.size() << '\n';
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
  std::ifstream frames(cache + "/frames.bin", std::ios::binary), imu_file(cache + "/imu.bin", std::ios::binary);
  if (!frames || !imu_file) { throw std::runtime_error("cannot open cache"); }
  std::vector<ImuRecord> imus;
  ImuRecord rec;
  while (imu_file.read(reinterpret_cast<char*>(&rec), sizeof(rec))) { imus.push_back(rec); }
  std::ofstream out(argv[2]);
  if (!out) { throw std::runtime_error("cannot open output"); }
  out << std::setprecision(12);
  out << "stamp,x,y,z,qx,qy,qz,qw,state_x,state_y,state_z,vx,vy,vz,compute_ms,wait_ms,points,flow_count,flow_rms,map_count,map_rms,rescued,degenerate,photo_count,photo_rms,deskew,keyframes\n";
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
  size_t next_imu = 0, scans = 0;
  bool diverged = false;
  uint64_t stamp, rx, first = 0;
  const auto start = std::chrono::steady_clock::now();
  while (rclcpp::ok() && frames.read(reinterpret_cast<char*>(&stamp), 8) && frames.read(reinterpret_cast<char*>(&rx), 8)) {
    if (!first) { first = stamp; }
    if (seconds > 0 && (stamp - first) * 1e-9 > seconds) { break; }
    if (!frames.read(reinterpret_cast<char*>(msg->data.data()), msg->data.size())) { throw std::runtime_error("partial cache record"); }
    // One sample beyond the end of the sweep supplies the interpolation bracket.
    while (next_imu < imus.size()) {
      const auto& v = imus[next_imu++];
      auto m = std::make_shared<sensor_msgs::msg::Imu>();
      m->header.stamp = rclcpp::Time(v.stamp, RCL_ROS_TIME); m->header.frame_id = "os_imu";
      m->linear_acceleration.x = v.a[0]; m->linear_acceleration.y = v.a[1]; m->linear_acceleration.z = v.a[2];
      m->angular_velocity.x = v.w[0]; m->angular_velocity.y = v.w[1]; m->angular_velocity.z = v.w[2];
      set_clock(v.stamp);
      dlio::OdomNodeTestAccess::imu(*node, m);
      if (v.stamp >= stamp + 100000000) { break; }
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
