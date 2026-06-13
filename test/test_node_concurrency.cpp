// Concurrent live-node integration / TSan harness.
//
// Constructs a real OdomNode with ALL optional paths enabled and pumps
// synthetic IMU + organized (Ouster-like) point clouds + camera frames through
// a MultiThreadedExecutor, so the IMU callback, the LiDAR scan callback, the
// image callback, the 100 Hz pose timer, the background submap thread, and the
// dashboard thread all run concurrently and touch the shared members.
//
// Under a normal build this is a functional smoke test (the node must process
// scans, create keyframes, and shut down cleanly -- exercising the worker-thread
// joins in ~OdomNode). Built with -DDLIIO_SANITIZE=thread (and OMP_NUM_THREADS=1)
// it is a data-race detector for the cross-thread state this sweep fixed
// (imu_rates, the keyframe vectors vs the background submap concat, the
// dashboard-thread stats).

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>

#include "dlio/odom.h"

using namespace std::chrono_literals;

namespace {

constexpr int kH = 64;     // beams (organized rows)
constexpr int kW = 512;    // azimuth columns
constexpr double kWallR = 4.0;  // tunnel radius [m]

// Organized cylinder ("tunnel") scan with reflectivity texture and Ouster-style
// per-column timestamps. center_x slowly ramps so the node perceives forward
// motion and keeps creating keyframes (exercising the background submap path).
sensor_msgs::msg::PointCloud2 makeScan(const rclcpp::Time& stamp, double center_x,
                                       double scan_dt) {
  pcl::PointCloud<dlio::Point> cloud;
  cloud.width = kW; cloud.height = kH; cloud.is_dense = false;
  cloud.points.resize(static_cast<size_t>(kW) * kH);
  const double el0 = -0.35, el1 = 0.35;
  for (int row = 0; row < kH; ++row) {
    const double el = el0 + (el1 - el0) * row / (kH - 1);
    for (int col = 0; col < kW; ++col) {
      const double az = -M_PI + 2.0 * M_PI * col / kW;
      const double r = kWallR;
      dlio::Point p;
      p.x = static_cast<float>(r * std::cos(az) + center_x);
      p.y = static_cast<float>(r * std::sin(az));
      p.z = static_cast<float>(r * std::tan(el));
      // texture so the photometric / lidar-image terms see a gradient
      float tex = 128.f + 90.f * std::sin(5.f * static_cast<float>(az))
                              * std::cos(3.f * static_cast<float>(el));
      p.intensity = tex;
      p.reflectivity = tex;
      p.t = static_cast<std::uint32_t>(scan_dt * 1e9 * col / kW);  // Ouster ns
      cloud.points[static_cast<size_t>(row) * kW + col] = p;
    }
  }
  sensor_msgs::msg::PointCloud2 msg;
  pcl::toROSMsg(cloud, msg);
  msg.header.stamp = stamp;
  msg.header.frame_id = "lidar";
  return msg;
}

sensor_msgs::msg::Imu makeImu(const rclcpp::Time& stamp) {
  sensor_msgs::msg::Imu m;
  m.header.stamp = stamp;
  m.header.frame_id = "imu";
  m.linear_acceleration.x = 0.0;
  m.linear_acceleration.y = 0.0;
  m.linear_acceleration.z = 9.80665;  // stationary: gravity only
  m.angular_velocity.x = 0.0;
  m.angular_velocity.y = 0.0;
  m.angular_velocity.z = 0.0;
  return m;
}

sensor_msgs::msg::Image makeImage(const rclcpp::Time& stamp) {
  sensor_msgs::msg::Image img;
  img.header.stamp = stamp;
  img.header.frame_id = "camera";
  img.height = 240; img.width = 320;
  img.encoding = "mono8"; img.step = img.width; img.is_bigendian = 0;
  img.data.resize(static_cast<size_t>(img.height) * img.width);
  for (uint32_t y = 0; y < img.height; ++y)
    for (uint32_t x = 0; x < img.width; ++x)
      img.data[y * img.width + x] =
          static_cast<uint8_t>(128 + 100 * std::sin(0.1 * x) * std::cos(0.1 * y));
  return img;
}

rclcpp::NodeOptions makeOptions() {
  rclcpp::NodeOptions o;
  o.append_parameter_override("use_sim_time", false);
  o.append_parameter_override("imu/calibration", true);
  o.append_parameter_override("odom/imu/calibration/time", 0.3);
  o.append_parameter_override("odom/imu/approximateGravity", false);
  o.append_parameter_override("pointcloud/deskew", true);
  // tiny keyframe thresholds -> frequent keyframes -> frequent background submap
  // rebuilds, maximizing the keyframe-vector concurrency surface
  o.append_parameter_override("odom/keyframe/threshD", 0.05);
  o.append_parameter_override("odom/keyframe/threshR", 5.0);
  o.append_parameter_override("odom/keyframe/maxKeyframes", 25);  // exercise pruning too
  o.append_parameter_override("odom/debug/dashboard", true);      // exercise the dashboard thread
  // all photometric/visual paths on
  o.append_parameter_override("odom/gicp/photometricWeight", 0.1);
  o.append_parameter_override("odom/gicp/photometricChannel", std::string("reflectivity"));
  o.append_parameter_override("odom/visual/enabled", true);
  o.append_parameter_override("odom/visual/weight", 0.05);
  o.append_parameter_override("odom/visual/map/enabled", true);
  o.append_parameter_override("odom/visual/map/weight", 0.05);
  o.append_parameter_override("odom/lidar_image/enabled", true);
  o.append_parameter_override("odom/lidar_image/weight", 0.05);
  o.append_parameter_override("camera/intrinsics",
      std::vector<double>{200.0, 200.0, 160.0, 120.0});
  return o;
}

}  // namespace

TEST(NodeConcurrency, ConcurrentCallbacksRunRaceFreeAndShutDownClean) {
  auto node = std::make_shared<dlio::OdomNode>(makeOptions());

  rclcpp::executors::MultiThreadedExecutor exec(rclcpp::ExecutorOptions(), 4);
  exec.add_node(node);
  std::atomic<bool> spinning{true};
  std::thread spin_thread([&] { exec.spin(); });

  // Publishers on the node's (un-remapped) input topics, matching the node's
  // sensor QoS so best-effort subscriptions accept them.
  auto pub_node = std::make_shared<rclcpp::Node>("dliio_test_pub");
  auto cloud_pub = pub_node->create_publisher<sensor_msgs::msg::PointCloud2>(
      "pointcloud", rclcpp::SensorDataQoS());
  auto imu_pub = pub_node->create_publisher<sensor_msgs::msg::Imu>(
      "imu", rclcpp::SensorDataQoS());
  auto img_pub = pub_node->create_publisher<sensor_msgs::msg::Image>(
      "camera", rclcpp::SensorDataQoS());

  rclcpp::Clock clock(RCL_SYSTEM_TIME);
  const double scan_dt = 0.1;     // 10 Hz scans
  const double imu_dt = 0.004;    // 250 Hz IMU
  const int n_scans = 30;         // ~3 s

  // Prime the IMU buffer so the first scans have coverage.
  rclcpp::Time t = clock.now();
  for (int i = 0; i < 100; ++i) {
    imu_pub->publish(makeImu(t));
    t = t + rclcpp::Duration::from_seconds(imu_dt);
    std::this_thread::sleep_for(1ms);
  }

  int img_div = 0;
  for (int s = 0; s < n_scans && rclcpp::ok(); ++s) {
    // a burst of IMU spanning this scan period (keeps the buffer ahead of scans)
    for (int k = 0; k < static_cast<int>(scan_dt / imu_dt); ++k) {
      imu_pub->publish(makeImu(t));
      t = t + rclcpp::Duration::from_seconds(imu_dt);
    }
    const rclcpp::Time scan_stamp = t - rclcpp::Duration::from_seconds(scan_dt);
    cloud_pub->publish(makeScan(scan_stamp, 0.02 * s, scan_dt));  // ramping motion
    if (img_div++ % 2 == 0) img_pub->publish(makeImage(scan_stamp));
    // Retune live params mid-run (exercises the on_set callback on the executor
    // thread vs applyLiveParams on the scan thread -- a TSan/ASan target).
    if (s % 5 == 4) {
      node->set_parameters_atomically({
          rclcpp::Parameter("odom/gicp/photometricWeight", 0.05 + 0.01 * s),
          rclcpp::Parameter("odom/geo/Kp", 4.0 + 0.1 * s),
          rclcpp::Parameter("odom/gicp/degeneracyThreshRatio", 0.004 + 0.0001 * s)});
    }
    std::this_thread::sleep_for(std::chrono::duration<double>(scan_dt));
  }

  // Clean shutdown: cancel the executor, join the spin thread, then destroy the
  // node -- ~OdomNode joins its worker threads and the background submap future.
  exec.cancel();
  spinning = false;
  spin_thread.join();
  exec.remove_node(node);

  // No estimator output is asserted here -- this is SUCCEED(): the point is that
  // the concurrent pipeline ran to completion without a sanitizer abort (and, in
  // a normal build, that the worker-thread joins in ~OdomNode are clean).
  SUCCEED();
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  rclcpp::init(argc, argv);
  int rc = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return rc;
}
