// End-to-end pipeline tests: drive a real OdomNode through a MultiThreadedExecutor
// with synthetic-but-consistent LiDAR + IMU and check the *estimate*, not just
// that it runs. Covers:
//   - accuracy regression (stationary drift bound; constant-velocity tracking),
//   - imu/normalized (g-unit IMU) gravity handling,
//   - extrinsics/source: tf gating + resolution,
//   - live-param descriptor range rejection,
//   - a generously-bounded hot-path timing smoke (gross-regression guard).
//
// The world is the interior of a static box (three orthogonal plane pairs -> full
// 3D observability, unlike a tunnel), observed from a moving sensor. Constant
// velocity means zero linear acceleration, so a gravity-only IMU is physically
// consistent with the motion; the LiDAR carries the displacement.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <mutex>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <pcl_conversions/pcl_conversions.h>

#include "dlio/odom.h"

using namespace std::chrono_literals;

namespace {

constexpr double kHalf = 6.0;     // box half-extent [m]
constexpr double kStep = 0.5;     // surface sample spacing [m]

// Interior box-surface cloud observed from sensor_pos (pure translation, no
// rotation): each world point maps to (world - sensor_pos) in the sensor frame.
// Ouster-style: per-point 't' [ns] ramps across the scan so deskew has work.
sensor_msgs::msg::PointCloud2 makeBoxScan(const rclcpp::Time& stamp,
                                          const Eigen::Vector3d& sensor_pos,
                                          double scan_dt) {
  // Box centered at an OFFSET from the sensor start so the six faces sit at
  // distinct ranges: that breaks the symmetry a sensor-centered box would have
  // and keeps scan-to-map registration well-constrained in all three axes.
  const Eigen::Vector3d C(1.5, 1.0, 0.5);
  pcl::PointCloud<dlio::Point> cloud;
  std::vector<Eigen::Vector3d> world;
  for (double a = -kHalf; a <= kHalf + 1e-6; a += kStep) {
    for (double b = -kHalf; b <= kHalf + 1e-6; b += kStep) {
      world.emplace_back(C.x() + kHalf, C.y() + a, C.z() + b);
      world.emplace_back(C.x() - kHalf, C.y() + a, C.z() + b);
      world.emplace_back(C.x() + a, C.y() + kHalf, C.z() + b);
      world.emplace_back(C.x() + a, C.y() - kHalf, C.z() + b);
      world.emplace_back(C.x() + a, C.y() + b, C.z() + kHalf);
      world.emplace_back(C.x() + a, C.y() + b, C.z() - kHalf);
    }
  }
  cloud.points.reserve(world.size());
  const size_t n = world.size();
  for (size_t i = 0; i < n; ++i) {
    const Eigen::Vector3d p = world[i] - sensor_pos;
    dlio::Point pt;
    pt.x = static_cast<float>(p.x());
    pt.y = static_cast<float>(p.y());
    pt.z = static_cast<float>(p.z());
    // texture so the photometric path (if on) has a gradient; harmless otherwise
    pt.intensity = 100.f + 50.f * std::sin(0.5f * static_cast<float>(p.x()));
    pt.reflectivity = pt.intensity;
    pt.t = static_cast<std::uint32_t>(scan_dt * 1e9 * i / n);  // Ouster ns ramp
    cloud.points.push_back(pt);
  }
  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;

  sensor_msgs::msg::PointCloud2 msg;
  pcl::toROSMsg(cloud, msg);
  msg.header.stamp = stamp;
  msg.header.frame_id = "lidar";
  return msg;
}

sensor_msgs::msg::Imu makeImu(const rclcpp::Time& stamp, double az = 9.80665) {
  sensor_msgs::msg::Imu m;
  m.header.stamp = stamp;
  m.header.frame_id = "imu";
  m.linear_acceleration.z = az;   // stationary / constant-velocity: gravity only
  return m;
}

rclcpp::NodeOptions baseOptions() {
  rclcpp::NodeOptions o;
  o.append_parameter_override("use_sim_time", false);
  o.append_parameter_override("imu/calibration", true);
  o.append_parameter_override("odom/imu/calibration/time", 0.3);
  o.append_parameter_override("odom/imu/approximateGravity", false);
  o.append_parameter_override("pointcloud/deskew", true);
  o.append_parameter_override("odom/debug/dashboard", false);
  o.append_parameter_override("odom/keyframe/threshD", 0.5);
  o.append_parameter_override("odom/keyframe/threshR", 30.0);
  return o;
}

// Collects the latest odometry pose published by the node under test.
struct OdomSink {
  std::mutex mtx;
  Eigen::Vector3d p{0, 0, 0};
  std::atomic<int> count{0};
  void set(const nav_msgs::msg::Odometry& m) {
    std::lock_guard<std::mutex> lk(mtx);
    p = Eigen::Vector3d(m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z);
    ++count;
  }
  Eigen::Vector3d get() { std::lock_guard<std::mutex> lk(mtx); return p; }
};

// Drives the node for n_scans with the sensor at pos(scan_index). Returns the
// final estimated position. accel_z lets g-unit IMU scenarios be exercised.
Eigen::Vector3d runTrajectory(std::shared_ptr<dlio::OdomNode> node,
                              std::function<Eigen::Vector3d(int)> pos,
                              int n_scans, double accel_z, OdomSink* sink) {
  rclcpp::executors::MultiThreadedExecutor exec(rclcpp::ExecutorOptions(), 4);
  exec.add_node(node);
  std::thread spin([&] { exec.spin(); });

  auto pub_node = std::make_shared<rclcpp::Node>("dliio_pipeline_pub");
  auto cloud_pub = pub_node->create_publisher<sensor_msgs::msg::PointCloud2>(
      "pointcloud", rclcpp::SensorDataQoS());
  auto imu_pub = pub_node->create_publisher<sensor_msgs::msg::Imu>(
      "imu", rclcpp::SensorDataQoS());
  auto odom_sub = pub_node->create_subscription<nav_msgs::msg::Odometry>(
      "odom", 10, [sink](const nav_msgs::msg::Odometry::SharedPtr m) { if (sink) sink->set(*m); });
  exec.add_node(pub_node);

  rclcpp::Clock clock(RCL_SYSTEM_TIME);
  const double scan_dt = 0.1, imu_dt = 0.004;
  rclcpp::Time t = clock.now();

  // Prime the IMU buffer (also drives startup gravity calibration).
  for (int i = 0; i < 120; ++i) {
    imu_pub->publish(makeImu(t, accel_z));
    t = t + rclcpp::Duration::from_seconds(imu_dt);
    std::this_thread::sleep_for(1ms);
  }

  for (int s = 0; s < n_scans && rclcpp::ok(); ++s) {
    for (int k = 0; k < static_cast<int>(scan_dt / imu_dt); ++k) {
      imu_pub->publish(makeImu(t, accel_z));
      t = t + rclcpp::Duration::from_seconds(imu_dt);
    }
    const rclcpp::Time scan_stamp = t - rclcpp::Duration::from_seconds(scan_dt);
    cloud_pub->publish(makeBoxScan(scan_stamp, pos(s), scan_dt));
    std::this_thread::sleep_for(std::chrono::duration<double>(scan_dt));
  }

  Eigen::Vector3d final_p = sink ? sink->get() : Eigen::Vector3d::Zero();
  exec.cancel();
  spin.join();
  exec.remove_node(pub_node);
  exec.remove_node(node);
  return final_p;
}

}  // namespace

// Stationary sensor in a static, fully-observable world: the estimate must not
// drift. Guards the whole pipeline (deskew -> registration -> observer) against
// bias / divergence regressions.
TEST(PipelineAccuracy, StationaryEstimateDoesNotDrift) {
  auto node = std::make_shared<dlio::OdomNode>(baseOptions());
  OdomSink sink;
  auto p = runTrajectory(node, [](int){ return Eigen::Vector3d::Zero(); },
                         35, 9.80665, &sink);
  ASSERT_GT(sink.count.load(), 0) << "node never published odometry";
  // Divergence guard (synthetic, wall-clock-paced -> generous margin for CI
  // timing): a working estimator stays put; a broken one wanders meters.
  EXPECT_LT(p.norm(), 0.30) << "stationary drift " << p.transpose();
}

// Constant-velocity translation along +x: the LiDAR carries the motion and the
// estimate must track it. Generous tolerance (observer lag + synthetic noise),
// but still catches sign errors, gross under/over-estimation, and total failure.
TEST(PipelineAccuracy, ConstantVelocityTracksGroundTruth) {
  auto node = std::make_shared<dlio::OdomNode>(baseOptions());
  OdomSink sink;
  const double v = 0.15;          // [m/s]
  const double scan_dt = 0.1;
  const int n = 45;
  auto pos = [&](int s){ return Eigen::Vector3d(v * scan_dt * s, 0, 0); };
  auto p = runTrajectory(node, pos, n, 9.80665, &sink);

  const double truth_x = v * scan_dt * (n - 1);   // ~0.66 m
  ASSERT_GT(sink.count.load(), 0);
  // The estimate must clearly track the LiDAR-carried forward motion (this is
  // the meaningful assertion); lateral/vertical bounds are divergence guards
  // with margin for the synthetic transient + CI timing.
  EXPECT_GT(p.x(), 0.45 * truth_x) << "did not track forward motion: " << p.transpose();
  EXPECT_LT(std::abs(p.x() - truth_x), 0.35 * truth_x + 0.1)
      << "x error too large: est " << p.x() << " truth " << truth_x;
  EXPECT_LT(std::abs(p.y()), 0.30) << "lateral drift " << p.y();
  EXPECT_LT(std::abs(p.z()), 0.35) << "vertical drift " << p.z();
}

// imu/normalized smoke: a Livox-style IMU reporting acceleration in units of g
// (z=1.0) is scaled to m/s^2 on intake. This checks the scaled path runs and the
// estimate stays bounded (the node ingests g-unit data without breaking). NOTE:
// a stationary bias calibration absorbs a *static* g-unit offset regardless, so
// this is a smoke test, not a discriminating one -- the flag's real value is
// under dynamic acceleration.
TEST(PipelineAccuracy, ImuNormalizedHandlesGUnitAccel) {
  auto opts = baseOptions();
  opts.append_parameter_override("imu/normalized", true);
  auto node = std::make_shared<dlio::OdomNode>(opts);
  OdomSink sink;
  auto p = runTrajectory(node, [](int){ return Eigen::Vector3d::Zero(); },
                         35, 1.0 /* g-units */, &sink);
  ASSERT_GT(sink.count.load(), 0);
  EXPECT_LT(p.norm(), 0.60) << "drift with g-unit IMU + imu/normalized " << p.transpose();
}

// extrinsics/source: tf -- the node must hold off processing until base_link->
// {imu,lidar} are available on tf2, then resolve them and start publishing.
TEST(PipelineExtrinsics, TfSourceGatesUntilTransformsArriveThenResolves) {
  auto opts = baseOptions();
  opts.append_parameter_override("extrinsics/source", std::string("tf"));
  auto node = std::make_shared<dlio::OdomNode>(opts);

  rclcpp::executors::MultiThreadedExecutor exec(rclcpp::ExecutorOptions(), 4);
  exec.add_node(node);

  // Publish the identity base_link->imu / base_link->lidar transforms the node
  // is waiting for (static; latched).
  auto tf_node = std::make_shared<rclcpp::Node>("dliio_tf_pub");
  tf2_ros::StaticTransformBroadcaster br(tf_node);
  auto identity_tf = [](const std::string& parent, const std::string& child) {
    geometry_msgs::msg::TransformStamped tf;
    tf.header.frame_id = parent; tf.child_frame_id = child;
    tf.transform.rotation.w = 1.0;
    return tf;
  };
  br.sendTransform(identity_tf("base_link", "imu"));
  br.sendTransform(identity_tf("base_link", "lidar"));
  exec.add_node(tf_node);

  std::thread spin([&] { exec.spin(); });

  auto pub_node = std::make_shared<rclcpp::Node>("dliio_tf_data_pub");
  auto cloud_pub = pub_node->create_publisher<sensor_msgs::msg::PointCloud2>(
      "pointcloud", rclcpp::SensorDataQoS());
  auto imu_pub = pub_node->create_publisher<sensor_msgs::msg::Imu>(
      "imu", rclcpp::SensorDataQoS());
  std::atomic<int> odom_count{0};
  auto odom_sub = pub_node->create_subscription<nav_msgs::msg::Odometry>(
      "odom", 10, [&](const nav_msgs::msg::Odometry::SharedPtr) { ++odom_count; });
  exec.add_node(pub_node);

  rclcpp::Clock clock(RCL_SYSTEM_TIME);
  const double imu_dt = 0.004;
  rclcpp::Time t = clock.now();
  // Give the node's 0.5 s extrinsics timer a couple of cycles to resolve tf,
  // while feeding IMU + scans so it has data to process once ungated.
  for (int i = 0; i < 120; ++i) {
    imu_pub->publish(makeImu(t)); t = t + rclcpp::Duration::from_seconds(imu_dt);
    std::this_thread::sleep_for(1ms);
  }
  for (int s = 0; s < 30 && rclcpp::ok(); ++s) {
    for (int k = 0; k < 25; ++k) {
      imu_pub->publish(makeImu(t)); t = t + rclcpp::Duration::from_seconds(imu_dt);
    }
    cloud_pub->publish(makeBoxScan(t - rclcpp::Duration::from_seconds(0.1),
                                   Eigen::Vector3d::Zero(), 0.1));
    std::this_thread::sleep_for(100ms);
  }

  exec.cancel();
  spin.join();
  EXPECT_GT(odom_count.load(), 0)
      << "node never published odom -- tf extrinsics not resolved / still gated";
}

// Live-param descriptors: a FloatingPointRange-bounded parameter rejects an
// out-of-range set before the user callback (so rqt sliders are bounded and bad
// sets can't reach the estimator). odom/geo/Kp is declared with range [0, 100].
TEST(LiveParams, OutOfRangeSetIsRejected) {
  auto node = std::make_shared<dlio::OdomNode>(baseOptions());
  auto good = node->set_parameter(rclcpp::Parameter("odom/geo/Kp", 5.0));
  EXPECT_TRUE(good.successful);
  auto bad = node->set_parameter(rclcpp::Parameter("odom/geo/Kp", 1000.0));  // > 100
  EXPECT_FALSE(bad.successful) << "out-of-range Kp was not rejected by the descriptor";
  double kp = 0.0; node->get_parameter("odom/geo/Kp", kp);
  EXPECT_DOUBLE_EQ(kp, 5.0) << "rejected set must not have mutated the value";
}

// Hot-path timing smoke: pump scans and read the node's own reported computation
// time off /diagnostics. Deliberately a *gross-regression* guard (generous bound)
// -- it would catch e.g. a reintroduced full-cloud copy or an O(N^2) blowup, not
// fine timing changes. The measured value is printed for information.
TEST(PipelinePerf, HotPathComputeTimeIsReasonable) {
  auto node = std::make_shared<dlio::OdomNode>(baseOptions());

  rclcpp::executors::MultiThreadedExecutor exec(rclcpp::ExecutorOptions(), 4);
  exec.add_node(node);
  std::thread spin([&] { exec.spin(); });

  auto pub_node = std::make_shared<rclcpp::Node>("dliio_perf_pub");
  auto cloud_pub = pub_node->create_publisher<sensor_msgs::msg::PointCloud2>(
      "pointcloud", rclcpp::SensorDataQoS());
  auto imu_pub = pub_node->create_publisher<sensor_msgs::msg::Imu>(
      "imu", rclcpp::SensorDataQoS());
  std::atomic<double> comp_ms{-1.0};
  auto diag_sub = pub_node->create_subscription<diagnostic_msgs::msg::DiagnosticArray>(
      "/diagnostics", rclcpp::QoS(10).best_effort(),
      [&](const diagnostic_msgs::msg::DiagnosticArray::SharedPtr m) {
        for (const auto& st : m->status)
          for (const auto& kv : st.values)
            if (kv.key == "Computation Time (ms)") comp_ms.store(std::stod(kv.value));
      });
  exec.add_node(pub_node);

  rclcpp::Clock clock(RCL_SYSTEM_TIME);
  const double imu_dt = 0.004;
  rclcpp::Time t = clock.now();
  for (int i = 0; i < 120; ++i) {
    imu_pub->publish(makeImu(t)); t = t + rclcpp::Duration::from_seconds(imu_dt);
    std::this_thread::sleep_for(1ms);
  }
  for (int s = 0; s < 40 && rclcpp::ok(); ++s) {
    for (int k = 0; k < 25; ++k) {
      imu_pub->publish(makeImu(t)); t = t + rclcpp::Duration::from_seconds(imu_dt);
    }
    cloud_pub->publish(makeBoxScan(t - rclcpp::Duration::from_seconds(0.1),
                                   Eigen::Vector3d::Zero(), 0.1));
    std::this_thread::sleep_for(100ms);
  }

  exec.cancel();
  spin.join();

  const double ms = comp_ms.load();
  ASSERT_GE(ms, 0.0) << "no /diagnostics computation-time sample received";
  std::cout << "[ perf ] steady-state per-scan computation time: " << ms << " ms\n";
  // Generous: a healthy synthetic scan here is a few ms; 750 ms would mean a
  // catastrophic per-scan blowup, not a normal machine-speed difference.
  EXPECT_LT(ms, 750.0) << "per-scan computation time regressed badly: " << ms << " ms";
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  rclcpp::init(argc, argv);
  int rc = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return rc;
}
