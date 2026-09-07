#include <gtest/gtest.h>
#include "dlio/odom.h"

namespace dlio {
struct OdomNodeTestAccess {
  static void timedWiring(OdomNode& node) {
    node.imu_meas.stamp = 10.; node.imu_meas.dt = .0025;
    node.imu_meas.raw_accel = Eigen::Vector3f(1., 0., node.gravity_);
    node.imu_meas.raw_gyro.setZero(); node.imu_meas.raw_valid = true;
    node.imu_meas.lin_accel = node.imu_meas.raw_accel; node.imu_meas.ang_vel.setZero();
    const auto initial_position = node.state.p;
    node.propagateState(rclcpp::Time(10, 0, RCL_ROS_TIME));
    EXPECT_EQ(node.state.p, initial_position);  // uninitialized: stamp only, no motion
    EXPECT_EQ(node.imu_delivery_.snapshot().propagated, 0u);
    node.scan_stamp = 10.1; node.prev_scan_stamp = 9.8;
    node.imu_stamp = rclcpp::Time(10, 200000000, RCL_ROS_TIME);
    node.lidarPose.p.setZero(); node.lidarPose.q.setIdentity();
    node.state.v.lin.w.setZero(); node.state.b.accel.setZero(); node.state.b.gyro.setZero();
    for (int i = 0; i <= 81; ++i) {
      OdomNode::ImuMeas sample;
      sample.stamp = 10.+i*.0025; sample.dt = .0025;
      sample.raw_accel = Eigen::Vector3f(1., 0., node.gravity_);
      sample.raw_gyro.setZero(); sample.raw_valid = true;
      sample.lin_accel = sample.raw_accel; sample.ang_vel = sample.raw_gyro;
      node.imu_buffer.push_front(sample);
    }
    // Buffer already contains the next callback (10.2025), but it has not yet
    // propagated. Initialization must not replay that sample twice.
    node.updateState();
    ASSERT_TRUE(node.timed_observer_->initialized());
    ASSERT_TRUE(node.timed_observer_->latest().covariance_valid);
    EXPECT_DOUBLE_EQ(node.timed_observer_->latest().stamp, 10.2);
    EXPECT_NEAR(node.state.p.x(), .005, 1e-7);
    EXPECT_NEAR(node.state.v.lin.w.x(), .1, 1e-7);
    // First scan's callback has not yet set first_opt_done. A real IMU
    // callback must still advance the initialized model and its stamp together.
    ASSERT_FALSE(node.geo.first_opt_done);
    node.extrinsics_ready_ = true;
    node.imu_calibrated = true;
    node.last_imu_input_ns_ = 10200000000;
    node.prev_imu_stamp = 10.2;
    auto message = std::make_shared<sensor_msgs::msg::Imu>();
    message->header.stamp = rclcpp::Time(10, 202500000, RCL_ROS_TIME);
    message->linear_acceleration.x = 1.; message->linear_acceleration.z = node.gravity_;
    node.callbackImu(message);
    EXPECT_EQ(node.timed_input_rejected_.load(), 0u);
    EXPECT_EQ(node.imu_delivery_.snapshot().propagated, 1u);
    EXPECT_NEAR(node.state.p.x(), .5*.1025*.1025, 1e-7);
    node.publishPose();
    EXPECT_EQ(node.odom_ros.header.stamp, node.imu_stamp);
    const auto pose = TimedObserver::poseCovariance(node.timed_observer_->latest());
    const auto twist = TimedObserver::twistCovariance(node.timed_observer_->latest());
    for (int r = 0; r < 6; ++r) for (int c = 0; c < 6; ++c) {
      EXPECT_DOUBLE_EQ(node.odom_ros.pose.covariance[r*6+c], pose(r, c));
      EXPECT_DOUBLE_EQ(node.odom_ros.twist.covariance[r*6+c], twist(r, c));
    }
    EXPECT_GT(std::abs(pose(0, 4)), 1e-5);  // retain attitude/position coupling
    const auto published_stamp = node.odom_ros.header.stamp;
    node.timed_observer_->invalidate();
    node.imu_meas.stamp = 10.205;
    node.propagateState(rclcpp::Time(10, 205000000, RCL_ROS_TIME));
    node.publishPose();
    EXPECT_EQ(node.odom_ros.header.stamp, published_stamp);  // no configured fallback
  }
  static void perfectDelayedTurn(OdomNode& node, double anchor_time) {
    constexpr double scan = 10.1, latest = 10.35, omega = 1.8;
    auto position = [](double time) { return Eigen::Vector3f(time-10., 0., 0.); };
    auto rotation = [](double time) {
      return Eigen::Quaternionf(Eigen::AngleAxisf(omega*(time-10.), Eigen::Vector3f::UnitZ()));
    };
    node.scan_stamp = scan; node.prev_scan_stamp = 9.8;
    node.imu_stamp = rclcpp::Time(10, 350000000, RCL_ROS_TIME);
    node.state.p = position(latest); node.state.q = rotation(latest);
    node.state.v.lin.w = Eigen::Vector3f::UnitX();
    node.state.b.accel.setZero(); node.state.b.gyro.setZero();
    node.lidarPose.p = position(scan); node.lidarPose.q = rotation(scan);
    node.geo.prev_state_stamp = anchor_time;
    node.geo.prev_p = position(anchor_time); node.geo.prev_q = rotation(anchor_time);
    node.geo.prev_vel = Eigen::Vector3f::UnitX();
    for (int k = 0; k <= 280; ++k) {
      OdomNode::ImuMeas sample;
      sample.stamp = 9.7+k*.0025; sample.dt = .0025;
      sample.ang_vel = Eigen::Vector3f(0.f, 0.f, omega);
      sample.lin_accel = Eigen::Vector3f(0.f, 0.f, node.gravity_);
      node.imu_buffer.push_front(sample);
    }
    node.updateState();
    EXPECT_TRUE(node.observer_time_aligned_);
    EXPECT_LT(node.observer_position_innovation_.norm(), 1e-4);
    EXPECT_LT(Eigen::AngleAxisf(node.observer_orientation_innovation_).angle(), 1e-4);
    EXPECT_LT((node.state.p-position(latest)).norm(), 1e-4);
    EXPECT_LT(node.state.q.angularDistance(rotation(latest)), 1e-4);
    EXPECT_LT(node.state.b.accel.norm(), 1e-4);
    EXPECT_LT(node.state.b.gyro.norm(), 1e-4);
  }
};
}  // namespace dlio

TEST(ObserverTiming, PerfectPoseDoesNotCorrectMotionWhenAnchorPrecedesScan) {
  dlio::OdomNode node;
  dlio::OdomNodeTestAccess::perfectDelayedTurn(node, 10.0);
}

TEST(ObserverTiming, PerfectPoseDoesNotCorrectMotionWhenAnchorFollowsScan) {
  dlio::OdomNode node;
  dlio::OdomNodeTestAccess::perfectDelayedTurn(node, 10.2);
}

TEST(ObserverTiming, AnchorExactlyAtScanUsesAnchorInsteadOfLatestState) {
  dlio::OdomNode node;
  dlio::OdomNodeTestAccess::perfectDelayedTurn(node, 10.1);
}

TEST(ObserverTiming, TimedModeInitializesReplaysAndPublishesMatchingCovariance) {
  rclcpp::NodeOptions options;
  options.append_parameter_override("odom/observer/measurementTimeUpdates", true);
  dlio::OdomNode node(options);
  dlio::OdomNodeTestAccess::timedWiring(node);
}

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
