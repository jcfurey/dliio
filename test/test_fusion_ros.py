"""Actual RL frontend: correct attitude-induced climb while retaining a slope."""
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import Vector3Stamped
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from scipy.spatial.transform import Rotation
from tf2_msgs.msg import TFMessage


def test_tilt_ekf_preserves_true_vertical_motion_without_publishing_tf(tmp_path):
    namespace = '/fusion_'+uuid.uuid4().hex[:8]
    environment = dict(os.environ, ROS_LOG_DIR=str(tmp_path/'ros-log'))
    log = (tmp_path/'ekf.log').open('w')
    process = subprocess.Popen(['ros2', 'launch', 'direct_lidar_inertial_odometry', 'dlio_ekf.launch.py',
        'namespace:='+namespace, 'body_frame:=vehicle'], env=environment, stdout=log,
        stderr=subprocess.STDOUT, start_new_session=True)
    rclpy.init()
    node = rclpy.create_node('tilt_integration_test', namespace=namespace)
    odom = node.create_publisher(Odometry, 'dlio/odom_node/odom', 30)
    gravity = node.create_publisher(Vector3Stamped, 'dliio/auxiliary_gravity', 100)
    clock = node.create_publisher(Clock, '/clock', 10)
    received, transforms = [], []
    subscriptions = [node.create_subscription(Odometry, 'odometry/tilt_filtered', received.append, 100),
                     node.create_subscription(TFMessage, '/tf', transforms.append, 10),
                     node.create_subscription(TFMessage, 'tf', transforms.append, 10)]

    def spin(seconds):
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            assert process.poll() is None, (tmp_path/'ekf.log').read_text()
            rclpy.spin_once(node, timeout_sec=min(.01, max(0., deadline-time.monotonic())))

    try:
        deadline = time.monotonic()+30
        while not (odom.get_subscription_count() and gravity.get_subscription_count() and
                   node.count_subscribers('dlio/ekf_inputs/odom') >= 2 and
                   node.count_subscribers('dlio/ekf_inputs/tilt') and subscriptions[0].get_publisher_count()):
            if time.monotonic() > deadline:
                pytest.fail('EKF startup timeout: '+(tmp_path/'ekf.log').read_text())
            tick = Clock()
            tick.clock.sec = 99
            clock.publish(tick)
            spin(.05)
        # Vehicle travels 1 m/s forward and .05 m/s vertically. DLIO's mistaken
        # -10 degree pitch adds a false climb; the second IMU observes level
        # attitude. There is no constraint that fixes Z or vertical velocity.
        wrong = Rotation.from_euler('y', -10., degrees=True)
        q = wrong.as_quat()
        for i in range(401):
            seconds = i*.05
            stamp = 100_000_000_000+i*50_000_000
            tick = Clock()
            tick.clock.sec, tick.clock.nanosec = divmod(stamp, 1_000_000_000)
            clock.publish(tick)
            up = Vector3Stamped()
            up.header.frame_id = 'vehicle'
            up.header.stamp = tick.clock
            up.vector.z = 1.
            gravity.publish(up)
            if i % 4 == 0:
                message = Odometry()
                message.header.frame_id, message.child_frame_id = 'odom', 'vehicle'
                message.header.stamp = tick.clock
                xyz = wrong.apply([seconds, 0., .05*seconds])
                message.pose.pose.position.x, message.pose.pose.position.y, message.pose.pose.position.z = map(float, xyz)
                message.pose.pose.orientation.x, message.pose.pose.orientation.y, message.pose.pose.orientation.z, \
                    message.pose.pose.orientation.w = map(float, q)
                odom.publish(message)
            spin(.01)  # 5x acquisition-order replay, no original delivery jitter.
        spin(1.)
        assert received, (tmp_path/'ekf.log').read_text()
        state = received[-1]
        p, q = state.pose.pose.position, state.pose.pose.orientation
        assert state.header.frame_id == 'odom' and state.child_frame_id == 'vehicle'
        assert 18. < p.x < 21.
        assert .7 < p.z < 1.3  # Expected 1 m true climb; a flattened map fails.
        assert abs(p.y) < .1
        tilt = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')[:2]
        assert np.linalg.norm(tilt) < .01
        covariance = np.asarray(state.pose.covariance).reshape(6, 6)
        assert np.isfinite(covariance).all() and np.linalg.eigvalsh(covariance)[0] > 0
        assert not transforms
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
        log.close()
        del subscriptions
        node.destroy_node()
        rclpy.shutdown()
