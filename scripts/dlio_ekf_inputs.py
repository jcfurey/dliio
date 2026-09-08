#!/usr/bin/env python3
"""Prepare explicitly modeled DLIO increments and auxiliary tilt for an EKF.

This publishes comparison inputs only. It does not change raw measurements,
publish TF, modify maps, infer extrinsics, or feed the frontend observer.
"""
from copy import deepcopy
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Vector3Stamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

from dliio_fusion.inputs import StampGate, pose_covariance, positive_sigma, tilt_quaternion


class Inputs(Node):
    def __init__(self):
        super().__init__('dlio_ekf_inputs')
        defaults = dict(odom_topic='dlio/odom_node/odom', gravity_topic='dliio/auxiliary_gravity',
                        body_frame='base_link', odom_frame='odom', covariance_policy='assumed',
                        assumed_position_sigma=.3, assumed_angle_sigma=.1,
                        assumed_tilt_sigma=math.radians(5.), odom_period=.2, tilt_period=.05)
        self.config = {key: self.declare_parameter(key, value).value for key, value in defaults.items()}
        self.odom_gate = StampGate(self.config['odom_period'])
        self.tilt_gate = StampGate(self.config['tilt_period'])
        self.tilt_variance = positive_sigma(self.config['assumed_tilt_sigma'])**2
        # Fail at startup for a misspelled policy or invalid assumed sigma.
        pose_covariance(np.eye(6), self.config['covariance_policy'],
                        self.config['assumed_position_sigma'], self.config['assumed_angle_sigma'])
        self.rejected = 0
        self.odom_pub = self.create_publisher(Odometry, 'dlio/ekf_inputs/odom', 30)
        self.tilt_pub = self.create_publisher(Imu, 'dlio/ekf_inputs/tilt', 100)
        self.create_subscription(Odometry, self.config['odom_topic'], self.odometry, qos_profile_sensor_data)
        self.create_subscription(Vector3Stamped, self.config['gravity_topic'], self.gravity, qos_profile_sensor_data)
        self.get_logger().warning(
            f"Experimental EKF inputs: pose covariance={self.config['covariance_policy']}; "
            f"assumed tilt sigma={math.sqrt(self.tilt_variance):.6g} rad. "
            "These assumptions and decimation do not establish calibrated independent noise. "
            "Gravity must already be world +Z in the configured body frame; restart on bag rewind.")

    @staticmethod
    def stamp(message):
        return message.header.stamp.sec*1_000_000_000+message.header.stamp.nanosec

    def reject(self, error):
        self.rejected += 1
        if self.rejected & (self.rejected-1) == 0:
            self.get_logger().warning(f'Rejected EKF input ({self.rejected} total): {error}')

    def odometry(self, message):
        stamp = self.stamp(message)
        if not self.odom_gate.ready(stamp):
            return
        try:
            if message.header.frame_id != self.config['odom_frame'] or message.child_frame_id != self.config['body_frame']:
                raise ValueError('Odometry world/body frames do not match configured frames')
            p, q = message.pose.pose.position, message.pose.pose.orientation
            if not np.isfinite([p.x, p.y, p.z, q.x, q.y, q.z, q.w]).all() or \
                    abs(np.linalg.norm([q.x, q.y, q.z, q.w])-1.) > 1.e-3:
                raise ValueError('Odometry pose must be finite with a unit quaternion')
            covariance = pose_covariance(message.pose.covariance, self.config['covariance_policy'],
                self.config['assumed_position_sigma'], self.config['assumed_angle_sigma'])
            output = deepcopy(message)
            output.pose.covariance = covariance.ravel().tolist()
            # This topic contains pose inputs only; do not expose a misleading
            # copy of the observer twist as another independent measurement.
            output.twist = Odometry().twist
            output.twist.covariance = (np.eye(6)*1.e6).ravel().tolist()
            self.odom_pub.publish(output)
            self.odom_gate.accept(stamp)
        except ValueError as error:
            self.reject(error)

    def gravity(self, message):
        stamp = self.stamp(message)
        if not self.tilt_gate.ready(stamp):
            return
        try:
            if message.header.frame_id != self.config['body_frame']:
                raise ValueError('Gravity direction must already be expressed in the configured body frame')
            q = tilt_quaternion([message.vector.x, message.vector.y, message.vector.z])
            output = Imu()
            output.header = deepcopy(message.header)
            output.orientation.x, output.orientation.y, output.orientation.z, output.orientation.w = map(float, q)
            output.orientation_covariance = np.diag([self.tilt_variance]*2+[1.e6]).ravel().tolist()
            output.angular_velocity_covariance[0] = -1.
            output.linear_acceleration_covariance[0] = -1.
            self.tilt_pub.publish(output)
            self.tilt_gate.accept(stamp)
        except ValueError as error:
            self.reject(error)


def main():
    rclpy.init()
    node = Inputs()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
