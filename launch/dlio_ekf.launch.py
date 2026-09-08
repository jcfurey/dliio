"""Optional downstream tilt EKF for an existing DLIO session; owns no TF."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    defaults = dict(namespace='', use_sim_time='true', body_frame='base_link', odom_frame='odom',
                    odom_topic='dlio/odom_node/odom', gravity_topic='dliio/auxiliary_gravity',
                    filtered_topic='odometry/tilt_filtered', covariance_policy='assumed',
                    assumed_position_sigma='0.3', assumed_angle_sigma='0.1',
                    assumed_tilt_sigma='0.08726646259971647', odom_period='0.2', tilt_period='0.05')
    arguments = [DeclareLaunchArgument(key, default_value=value) for key, value in defaults.items()]
    inputs = {key: LaunchConfiguration(key) for key in
              ('body_frame', 'odom_frame', 'odom_topic', 'gravity_topic', 'covariance_policy')}
    inputs.update({key: ParameterValue(LaunchConfiguration(key), value_type=float) for key in
                   ('assumed_position_sigma', 'assumed_angle_sigma', 'assumed_tilt_sigma', 'odom_period', 'tilt_period')})
    sim_time = ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)
    inputs['use_sim_time'] = sim_time
    ekf = PathJoinSubstitution([FindPackageShare('direct_lidar_inertial_odometry'),
                               'cfg', 'robot_localization_tilt.yaml'])
    return LaunchDescription(arguments + [
        Node(package='direct_lidar_inertial_odometry', executable='dlio_ekf_inputs.py',
             namespace=LaunchConfiguration('namespace'),
             name='dlio_ekf_inputs', parameters=[inputs], output='screen'),
        Node(package='robot_localization', executable='ekf_node', name='ekf_tilt_node',
             namespace=LaunchConfiguration('namespace'),
             parameters=[ekf, {'use_sim_time': sim_time, 'publish_tf': False,
                              'base_link_frame': LaunchConfiguration('body_frame'),
                              'odom_frame': LaunchConfiguration('odom_frame'),
                              'world_frame': LaunchConfiguration('odom_frame')}],
             remappings=[('odometry/filtered', LaunchConfiguration('filtered_topic'))], output='screen'),
    ])
