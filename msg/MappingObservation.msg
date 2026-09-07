# One scan-time snapshot. Registered pose and cloud are in header.frame_id
# (odom); cloud.header must match header exactly. The receiver recovers the
# immutable base-local cloud with inverse(registered_pose) * cloud.
std_msgs/Header header
string<=36 source_session_id
uint64 observation_id
string<=256 base_frame_id
geometry_msgs/Pose registered_pose
sensor_msgs/PointCloud2 cloud

# RIGHT/local SE(3) error: T_true = registered_pose * Exp(delta).
# Order [tx,ty,tz,rx,ry,rz], units metres/radians, full row-major 6x6.
# This is NOT ROS fixed-axis/world PoseWithCovariance. Kind UNKNOWN means
# unavailable; its all-zero wire payload MUST NOT become a zero-noise factor.
uint8 UNKNOWN=0
uint8 ASSUMED=1
uint8 CONDITIONAL=2
uint8 CALIBRATED=3
uint8 covariance_kind
string<=256 covariance_model
float64[36] pose_covariance

bool registration_converged
uint8 degenerate_translation_modes
uint8 degenerate_rotation_modes
