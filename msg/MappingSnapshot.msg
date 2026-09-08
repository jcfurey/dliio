# Geometry and its entire retained pose trajectory belong to ONE committed
# revision. Consumers must not pair this cloud with an older trajectory/TF.
std_msgs/Header header
string<=36 session_id
uint64 pose_revision
sensor_msgs/PointCloud2 cloud
# Matching observation stamps/order; originals in odom, optimized poses in map.
nav_msgs/Path original_trajectory
nav_msgs/Path optimized_trajectory
