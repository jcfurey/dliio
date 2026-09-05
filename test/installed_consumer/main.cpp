#include <direct_lidar_inertial_odometry/srv/save_pcd.hpp>
#include <rosidl_typesupport_cpp/message_type_support.hpp>

int main() {
  // Exercise both installed headers and the generated shared-library link.
  return rosidl_typesupport_cpp::get_message_type_support_handle<
      direct_lidar_inertial_odometry::srv::SavePCD_Request>() ? 0 : 1;
}
