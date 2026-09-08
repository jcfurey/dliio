#include <direct_lidar_inertial_odometry/srv/save_pcd.hpp>
#include <direct_lidar_inertial_odometry/msg/mapping_observation.hpp>
#include <direct_lidar_inertial_odometry/srv/map_archive.hpp>
#include <direct_lidar_inertial_odometry/srv/update_pose_graph.hpp>
#include <rosidl_typesupport_cpp/message_type_support.hpp>

int main() {
  // Exercise both installed headers and the generated shared-library link.
  using namespace direct_lidar_inertial_odometry;
  return rosidl_typesupport_cpp::get_message_type_support_handle<srv::SavePCD_Request>() &&
         rosidl_typesupport_cpp::get_message_type_support_handle<msg::MappingObservation>() &&
         rosidl_typesupport_cpp::get_message_type_support_handle<srv::MapArchive_Request>() &&
         rosidl_typesupport_cpp::get_message_type_support_handle<srv::UpdatePoseGraph_Request>() ? 0 : 1;
}
