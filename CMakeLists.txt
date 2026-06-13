###########################################################
#                                                         #
# Copyright (c)                                           #
#                                                         #
# The Verifiable & Control-Theoretic Robotics (VECTR) Lab #
# University of California, Los Angeles                   #
#                                                         #
# Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez   #
# Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu         #
#                                                         #
###########################################################

cmake_minimum_required(VERSION 3.12.4)
project(direct_lidar_inertial_odometry)

# Default to Release but respect whatever the user/colcon asked for
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)

# Reproducible sanitizer builds:  colcon build --cmake-args -DDLIIO_SANITIZE=address
# Accepts: address (ASan+UBSan), thread (TSan), or undefined (UBSan only).
# For TSan, also force single-threaded OpenMP at runtime (OMP_NUM_THREADS=1) to
# avoid false positives from the OpenMP runtime; the node's own callback threads
# are what TSan should observe.
set(DLIIO_SANITIZE "" CACHE STRING "Sanitizer: address|thread|undefined (empty=off)")
if(DLIIO_SANITIZE STREQUAL "address")
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address,undefined)
elseif(DLIIO_SANITIZE STREQUAL "thread")
  add_compile_options(-fsanitize=thread -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=thread)
elseif(DLIIO_SANITIZE STREQUAL "undefined")
  add_compile_options(-fsanitize=undefined -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=undefined)
endif()
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)

# ROS deps are pulled from package.xml; non-ROS deps found explicitly below.
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(rosidl_default_generators REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io registration kdtree search features surface filters)
find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)
# OpenCV: direct visual photometric residual (nano_gicp) + image undistort (odom)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

# Optional: raw Livox CustomMsg ingestion. Only built when livox_ros_driver2 is
# present in the workspace; without it the node consumes Livox data already in
# PointCloud2 form (the default), so this is NOT a required build dependency.
find_package(livox_ros_driver2 QUIET)
if(livox_ros_driver2_FOUND)
  message(STATUS "livox_ros_driver2 found: enabling raw Livox CustomMsg ingestion.")
else()
  message(STATUS "livox_ros_driver2 not found: Livox CustomMsg ingestion disabled "
                 "(feed Livox as PointCloud2 instead).")
endif()

# OpenMP via the imported target (carries compile options and link flags to
# every dependent instead of mutating global CMAKE_*_FLAGS)
find_package(OpenMP)
if(NOT OpenMP_CXX_FOUND)
  message(WARNING "OpenMP could not be found; building single-threaded.")
endif()

# CPUID test
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
file(WRITE ${CMAKE_BINARY_DIR}/test_cpuid.cpp "#include <cpuid.h>")
try_compile(HAS_CPUID ${CMAKE_BINARY_DIR} ${CMAKE_BINARY_DIR}/test_cpuid.cpp)
file(REMOVE ${CMAKE_BINARY_DIR}/test_cpuid.cpp)

if(HAS_CPUID)
  add_compile_definitions(HAS_CPUID)
endif()

# ROS IDL
rosidl_generate_interfaces(
    ${PROJECT_NAME}
        srv/SavePCD.srv
    DEPENDENCIES
)

rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} rosidl_typesupport_cpp)

# Include directories (project headers are target-scoped; see below)
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# NanoFLANN
# (PIC: linked into the shared component library)
add_library(nanoflann STATIC src/nano_gicp/nanoflann.cc)
set_target_properties(nanoflann PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(nanoflann PRIVATE PCL_NO_PRECOMPILE)
target_include_directories(nanoflann PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(nanoflann PUBLIC ${PCL_LIBRARIES})

# NanoGICP
add_library(nano_gicp STATIC
  src/nano_gicp/nano_gicp.cc
)
set_target_properties(nano_gicp PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(nano_gicp PRIVATE PCL_NO_PRECOMPILE)
target_include_directories(nano_gicp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(nano_gicp PUBLIC
  nanoflann
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

# Component library: both nodes, loadable into a component container for
# intra-process communication; the classic executables link it too.
ament_auto_add_library(dlio_components SHARED
  src/dlio/odom.cc
  src/dlio/map.cc
  src/dlio/components.cc
)
target_compile_definitions(dlio_components PRIVATE PCL_NO_PRECOMPILE)
target_link_libraries(dlio_components
  nano_gicp
  Threads::Threads
  ${PCL_LIBRARIES}
  ${cpp_typesupport_target}
)
if(livox_ros_driver2_FOUND)
  # PRIVATE: only odom.cc compiles the Livox path; odom.h stays livox-free so the
  # OdomNode layout is identical for every translation unit (no ODR mismatch).
  target_compile_definitions(dlio_components PRIVATE HAVE_LIVOX_ROS_DRIVER2)
  ament_target_dependencies(dlio_components livox_ros_driver2)
endif()
rclcpp_components_register_nodes(dlio_components "dlio::OdomNode" "dlio::MapNode")

# Standalone executables
ament_auto_add_executable(dlio_odom_node src/dlio/odom_node.cc)
target_compile_definitions(dlio_odom_node PRIVATE PCL_NO_PRECOMPILE)
target_link_libraries(dlio_odom_node dlio_components)

ament_auto_add_executable(dlio_map_node src/dlio/map_node.cc)
target_compile_definitions(dlio_map_node PRIVATE PCL_NO_PRECOMPILE)
target_link_libraries(dlio_map_node dlio_components)

install(DIRECTORY cfg launch
  DESTINATION share/${PROJECT_NAME}
)

# Tests
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_nano_gicp test/test_nano_gicp.cpp)
  target_compile_definitions(test_nano_gicp PRIVATE PCL_NO_PRECOMPILE)
  target_link_libraries(test_nano_gicp
    nano_gicp
    ${PCL_LIBRARIES}
  )

  ament_add_gtest(test_imu_integration test/test_imu_integration.cpp)
  target_compile_definitions(test_imu_integration PRIVATE PCL_NO_PRECOMPILE)
  target_link_libraries(test_imu_integration
    dlio_components
  )

  ament_add_gtest(test_intensity test/test_intensity.cpp)
  target_compile_definitions(test_intensity PRIVATE PCL_NO_PRECOMPILE)
  target_link_libraries(test_intensity
    dlio_components
  )

  ament_add_gtest(test_visual_residual test/test_visual_residual.cpp)
  target_compile_definitions(test_visual_residual PRIVATE PCL_NO_PRECOMPILE)
  target_link_libraries(test_visual_residual
    nano_gicp
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
  )

  # Live-node concurrency / TSan harness (longer; generous timeout).
  ament_add_gtest(test_node_concurrency test/test_node_concurrency.cpp TIMEOUT 120)
  target_compile_definitions(test_node_concurrency PRIVATE PCL_NO_PRECOMPILE)
  target_link_libraries(test_node_concurrency
    dlio_components
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
  )
endif()

ament_auto_package()
