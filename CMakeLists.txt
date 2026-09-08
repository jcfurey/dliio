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

cmake_minimum_required(VERSION 3.16)
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
find_package(ament_cmake_python REQUIRED)
# VTK::jsoncpp (pulled transitively via PCL/pcl_ros) lists JsonCpp::JsonCpp in its
# link interface but VTK doesn't find_package(jsoncpp) itself — define the imported
# target first so VTK-targets.cmake resolves it.
find_package(jsoncpp REQUIRED)
ament_auto_find_build_dependencies()

find_package(rosidl_default_generators REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io registration kdtree search features surface filters)
find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)
find_package(GTSAM 4.2 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)
# OpenCV: direct visual photometric residual (nano_gicp) + image undistort (odom)
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d)

# Optional: raw Livox CustomMsg ingestion. Only built when livox_ros_driver2 is
# present in the workspace; without it the node consumes Livox data already in
# PointCloud2 form (the default), so this is NOT a required build dependency.
option(DLIIO_ENABLE_LIVOX "Enable Livox CustomMsg ingestion when its message package is available" ON)
if(DLIIO_ENABLE_LIVOX)
  find_package(livox_ros_driver2 QUIET)
endif()
if(DLIIO_ENABLE_LIVOX AND livox_ros_driver2_FOUND)
  message(STATUS "livox_ros_driver2 found: enabling raw Livox CustomMsg ingestion.")
else()
  message(STATUS "Livox CustomMsg adapter unavailable or disabled; PointCloud2 ingestion remains available.")
endif()

# OpenMP via the imported target (carries compile options and link flags to
# every dependent instead of mutating global CMAKE_*_FLAGS)
find_package(OpenMP REQUIRED)

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
        srv/MapArchive.srv
        srv/ApplyPoseRevision.srv
        srv/RestorePoseRevision.srv
        srv/UpdatePoseGraph.srv
        msg/MappingObservation.msg
        msg/MappingSnapshot.msg
    DEPENDENCIES std_msgs geometry_msgs sensor_msgs nav_msgs
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
target_link_libraries(nanoflann ${PCL_LIBRARIES})

# NanoGICP
add_library(nano_gicp STATIC
  src/nano_gicp/nano_gicp.cc
  src/nano_gicp/surface_texture.cc
  src/nano_gicp/geometry_diagnostics.cc
)
set_target_properties(nano_gicp PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(nano_gicp PRIVATE PCL_NO_PRECOMPILE)
target_include_directories(nano_gicp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(nano_gicp
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
  src/dlio/timed_observer.cc
)
target_compile_definitions(dlio_components PRIVATE PCL_NO_PRECOMPILE)
target_link_libraries(dlio_components
  nano_gicp
  Threads::Threads
  ${PCL_LIBRARIES}
  ${cpp_typesupport_target}
)
if(DLIIO_ENABLE_LIVOX AND livox_ros_driver2_FOUND)
  # PRIVATE: only odom.cc compiles the Livox path; odom.h stays livox-free so the
  # OdomNode layout is identical for every translation unit (no ODR mismatch).
  target_compile_definitions(dlio_components PRIVATE HAVE_LIVOX_ROS_DRIVER2)
  # ament_target_dependencies was removed in Rolling/Lyrical. livox_ros_driver2
  # is a rosidl message package (CustomMsg/CustomPoint), so link its modern
  # _TARGETS and include the legacy _INCLUDE_DIRS; each is a no-op when undefined,
  # so this works Humble..Lyrical. Plain signature to match the
  # target_link_libraries above (CMake forbids mixing keyword/plain styles).
  target_link_libraries(dlio_components ${livox_ros_driver2_TARGETS})
  target_include_directories(dlio_components PRIVATE ${livox_ros_driver2_INCLUDE_DIRS})
  ament_export_dependencies(livox_ros_driver2)
endif()
rclcpp_components_register_nodes(dlio_components "dlio::OdomNode" "dlio::MapNode")

# Standalone executables
pybind11_add_module(_dliio_pose_graph MODULE src/dlio/pose_graph.cc src/dlio/loop_features.cc
  src/dlio/loop_registration.cc src/dlio/map_geometry.cc)
target_include_directories(_dliio_pose_graph PRIVATE include)
target_link_libraries(_dliio_pose_graph PRIVATE gtsam Eigen3::Eigen ${PCL_LIBRARIES})
# ROSIDL already registers this Python directory in the package environment.
# Installing next to script symlinks would not be importable: Python resolves
# their source directory when constructing sys.path.
ament_get_python_install_dir(DLIIO_PYTHON_INSTALL_DIR)
install(TARGETS _dliio_pose_graph LIBRARY DESTINATION ${DLIIO_PYTHON_INSTALL_DIR})

ament_auto_add_executable(dlio_odom_node src/dlio/odom_node.cc)
target_compile_definitions(dlio_odom_node PRIVATE PCL_NO_PRECOMPILE)
target_link_libraries(dlio_odom_node dlio_components)

ament_auto_add_executable(dlio_map_node src/dlio/map_node.cc)
target_compile_definitions(dlio_map_node PRIVATE PCL_NO_PRECOMPILE)
target_link_libraries(dlio_map_node dlio_components)

install(DIRECTORY cfg launch doc
  DESTINATION share/${PROJECT_NAME}
)
install(FILES README.md ouster.repos LICENSE THIRD_PARTY_LICENSES.md
  DESTINATION share/${PROJECT_NAME})
install(DIRECTORY test/installed_consumer DESTINATION share/${PROJECT_NAME}/test)
install(PROGRAMS scripts/inspect_replay.py scripts/verify_install.py scripts/verify_packet_replay.py
  scripts/dlio_mapping_node.py scripts/mapping_archive.py scripts/verify_mapping.py
  scripts/verify_mapping_replay.py scripts/verify_pose_revisions.py
  scripts/verify_pose_graph.py scripts/mapping_loop_closure.py scripts/dlio_loop_closure_node.py
  scripts/wait_for_mapping.py
  scripts/dlio_ekf_inputs.py
  scripts/verify_integration.py
  DESTINATION lib/${PROJECT_NAME})
ament_python_install_package(dliio_mapping PACKAGE_DIR scripts/dliio_mapping)
ament_python_install_package(dliio_fusion PACKAGE_DIR scripts/dliio_fusion)
install(DIRECTORY scripts DESTINATION share/${PROJECT_NAME})

# Tests
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_handoff_launch test/test_handoff_launch.py TIMEOUT 60)
  ament_add_pytest_test(test_fusion_inputs test/test_fusion_inputs.py TIMEOUT 60)
  ament_add_pytest_test(test_fusion_ros test/test_fusion_ros.py TIMEOUT 90)
  # This integration test owns /clock and checks /tf silence in its test domain.
  set_tests_properties(test_fusion_ros PROPERTIES RUN_SERIAL TRUE)
  ament_add_pytest_test(test_mapping_core test/test_mapping_core.py TIMEOUT 60)
  ament_add_pytest_test(test_mapping_input test/test_mapping_input.py TIMEOUT 60)
  ament_add_pytest_test(test_mapping_revisions test/test_mapping_revisions.py TIMEOUT 120
    APPEND_ENV PYTHONPATH=$<TARGET_FILE_DIR:_dliio_pose_graph>)
  ament_add_pytest_test(test_map_geometry test/test_map_geometry.py TIMEOUT 60
    APPEND_ENV PYTHONPATH=$<TARGET_FILE_DIR:_dliio_pose_graph>)
  ament_add_pytest_test(test_mapping_uncertainty test/test_mapping_uncertainty.py TIMEOUT 60)
  ament_add_pytest_test(test_mapping_graph test/test_mapping_graph.py TIMEOUT 120
    APPEND_ENV PYTHONPATH=$<TARGET_FILE_DIR:_dliio_pose_graph>)
  ament_add_pytest_test(test_loop_validation test/test_loop_validation.py TIMEOUT 60)
  ament_add_pytest_test(test_loop_inputs test/test_loop_inputs.py TIMEOUT 120
    APPEND_ENV PYTHONPATH=$<TARGET_FILE_DIR:_dliio_pose_graph>)
  ament_add_pytest_test(test_loop_retrieval test/test_loop_retrieval.py TIMEOUT 120
    APPEND_ENV PYTHONPATH=$<TARGET_FILE_DIR:_dliio_pose_graph>)
  ament_add_pytest_test(test_loop_registration test/test_loop_registration.py TIMEOUT 120
    APPEND_ENV PYTHONPATH=$<TARGET_FILE_DIR:_dliio_pose_graph>)
  ament_add_pytest_test(test_live_loop_pipeline test/test_live_loop_pipeline.py TIMEOUT 180
    APPEND_ENV PYTHONPATH=$<TARGET_FILE_DIR:_dliio_pose_graph>)
  ament_add_gtest(test_imu_delivery test/test_imu_delivery.cpp)
  target_include_directories(test_imu_delivery PRIVATE include)
  find_package(Eigen3 REQUIRED)
  ament_add_gtest(test_auxiliary_gravity test/test_auxiliary_gravity.cpp)
  target_include_directories(test_auxiliary_gravity PRIVATE include)
  target_link_libraries(test_auxiliary_gravity Eigen3::Eigen)
  ament_add_gtest(test_observer_timing test/test_observer_timing.cpp)
  target_compile_definitions(test_observer_timing PRIVATE PCL_NO_PRECOMPILE)
  target_link_libraries(test_observer_timing dlio_components)
  ament_add_gtest(test_timed_observer test/test_timed_observer.cpp)
  target_link_libraries(test_timed_observer dlio_components)
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

  ament_add_gtest(test_kdtree test/test_kdtree.cpp)
  target_compile_definitions(test_kdtree PRIVATE PCL_NO_PRECOMPILE)
  target_link_libraries(test_kdtree
    nano_gicp
    ${PCL_LIBRARIES}
  )

  ament_add_gtest(test_degeneracy_governor test/test_degeneracy_governor.cpp)
  target_compile_definitions(test_degeneracy_governor PRIVATE PCL_NO_PRECOMPILE)
  target_link_libraries(test_degeneracy_governor
    dlio_components
  )

  ament_add_gtest(test_degeneracy_observer test/test_degeneracy_observer.cpp)
  target_compile_definitions(test_degeneracy_observer PRIVATE PCL_NO_PRECOMPILE)
  target_link_libraries(test_degeneracy_observer
    dlio_components
  )

  ament_add_gtest(test_elevation_lut test/test_elevation_lut.cpp)
  target_compile_definitions(test_elevation_lut PRIVATE PCL_NO_PRECOMPILE)
  target_link_libraries(test_elevation_lut
    dlio_components
  )

  ament_add_gtest(test_physics_fuse test/test_physics_fuse.cpp)
  target_compile_definitions(test_physics_fuse PRIVATE PCL_NO_PRECOMPILE)
  target_link_libraries(test_physics_fuse
    dlio_components
  )

  ament_add_gtest(test_slosh_guard test/test_slosh_guard.cpp)
  target_compile_definitions(test_slosh_guard PRIVATE PCL_NO_PRECOMPILE)
  target_link_libraries(test_slosh_guard
    dlio_components
  )

  ament_add_gtest(test_genz_weight test/test_genz_weight.cpp)
  target_compile_definitions(test_genz_weight PRIVATE PCL_NO_PRECOMPILE)
  target_link_libraries(test_genz_weight
    dlio_components
  )

  ament_add_gtest(test_xicp_localizability test/test_xicp_localizability.cpp)
  target_compile_definitions(test_xicp_localizability PRIVATE PCL_NO_PRECOMPILE)
  target_link_libraries(test_xicp_localizability
    dlio_components
  )

  ament_add_gtest(test_saliency_weight test/test_saliency_weight.cpp)
  target_compile_definitions(test_saliency_weight PRIVATE PCL_NO_PRECOMPILE)
  target_link_libraries(test_saliency_weight
    dlio_components
  )

  ament_add_gtest(test_intensity test/test_intensity.cpp)
  target_compile_definitions(test_intensity PRIVATE PCL_NO_PRECOMPILE)
  target_link_libraries(test_intensity
    dlio_components
  )

  ament_add_gtest(test_surface_texture test/test_surface_texture.cpp)
  target_link_libraries(test_surface_texture nano_gicp)

  ament_add_gtest(test_geometry_diagnostics test/test_geometry_diagnostics.cpp)
  target_link_libraries(test_geometry_diagnostics nano_gicp)

  # MapNode creates SavePCD; make its generated runtime type support available
  # even when testing the build tree before installation.
  ament_add_gtest(test_pointcloud_channels test/test_pointcloud_channels.cpp
    APPEND_LIBRARY_DIRS "${CMAKE_CURRENT_BINARY_DIR}")
  target_compile_definitions(test_pointcloud_channels PRIVATE PCL_NO_PRECOMPILE)
  target_link_libraries(test_pointcloud_channels dlio_components)

  ament_add_gtest(test_output_contract test/test_output_contract.cpp
    APPEND_LIBRARY_DIRS "${CMAKE_CURRENT_BINARY_DIR}" TIMEOUT 60)
  target_compile_definitions(test_output_contract PRIVATE PCL_NO_PRECOMPILE)
  target_link_libraries(test_output_contract dlio_components)

  ament_add_gtest(test_sensor_detection test/test_sensor_detection.cpp)
  target_compile_definitions(test_sensor_detection PRIVATE PCL_NO_PRECOMPILE)
  target_link_libraries(test_sensor_detection
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

  # End-to-end pipeline: accuracy regression, feature behavior, perf smoke
  # (drives a real node; longer wall-clock, generous timeout).
  ament_add_gtest(test_pipeline test/test_pipeline.cpp TIMEOUT 180)
  target_compile_definitions(test_pipeline PRIVATE PCL_NO_PRECOMPILE)
  target_link_libraries(test_pipeline
    dlio_components
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
  )
endif()

ament_auto_package()
