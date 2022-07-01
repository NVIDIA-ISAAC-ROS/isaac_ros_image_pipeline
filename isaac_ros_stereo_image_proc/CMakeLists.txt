# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

cmake_minimum_required(VERSION 3.5)
project(isaac_ros_stereo_image_proc CUDA CXX)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Default to Release build
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()
message( STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}" )

execute_process(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE)
message( STATUS "Architecture: ${ARCHITECTURE}" )

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Find VPI dependency
find_package(vpi REQUIRED)

set(CUDA_MIN_VERSION "11.4")
find_package(CUDA ${CUDA_MIN_VERSION} REQUIRED)

# Explicitly include directories for NVCC
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(include)

add_library(libgxf_point_cloud SHARED IMPORTED)
if( ${ARCHITECTURE} STREQUAL "x86_64" )
  set_property(TARGET libgxf_point_cloud PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/lib/x86_64/libgxf_point_cloud.so)
elseif( ${ARCHITECTURE} STREQUAL "aarch64" )
  set_property(TARGET libgxf_point_cloud PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/lib/jetpack50/libgxf_point_cloud.so)
endif()

# point_cloud_node
ament_auto_add_library(point_cloud_node SHARED
  src/point_cloud_node.cpp)
target_compile_definitions(point_cloud_node
  PRIVATE "COMPOSITION_BUILDING_DLL"
)
target_link_libraries(point_cloud_node libgxf_point_cloud)
rclcpp_components_register_nodes(point_cloud_node "nvidia::isaac_ros::stereo_image_proc::PointCloudNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::stereo_image_proc::PointCloudNode;$<TARGET_FILE:point_cloud_node>\n")

# disparity_node
add_library(libgxf_sgm_disparity SHARED IMPORTED)
if( ${ARCHITECTURE} STREQUAL "x86_64" )
  set_property(TARGET libgxf_sgm_disparity
  PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/lib/x86_64/libgxf_disparity_extension.so)
elseif( ${ARCHITECTURE} STREQUAL "aarch64" )
  set_property(TARGET libgxf_sgm_disparity
  PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/lib/jetpack50/libgxf_disparity_extension.so)
endif()


ament_auto_add_library(disparity_node SHARED
src/disparity_node.cpp)
target_compile_definitions(disparity_node
  PRIVATE "COMPOSITION_BUILDING_DLL"
)
target_link_libraries(disparity_node vpi libgxf_sgm_disparity)

rclcpp_components_register_nodes(disparity_node "nvidia::isaac_ros::stereo_image_proc::DisparityNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::stereo_image_proc::DisparityNode;$<TARGET_FILE:disparity_node>\n")

# Install config directory
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# Install cvcore tensorops extensions directory
if( ${ARCHITECTURE} STREQUAL "x86_64" )
  install(DIRECTORY lib/x86_64/ DESTINATION share/${PROJECT_NAME}/lib)
elseif( ${ARCHITECTURE} STREQUAL "aarch64" )
  install(DIRECTORY lib/jetpack50/ DESTINATION share/${PROJECT_NAME}/lib)
endif()

install(TARGETS disparity_node point_cloud_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  # Ignore copyright notices since we use custom NVIDIA Isaac ROS Software License
  set(ament_cmake_copyright_FOUND TRUE)

  ament_lint_auto_find_test_dependencies()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/isaac_ros_stereo_disparity_pol.py)
  add_launch_test(test/isaac_ros_stereo_point_cloud_pol.py)
  add_launch_test(test/isaac_ros_stereo_pipeline_pol.py)
  add_launch_test(test/isaac_ros_stereo_point_cloud_OSS_test.py)
  add_launch_test(test/isaac_ros_stereo_pipeline_output_compare.py)
  add_launch_test(test/isaac_ros_stereo_point_cloud_output_compare.py)
endif()

ament_auto_package(INSTALL_TO_SHARE launch)
