# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

set(CUDA_MIN_VERSION "10.2")
find_package(CUDA ${CUDA_MIN_VERSION} REQUIRED)

# Explicitly include directories for NVCC
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(include)

# point_cloud_node_cuda
cuda_add_library(point_cloud_node_cuda SHARED
  src/point_cloud_node_cuda.cu)
ament_target_dependencies(point_cloud_node_cuda CUDA)

# point_cloud_node
ament_auto_add_library(point_cloud_node SHARED
  src/point_cloud_node.cpp)
add_dependencies(point_cloud_node point_cloud_node_cuda)
target_compile_definitions(point_cloud_node
  PRIVATE "COMPOSITION_BUILDING_DLL"
)
target_link_libraries(point_cloud_node point_cloud_node_cuda ${CUDA_LIBRARIES})
rclcpp_components_register_nodes(point_cloud_node "isaac_ros::stereo_image_proc::PointCloudNode")
set(node_plugins "${node_plugins}isaac_ros::stereo_image_proc::PointCloudNode;$<TARGET_FILE:point_cloud_node>\n")

# disparity_node
ament_auto_add_library(disparity_node SHARED
src/disparity_node.cpp)
target_compile_definitions(disparity_node
  PRIVATE "COMPOSITION_BUILDING_DLL"
)
target_link_libraries(disparity_node vpi)
rclcpp_components_register_nodes(disparity_node "isaac_ros::stereo_image_proc::DisparityNode")
set(node_plugins "${node_plugins}isaac_ros::stereo_image_proc::DisparityNode;$<TARGET_FILE:disparity_node>\n")

# isaac_ros_stereo_image_proc executable
ament_auto_add_executable(${PROJECT_NAME}
  src/stereo_image_proc_main.cpp
)

file(GLOB SOURCES "src/*.cpp")

install(TARGETS disparity_node point_cloud_node point_cloud_node_cuda
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  # Ignore copyright notices since we use custom JetPack EULA
  set(ament_cmake_copyright_FOUND TRUE)

  ament_lint_auto_find_test_dependencies()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/isaac_ros_stereo_disparity_pol.py)
  add_launch_test(test/isaac_ros_stereo_pipeline_pol.py)
  add_launch_test(test/isaac_ros_stereo_point_cloud_OSS_test.py)
  add_launch_test(test/isaac_ros_stereo_point_cloud_pol.py)
endif()

ament_auto_package(INSTALL_TO_SHARE launch)
