# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

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

set(CUDA_MIN_VERSION "11.4")
find_package(CUDA ${CUDA_MIN_VERSION} REQUIRED)

# Explicitly include directories for NVCC
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(include)

# point_cloud_node
ament_auto_add_library(point_cloud_node SHARED
  src/point_cloud_node.cpp)
target_compile_definitions(point_cloud_node
  PRIVATE "COMPOSITION_BUILDING_DLL"
)
rclcpp_components_register_nodes(point_cloud_node "nvidia::isaac_ros::stereo_image_proc::PointCloudNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::stereo_image_proc::PointCloudNode;$<TARGET_FILE:point_cloud_node>\n")

# disparity_node
ament_auto_add_library(disparity_node SHARED
src/disparity_node.cpp)
target_compile_definitions(disparity_node
  PRIVATE "COMPOSITION_BUILDING_DLL"
)

rclcpp_components_register_nodes(disparity_node "nvidia::isaac_ros::stereo_image_proc::DisparityNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::stereo_image_proc::DisparityNode;$<TARGET_FILE:disparity_node>\n")

# Install config directory
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(TARGETS disparity_node point_cloud_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/isaac_ros_stereo_disparity_pol.py)
  add_launch_test(test/isaac_ros_stereo_point_cloud_pol.py)
  add_launch_test(test/isaac_ros_stereo_pipeline_pol.py)
  add_launch_test(test/isaac_ros_stereo_point_cloud_OSS_test.py)
  add_launch_test(test/isaac_ros_stereo_pipeline_output_compare.py)
  add_launch_test(test/isaac_ros_stereo_point_cloud_output_compare.py)
endif()

ament_auto_package(INSTALL_TO_SHARE launch config)
