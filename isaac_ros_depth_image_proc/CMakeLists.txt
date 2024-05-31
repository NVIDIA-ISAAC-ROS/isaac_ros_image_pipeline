# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.22.1)
project(isaac_ros_depth_image_proc CUDA CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# PointCloudXyzNode
ament_auto_add_library(point_cloud_xyz_node SHARED src/point_cloud_xyz_node.cpp)
rclcpp_components_register_nodes(point_cloud_xyz_node "nvidia::isaac_ros::depth_image_proc::PointCloudXyzNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::depth_image_proc::PointCloudXyzNode;$<TARGET_FILE:point_cloud_xyz_node>\n")
set_target_properties(point_cloud_xyz_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

# PointCloudXyzrgbNode
ament_auto_add_library(point_cloud_xyzrgb_node SHARED src/point_cloud_xyzrgb_node.cpp)
rclcpp_components_register_nodes(point_cloud_xyzrgb_node "nvidia::isaac_ros::depth_image_proc::PointCloudXyzrgbNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::depth_image_proc::PointCloudXyzrgbNode;$<TARGET_FILE:point_cloud_xyzrgb_node>\n")
set_target_properties(point_cloud_xyzrgb_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

# convert_metric_node
ament_auto_add_library(convert_metric_node SHARED src/convert_metric_node.cpp)
rclcpp_components_register_nodes(convert_metric_node "nvidia::isaac_ros::depth_image_proc::ConvertMetricNode")
target_link_libraries(convert_metric_node cvcuda nvcv_types)
set(node_plugins "${node_plugins}nvidia::isaac_ros::depth_image_proc::ConvertMetricNode;$<TARGET_FILE:convert_metric_node>\n")
set_target_properties(convert_metric_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  # The FindPythonInterp and FindPythonLibs modules are removed
  if(POLICY CMP0148)
    cmake_policy(SET CMP0148 OLD)
  endif()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/isaac_ros_depth_image_proc_point_cloud_xyz_pol.py)
  add_launch_test(test/isaac_ros_depth_image_proc_point_cloud_xyz_OSS_compare.py)
  add_launch_test(test/isaac_ros_depth_image_proc_point_cloud_xyzrgb_pol.py)
  add_launch_test(test/isaac_ros_depth_image_proc_point_cloud_xyzrgb_OSS_compare.py)
  add_launch_test(test/isaac_ros_depth_image_proc_convert_metric_pol.py)
endif()

ament_auto_package(INSTALL_TO_SHARE launch config)
