# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(isaac_ros_image_proc LANGUAGES C CXX CUDA)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Find VPI dependency
find_package(vpi REQUIRED)

## Enable CUDA
#find_package(CUDAToolkit REQUIRED)
#enable_language(CUDA)
#
# alpha blend node
ament_auto_add_library(alpha_blend_node SHARED src/alpha_blend_node.cpp src/alpha_blend.cu.cpp)
rclcpp_components_register_nodes(alpha_blend_node "nvidia::isaac_ros::image_proc::AlphaBlendNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::image_proc::AlphaBlendNode;$<TARGET_FILE:alpha_blend_node>\n")
set_target_properties(alpha_blend_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)
# Mark as CUDA files with non-standard extensions
set_source_files_properties(
  src/alpha_blend.cu.cpp
  include/isaac_ros_image_proc/alpha_blend.cu.hpp
  PROPERTIES LANGUAGE CUDA
)
# Disable -Wpedantic flag
get_target_property(target_options alpha_blend_node COMPILE_OPTIONS)
list(REMOVE_ITEM target_options "-Wpedantic")
set_property(TARGET alpha_blend_node PROPERTY COMPILE_OPTIONS ${target_options})

# image_format_converter_node
ament_auto_add_library(image_format_converter_node SHARED src/image_format_converter_node.cpp)
target_link_libraries(image_format_converter_node vpi)
rclcpp_components_register_nodes(image_format_converter_node "nvidia::isaac_ros::image_proc::ImageFormatConverterNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::image_proc::ImageFormatConverterNode;$<TARGET_FILE:image_format_converter_node>\n")
set_target_properties(image_format_converter_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

# rectify_node
ament_auto_add_library(rectify_node SHARED src/rectify_node.cpp)
target_link_libraries(rectify_node vpi)
rclcpp_components_register_nodes(rectify_node "nvidia::isaac_ros::image_proc::RectifyNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::image_proc::RectifyNode;$<TARGET_FILE:rectify_node>\n")
set_target_properties(rectify_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

# resize_node
ament_auto_add_library(resize_node SHARED src/resize_node.cpp)
target_link_libraries(resize_node vpi)
rclcpp_components_register_nodes(resize_node "nvidia::isaac_ros::image_proc::ResizeNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::image_proc::ResizeNode;$<TARGET_FILE:resize_node>\n")
set_target_properties(resize_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

# flip_node
ament_auto_add_library(flip_node SHARED src/image_flip_node.cpp)
target_link_libraries(flip_node vpi)
rclcpp_components_register_nodes(flip_node "nvidia::isaac_ros::image_proc::ImageFlipNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::image_proc::ImageFlipNode;$<TARGET_FILE:flip_node>\n")
set_target_properties(flip_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

# crop_node
ament_auto_add_library(crop_node SHARED src/crop_node.cpp)
target_link_libraries(crop_node vpi)
rclcpp_components_register_nodes(crop_node "nvidia::isaac_ros::image_proc::CropNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::image_proc::CropNode;$<TARGET_FILE:crop_node>\n")
set_target_properties(crop_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

# pad_node
ament_auto_add_library(pad_node SHARED src/pad_node.cpp)
rclcpp_components_register_nodes(pad_node "nvidia::isaac_ros::image_proc::PadNode")
target_link_libraries(pad_node cvcuda nvcv_types)
set(node_plugins "${node_plugins}nvidia::isaac_ros::image_proc::PadNode;$<TARGET_FILE:pad_node>\n")

# image_normalize_node
ament_auto_add_library(image_normalize_node SHARED src/image_normalize_node.cpp)
  rclcpp_components_register_nodes(image_normalize_node
    "nvidia::isaac_ros::image_proc::ImageNormalizeNode")
  set(node_plugins
    "${node_plugins}nvidia::isaac_ros::image_proc::ImageNormalizeNode;$<TARGET_FILE:image_normalize_node>\n")
  target_link_libraries(image_normalize_node cvcuda nvcv_types)
  set_target_properties(image_normalize_node PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    BUILD_RPATH_USE_ORIGIN TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE)

set_target_properties(pad_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  # Gtest for alpha blend node
  ament_add_gtest(alpha_blend_test test/alpha_blend_test.cpp)
  target_link_libraries(alpha_blend_test alpha_blend_node)
  target_include_directories(alpha_blend_node PUBLIC include/isaac_ros_image_proc/)
  ament_target_dependencies(alpha_blend_node rclcpp)
  ament_target_dependencies(alpha_blend_node isaac_ros_nitros)

  # Gtest for image format converter node
  ament_add_gtest(image_format_converter_node_test test/image_format_converter_node_test.cpp)
  target_link_libraries(image_format_converter_node_test image_format_converter_node)
  target_include_directories(image_format_converter_node_test PUBLIC include/isaac_ros_image_proc/)
  ament_target_dependencies(image_format_converter_node_test rclcpp)
  ament_target_dependencies(image_format_converter_node_test isaac_ros_nitros)

  # Gtest for crop node
  ament_add_gtest(crop_node_test test/crop_node_test.cpp)
  target_link_libraries(crop_node_test crop_node)
  target_include_directories(crop_node_test PUBLIC include/isaac_ros_image_proc/)
  ament_target_dependencies(crop_node_test rclcpp)
  ament_target_dependencies(crop_node_test isaac_ros_nitros)

  # Gtest for pad node
  ament_add_gtest(pad_node_test test/pad_node_test.cpp)
  target_link_libraries(pad_node_test pad_node)
  target_include_directories(pad_node_test PUBLIC include/isaac_ros_image_proc/)
  ament_target_dependencies(pad_node_test rclcpp)
  ament_target_dependencies(pad_node_test isaac_ros_nitros)

  # Gtest for resize node
  ament_add_gtest(resize_node_test test/resize_node_test.cpp)
  target_link_libraries(resize_node_test resize_node)
  target_include_directories(resize_node_test PUBLIC include/isaac_ros_image_proc/)
  ament_target_dependencies(resize_node_test rclcpp)
  ament_target_dependencies(resize_node_test isaac_ros_nitros)

  # The FindPythonInterp and FindPythonLibs modules are removed
  if(POLICY CMP0148)
    cmake_policy(SET CMP0148 OLD)
  endif()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/isaac_ros_alpha_blend_bgr_test.py)
  add_launch_test(test/isaac_ros_alpha_blend_mono_test.py)
  add_launch_test(test/isaac_ros_image_format_converter_test.py)
  add_launch_test(test/isaac_ros_image_format_converter_4_channel_test.py)
  add_launch_test(test/isaac_ros_image_format_converter_grayscale_test.py)
  add_launch_test(test/isaac_ros_image_format_converter_nv24_test.py)
  add_launch_test(test/isaac_ros_rectify_test.py)
  add_launch_test(test/isaac_ros_rectify_timestamp_match_test.py)
  add_launch_test(test/isaac_ros_rectify_oss_comparison_test.py)
  add_launch_test(test/isaac_ros_rectify_stereo_epipolar_test.py)
  add_launch_test(test/isaac_ros_resize_test.py)
  add_launch_test(test/isaac_ros_resize_bgra_test.py)
  add_launch_test(test/isaac_ros_resize_invalid_test.py)
  add_launch_test(test/isaac_ros_resize_rectify_stereo_epipolar_test.py)
  add_launch_test(test/isaac_ros_image_flip_180_test.py)
  add_launch_test(test/isaac_ros_image_flip_hori_test.py)
  add_launch_test(test/isaac_ros_image_flip_vert_test.py)
  add_launch_test(test/isaac_ros_crop_bottom_left_test.py)
  add_launch_test(test/isaac_ros_crop_bottom_right_test.py)
  add_launch_test(test/isaac_ros_crop_bottom_test.py)
  add_launch_test(test/isaac_ros_crop_center_test.py)
  add_launch_test(test/isaac_ros_crop_left_test.py)
  add_launch_test(test/isaac_ros_crop_right_test.py)
  add_launch_test(test/isaac_ros_crop_top_left_test.py)
  add_launch_test(test/isaac_ros_crop_top_right_test.py)
  add_launch_test(test/isaac_ros_crop_top_test.py)
  add_launch_test(test/isaac_ros_pad_node_bottom_left_pad_test.py)
  add_launch_test(test/isaac_ros_pad_node_bottom_right_pad_test.py)
  add_launch_test(test/isaac_ros_pad_node_top_left_pad_test.py)
  add_launch_test(test/isaac_ros_pad_node_top_right_pad_test.py)
  add_launch_test(test/isaac_ros_pad_node_center_pad_test.py)
  add_launch_test(test/isaac_ros_image_normalize_test.py)
  add_launch_test(test/isaac_ros_wide_fov_test.py)
endif()


# Embed versioning information into installed files
ament_index_get_resource(ISAAC_ROS_COMMON_CMAKE_PATH isaac_ros_common_cmake_path isaac_ros_common)
include("${ISAAC_ROS_COMMON_CMAKE_PATH}/isaac_ros_common-version-info.cmake")
generate_version_info(${PROJECT_NAME})

ament_auto_package(INSTALL_TO_SHARE launch config)
