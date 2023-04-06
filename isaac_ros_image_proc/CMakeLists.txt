# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.23.2)
project(isaac_ros_image_proc LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Find VPI dependency
find_package(vpi REQUIRED)

# image_format_converter_node
ament_auto_add_library(image_format_converter_node SHARED src/image_format_converter_node.cpp)
target_link_libraries(image_format_converter_node vpi)
rclcpp_components_register_nodes(image_format_converter_node "nvidia::isaac_ros::image_proc::ImageFormatConverterNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::image_proc::ImageFormatConverterNode;$<TARGET_FILE:image_format_converter_node>\n")
set_target_properties(image_format_converter_node PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)

# rectify_node
ament_auto_add_library(rectify_node SHARED src/rectify_node.cpp)
target_link_libraries(rectify_node vpi)
rclcpp_components_register_nodes(rectify_node "nvidia::isaac_ros::image_proc::RectifyNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::image_proc::RectifyNode;$<TARGET_FILE:rectify_node>\n")
set_target_properties(rectify_node PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)

# resize_node
ament_auto_add_library(resize_node SHARED src/resize_node.cpp)
target_link_libraries(resize_node vpi)
rclcpp_components_register_nodes(resize_node "nvidia::isaac_ros::image_proc::ResizeNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::image_proc::ResizeNode;$<TARGET_FILE:resize_node>\n")
set_target_properties(resize_node PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)

# flip_node
ament_auto_add_library(flip_node SHARED src/image_flip_node.cpp)
target_link_libraries(flip_node vpi)
rclcpp_components_register_nodes(flip_node "nvidia::isaac_ros::image_proc::ImageFlipNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::image_proc::ImageFlipNode;$<TARGET_FILE:flip_node>\n")

# isaac_ros_image_proc executable
ament_auto_add_executable(${PROJECT_NAME} src/image_proc_main.cpp)
target_link_libraries(${PROJECT_NAME}
  image_format_converter_node
  rectify_node
  resize_node
  ament_index_cpp::ament_index_cpp
  vpi
)

### Install extensions built from source

# ImageFlip
add_subdirectory(gxf/image_flip)
install(TARGETS gxf_image_flip DESTINATION share/${PROJECT_NAME}/gxf/lib/image_proc)

# TensorOps
add_subdirectory(gxf/tensorops)
install(TARGETS gxf_tensorops DESTINATION share/${PROJECT_NAME}/gxf/lib/image_proc)

### End extensions

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/isaac_ros_image_proc_test.py)
  add_launch_test(test/isaac_ros_image_format_converter_test.py)
  add_launch_test(test/isaac_ros_image_format_converter_4_channel_test.py)
  add_launch_test(test/isaac_ros_image_format_converter_grayscale_test.py)
  add_launch_test(test/isaac_ros_image_format_converter_nv24_test.py)
  add_launch_test(test/isaac_ros_rectify_test.py)
  add_launch_test(test/isaac_ros_resize_test.py)
  add_launch_test(test/isaac_ros_resize_invalid_test.py)
  add_launch_test(test/isaac_ros_image_flip_180_test.py)
  add_launch_test(test/isaac_ros_image_flip_hori_test.py)
  add_launch_test(test/isaac_ros_image_flip_vert_test.py)

endif()

ament_auto_package(INSTALL_TO_SHARE launch config)
