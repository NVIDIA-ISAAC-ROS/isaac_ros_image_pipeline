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

cmake_minimum_required(VERSION 3.8)
project(isaac_ros_image_proc)

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

# image_format_converter_node
ament_auto_add_library(image_format_converter_node SHARED src/image_format_converter_node.cpp)
target_compile_definitions(image_format_converter_node
  PRIVATE "COMPOSITION_BUILDING_DLL"
)
target_link_libraries(image_format_converter_node vpi)
rclcpp_components_register_nodes(image_format_converter_node "nvidia::isaac_ros::image_proc::ImageFormatConverterNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::image_proc::ImageFormatConverterNode;$<TARGET_FILE:image_format_converter_node>\n")

# rectify_node
ament_auto_add_library(rectify_node SHARED src/rectify_node.cpp)
target_compile_definitions(rectify_node
  PRIVATE "COMPOSITION_BUILDING_DLL"
)
target_link_libraries(rectify_node vpi)
rclcpp_components_register_nodes(rectify_node "nvidia::isaac_ros::image_proc::RectifyNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::image_proc::RectifyNode;$<TARGET_FILE:rectify_node>\n")

# resize_node
ament_auto_add_library(resize_node SHARED src/resize_node.cpp)
target_compile_definitions(resize_node
  PRIVATE "COMPOSITION_BUILDING_DLL"
)
target_link_libraries(resize_node vpi)
rclcpp_components_register_nodes(resize_node "nvidia::isaac_ros::image_proc::ResizeNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::image_proc::ResizeNode;$<TARGET_FILE:resize_node>\n")

# flip_node
ament_auto_add_library(flip_node SHARED src/image_flip_node.cpp)
target_compile_definitions(flip_node
  PRIVATE "COMPOSITION_BUILDING_DLL"
)
target_link_libraries(flip_node vpi)
rclcpp_components_register_nodes(flip_node "nvidia::isaac_ros::image_proc::ImageFlipNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::image_proc::ImageFlipNode;$<TARGET_FILE:flip_node>\n")

# Install config directory
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

set_target_properties(resize_node PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
install(TARGETS resize_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

set_target_properties(rectify_node PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
install(TARGETS rectify_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

set_target_properties(image_format_converter_node PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
install(TARGETS image_format_converter_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# isaac_ros_image_proc executable
ament_auto_add_executable(${PROJECT_NAME}
  src/image_proc_main.cpp
)
target_link_libraries(${PROJECT_NAME}
  image_format_converter_node
  rectify_node
  resize_node
  ament_index_cpp::ament_index_cpp
  vpi
)

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

ament_auto_package(INSTALL_TO_SHARE launch)
