# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

cmake_minimum_required(VERSION 3.5)
project(isaac_ros_image_proc)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Find VPI dependency
find_package(vpi REQUIRED)

# image_format_converter_node
ament_auto_add_library(image_format_converter_node SHARED
  src/image_format_converter_node.cpp)
target_compile_definitions(image_format_converter_node
  PRIVATE "COMPOSITION_BUILDING_DLL"
)
target_link_libraries(image_format_converter_node vpi)
rclcpp_components_register_nodes(image_format_converter_node "isaac_ros::image_proc::ImageFormatConverterNode")
set(node_plugins "${node_plugins}isaac_ros::image_proc::ImageFormatConverterNode;$<TARGET_FILE:image_format_converter_node>\n")

# rectify_node
ament_auto_add_library(rectify_node SHARED
  src/rectify_node.cpp)
target_compile_definitions(rectify_node
  PRIVATE "COMPOSITION_BUILDING_DLL"
)
target_link_libraries(rectify_node vpi)
rclcpp_components_register_nodes(rectify_node "isaac_ros::image_proc::RectifyNode")
set(node_plugins "${node_plugins}isaac_ros::image_proc::RectifyNode;$<TARGET_FILE:rectify_node>\n")

# resize_node
ament_auto_add_library(resize_node SHARED
  src/resize_node.cpp)
target_compile_definitions(resize_node
  PRIVATE "COMPOSITION_BUILDING_DLL"
)
target_link_libraries(resize_node vpi)
rclcpp_components_register_nodes(resize_node "isaac_ros::image_proc::ResizeNode")
set(node_plugins "${node_plugins}isaac_ros::image_proc::ResizeNode;$<TARGET_FILE:resize_node>\n")

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

install(TARGETS ${PROJECT_NAME}
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
  add_launch_test(test/isaac_ros_image_proc_test.py)
  add_launch_test(test/isaac_ros_image_format_converter_test.py)
  add_launch_test(test/isaac_ros_image_format_converter_4_channel_test.py)
  add_launch_test(test/isaac_ros_image_format_converter_grayscale_test.py)
  add_launch_test(test/isaac_ros_rectify_test.py)
  add_launch_test(test/isaac_ros_resize_test.py)
  add_launch_test(test/isaac_ros_resize_aspect_ratio_test.py)
  add_launch_test(test/isaac_ros_resize_invalid_test.py)

endif()

ament_auto_package()
