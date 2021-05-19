/**
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <memory>

#include "rclcpp/rclcpp.hpp"

#include "isaac_ros_image_proc/image_format_converter_node.hpp"
#include "isaac_ros_image_proc/rectify_node.hpp"

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  rclcpp::executors::MultiThreadedExecutor exec;

  // Make the raw images available in grayscale from /image_raw to /image_mono
  rclcpp::NodeOptions format_mono_options;
  format_mono_options.arguments(
  {
    "--ros-args",
    "-r", "__node:=image_format_mono",
    "-r", "image:=image_mono",
    "-p", "encoding_desired:=mono8"
  });
  auto format_mono_node =
    std::make_shared<isaac_ros::image_proc::ImageFormatConverterNode>(format_mono_options);
  exec.add_node(format_mono_node);

  // Also make the raw images available in color from /image_raw to /image_color
  rclcpp::NodeOptions format_color_options;
  format_color_options.arguments(
  {
    "--ros-args",
    "-r", "__node:=image_format_color",
    "-r", "image:=image_color"
  });
  auto format_color_node =
    std::make_shared<isaac_ros::image_proc::ImageFormatConverterNode>(format_color_options);
  exec.add_node(format_color_node);

  // Rectify and undistort grayscale images from /image_mono to /image_rect
  rclcpp::NodeOptions rectify_mono_options;
  rectify_mono_options.arguments(
  {
    "--ros-args",
    "-r", "__node:=rectify_mono",
    "-r", "image:=image_mono",
    "-r", "image_rect:=image_rect"
  });
  auto rectify_mono_node = std::make_shared<isaac_ros::image_proc::RectifyNode>(
    rectify_mono_options);
  exec.add_node(rectify_mono_node);

  // Also rectify and undistort color images from /image_color to /image_rect_color
  rclcpp::NodeOptions rectify_color_options;
  rectify_color_options.arguments(
  {
    "--ros-args",
    "-r", "__node:=rectify_color",
    "-r", "image:=image_color",
    "-r", "image_rect:=image_rect_color"
  });
  auto rectify_color_node = std::make_shared<isaac_ros::image_proc::RectifyNode>(
    rectify_color_options);
  exec.add_node(rectify_color_node);

  // Spin with all the components loaded
  exec.spin();

  rclcpp::shutdown();
  return 0;
}
