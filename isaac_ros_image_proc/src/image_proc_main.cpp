// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

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
    std::make_shared<nvidia::isaac_ros::image_proc::ImageFormatConverterNode>(format_mono_options);
  exec.add_node(format_mono_node);

  // Also make the raw images available in color from /image_raw to /image_color
  rclcpp::NodeOptions format_color_options;
  format_color_options.arguments(
  {
    "--ros-args",
    "-r", "__node:=image_format_color",
    "-r", "image:=image_color",
    "-p", "encoding_desired:=rgb8"
  });
  auto format_color_node =
    std::make_shared<nvidia::isaac_ros::image_proc::ImageFormatConverterNode>(format_color_options);
  exec.add_node(format_color_node);

  // Also rectify and undistort color images from /image_color to /image_rect_color
  rclcpp::NodeOptions rectify_color_options;
  rectify_color_options.arguments(
  {
    "--ros-args",
    "-r", "__node:=rectify_color",
    "-r", "image_raw:=image_color",
    "-r", "image_rect:=image_rect_color",
    "-p", "encoding_desired:=rgb8",
    "-p", "output_width:=640",
    "-p", "output_height:=480"
  });
  auto rectify_color_node = std::make_shared<nvidia::isaac_ros::image_proc::RectifyNode>(
    rectify_color_options);
  exec.add_node(rectify_color_node);

  // Spin with all the components loaded
  exec.spin();

  rclcpp::shutdown();
  return 0;
}
