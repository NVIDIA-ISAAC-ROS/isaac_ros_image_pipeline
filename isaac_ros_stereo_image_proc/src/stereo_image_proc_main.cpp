/**
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <rclcpp/rclcpp.hpp>
#include <memory>
#include "isaac_ros_stereo_image_proc/disparity_node.hpp"
#include "isaac_ros_stereo_image_proc/point_cloud_node.hpp"


int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  rclcpp::executors::SingleThreadedExecutor exec;

  // Find the disparity image from the left and right rectified images
  rclcpp::NodeOptions disparity_options;
  auto disparity_node = std::make_shared<isaac_ros::stereo_image_proc::DisparityNode>(
    disparity_options);
  exec.add_node(disparity_node);

  // Find the pointcloud from the disparity image
  rclcpp::NodeOptions point_cloud_options;
  point_cloud_options.arguments(
  {
    "--ros-args",
    "-r", "left/image_rect_color:=left/image_rect"
  });
  auto point_cloud_node = std::make_shared<isaac_ros::stereo_image_proc::PointCloudNode>(
    point_cloud_options);
  exec.add_node(point_cloud_node);

  // Spin with all the components loaded
  exec.spin();

  rclcpp::shutdown();
  return 0;
}
