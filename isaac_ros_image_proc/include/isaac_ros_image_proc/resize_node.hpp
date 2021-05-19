/**
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef ISAAC_ROS_IMAGE_PROC__RESIZE_NODE_HPP_
#define ISAAC_ROS_IMAGE_PROC__RESIZE_NODE_HPP_

#include <string>

#include "image_transport/image_transport.hpp"
#include "rclcpp/rclcpp.hpp"
#include "vpi/VPI.h"

namespace isaac_ros
{
namespace image_proc
{

class ResizeNode : public rclcpp::Node
{
public:
  explicit ResizeNode(const rclcpp::NodeOptions &);

private:
  /**
   * @brief Callback to resize the image received by subscription
   *
   * @param image_msg The image message received
   * @param info_msg The information of the camera that produced the image
   */
  void ResizeCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr & image_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & info_msg);

  // ROS2 Camera subscriber for input and Image publisher for output
  image_transport::CameraSubscriber sub_;
  image_transport::CameraPublisher pub_;

  // ROS2 parameters for configuring the resize operation
  bool use_relative_scale_{};
  double scale_height_{};
  double scale_width_{};
  int height_{};
  int width_{};

  // ROS2 parameter for VPI backend flags
  uint32_t vpi_backends_{};
};

}  // namespace image_proc
}  // namespace isaac_ros

#endif  // ISAAC_ROS_IMAGE_PROC__RESIZE_NODE_HPP_
