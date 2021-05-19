/**
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef ISAAC_ROS_IMAGE_PROC__IMAGE_FORMAT_CONVERTER_NODE_HPP_
#define ISAAC_ROS_IMAGE_PROC__IMAGE_FORMAT_CONVERTER_NODE_HPP_

#include <string>

#include "image_transport/image_transport.hpp"
#include "rclcpp/rclcpp.hpp"

namespace isaac_ros
{
namespace image_proc
{

class ImageFormatConverterNode : public rclcpp::Node
{
public:
  explicit ImageFormatConverterNode(const rclcpp::NodeOptions &);

private:
  /**
   * @brief Callback to change the format of the image received by subscription
   *
   * @param image_msg The image message received
   */
  void FormatCallback(const sensor_msgs::msg::Image::ConstSharedPtr & image_msg);

  // ROS2 Image subscriber for input and Image publisher for output
  image_transport::Subscriber sub_;
  image_transport::Publisher pub_;

  // ROS2 parameter for specifying desired encoding
  std::string encoding_desired_{};

  // ROS2 parameter for VPI backend flags
  uint32_t vpi_backends_{};
};

}  // namespace image_proc
}  // namespace isaac_ros

#endif  // ISAAC_ROS_IMAGE_PROC__IMAGE_FORMAT_CONVERTER_NODE_HPP_
