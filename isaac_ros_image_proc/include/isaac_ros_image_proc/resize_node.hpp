/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
#include <chrono>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "isaac_ros_nitros/nitros_node.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace image_proc
{

class ResizeNode : public nitros::NitrosNode
{
public:
  explicit ResizeNode(const rclcpp::NodeOptions &);

  ~ResizeNode();

  ResizeNode(const ResizeNode &) = delete;

  ResizeNode & operator=(const ResizeNode &) = delete;

  // The callback to be implemented by users for any required initialization
  void postLoadGraphCallback() override;

private:
  // Resize node parameters
  int64_t output_width_;
  int64_t output_height_;
  bool keep_aspect_ratio_;
};

}  // namespace image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_IMAGE_PROC__RESIZE_NODE_HPP_
