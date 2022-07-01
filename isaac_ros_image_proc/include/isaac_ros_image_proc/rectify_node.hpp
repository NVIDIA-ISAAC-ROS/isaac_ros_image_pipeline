/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#ifndef ISAAC_ROS_IMAGE_PROC__RECTIFY_NODE_HPP_
#define ISAAC_ROS_IMAGE_PROC__RECTIFY_NODE_HPP_

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

class RectifyNode : public nitros::NitrosNode
{
public:
  explicit RectifyNode(const rclcpp::NodeOptions &);

  ~RectifyNode();

  RectifyNode(const RectifyNode &) = delete;

  RectifyNode & operator=(const RectifyNode &) = delete;

  // The callback to be implemented by users for any required initialization
  void preLoadGraphCallback() override;
  void postLoadGraphCallback() override;

private:
  int16_t output_width_;
  int16_t output_height_;
};

}  // namespace image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_IMAGE_PROC__RECTIFY_NODE_HPP_
