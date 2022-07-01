/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef ISAAC_ROS_STEREO_IMAGE_PROC__POINT_CLOUD_NODE_HPP_
#define ISAAC_ROS_STEREO_IMAGE_PROC__POINT_CLOUD_NODE_HPP_

#include <string>
#include <chrono>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "isaac_ros_nitros/nitros_node.hpp"

using StringList = std::vector<std::string>;

namespace nvidia
{
namespace isaac_ros
{
namespace stereo_image_proc
{

class PointCloudNode : public nitros::NitrosNode
{
public:
  explicit PointCloudNode(const rclcpp::NodeOptions &);

  ~PointCloudNode();

  PointCloudNode(const PointCloudNode &) = delete;

  PointCloudNode & operator=(const PointCloudNode &) = delete;

  // The callback to be implemented by users for any required initialization
  void preLoadGraphCallback() override;
  void postLoadGraphCallback() override;

private:
  // Point cloud node parameters
  bool use_color_;
  float unit_scaling_;
};

}  // namespace stereo_image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_STEREO_IMAGE_PROC__POINT_CLOUD_NODE_HPP_
