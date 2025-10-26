// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef ISAAC_ROS_STEREO_IMAGE_PROC__DISPARITY_NODE_HPP_
#define ISAAC_ROS_STEREO_IMAGE_PROC__DISPARITY_NODE_HPP_

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

class DisparityNode : public nitros::NitrosNode
{
public:
  explicit DisparityNode(const rclcpp::NodeOptions &);

  ~DisparityNode();

  DisparityNode(const DisparityNode &) = delete;

  DisparityNode & operator=(const DisparityNode &) = delete;

  // The callback to be implemented by users for any required initialization
  void preLoadGraphCallback() override;
  void postLoadGraphCallback() override;

private:
  // Disparity node parameters
  const std::string vpi_backend_;
  const float max_disparity_;
  const int confidence_threshold_;
  const int confidence_type_;
  const int window_size_;
  const int num_passes_;
  const int p1_;
  const int p2_;
  const int p2_alpha_;
  const int quality_;
};

}  // namespace stereo_image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_STEREO_IMAGE_PROC__DISPARITY_NODE_HPP_
