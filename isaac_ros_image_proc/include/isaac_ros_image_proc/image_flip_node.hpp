// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef ISAAC_ROS_IMAGE_PROC__IMAGE_FLIP_NODE_HPP_
#define ISAAC_ROS_IMAGE_PROC__IMAGE_FLIP_NODE_HPP_

#include <string>

#include "rclcpp/rclcpp.hpp"
#include "isaac_ros_nitros/nitros_node.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace image_proc
{

class ImageFlipNode : public nitros::NitrosNode
{
public:
  explicit ImageFlipNode(const rclcpp::NodeOptions &);

  ~ImageFlipNode() = default;

  ImageFlipNode(const ImageFlipNode &) = delete;

  ImageFlipNode & operator=(const ImageFlipNode &) = delete;

  // The callback to be implemented by users for any required initialization
  void postLoadGraphCallback() override;

private:
  const std::string flip_mode_;
};

}  // namespace image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_IMAGE_PROC__IMAGE_FLIP_NODE_HPP_
