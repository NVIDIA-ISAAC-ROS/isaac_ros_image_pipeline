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
#ifndef ISAAC_ROS_IMAGE_PROC__CROP_NODE_HPP_
#define ISAAC_ROS_IMAGE_PROC__CROP_NODE_HPP_

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

enum class CropMode
{
  kCenter, kLeft, kRight, kTop, kBottom,
  kTopLeft, kTopRight, kBottomLeft, kBottomRight, kBBox
};

struct BBox
{
  size_t top_left_x;
  size_t top_left_y;
  size_t width;
  size_t height;
};

class CropNode : public nitros::NitrosNode
{
public:
  explicit CropNode(const rclcpp::NodeOptions &);

  ~CropNode();

  CropNode(const CropNode &) = delete;

  CropNode & operator=(const CropNode &) = delete;

  // The callback to be implemented by users for any required initialization
  void postLoadGraphCallback() override;

private:
  void CalculateResizeAndCropParams(const CropMode & crop_mode);
  uint16_t GetResizeScalar();
  // Node parameters
  const int64_t input_width_;
  const int64_t input_height_;
  const int64_t crop_width_;
  const int64_t crop_height_;
  int64_t num_blocks_;
  const std::string crop_mode_;
  BBox roi_;
};

}  // namespace image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_IMAGE_PROC__CROP_NODE_HPP_
