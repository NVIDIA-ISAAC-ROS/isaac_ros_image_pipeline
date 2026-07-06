// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef ISAAC_ROS_IMAGE_PROC__PAD_NODE_HPP_
#define ISAAC_ROS_IMAGE_PROC__PAD_NODE_HPP_

#include <memory>
#include <string>
#include <vector>

#include "cvcuda/OpCopyMakeBorder.hpp"
#include "rclcpp/rclcpp.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "isaac_ros_nitros/types/cuda_memory_pool.hpp"
#include "nvcv/BorderType.h"

namespace nvidia
{
namespace isaac_ros
{
namespace image_proc
{

enum class PaddingType
{
  kCenter,
  kTopLeft,
  kTopRight,
  kBottomLeft,
  kBottomRight
};

class PadNode : public rclcpp::Node
{
public:
  explicit PadNode(const rclcpp::NodeOptions & options);

  ~PadNode();
  PadNode(const PadNode &) = delete;
  PadNode & operator=(const PadNode &) = delete;

private:
  void imageSubCallback(const nvidia::isaac_ros::nitros::NitrosImage::SharedPtr msg);

  // QoS settings
  rclcpp::QoS input_qos_;
  rclcpp::QoS output_qos_;

  // Subscription to input NitrosImage messages
  rclcpp::Subscription<nvidia::isaac_ros::nitros::NitrosImage>::SharedPtr image_sub_;

  // Publisher for output NitrosImage messages
  rclcpp::Publisher<nvidia::isaac_ros::nitros::NitrosImage>::SharedPtr image_pub_;

  // Pad node parameters
  const uint16_t output_image_width_;
  const uint16_t output_image_height_;
  const std::string padding_type_;
  const std::string border_type_;
  // Param to store the channel values for each pixel for border.
  // Needed for CENTER CONSTANT padding
  const std::vector<double> border_pixel_color_value_;
  const int64_t memory_pool_block_size_;
  const int64_t memory_pool_num_blocks_;

  PaddingType padding_type_val_;
  NVCVBorderType border_type_val_;
  std::vector<float> border_values_float_;
  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;
  nvidia::isaac_ros::nitros::CUDAMemoryPool pool_;

  // Pad node CVCUDA operation
  cvcuda::CopyMakeBorder make_border_op_;
};

}  // namespace image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_IMAGE_PROC__PAD_NODE_HPP_
