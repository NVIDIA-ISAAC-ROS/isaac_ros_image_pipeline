// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef ISAAC_ROS_DEPTH_IMAGE_PROC__CONVERT_METRIC_NODE_HPP_
#define ISAAC_ROS_DEPTH_IMAGE_PROC__CONVERT_METRIC_NODE_HPP_

#include <string>
#include <memory>

#include "cvcuda/OpConvertTo.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "isaac_ros_nitros/types/cuda_memory_pool.hpp"
#include "rclcpp/rclcpp.hpp"

#include "cuda_runtime.h" // NOLINT

namespace nvidia
{
namespace isaac_ros
{
namespace depth_image_proc
{

class ConvertMetricNode : public rclcpp::Node
{
public:
  explicit ConvertMetricNode(const rclcpp::NodeOptions options = rclcpp::NodeOptions());
  ~ConvertMetricNode();

private:
  void DepthCallback(const nvidia::isaac_ros::nitros::NitrosImage::SharedPtr msg);

  // ROS Node parameters
  const int64_t memory_pool_block_size_;
  const int64_t memory_pool_num_blocks_;
  const uint16_t input_queue_size_;
  const uint16_t output_queue_size_;

  rclcpp::Subscription<nvidia::isaac_ros::nitros::NitrosImage>::SharedPtr image_sub_;
  rclcpp::Publisher<nvidia::isaac_ros::nitros::NitrosImage>::SharedPtr image_pub_;

  // CUDA resources
  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;
  nvidia::isaac_ros::nitros::CUDAMemoryPool pool_;

  // CVCUDA operation
  cvcuda::ConvertTo convert_op_;
};

}  // namespace depth_image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_DEPTH_IMAGE_PROC__CONVERT_METRIC_NODE_HPP_
