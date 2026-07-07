// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <string>
#include <chrono>
#include <utility>
#include <vector>

#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_nitros/types/cuda_memory_pool.hpp"
#include "isaac_ros_nitros_disparity_image_type/nitros_disparity_image.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "rclcpp/rclcpp.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace stereo_image_proc
{

class DisparityToDepthNode : public rclcpp::Node
{
public:
  explicit DisparityToDepthNode(const rclcpp::NodeOptions & options);

  ~DisparityToDepthNode();

  DisparityToDepthNode(const DisparityToDepthNode &) = delete;

  DisparityToDepthNode & operator=(const DisparityToDepthNode &) = delete;

private:
  void DisparityToDepthCallback(
    const nvidia::isaac_ros::nitros::NitrosDisparityImage::ConstSharedPtr & disparity_msg);

  // Parameters
  const int64_t memory_pool_block_size_;
  const int64_t memory_pool_num_blocks_;
  rclcpp::QoS input_qos_;
  rclcpp::QoS output_qos_;

  // Subscribers and publishers
  rclcpp::Subscription<nvidia::isaac_ros::nitros::NitrosDisparityImage>::SharedPtr disparity_sub_;
  rclcpp::Publisher<nvidia::isaac_ros::nitros::NitrosImage>::SharedPtr depth_pub_;

  // Resources
  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;
  nvidia::isaac_ros::nitros::CUDAMemoryPool pool_;
};

}  // namespace stereo_image_proc
}  // namespace isaac_ros
}  // namespace nvidia
