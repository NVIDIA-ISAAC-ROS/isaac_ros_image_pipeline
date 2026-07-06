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

#ifndef ISAAC_ROS_IMAGE_PROC__ALPHA_BLEND_NODE_HPP_
#define ISAAC_ROS_IMAGE_PROC__ALPHA_BLEND_NODE_HPP_

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

#include <memory>

#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_image_proc/alpha_blend.cu.hpp"
#include "isaac_ros_nitros/types/cuda_memory_pool.hpp"
#include "isaac_ros_nitros/types/nitros_type_message_filter_traits.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "rclcpp/rclcpp.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace image_proc
{

class AlphaBlendNode : public rclcpp::Node
{
public:
  explicit AlphaBlendNode(const rclcpp::NodeOptions & options);

  ~AlphaBlendNode();

private:
  // Callback function
  void InputCallback(
    const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & img_ptr,
    const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & mask_ptr);

  // Alpha blend node parameters
  double alpha_;
  int memory_pool_block_size_;
  int memory_pool_num_blocks_;
  int64_t input_queue_size_;
  int64_t output_queue_size_;

  // Subscribers and publishers
  message_filters::Subscriber<nvidia::isaac_ros::nitros::NitrosImage> image_sub_;
  message_filters::Subscriber<nvidia::isaac_ros::nitros::NitrosImage> mask_sub_;
  rclcpp::Publisher<nvidia::isaac_ros::nitros::NitrosImage>::SharedPtr image_pub_;

  // Exact message sync policy
  using ExactPolicy = ::message_filters::sync_policies::ExactTime<
    nvidia::isaac_ros::nitros::NitrosImage, nvidia::isaac_ros::nitros::NitrosImage>;
  message_filters::Synchronizer<ExactPolicy> sync_;

  // Resources
  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;
  nvidia::isaac_ros::nitros::CUDAMemoryPool pool_;
};

}  // namespace image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_IMAGE_PROC__ALPHA_BLEND_NODE_HPP_
