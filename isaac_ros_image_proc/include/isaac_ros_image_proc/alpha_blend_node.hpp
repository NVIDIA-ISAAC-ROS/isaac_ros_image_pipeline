// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_image_proc/alpha_blend.cu.hpp"

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_message_filters_subscriber.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_subscriber.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_publisher.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "isaac_ros_nitros_image_type/nitros_image_view.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace image_proc
{

class AlphaBlendNode : public rclcpp::Node
{
public:
  explicit AlphaBlendNode(const rclcpp::NodeOptions options = rclcpp::NodeOptions());

  ~AlphaBlendNode();

private:
  cudaStream_t stream_;
  double alpha_;
  int mask_queue_size_;
  int image_queue_size_;
  int sync_queue_size_;

  // Subscribers for input images
  nvidia::isaac_ros::nitros::message_filters::Subscriber<nvidia::isaac_ros::nitros::NitrosImageView>
  mask_sub_;
  nvidia::isaac_ros::nitros::message_filters::Subscriber<nvidia::isaac_ros::nitros::NitrosImageView>
  image_sub_;

  // Publisher for output image
  std::shared_ptr<
    nvidia::isaac_ros::nitros::ManagedNitrosPublisher<nvidia::isaac_ros::nitros::NitrosImage>>
  image_pub_;

  // Exact message sync policy
  using ExactPolicyMode = ::message_filters::sync_policies::ExactTime<
    nvidia::isaac_ros::nitros::NitrosImage, nvidia::isaac_ros::nitros::NitrosImage>;
  using ExactSyncMode = ::message_filters::Synchronizer<ExactPolicyMode>;
  std::shared_ptr<ExactSyncMode> sync_mode_;

  // Callback function
  void InputCallback(
    const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & mask_ptr,
    const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & img_ptr);
};

}  // namespace image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_IMAGE_PROC__ALPHA_BLEND_NODE_HPP_
