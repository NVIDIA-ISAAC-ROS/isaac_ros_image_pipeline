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

#ifndef ISAAC_ROS_DEPTH_IMAGE_PROC__CONVERT_METRIC_NODE_HPP_
#define ISAAC_ROS_DEPTH_IMAGE_PROC__CONVERT_METRIC_NODE_HPP_

#include <string>
#include <memory>

#include "cvcuda/OpConvertTo.hpp"
#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_publisher.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_subscriber.hpp"
#include "isaac_ros_nitros_image_type/nitros_image_view.hpp"
#include "nvcv/Tensor.hpp"
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
  void DepthCallback(const ::nvidia::isaac_ros::nitros::NitrosImageView & img_msg);

  rclcpp::QoS input_qos_;
  rclcpp::QoS output_qos_;

  std::shared_ptr<::nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
      ::nvidia::isaac_ros::nitros::NitrosImageView>>
  nitros_img_sub_;

  std::shared_ptr<::nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
      ::nvidia::isaac_ros::nitros::NitrosImage>>
  nitros_img_pub_;

  cvcuda::ConvertTo convert_op_;

  // CUDA stream to process dynamics detection on
  cudaStream_t cuda_stream_;
};

}  // namespace depth_image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_DEPTH_IMAGE_PROC__CONVERT_METRIC_NODE_HPP_
