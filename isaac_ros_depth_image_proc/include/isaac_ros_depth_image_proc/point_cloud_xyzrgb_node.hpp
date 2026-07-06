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

#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/exact_time.h"

#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_depth_image_proc/depth_to_point_cloud_cuda.cu.hpp"
#include "isaac_ros_nitros/types/nitros_type_message_filter_traits.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace depth_image_proc
{

class PointCloudXyzrgbNode : public rclcpp::Node
{
public:
  explicit PointCloudXyzrgbNode(const rclcpp::NodeOptions & options);

  ~PointCloudXyzrgbNode();

  PointCloudXyzrgbNode(const PointCloudXyzrgbNode &) = delete;

  PointCloudXyzrgbNode & operator=(const PointCloudXyzrgbNode &) = delete;

private:
  // Callback
  void OnSynchronizedInputs(
    const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & depth_msg,
    const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & rgb_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_msg);

  // Create PointCloudProperties
  PointCloudProperties CreatePointCloudProperties(
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & depth_info_msg,
    const int skip);

  // Create DepthProperties
  DepthProperties CreateDepthProperties(
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & depth_info_msg);

  // PointCloudXyzNode node parameters
  int skip_;  // Paramter to limit the number of pixels converted to points
  uint16_t output_height_;
  uint16_t output_width_;
  int64_t memory_pool_block_size_;
  int64_t memory_pool_num_blocks_;
  int32_t input_qos_size_;
  int32_t output_qos_size_;

  // Subscribers
  message_filters::Subscriber<nvidia::isaac_ros::nitros::NitrosImage> depth_sub_;
  message_filters::Subscriber<nvidia::isaac_ros::nitros::NitrosImage> rgb_sub_;
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> camera_info_sub_;

  // Publisher
  rclcpp::Publisher<nvidia::isaac_ros::nitros::NitrosPointCloud>::SharedPtr point_cloud_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub_;

  using ExactSyncPolicy = message_filters::sync_policies::ExactTime<
    nvidia::isaac_ros::nitros::NitrosImage,
    nvidia::isaac_ros::nitros::NitrosImage,
    sensor_msgs::msg::CameraInfo
  >;
  message_filters::Synchronizer<ExactSyncPolicy> exact_sync_;

  // CUDA Resources
  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;
  nvidia::isaac_ros::nitros::CUDAMemoryPool pool_;

  depth_image_proc::DepthToPointCloudCUDA cloud_compute_;
};

}  // namespace depth_image_proc
}  // namespace isaac_ros
}  // namespace nvidia
