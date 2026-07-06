// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <string>
#include <chrono>
#include <utility>
#include <vector>

#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/exact_time.h"

#include "isaac_ros_nitros/types/nitros_type_message_filter_traits.hpp"
#include "isaac_ros_nitros_disparity_image_type/nitros_disparity_image.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "isaac_ros_nitros_point_cloud_type/nitros_point_cloud.hpp"
#include "isaac_ros_stereo_image_proc/point_cloud_cuda.cu.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"

using StringList = std::vector<std::string>;

namespace nvidia
{
namespace isaac_ros
{
namespace stereo_image_proc
{

class PointCloudNode : public rclcpp::Node
{
public:
  explicit PointCloudNode(const rclcpp::NodeOptions & options);

  ~PointCloudNode();

  PointCloudNode(const PointCloudNode &) = delete;

  PointCloudNode & operator=(const PointCloudNode &) = delete;

private:
  void PointCloudCallback(
    const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & left_image_msg,
    const nvidia::isaac_ros::nitros::NitrosDisparityImage::ConstSharedPtr & disparity_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & left_camera_info_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & right_camera_info_msg);

  PointCloudProperties CreateCloudProperties(
    const nvidia::isaac_ros::nitros::NitrosDisparityImage::ConstSharedPtr & disparity_msg);
  DisparityProperties CreateDisparityProperties(
    const nvidia::isaac_ros::nitros::NitrosDisparityImage::ConstSharedPtr & disparity_msg);
  CameraIntrinsics CreateCameraIntrinsics(
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & left_camera_info_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & right_camera_info_msg);
  RGBProperties CreateRGBProperties(
    const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & rgb_msg);
  bool SelectDisparityFormatAndCompute(
    float * point_cloud_output,
    const PointCloudProperties & cloud_properties,
    const nvidia::isaac_ros::nitros::NitrosDisparityImage::ConstSharedPtr & disparity_msg,
    const DisparityProperties & disparity_properties,
    const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & rgb_msg,
    const RGBProperties & rgb_properties,
    const CameraIntrinsics & intrinsics);

  // Point cloud node parameters
  bool use_color_;
  float unit_scaling_;
  const int64_t memory_pool_block_size_;
  const int64_t memory_pool_num_blocks_;
  const int64_t input_queue_size_;
  const int64_t output_queue_size_;

  // Subscribers and publishers
  message_filters::Subscriber<nvidia::isaac_ros::nitros::NitrosImage> left_image_sub_;
  message_filters::Subscriber<nvidia::isaac_ros::nitros::NitrosDisparityImage> disparity_sub_;
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> left_camera_info_sub_;
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> right_camera_info_sub_;
  rclcpp::Publisher<nvidia::isaac_ros::nitros::NitrosPointCloud>::SharedPtr point_cloud_pub_;

  using ExactSyncPolicy = message_filters::sync_policies::ExactTime<
    nvidia::isaac_ros::nitros::NitrosImage,
    nvidia::isaac_ros::nitros::NitrosDisparityImage,
    sensor_msgs::msg::CameraInfo,
    sensor_msgs::msg::CameraInfo
  >;
  message_filters::Synchronizer<ExactSyncPolicy> exact_sync_;

  // Resources
  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;
  nvidia::isaac_ros::nitros::CUDAMemoryPool pool_;

  PointCloudNodeCUDA cloud_compute_;
};

}  // namespace stereo_image_proc
}  // namespace isaac_ros
}  // namespace nvidia
