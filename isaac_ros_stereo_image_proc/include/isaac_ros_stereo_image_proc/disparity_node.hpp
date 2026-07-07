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
#include "isaac_ros_vpi_utils/vpi_handle.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "vpi/algo/ConvertImageFormat.h"
#include "vpi/algo/Rescale.h"
#include "vpi/algo/StereoDisparity.h"
#include "vpi/CUDAInterop.h"
#include "vpi/VPI.h"

using StringList = std::vector<std::string>;

namespace nvidia
{
namespace isaac_ros
{
namespace stereo_image_proc
{
using nvidia::isaac_ros::nitros::ReadHandle;
using nvidia::isaac_ros::nitros::WriteHandle;
using nvidia::isaac_ros::nitros::NitrosImage;
using nvidia::isaac_ros::nitros::NitrosDisparityImage;
using nvidia::isaac_ros::vpi_utils::VPIImageHandle;

struct Impl
{
  // Shared VPI stream for submitting all operations
  VPIStream stream;

  VPIImageHandle<ReadHandle> left_input;
  VPIImageHandle<ReadHandle> right_input;

  // Left and right VPI images in stereo algorithm-specific format
  VPIImage left_formatted;
  VPIImage right_formatted;
  // Raw disparity, resized disparity, and confidence map in VPI-specific format
  VPIImage disparity_raw;
  VPIImage disparity_resized;
  VPIImage confidence_map;

  // VPI algorithm parameters
  VPIConvertImageFormatParams stereo_input_scale_params;
  VPIStereoDisparityEstimatorCreationParams disparity_params;
  VPIStereoDisparityEstimatorParams disparity_context_params;
  VPIStereoDisparityConfidenceType disparity_confidence_type;
  VPIConvertImageFormatParams disparity_scale_params;
  // VPI stereo calculation parameters
  VPIPayload stereo_payload;
  VPIImageFormat stereo_format;
  // VPI backends
  uint32_t vpi_backends;
  uint64_t vpi_flags;
  // Cached values from previous iteration to compare against
  int32_t prev_height;
  int32_t prev_width;
};

class DisparityNode : public rclcpp::Node
{
public:
  explicit DisparityNode(const rclcpp::NodeOptions & options);

  ~DisparityNode();

  DisparityNode(const DisparityNode &) = delete;

  DisparityNode & operator=(const DisparityNode &) = delete;

private:
  void InputCallback(
    const NitrosImage::ConstSharedPtr & left_image_msg,
    const NitrosImage::ConstSharedPtr & right_image_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & left_camera_info_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & right_camera_info_msg);
  bool InitVPIImages(
    const NitrosImage & left_image_msg,
    const NitrosImage & right_image_msg);
  bool initialize();
  void deinitialize();

  // Disparity node parameters
  const std::string backend_;
  const int64_t memory_pool_block_size_;
  const int64_t memory_pool_num_blocks_;
  const float max_disparity_;
  const int32_t confidence_threshold_;
  const int32_t confidence_type_;
  const int32_t window_size_;
  const int32_t num_passes_;
  const int32_t p1_;
  const int32_t p2_;
  const int32_t p2_alpha_;
  const int32_t quality_;
  const uint16_t input_queue_size_;
  const uint16_t output_queue_size_;

  // Subscribers and publishers
  message_filters::Subscriber<nvidia::isaac_ros::nitros::NitrosImage> left_image_sub_;
  message_filters::Subscriber<nvidia::isaac_ros::nitros::NitrosImage> right_image_sub_;
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> left_camera_info_sub_;
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> right_camera_info_sub_;

  rclcpp::Publisher<nvidia::isaac_ros::nitros::NitrosDisparityImage>::SharedPtr disparity_pub_;

  using ExactPolicy = message_filters::sync_policies::ExactTime<
    nvidia::isaac_ros::nitros::NitrosImage,
    nvidia::isaac_ros::nitros::NitrosImage,
    sensor_msgs::msg::CameraInfo,
    sensor_msgs::msg::CameraInfo>;
  message_filters::Synchronizer<ExactPolicy> exact_sync_;

  // CUDA resources
  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;
  nvidia::isaac_ros::nitros::CUDAMemoryPool pool_;

  // VPI internal parameters
  Impl impl_{};
};

}  // namespace stereo_image_proc
}  // namespace isaac_ros
}  // namespace nvidia
