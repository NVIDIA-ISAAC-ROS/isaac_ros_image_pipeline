// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <vector>

#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/exact_time.h"
#include "nvcv/Rect.h"

#include "cvcuda/OpCustomCrop.hpp"
#include "isaac_ros_cvcuda_utils/cvcuda_handle.hpp"
#include "isaac_ros_nitros/types/cuda_memory_pool.hpp"
#include "isaac_ros_nitros/types/nitros_type_message_filter_traits.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "nvcv/Tensor.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"

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

class CropNode : public rclcpp::Node
{
public:
  explicit CropNode(const rclcpp::NodeOptions & options);

  ~CropNode();

  CropNode(const CropNode &) = delete;

  CropNode & operator=(const CropNode &) = delete;

private:
  void CalculateResizeAndCropParams(const CropMode & crop_mode);
  void InputCallback(
    const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & nitros_image,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info);
  void UpdateCameraInfo(
    const sensor_msgs::msg::CameraInfo & input_camera_info,
    sensor_msgs::msg::CameraInfo & output_camera_info);

  // Node parameters
  const int64_t input_width_;
  const int64_t input_height_;
  const int64_t crop_width_;
  const int64_t crop_height_;
  const int64_t roi_top_left_x_;
  const int64_t roi_top_left_y_;
  const std::string crop_mode_;
  const int64_t memory_pool_block_size_;
  const int64_t memory_pool_num_blocks_;
  const uint32_t input_queue_size_;
  const uint32_t output_queue_size_;

  // top-left corner {x, y} and width, height of the rectangle
  NVCVRectI roi_;

  // Crop node subscribers and publishers
  message_filters::Subscriber<nvidia::isaac_ros::nitros::NitrosImage> image_sub_;
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> camera_info_sub_;
  rclcpp::Publisher<nvidia::isaac_ros::nitros::NitrosImage>::SharedPtr image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub_;

  using ExactPolicy = message_filters::sync_policies::ExactTime<
    nvidia::isaac_ros::nitros::NitrosImage,
    sensor_msgs::msg::CameraInfo>;
  message_filters::Synchronizer<ExactPolicy> exact_sync_;

  // Resources
  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;
  nvidia::isaac_ros::nitros::CUDAMemoryPool pool_;

  // Crop node CVCUDA operation
  cvcuda::CustomCrop crop_op_;
};

}  // namespace image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_IMAGE_PROC__CROP_NODE_HPP_
