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
#ifndef ISAAC_ROS_IMAGE_PROC__RESIZE_NODE_HPP_
#define ISAAC_ROS_IMAGE_PROC__RESIZE_NODE_HPP_

#include <string>
#include <chrono>
#include <utility>
#include <vector>

#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/exact_time.h"

#include "cvcuda/OpResize.hpp"
#include "cvcuda/OpCopyMakeBorder.hpp"
#include "isaac_ros_cvcuda_utils/cvcuda_handle.hpp"
#include "isaac_ros_nitros/types/cuda_memory_pool.hpp"
#include "isaac_ros_nitros/types/nitros_type_message_filter_traits.hpp"
#include "isaac_ros_nitros_camera_info_type/nitros_camera_info.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "nvcv/Tensor.hpp"
#include "rclcpp/rclcpp.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace image_proc
{

class ResizeNode : public rclcpp::Node
{
public:
  explicit ResizeNode(const rclcpp::NodeOptions & options);

  ~ResizeNode();

  ResizeNode(const ResizeNode &) = delete;

  ResizeNode & operator=(const ResizeNode &) = delete;

private:
  void InputCallback(
    const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & nitros_image,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info);
  void UpdateCameraInfo(
    const sensor_msgs::msg::CameraInfo & input_camera_info,
    sensor_msgs::msg::CameraInfo & output_camera_info);
  void CalculateOutputDims(
    const int64_t input_width, const int64_t input_height,
    int64_t & output_width, int64_t & output_height) const;

  // Resize node parameters
  int64_t output_width_;
  int64_t output_height_;
  const int64_t memory_pool_block_size_;
  const int64_t memory_pool_num_blocks_;
  std::string interp_type_;
  std::string border_type_;
  bool keep_aspect_ratio_;
  // Parameters to support aspect ratio preservation w/o padding
  bool disable_padding_;
  int64_t input_queue_size_;
  int64_t output_queue_size_;

  // Subscriptions and publishers
  message_filters::Subscriber<nvidia::isaac_ros::nitros::NitrosImage> image_sub_;
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> camera_info_sub_;
  rclcpp::Publisher<nvidia::isaac_ros::nitros::NitrosImage>::SharedPtr image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub_;

  using ExactPolicy = message_filters::sync_policies::ExactTime<
    nvidia::isaac_ros::nitros::NitrosImage,
    sensor_msgs::msg::CameraInfo>;
  message_filters::Synchronizer<ExactPolicy> exact_sync_;

  // CUDA resources
  nvidia::isaac_ros::nitros::CUDAMemoryPool pool_;
  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;

  // Intermediate tensor for resized image
  int64_t resize_out_img_width_;
  int64_t resize_out_img_height_;
  nvcv::Tensor resized_tensor_;

  // CVCUDA operation
  cvcuda::Resize resize_op_;
  cvcuda::CopyMakeBorder copy_make_border_op_;
};

}  // namespace image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_IMAGE_PROC__RESIZE_NODE_HPP_
