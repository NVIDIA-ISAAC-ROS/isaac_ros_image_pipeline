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
#ifndef ISAAC_ROS_IMAGE_PROC__RECTIFY_NODE_HPP_
#define ISAAC_ROS_IMAGE_PROC__RECTIFY_NODE_HPP_

#include <functional>
#include <memory>
#include <string>
#include <chrono>
#include <utility>
#include <vector>

#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/exact_time.h"

#include "cvcuda/OpRemap.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_cvcuda_utils/cvcuda_handle.hpp"
#include "isaac_ros_nitros/types/cuda_memory_pool.hpp"
#include "isaac_ros_nitros/types/nitros_type_message_filter_traits.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "nvcv/Tensor.hpp"
#include "opencv2/calib3d.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace image_proc
{

class RectifyNode : public rclcpp::Node
{
public:
  explicit RectifyNode(const rclcpp::NodeOptions & options);

  ~RectifyNode();

  RectifyNode(const RectifyNode &) = delete;

  RectifyNode & operator=(const RectifyNode &) = delete;

private:
  nvcv::Tensor WrapOpencvCVMapToCVCUDATensor(const cv::Mat & mat, void * map_buffer);
  void InitRemapMapAndOutputCameraInfo(const sensor_msgs::msg::CameraInfo & camera_info);
  void InputCallback(
    const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & image,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info);

  int16_t output_width_;
  int16_t output_height_;
  int16_t horizontal_interval_;
  int16_t vertical_interval_;
  std::string interpolation_;
  std::string border_type_;
  std::vector<double> border_constant_;
  bool align_corners_;
  std::string map_value_type_;
  std::string map_interpolation_type_;
  const int64_t memory_pool_block_size_;
  const int64_t memory_pool_num_blocks_;
  int64_t input_queue_size_;
  int64_t output_queue_size_;

  // Subscriptions and publishers
  message_filters::Subscriber<nvidia::isaac_ros::nitros::NitrosImage> image_sub_;
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> camera_info_sub_;

  rclcpp::Publisher<nvidia::isaac_ros::nitros::NitrosImage>::SharedPtr image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub_;
  using ExactPolicy = message_filters::sync_policies::ExactTime<
    nvidia::isaac_ros::nitros::NitrosImage, sensor_msgs::msg::CameraInfo>;
  message_filters::Synchronizer<ExactPolicy> exact_sync_;

  // CUDA resources
  nvidia::isaac_ros::nitros::CUDAMemoryPool pool_;
  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;

  // CVCUDA operation
  cvcuda::Remap remap_op_;
  nvcv::Tensor remap_map_;

  sensor_msgs::msg::CameraInfo cached_input_camera_info_{};
  sensor_msgs::msg::CameraInfo cached_output_camera_info_{};
  NVCVBorderType border_;
  float4 borderValue_;
  NVCVInterpolationType inInterp_;
  NVCVInterpolationType mapInterp_;
  NVCVRemapMapValueType mapValueType_;
  void * map_buffer_{nullptr};
};

}  // namespace image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_IMAGE_PROC__RECTIFY_NODE_HPP_
