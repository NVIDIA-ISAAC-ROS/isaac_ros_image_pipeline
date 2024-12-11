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

#include "isaac_ros_image_proc/alpha_blend_node.hpp"

#include <string>
#include <stdexcept>

#include "isaac_ros_nitros_image_type/nitros_image_builder.hpp"
#include "isaac_ros_common/qos.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace image_proc
{
namespace
{
#define CHECK_CUDA_ERRORS(result){CheckCudaErrors(result, __FILE__, __LINE__); \
}
inline void CheckCudaErrors(cudaError_t code, const char * file, const int line)
{
  if (code != cudaSuccess) {
    const std::string message = "CUDA error returned at " + std::string(file) + ":" +
      std::to_string(line) + ", Error code: " + std::to_string(code) +
      " (" + std::string(cudaGetErrorString(code)) + ")";
    throw std::runtime_error(message);
  }
}
constexpr const char kDefaultQoS[] = "SENSOR_DATA";
}  // namespace

AlphaBlendNode::AlphaBlendNode(const rclcpp::NodeOptions options)
: rclcpp::Node("alpha_blend_node", options),
  alpha_(declare_parameter<double>("alpha", 0.5)),
  mask_queue_size_(declare_parameter<int>("mask_queue_size", 10)),
  image_queue_size_(declare_parameter<int>("image_queue_size", 10)),
  sync_queue_size_(declare_parameter<int>("sync_queue_size", 10))
{
  if (alpha_ < 0 || alpha_ > 1) {
    RCLCPP_ERROR(get_logger(), "[AlphaBlendNode] Alpha must be between 0 and 1");
    throw std::invalid_argument(
            "[AlphaBlendNode] Invalid alpha parameter "
            "Alpha must be between 0 and 1.");
  }

  // Mask and image QoS profiles
  const rmw_qos_profile_t mask_qos_profile = ::isaac_ros::common::AddQosParameter(
    *this, kDefaultQoS, "mask_qos").keep_last(mask_queue_size_).get_rmw_qos_profile();
  const rmw_qos_profile_t image_qos_profile = ::isaac_ros::common::AddQosParameter(
    *this, kDefaultQoS, "image_qos").keep_last(image_queue_size_).get_rmw_qos_profile();

  // Subscribers for input images
  mask_sub_.subscribe(this, "mask_input", mask_qos_profile);
  image_sub_.subscribe(this, "image_input", image_qos_profile);
  sync_mode_ = std::make_shared<ExactSyncMode>(
    ExactPolicyMode(sync_queue_size_), mask_sub_, image_sub_);
  sync_mode_->registerCallback(
    std::bind(
      &AlphaBlendNode::InputCallback, this,
      std::placeholders::_1, std::placeholders::_2));

  // Publisher for output image
  image_pub_ = std::make_shared<
    nvidia::isaac_ros::nitros::ManagedNitrosPublisher<nvidia::isaac_ros::nitros::NitrosImage>>(
    this, "blended_image",
    nvidia::isaac_ros::nitros::nitros_image_rgb8_t::supported_type_name);

  CHECK_CUDA_ERRORS(cudaStreamCreate(&stream_));
}

AlphaBlendNode::~AlphaBlendNode()
{
  CHECK_CUDA_ERRORS(cudaStreamDestroy(stream_));
}

void AlphaBlendNode::InputCallback(
  const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & mask_ptr,
  const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & img_ptr)
{
  // Create NitrosImageView to access image data
  auto mask_view = nvidia::isaac_ros::nitros::NitrosImageView(*mask_ptr);
  auto img_view = nvidia::isaac_ros::nitros::NitrosImageView(*img_ptr);

  // Throw error if two images are not the same size
  if (mask_view.GetWidth() != img_view.GetWidth() ||
    mask_view.GetHeight() != img_view.GetHeight())
  {
    RCLCPP_ERROR(
      get_logger(),
      "[AlphaBlendNode] Input image and mask must have matching width and height");
    throw std::runtime_error(
            "[AlphaBlendNode] Invalid input image dimensions "
            "Input image and mask must have matching width and height.");
  }

  // Image properties
  int width = img_view.GetWidth();
  int height = img_view.GetHeight();
  size_t bytes = img_view.GetSizeInBytes();

  // Allocate GPU memory for output image
  uint8_t * output_image;
  CHECK_CUDA_ERRORS(cudaMallocAsync(&output_image, bytes, stream_));

  // Run alpha blending on GPU using CUDA
  bool is_mono = mask_view.GetEncoding() == sensor_msgs::image_encodings::MONO8;
  AlphaBlend(
    output_image, mask_view.GetGpuData(), img_view.GetGpuData(),
    width, height, alpha_, is_mono, stream_);
  CHECK_CUDA_ERRORS(cudaGetLastError());
  CHECK_CUDA_ERRORS(cudaStreamSynchronize(stream_));

  // Build the output Nitros image
  std_msgs::msg::Header header;
  header.stamp.sec = mask_view.GetTimestampSeconds();
  header.stamp.nanosec = mask_view.GetTimestampNanoseconds();
  header.frame_id = mask_view.GetFrameId();
  nvidia::isaac_ros::nitros::NitrosImage nitros_image =
    nvidia::isaac_ros::nitros::NitrosImageBuilder()
    .WithHeader(header)
    .WithEncoding(img_view.GetEncoding())
    .WithDimensions(height, width)
    .WithGpuData(output_image)
    .Build();

  // Publish Nitros image
  image_pub_->publish(nitros_image);
}
}  // namespace image_proc
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::image_proc::AlphaBlendNode)
