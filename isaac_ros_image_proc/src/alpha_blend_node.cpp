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

#include "isaac_ros_image_proc/alpha_blend_node.hpp"

#include <string>
#include <stdexcept>

#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_common/qos.hpp"
#include "sensor_msgs/image_encodings.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace image_proc
{
using nvidia::isaac_ros::nitros::NitrosImage;
namespace
{
constexpr const char kDefaultQoS[] = "SENSOR_DATA";
}  // namespace

AlphaBlendNode::AlphaBlendNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("alpha_blend_node", options),
  alpha_(declare_parameter<double>("alpha", 0.5)),
  memory_pool_block_size_(declare_parameter<int>("memory_pool_block_size", 1920 * 1200 * 4)),
  memory_pool_num_blocks_(declare_parameter<int>("memory_pool_num_blocks", 40)),
  input_queue_size_(declare_parameter<int64_t>("input_queue_size", 10)),
  output_queue_size_(declare_parameter<int64_t>("output_queue_size", 10)),
  image_sub_{},
  mask_sub_{},
  sync_{ExactPolicy(input_queue_size_), image_sub_, mask_sub_}
{
  if (alpha_ < 0 || alpha_ > 1) {
    RCLCPP_ERROR(get_logger(), "[AlphaBlendNode] Alpha must be between 0 and 1");
    throw std::invalid_argument(
            "[AlphaBlendNode] Invalid alpha parameter "
            "Alpha must be between 0 and 1.");
  }

  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("AlphaBlendNode");

  // Create CUDA memory pool
  cudaError_t err = pool_.create(
    static_cast<size_t>(memory_pool_block_size_),
    static_cast<size_t>(memory_pool_num_blocks_),
    nvidia::isaac_ros::nitros::CUDAMemoryPool::MemoryType::Device);
  CHECK_CUDA_ERROR(err, "Failed to create CUDA memory pool");

  rclcpp::QoS input_qos = ::isaac_ros::common::AddQosParameter(*this, kDefaultQoS, "input_qos")
    .keep_last(input_queue_size_);
  rclcpp::QoS output_qos = ::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos")
    .keep_last(output_queue_size_);
  const rmw_qos_profile_t input_qos_profile = input_qos.get_rmw_qos_profile();

  // Subscription options
  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  // Publisher options
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  sync_.registerCallback(
    std::bind(
      &AlphaBlendNode::InputCallback, this,
      std::placeholders::_1, std::placeholders::_2));
  image_sub_.subscribe(this, "image_input", input_qos_profile, sub_options);
  mask_sub_.subscribe(this, "mask_input", input_qos_profile, sub_options);

  // Publisher for output image
  image_pub_ = create_publisher<nvidia::isaac_ros::nitros::NitrosImage>(
    "blended_image", output_qos, pub_options);

  RCLCPP_INFO(get_logger(), "[AlphaBlendNode] Alpha blend node initialized");
}

AlphaBlendNode::~AlphaBlendNode() {}

void AlphaBlendNode::InputCallback(
  const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & img,
  const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & mask)
{
  // Throw error if two images are not the same size
  if (mask->width != img->width ||
    mask->height != img->height)
  {
    RCLCPP_ERROR(
      get_logger(),
      "[AlphaBlendNode] Input image and mask must have matching width and height");
    throw std::runtime_error(
            "[AlphaBlendNode] Invalid input image dimensions "
            "Input image and mask must have matching width and height.");
  }

  // Image properties
  int width = img->width;
  int height = img->height;
  // Input image and mask pointers
  auto input_mask_handle = mask->get_read_handle(*cuda_stream_);
  auto input_img_handle = img->get_read_handle(*cuda_stream_);
  const uint8_t * input_mask = static_cast<const uint8_t *>(input_mask_handle.get_ptr());
  const uint8_t * input_img = static_cast<const uint8_t *>(input_img_handle.get_ptr());

  // Create output image
  auto output_msg = std::make_unique<nvidia::isaac_ros::nitros::NitrosImage>();
  auto output_write_handle = output_msg->from_pool(
    pool_, width, height, img->step, img->encoding, *cuda_stream_);
  uint8_t * output_image = static_cast<uint8_t *>(output_write_handle.get_ptr());

  // Run alpha blending on GPU using CUDA
  bool is_mono = sensor_msgs::image_encodings::isMono(mask->encoding);

  AlphaBlend(
    output_image, input_mask, input_img,
    width, height, alpha_, is_mono, *cuda_stream_);
  CHECK_CUDA_ERROR(cudaGetLastError(), "Failed to execute alpha blending");
  CHECK_CUDA_ERROR(cudaStreamSynchronize(*cuda_stream_), "Failed to synchronize CUDA stream");

  output_msg->timestamp_sec = img->timestamp_sec;
  output_msg->timestamp_nsec = img->timestamp_nsec;
  output_msg->frame_id = img->frame_id;

  // Publish output image
  image_pub_->publish(std::move(output_msg));
}

}  // namespace image_proc
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::image_proc::AlphaBlendNode)
