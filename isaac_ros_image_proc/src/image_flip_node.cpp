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

#include "isaac_ros_image_proc/image_flip_node.hpp"

#include <cuda_runtime.h>
#include <string>

#include "cvcuda/OpFlip.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_cvcuda_utils/cvcuda_utilities.hpp"
#include "rclcpp/rclcpp.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace image_proc
{
ImageFlipNode::ImageFlipNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("image_flip_node", options),
  flip_mode_(declare_parameter<std::string>("flip_mode", "BOTH")),
  memory_pool_block_size_(declare_parameter<int64_t>("memory_pool_block_size", 1920 * 1200 * 4)),
  memory_pool_num_blocks_(declare_parameter<int64_t>("memory_pool_num_blocks", 40)),
  input_qos_(::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos", 10)),
  output_qos_(::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos", 10))
{
  RCLCPP_DEBUG(get_logger(), "[ImageFlipNode] Constructor");

  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("ImageFlipNode");

  // Create CUDA memory pool
  cudaError_t err = pool_.create(
    static_cast<size_t>(memory_pool_block_size_),
    static_cast<size_t>(memory_pool_num_blocks_),
    nvidia::isaac_ros::nitros::CUDAMemoryPool::MemoryType::Device);
  if (err != cudaSuccess) {
    RCLCPP_ERROR(get_logger(), "Failed to create CUDA memory pool: %s", cudaGetErrorString(err));
    throw std::runtime_error("Failed to create CUDA memory pool");
  }
  // Subscription options
  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  // Publisher options
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  // Create subscribers and publishers
  image_sub_ = create_subscription<nvidia::isaac_ros::nitros::NitrosImage>(
    "image", input_qos_,
    std::bind(&ImageFlipNode::imageSubCallback, this, std::placeholders::_1), sub_options);
  image_pub_ = create_publisher<nvidia::isaac_ros::nitros::NitrosImage>(
    "image_flipped", output_qos_, pub_options);
}

ImageFlipNode::~ImageFlipNode() {}

void ImageFlipNode::imageSubCallback(const nvidia::isaac_ros::nitros::NitrosImage::SharedPtr msg)
{
  const int num_channels{sensor_msgs::image_encodings::numChannels(msg->encoding)};
  const int bytes_per_channel = sensor_msgs::image_encodings::bitDepth(msg->encoding) / CHAR_BIT;
  const cvcuda_utils::NVCVImageFormat format = cvcuda_utils::ToNVCVFormat(msg->encoding);
  RCLCPP_DEBUG(get_logger(),
    "[ImageFlipNode] Input width: %d, height: %d, num_channels: %d, bytes_per_channel: %d",
    msg->width, msg->height, num_channels, bytes_per_channel);
  auto input_handle = cvcuda_utils::WrapCVCUDATensor(
    *msg, msg->get_read_handle(*cuda_stream_),
    format.format, num_channels, bytes_per_channel);

  auto output_msg = std::make_unique<nvidia::isaac_ros::nitros::NitrosImage>();
  auto output_write_handle = output_msg->from_pool(
    pool_, msg->width, msg->height, msg->step, msg->encoding, *cuda_stream_);
  auto output_handle = cvcuda_utils::WrapCVCUDATensor(
    *output_msg, std::move(output_write_handle), format.format, num_channels, bytes_per_channel);

  // Execute flip operation
  int32_t flip_flag = cvcuda_utils::ToNVCVFlipMode(flip_mode_);
  flip_op_(*cuda_stream_, input_handle.get_tensor(), output_handle.get_tensor(), flip_flag);

  output_msg->timestamp_sec = msg->timestamp_sec;
  output_msg->timestamp_nsec = msg->timestamp_nsec;
  output_msg->frame_id = msg->frame_id;

  // Publish output image
  image_pub_->publish(std::move(output_msg));
}

}  // namespace image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::image_proc::ImageFlipNode)
