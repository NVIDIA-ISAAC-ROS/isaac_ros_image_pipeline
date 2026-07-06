// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_depth_image_proc/convert_metric_node.hpp"

#include <climits>

#include "isaac_ros_nitros_image_type/nitros_image_builder.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "isaac_ros_cvcuda_utils/cvcuda_utilities.hpp"
#include "isaac_ros_cvcuda_utils/cvcuda_handle.hpp"
#include "isaac_ros_common/cuda_stream.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace depth_image_proc
{

namespace
{
constexpr float kMillimetresToMetres = 0.001f;
constexpr float kConvertOpBeta = 0.0f;
}  // namespace

ConvertMetricNode::ConvertMetricNode(const rclcpp::NodeOptions options)
: rclcpp::Node("convert_metric_node", options),
  memory_pool_block_size_(declare_parameter<int64_t>("memory_pool_block_size", 1920 * 1200 * 4)),
  memory_pool_num_blocks_(declare_parameter<int64_t>("memory_pool_num_blocks", 40)),
  input_queue_size_(declare_parameter<uint16_t>("input_queue_size", 10)),
  output_queue_size_(declare_parameter<uint16_t>("output_queue_size", 10))
{
  // Create CUDA stream
  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("ConvertMetricNode");

  // Create CUDA memory pool
  CHECK_CUDA_ERROR(pool_.create(
    static_cast<size_t>(memory_pool_block_size_),
    static_cast<size_t>(memory_pool_num_blocks_),
    nvidia::isaac_ros::nitros::CUDAMemoryPool::MemoryType::Device),
    "Failed to create CUDA memory pool");

  const rclcpp::QoS input_qos = rclcpp::QoS(input_queue_size_).keep_last(input_queue_size_);
  const rclcpp::QoS output_qos = rclcpp::QoS(output_queue_size_).keep_last(output_queue_size_);

  // Create subscribers and publishers
  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  image_sub_ = create_subscription<nvidia::isaac_ros::nitros::NitrosImage>(
    "image_raw", input_qos,
    std::bind(&ConvertMetricNode::DepthCallback,
      this, std::placeholders::_1), sub_options);
  image_pub_ = create_publisher<nvidia::isaac_ros::nitros::NitrosImage>(
    "image", output_qos, pub_options);
}

void ConvertMetricNode::DepthCallback(
  const nvidia::isaac_ros::nitros::NitrosImage::SharedPtr msg)
{
  if (msg->encoding != sensor_msgs::image_encodings::MONO16 &&
    msg->encoding != sensor_msgs::image_encodings::TYPE_16UC1)
  {
    RCLCPP_ERROR(
      get_logger(),
      "Input image format is not MONO16 or TYPE_16UC1 image."
      "This node only supports MONO16 or TYPE_16UC1 image."
      "The current image input is %s", msg->encoding.c_str());
    return;
  }

  const uint32_t img_width{msg->width};
  const uint32_t img_height{msg->height};
  const int img_channels{sensor_msgs::image_encodings::numChannels(msg->encoding)};
  const int bytes_per_channel{sensor_msgs::image_encodings::bitDepth(msg->encoding) / CHAR_BIT};
  const cvcuda_utils::NVCVImageFormat input_format = cvcuda_utils::ToNVCVFormat(msg->encoding);

  // Create input buffer handle
  auto input_handle = cvcuda_utils::WrapCVCUDATensor(
    *msg, msg->get_read_handle(*cuda_stream_), input_format.format, img_channels,
    bytes_per_channel);

  // Allocate output image from pool (32FC1: 1 channel, 4 bytes per channel)
  const uint32_t output_step = img_width * sizeof(float);
  auto output_msg = std::make_unique<nvidia::isaac_ros::nitros::NitrosImage>();
  auto output_write_handle = output_msg->from_pool(
    pool_, img_width, img_height, output_step,
    sensor_msgs::image_encodings::TYPE_32FC1, *cuda_stream_);

  // Create output buffer handle
  auto output_handle = cvcuda_utils::WrapCVCUDATensor(
    *output_msg, std::move(output_write_handle), nvcv::FMT_F32, img_channels,
    static_cast<int>(sizeof(float)));

  // Convert from uint16_t -> float32.
  // And divide by 1000 to convert from millimeters -> meters
  convert_op_(
    *cuda_stream_, input_handle.get_tensor(), output_handle.get_tensor(),
    kMillimetresToMetres, kConvertOpBeta);

  // Copy header from input
  output_msg->timestamp_sec = msg->get_timestamp_sec();
  output_msg->timestamp_nsec = msg->get_timestamp_nsec();
  output_msg->frame_id = msg->get_frame_id();

  // Publish the output image
  image_pub_->publish(std::move(output_msg));
}

ConvertMetricNode::~ConvertMetricNode() {}

}  // namespace depth_image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::depth_image_proc::ConvertMetricNode)
