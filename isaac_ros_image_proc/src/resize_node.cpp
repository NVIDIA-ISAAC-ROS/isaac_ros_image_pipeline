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

#include "isaac_ros_image_proc/resize_node.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <memory>
#include <string>
#include <utility>

#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_cvcuda_utils/cvcuda_utilities.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace image_proc
{

void ResizeNode::CalculateOutputDims(
  const int64_t input_width, const int64_t input_height,
  int64_t & output_width, int64_t & output_height) const
{
  // The output dimensions are provided through the node parameters,
  // in case of aspect ratio preservation, we need to calculate the output dimensions based
  // on the input dimensions and the output dimensions.
  float height_factor = static_cast<float>(output_height) / input_height;
  float width_factor = static_cast<float>(output_width) / input_width;
  if (height_factor < width_factor) {
    output_width = input_width * height_factor;
    // To make sure output width is even
    if (output_width % 2 != 0) {output_width++;}
  } else if (width_factor < height_factor) {
    output_height = input_height * width_factor;
    if (output_height % 2 != 0) {output_height++;}
  }
}

void ResizeNode::UpdateCameraInfo(
  const sensor_msgs::msg::CameraInfo & input_camera_info,
  sensor_msgs::msg::CameraInfo & output_camera_info)
{
  const int64_t input_width = input_camera_info.width;
  const int64_t input_height = input_camera_info.height;

  output_camera_info = input_camera_info;
  float scaler_x = static_cast<float>(output_width_) / static_cast<float>(input_width);
  float scaler_y = static_cast<float>(output_height_) / static_cast<float>(input_height);
  float min_scaler = std::min(scaler_x, scaler_y);
  float pixel_center = 0.5f;
  output_camera_info.width = output_width_;
  output_camera_info.height = output_height_;

  // Update the focal length
  if (keep_aspect_ratio_) {
    output_camera_info.k[0] = input_camera_info.k[0] * min_scaler;
    output_camera_info.k[4] = input_camera_info.k[4] * min_scaler;
  } else {
    output_camera_info.k[0] = input_camera_info.k[0] * scaler_x;
    output_camera_info.k[4] = input_camera_info.k[4] * scaler_y;
  }
  // Update the principal point
  output_camera_info.k[2] = (input_camera_info.k[2] + pixel_center) * scaler_x - pixel_center;
  output_camera_info.k[5] = (input_camera_info.k[5] + pixel_center) * scaler_y - pixel_center;

  output_camera_info.p[0] = output_camera_info.k[0];
  output_camera_info.p[1] = 0;
  output_camera_info.p[2] = output_camera_info.k[2];
  output_camera_info.p[3] = (input_camera_info.p[0] != 0.0) ?
    input_camera_info.p[3] * (output_camera_info.p[0] / input_camera_info.p[0]) :
    0.0;
  output_camera_info.p[4] = 0;
  output_camera_info.p[5] = output_camera_info.k[4];
  output_camera_info.p[6] = output_camera_info.k[5];
  output_camera_info.p[7] = (input_camera_info.p[5] != 0.0) ?
    input_camera_info.p[7] * (output_camera_info.p[5] / input_camera_info.p[5]) :
    0.0;
  output_camera_info.p[8] = 0;
  output_camera_info.p[9] = 0;
  output_camera_info.p[10] = 1;
  output_camera_info.p[11] = input_camera_info.p[11];

  // No subsampling, full resolution after transform
  output_camera_info.binning_x = 1;
  output_camera_info.binning_y = 1;
  output_camera_info.roi.height = output_camera_info.height;
  output_camera_info.roi.width = output_camera_info.width;
}

ResizeNode::ResizeNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("ResizeNode", options),
  output_width_(declare_parameter<int64_t>("output_width", 1080)),
  output_height_(declare_parameter<int64_t>("output_height", 720)),
  memory_pool_block_size_(declare_parameter<int64_t>("memory_pool_block_size", 1920 * 1200 * 4)),
  memory_pool_num_blocks_(declare_parameter<int64_t>("memory_pool_num_blocks", 40)),
  interp_type_(declare_parameter<std::string>("interp_type", "linear")),
  border_type_(declare_parameter<std::string>("border_type", "zero")),
  keep_aspect_ratio_(declare_parameter<bool>("keep_aspect_ratio", false)),
  disable_padding_(declare_parameter<bool>("disable_padding", false)),
  input_queue_size_(declare_parameter<int64_t>("input_queue_size", 10)),
  output_queue_size_(declare_parameter<int64_t>("output_queue_size", 10)),
  image_sub_{},
  camera_info_sub_{},
  exact_sync_{ExactPolicy(input_queue_size_), image_sub_, camera_info_sub_}
{
  RCLCPP_DEBUG(get_logger(), "[ResizeNode] Constructor");

  // Validate configuration
  if (output_width_ <= 0 || output_height_ <= 0) {
    RCLCPP_ERROR(
      get_logger(),
      "[ResizeNode] Width and height need to be non-zero positive number");
    throw std::invalid_argument(
      "[ResizeNode] Invalid output dimension "
      "Width and height need to be non-zero positive number.");
  }

  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("ResizeNode");

  cudaError_t pool_err = pool_.create(
    static_cast<size_t>(memory_pool_block_size_),
    static_cast<size_t>(memory_pool_num_blocks_),
    nvidia::isaac_ros::nitros::CUDAMemoryPool::MemoryType::Device);
  CHECK_CUDA_ERROR(pool_err, "[ResizeNode] Failed to create CUDA memory pool");

  const rclcpp::QoS input_qos =
    ::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos")
    .keep_last(input_queue_size_);
  const rclcpp::QoS output_qos =
    ::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos")
    .keep_last(output_queue_size_);
  const rmw_qos_profile_t input_qos_profile = input_qos.get_rmw_qos_profile();

  // Create subscribers
  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  exact_sync_.registerCallback(
    std::bind(
      &ResizeNode::InputCallback, this,
      std::placeholders::_1, std::placeholders::_2));
  image_sub_.subscribe(this, "image", input_qos_profile, sub_options);
  camera_info_sub_.subscribe(this, "camera_info", input_qos_profile, sub_options);

  image_pub_ = create_publisher<nvidia::isaac_ros::nitros::NitrosImage>(
    "resize/image", output_qos, pub_options);
  camera_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(
    "resize/camera_info", output_qos, pub_options);

  RCLCPP_INFO(get_logger(), "[ResizeNode] ResizeNode initialized");
}

ResizeNode::~ResizeNode() {}

void ResizeNode::InputCallback(
  const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & nitros_image,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info)
{
  RCLCPP_DEBUG(get_logger(), "[ResizeNode] InputCallback - SYNCHRONIZED!");

  if (!nitros_image || !camera_info) {
    throw std::runtime_error("[ResizeNode] No inputs received");
  }

  const cvcuda_utils::NVCVImageFormat format = cvcuda_utils::ToNVCVFormat(nitros_image->encoding);
  if (keep_aspect_ratio_ && disable_padding_) {
    CalculateOutputDims(nitros_image->width, nitros_image->height, output_width_, output_height_);
  }

  if (keep_aspect_ratio_ && !disable_padding_) {
    resize_out_img_width_ = output_width_;
    resize_out_img_height_ = output_height_;
    CalculateOutputDims(nitros_image->width, nitros_image->height,
      resize_out_img_width_, resize_out_img_height_);
    if (resize_out_img_width_ <= 0 || resize_out_img_height_ <= 0) {
      RCLCPP_ERROR(get_logger(),
        "[ResizeNode] Invalid resized dimensions %ldx%ld (input %dx%d)",
        resize_out_img_width_, resize_out_img_height_,
        nitros_image->width, nitros_image->height);
      return;
    }
    resized_tensor_ = nvcv::Tensor(1,
        {static_cast<int32_t>(resize_out_img_width_),
          static_cast<int32_t>(resize_out_img_height_)},
      format.format);
  }
  const int num_channels{sensor_msgs::image_encodings::numChannels(nitros_image->encoding)};
  const int bytes_per_channel =
    sensor_msgs::image_encodings::bitDepth(nitros_image->encoding) / CHAR_BIT;
  auto input_handle = cvcuda_utils::WrapCVCUDATensor(
    *nitros_image, nitros_image->get_read_handle(*cuda_stream_),
    format.format, num_channels, bytes_per_channel);

  auto output_image = std::make_unique<nvidia::isaac_ros::nitros::NitrosImage>();
  size_t output_step = num_channels * bytes_per_channel * output_width_;
  auto output_write_handle = output_image->from_pool(
    pool_, output_width_, output_height_, output_step, nitros_image->encoding, *cuda_stream_);
  auto output_handle = cvcuda_utils::WrapCVCUDATensor(
    *output_image, std::move(output_write_handle),
    format.format, num_channels, bytes_per_channel);

  const NVCVInterpolationType interp_type = cvcuda_utils::ToNVCVInterpolationType(interp_type_);
  if (keep_aspect_ratio_ && !disable_padding_) {
    resize_op_(*cuda_stream_, input_handle.get_tensor(), resized_tensor_, interp_type);

    float4 border_value = {0.0f, 0.0f, 0.0f, 0.0f};
    int32_t top = (output_height_ - resize_out_img_height_) / 2;
    int32_t left = (output_width_ - resize_out_img_width_) / 2;
    copy_make_border_op_(*cuda_stream_, resized_tensor_, output_handle.get_tensor(), top, left,
      NVCV_BORDER_CONSTANT, border_value);
  } else {
    resize_op_(*cuda_stream_, input_handle.get_tensor(), output_handle.get_tensor(), interp_type);
  }
  output_image->timestamp_sec = nitros_image->timestamp_sec;
  output_image->timestamp_nsec = nitros_image->timestamp_nsec;
  output_image->frame_id = nitros_image->frame_id;

  auto camera_info_msg = std::make_unique<sensor_msgs::msg::CameraInfo>();
  UpdateCameraInfo(*camera_info, *camera_info_msg);

  camera_info_pub_->publish(std::move(camera_info_msg));
  RCLCPP_DEBUG(get_logger(), "[ResizeNode] Camera info published");
  image_pub_->publish(std::move(output_image));
  RCLCPP_DEBUG(get_logger(), "[ResizeNode] Image published");
}

}  // namespace image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::image_proc::ResizeNode)
