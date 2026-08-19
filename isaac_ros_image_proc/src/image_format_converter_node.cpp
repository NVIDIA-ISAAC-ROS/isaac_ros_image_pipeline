// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_image_proc/image_format_converter_node.hpp"

#include <climits>
#include <cstdio>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "cvcuda/OpCvtColor.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_cvcuda_utils/cvcuda_utilities.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/image_encodings.hpp"

using nvidia::isaac_ros::nitros::NitrosImage;
using nvidia::isaac_ros::nitros::CUDAMemoryPool;

namespace nvidia
{
namespace isaac_ros
{
namespace image_proc
{
namespace
{
int bytesPerPixel(const std::string & encoding)
{
  try {
    return sensor_msgs::image_encodings::numChannels(encoding) *
           (sensor_msgs::image_encodings::bitDepth(encoding) / CHAR_BIT);
  } catch (const std::runtime_error &) {
    RCLCPP_WARN(
      rclcpp::get_logger("ImageFormatConverterNode"),
      "Unrecognized encoding '%s'; using 4 bpp for pool sizing. "
      "Check encoding_desired parameter.",
      encoding.c_str());
    return 4;
  }
}

// As of writing, CV-CUDA 0.14 AdvCvtColor only accepts {BT601, BT709, BT2020} in
// limited range; the full-range (_ER) enumerants exist in nvcv/ColorSpec.h
// but are rejected at the operator's isSupportedColorSpec() gate.
// We therefore do not expose _ER variants here. Full-range inputs will get
// the correct matrix but a slight Y contrast mismatch.
constexpr char kDefaultYuvColorSpec[] = "bt601";
}  // namespace

NVCVColorSpec ImageFormatConverterNode::ParseYuvColorSpec(const std::string & name)
{
  static const std::unordered_map<std::string, NVCVColorSpec> kNameToSpec = {
    {"bt601", NVCV_COLOR_SPEC_BT601},
    {"bt709", NVCV_COLOR_SPEC_BT709},
    {"bt2020", NVCV_COLOR_SPEC_BT2020},
  };
  const auto it = kNameToSpec.find(name);
  if (it == kNameToSpec.end()) {
    // Build supported values string for error message.
    std::string supported;
    for (const auto & entry : kNameToSpec) {
      if (!supported.empty()) {
        supported += ", ";
      }
      supported += entry.first;
    }
    throw std::invalid_argument(
            "Unsupported yuv_color_spec: '" + name +
            "'. Supported values: " + supported + ".");
  }
  return it->second;
}

// Pool block size is computed automatically from image_width, image_height, and encoding_desired.
// Set image_width and image_height to match your camera resolution; memory_pool_block_size
// can be omitted unless you need an explicit override.
ImageFormatConverterNode::ImageFormatConverterNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("image_format_converter_node", options),
  encoding_desired_(declare_parameter<std::string>("encoding_desired", "rgb8")),
  image_width_(declare_parameter<int32_t>("image_width", 1920)),
  image_height_(declare_parameter<int32_t>("image_height", 1200)),
  memory_pool_block_size_(declare_parameter<int64_t>("memory_pool_block_size",
    static_cast<int64_t>(image_width_) * image_height_ * bytesPerPixel(encoding_desired_))),
  memory_pool_num_blocks_(declare_parameter<int64_t>("memory_pool_num_blocks", 40)),
  yuv_color_spec_(ParseYuvColorSpec(
      declare_parameter<std::string>("yuv_color_spec", kDefaultYuvColorSpec))),
  input_qos_(::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos", 10)),
  output_qos_(::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos", 10))
{
  RCLCPP_INFO(get_logger(),
    "[ImageFormatConverterNode] Pool sized from image_width=%d, image_height=%d, "
    "encoding=%s: block_size=%ld bytes (%ld blocks)",
    image_width_, image_height_, encoding_desired_.c_str(),
    memory_pool_block_size_, memory_pool_num_blocks_);

  if (image_width_ <= 0 || image_height_ <= 0) {
    RCLCPP_ERROR(get_logger(),
      "image_width (%d) and image_height (%d) must be positive",
      image_width_, image_height_);
    throw std::invalid_argument("image_width and image_height must be positive");
  }
  if (memory_pool_block_size_ <= 0) {
    RCLCPP_ERROR(get_logger(),
      "memory_pool_block_size (%ld) must be positive", memory_pool_block_size_);
    throw std::invalid_argument("memory_pool_block_size must be positive");
  }
  if (memory_pool_num_blocks_ <= 0) {
    RCLCPP_ERROR(get_logger(),
      "memory_pool_num_blocks (%ld) must be positive", memory_pool_num_blocks_);
    throw std::invalid_argument("memory_pool_num_blocks must be positive");
  }

  // check if the encoding is supported. Multiplanar outputs (e.g. nv12) are not in
  // the packed-format table, so validate them separately and skip the packed check.
  if (!encoding_desired_.empty() && !cvcuda_utils::IsMultiplanarEncoding(encoding_desired_)) {
    try {
      const cvcuda_utils::NVCVImageFormat format = cvcuda_utils::ToNVCVFormat(encoding_desired_);
    } catch (const std::invalid_argument & e) {
      RCLCPP_ERROR(get_logger(), "Unsupported encoding: %s", encoding_desired_.c_str());
      throw std::invalid_argument("Unsupported encoding: " + encoding_desired_);
    }
  }

  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("ImageFormatConverterNode");

  // Create CUDA memory pool
  cudaError_t err = pool_.create(
    static_cast<size_t>(memory_pool_block_size_),
    static_cast<size_t>(memory_pool_num_blocks_),
    CUDAMemoryPool::MemoryType::Device);
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
  image_sub_ = create_subscription<NitrosImage>(
    "image_raw", input_qos_,
    std::bind(&ImageFormatConverterNode::imageSubCallback, this, std::placeholders::_1),
    sub_options);
  image_pub_ = create_publisher<NitrosImage>(
    "image", output_qos_, pub_options);
}

ImageFormatConverterNode::~ImageFormatConverterNode() {}

std::pair<std::unique_ptr<NitrosImage>, OutputTensorHandle>
ImageFormatConverterNode::allocateOutput(const NitrosImage & msg)
{
  int num_channels = sensor_msgs::image_encodings::numChannels(encoding_desired_);
  int bpc = sensor_msgs::image_encodings::bitDepth(encoding_desired_) / CHAR_BIT;
  size_t output_step = static_cast<size_t>(num_channels) * bpc * msg.width;

  auto output_msg = std::make_unique<NitrosImage>();
  auto write_handle = output_msg->from_pool(
    pool_, msg.width, msg.height, output_step, encoding_desired_, *cuda_stream_);

  const cvcuda_utils::NVCVImageFormat output_format = cvcuda_utils::ToNVCVFormat(encoding_desired_);
  auto output_handle = cvcuda_utils::WrapCVCUDATensor(
    *output_msg, std::move(write_handle), output_format.format, num_channels, bpc);

  return {std::move(output_msg), std::move(output_handle)};
}

void ImageFormatConverterNode::publishOutput(
  std::unique_ptr<NitrosImage> output_msg, const NitrosImage & input_msg)
{
  output_msg->timestamp_sec = input_msg.timestamp_sec;
  output_msg->timestamp_nsec = input_msg.timestamp_nsec;
  output_msg->frame_id = input_msg.frame_id;
  image_pub_->publish(std::move(output_msg));
}

void ImageFormatConverterNode::imageSubCallback(const NitrosImage::SharedPtr msg)
{
  auto input_encoding = msg->encoding;

  // If input already matches desired encoding, forward without conversion.
  if (input_encoding == encoding_desired_) {
    RCLCPP_WARN_ONCE(get_logger(),
      "Input encoding '%s' already matches desired encoding. "
      "Passing through without conversion.",
      input_encoding.c_str());
    image_pub_->publish(*msg);
    return;
  }

  if (cvcuda_utils::IsMultiplanarEncoding(input_encoding)) {
    convertMultiplanar(msg);
    return;
  }

  // mono8 -> nv12: CV-CUDA exposes no GRAY->NV12 conversion, so handle it directly
  // via a luma Y-plane copy plus a neutral chroma fill. This is the inverse of the
  // NV12->MONO8 path in convertMultiplanar() and lets infrared (mono8) streams feed
  // the H.264 encoder's native NV12 input.
  if (encoding_desired_ == cvcuda_utils::kEncodingNV12) {
    if (input_encoding == sensor_msgs::image_encodings::MONO8) {
      convertMono8ToNV12(msg);
      return;
    }
    RCLCPP_ERROR(get_logger(),
      "Conversion to nv12 is only supported from mono8 input, got '%s'",
      input_encoding.c_str());
    throw std::invalid_argument(
            "Unsupported conversion to nv12 from input encoding: " + input_encoding);
  }

  // Packed-format path (rgb8, bgr8, mono8, etc.)
  cvcuda_utils::NVCVImageFormat input_format;
  try {
    input_format = cvcuda_utils::ToNVCVFormat(input_encoding);
  } catch (const std::invalid_argument & e) {
    RCLCPP_ERROR(get_logger(), "Unsupported input encoding: %s", input_encoding.c_str());
    throw std::invalid_argument("Unsupported input encoding: " + input_encoding);
  }

  int num_channels{sensor_msgs::image_encodings::numChannels(input_encoding)};
  int bytes_per_channel = sensor_msgs::image_encodings::bitDepth(input_encoding) / CHAR_BIT;
  RCLCPP_DEBUG(get_logger(),
    "[ImageFormatConverterNode] Input width: %d, height: %d, num_channels: %d,"
    "bytes_per_channel: %d",
    msg->width, msg->height, num_channels, bytes_per_channel);
  auto input_handle = cvcuda_utils::WrapCVCUDATensor(
    *msg, msg->get_read_handle(*cuda_stream_), input_format.format, num_channels,
    bytes_per_channel);

  auto [output_msg, output_handle] = allocateOutput(*msg);

  const auto conversion_code = cvcuda_utils::ToNVCVColorConversionCode(input_encoding,
    encoding_desired_);
  cvt_color_op_(*cuda_stream_, input_handle.get_tensor(), output_handle.get_tensor(),
    conversion_code);

  publishOutput(std::move(output_msg), *msg);
}

void ImageFormatConverterNode::convertMultiplanar(const NitrosImage::SharedPtr & msg)
{
  if (msg->encoding != cvcuda_utils::kEncodingNV12) {
    RCLCPP_ERROR(get_logger(),
      "Multiplanar encoding '%s' is not yet supported, only nv12", msg->encoding.c_str());
    throw std::invalid_argument("Unsupported multiplanar encoding: " + msg->encoding);
  }

  RCLCPP_DEBUG(get_logger(),
    "[ImageFormatConverterNode] NV12 input: %dx%d -> %s",
    msg->width, msg->height, encoding_desired_.c_str());

  if (encoding_desired_ == sensor_msgs::image_encodings::MONO8) {
    // NV12 Y plane is already monochrome so we can just copy this to output as AdvCvtColor does
    // not support it. We special handle this because CV-CUDA does not expose an NV12->MONO8
    // conversion code.
    auto read_handle = msg->get_read_handle(*cuda_stream_);
    auto [output_msg, output_handle] = allocateOutput(*msg);
    auto tensor_data =
      output_handle.get_tensor().exportData<nvcv::TensorDataStridedCuda>();
    if (tensor_data == nullptr) {
      RCLCPP_ERROR(get_logger(), "NV12->MONO8 failed: could not export output tensor data");
      throw std::runtime_error("NV12->MONO8 failed: could not export output tensor data");
    }
    cudaError_t err = cudaMemcpy2DAsync(
      tensor_data->basePtr(),
      output_msg->step,
      read_handle.get_ptr(),
      msg->step,
      msg->width,
      msg->height,
      cudaMemcpyDeviceToDevice, *cuda_stream_);
    if (err != cudaSuccess) {
      RCLCPP_ERROR(get_logger(), "NV12->MONO8 cudaMemcpy2DAsync failed: %s",
        cudaGetErrorString(err));
      throw std::runtime_error(
        std::string("NV12->MONO8 cudaMemcpy2DAsync failed: ") + cudaGetErrorString(err));
    }
    publishOutput(std::move(output_msg), *msg);
    return;
  }

  auto input_handle = cvcuda_utils::WrapCVCUDATensorNV12(
    *msg, msg->get_read_handle(*cuda_stream_));

  auto [output_msg, output_handle] = allocateOutput(*msg);

  const auto conversion_code = cvcuda_utils::GetNV12ConversionCode(encoding_desired_);
  adv_cvt_color_op_(
    *cuda_stream_, input_handle.get_tensor(), output_handle.get_tensor(),
    conversion_code, yuv_color_spec_);

  publishOutput(std::move(output_msg), *msg);
}

void ImageFormatConverterNode::convertMono8ToNV12(const NitrosImage::SharedPtr & msg)
{
  // mono8 luma maps directly onto the NV12 Y plane, so no color-space conversion is
  // needed: copy the input into the Y plane and fill the chroma plane with neutral
  // 0x80. CV-CUDA does not expose a GRAY->NV12 code, mirroring the NV12->MONO8 case
  // above which is also hand-written.
  RCLCPP_DEBUG(get_logger(),
    "[ImageFormatConverterNode] mono8 input: %dx%d -> nv12", msg->width, msg->height);

  auto read_handle = msg->get_read_handle(*cuda_stream_);
  const uint8_t * src = read_handle.get_ptr();
  if (src == nullptr) {
    RCLCPP_ERROR(get_logger(), "mono8->NV12 failed: input buffer pointer is null");
    throw std::runtime_error("mono8->NV12 failed: input buffer pointer is null");
  }

  // Compact NV12 layout: the Y plane stride equals the width (1 byte/pixel luma).
  // NitrosImage rejects odd dimensions for nv12, matching the 4:2:0 requirement.
  auto output_msg = std::make_unique<NitrosImage>();
  auto write_handle = output_msg->from_pool(
    pool_, msg->width, msg->height, msg->width, cvcuda_utils::kEncodingNV12, *cuda_stream_);
  uint8_t * dst = write_handle.get_ptr();
  if (dst == nullptr) {
    RCLCPP_ERROR(get_logger(), "mono8->NV12 failed: output buffer pointer is null");
    throw std::runtime_error("mono8->NV12 failed: output buffer pointer is null");
  }

  const auto & y_plane = output_msg->get_plane(0);
  const auto & uv_plane = output_msg->get_plane(1);

  // Copy the mono8 luma into the Y plane (honoring input/output row strides).
  cudaError_t err = cudaMemcpy2DAsync(
    dst + y_plane.offset, y_plane.stride,
    src, msg->step,
    msg->width, msg->height,
    cudaMemcpyDeviceToDevice, *cuda_stream_);
  if (err != cudaSuccess) {
    RCLCPP_ERROR(get_logger(), "mono8->NV12 Y-plane cudaMemcpy2DAsync failed: %s",
      cudaGetErrorString(err));
    throw std::runtime_error(
            std::string("mono8->NV12 Y-plane cudaMemcpy2DAsync failed: ") +
            cudaGetErrorString(err));
  }

  // Fill the interleaved UV plane with 0x80 (neutral chroma) for a true grayscale frame.
  err = cudaMemset2DAsync(
    dst + uv_plane.offset, uv_plane.stride,
    0x80, static_cast<size_t>(uv_plane.width) * 2, uv_plane.height, *cuda_stream_);
  if (err != cudaSuccess) {
    RCLCPP_ERROR(get_logger(), "mono8->NV12 UV-plane cudaMemset2DAsync failed: %s",
      cudaGetErrorString(err));
    throw std::runtime_error(
            std::string("mono8->NV12 UV-plane cudaMemset2DAsync failed: ") +
            cudaGetErrorString(err));
  }

  publishOutput(std::move(output_msg), *msg);
}

}  // namespace image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::image_proc::ImageFormatConverterNode)
