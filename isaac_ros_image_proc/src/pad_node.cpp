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

#include "isaac_ros_image_proc/pad_node.hpp"

#include <cuda_runtime.h>
#include "nvcv/BorderType.h"

#include "isaac_ros_nitros_image_type/nitros_image_view.hpp"
#include "isaac_ros_nitros_image_type/nitros_image_builder.hpp"
#include "sensor_msgs/image_encodings.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace image_proc
{

namespace
{
const std::unordered_map<std::string, nvcv::ImageFormat> kStringEncToNVCVImageFormatMap({
          {sensor_msgs::image_encodings::RGB8, nvcv::FMT_RGB8},
          {sensor_msgs::image_encodings::BGR8, nvcv::FMT_BGR8},
          {sensor_msgs::image_encodings::RGBA8, nvcv::FMT_RGBA8},
          {sensor_msgs::image_encodings::BGRA8, nvcv::FMT_BGRA8},
          {sensor_msgs::image_encodings::MONO8, nvcv::FMT_U8},
          {sensor_msgs::image_encodings::TYPE_32FC3, nvcv::FMT_RGBf32},
          {sensor_msgs::image_encodings::TYPE_32FC1, nvcv::FMT_F32}
        });

const std::unordered_map<std::string, PaddingType> kStringToPaddingTypeMap({
          {"CENTER", PaddingType::kCenter},
          {"TOP_LEFT", PaddingType::kTopLeft},
          {"TOP_RIGHT", PaddingType::kTopRight},
          {"BOTTOM_LEFT", PaddingType::kBottomLeft},
          {"BOTTOM_RIGHT", PaddingType::kBottomRight}
        });

const std::unordered_map<std::string, NVCVBorderType> kStringToBorderTypeMap({
          {"CONSTANT", NVCVBorderType::NVCV_BORDER_CONSTANT},
          {"REPLICATE", NVCVBorderType::NVCV_BORDER_REPLICATE},
          {"REFLECT", NVCVBorderType::NVCV_BORDER_REFLECT},
          {"WRAP", NVCVBorderType::NVCV_BORDER_WRAP},
          {"REFLECT101", NVCVBorderType::NVCV_BORDER_REFLECT101}
        });

constexpr uint8_t kBitsInByte = 8;
constexpr uint8_t kBatchSize = 1;

uint32_t CalculateOffset(
  const uint16_t input_width,
  const uint16_t input_height,
  const uint16_t output_width,
  const uint16_t output_height,
  const PaddingType & padding_type,
  const nvcv::TensorDataStridedCuda::Buffer & output_buffer
)
{
  uint32_t offset = 0;
  switch (padding_type) {
    case PaddingType::kBottomRight: {
        offset = 0;
        return offset;
      }
    case PaddingType::kBottomLeft: {
        uint32_t start_x = output_width - input_width;
        offset = start_x * output_buffer.strides[2];
        return offset;
      }
    case PaddingType::kTopRight: {
        uint32_t start_y = output_height - input_height;
        offset = start_y * output_buffer.strides[1];
        return offset;
      }
    case PaddingType::kTopLeft: {
        uint32_t start_y = output_height - input_height;
        uint32_t start_x = output_width - input_width;
        offset = start_y * output_buffer.strides[1];
        offset += start_x * output_buffer.strides[2];
        return offset;
      }
    default: {
        throw std::invalid_argument("[PadNode] Unsupported Padding Type");
      }
  }
}

void InitializeBuffer(
  const nvcv::ImageFormat & fmt,
  const uint16_t width,
  const uint16_t height,
  nvcv::TensorDataStridedCuda::Buffer * buffer,
  nvcv::Tensor::Requirements * reqs
)
{
  *reqs = nvcv::Tensor::CalcRequirements(kBatchSize, {width, height}, fmt);
  uint32_t input_image_channels = fmt.numChannels();
  uint32_t bytes_per_pixel = (
    nvcv::DataType{reqs->dtype}.bitsPerPixel() + kBitsInByte - 1) / kBitsInByte;

  buffer->strides[3] = bytes_per_pixel;
  buffer->strides[2] = input_image_channels * buffer->strides[3];
  buffer->strides[1] = width * buffer->strides[2];
  buffer->strides[0] = height * buffer->strides[1];
}

void checkCudaErrors(cudaError_t err)
{
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
}

}  // namespace

PadNode::PadNode(const rclcpp::NodeOptions options)
: rclcpp::Node("padding_node", options),
  input_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos")},
  output_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos")},
  nitros_sub_{std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
        nvidia::isaac_ros::nitros::NitrosImageView>>(
      this, "image", nvidia::isaac_ros::nitros::nitros_image_rgb8_t::supported_type_name,
      std::bind(&PadNode::InputCallback, this,
      std::placeholders::_1), nvidia::isaac_ros::nitros::NitrosDiagnosticsConfig{}, input_qos_)},
  nitros_pub_{std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
        nvidia::isaac_ros::nitros::NitrosImage>>(
      this, "padded_image",
      nvidia::isaac_ros::nitros::nitros_image_rgb8_t::supported_type_name,
      nvidia::isaac_ros::nitros::NitrosDiagnosticsConfig{}, output_qos_)},
  output_image_width_(declare_parameter<uint16_t>("output_image_width", 1200)),
  output_image_height_(declare_parameter<uint16_t>("output_image_height", 1024)),
  padding_type_(declare_parameter<std::string>("padding_type", "CENTER")),
  border_type_(declare_parameter<std::string>("border_type", "CONSTANT")),
  border_pixel_color_value_(
    declare_parameter<std::vector<double>>("border_pixel_color_value", {0.0, 0.0, 0.0, 0.0}))
{
  auto img_padding_itr = kStringToPaddingTypeMap.find(padding_type_);
  if (img_padding_itr == std::end(kStringToPaddingTypeMap)) {
    RCLCPP_ERROR(get_logger(), "[PadNode] Unsupported padding type [%s]", padding_type_.c_str());
    throw std::invalid_argument("[PadNode] Unsupported padding type");
  }
  padding_type_val_ = img_padding_itr->second;

  auto img_border_itr = kStringToBorderTypeMap.find(border_type_);
  if (img_border_itr == std::end(kStringToBorderTypeMap)) {
    RCLCPP_ERROR(get_logger(), "[PadNode] Unsupported border type [%s]", border_type_.c_str());
    throw std::invalid_argument("[PadNode] Unsupported border type");
  }
  border_type_val_ = img_border_itr->second;

  if (border_pixel_color_value_.size() != 4) {
    RCLCPP_ERROR(
      get_logger(),
      "[PadNode] Invalid length of border_pixel_channel_values. Needed 4, given %ld",
      border_pixel_color_value_.size());
    throw std::invalid_argument("[PadNode] Invalid length of border_pixel_channel_values");
  }

  border_values_float_.push_back(static_cast<float>(border_pixel_color_value_[0]));
  border_values_float_.push_back(static_cast<float>(border_pixel_color_value_[1]));
  border_values_float_.push_back(static_cast<float>(border_pixel_color_value_[2]));
  border_values_float_.push_back(static_cast<float>(border_pixel_color_value_[3]));

  checkCudaErrors(cudaStreamCreate(&stream_));
}

PadNode::~PadNode()
{
  checkCudaErrors(cudaStreamDestroy(stream_));
}

void PadNode::InputCallback(const nvidia::isaac_ros::nitros::NitrosImageView & view)
{
  uint16_t input_width = view.GetWidth();
  uint16_t input_height = view.GetHeight();

  if ((input_width > output_image_width_) || (input_height > output_image_height_)) {
    RCLCPP_ERROR(
      get_logger(),
      "Input image dims > Output image dims."
    );
    throw std::runtime_error(
            "Error: Input image dims > Output image dims.");
  }
  auto input_image_format_itr = kStringEncToNVCVImageFormatMap.find(view.GetEncoding());
  if (input_image_format_itr == std::end(kStringEncToNVCVImageFormatMap)) {
    RCLCPP_ERROR(
      get_logger(),
      "Unsupported image encoding."
    );
    throw std::invalid_argument("[PadNode] Unsupported image encoding.");
  }
  nvcv::ImageFormat input_image_format = input_image_format_itr->second;

  nvcv::TensorDataStridedCuda::Buffer input_image_buffer;
  nvcv::TensorDataStridedCuda::Buffer output_image_buffer;
  nvcv::Tensor::Requirements out_reqs;
  nvcv::Tensor::Requirements in_reqs;
  nvcv::Tensor input_image_tensor;
  nvcv::Tensor output_image_tensor;

  InitializeBuffer(
    input_image_format, input_width, input_height,
    &input_image_buffer, &in_reqs
  );
  InitializeBuffer(
    input_image_format, output_image_width_, output_image_height_,
    &output_image_buffer, &out_reqs
  );

  // wrap the incoming image in CV-CUDA Tensor.
  input_image_buffer.basePtr = const_cast<NVCVByte *>(
    reinterpret_cast<const NVCVByte *>(view.GetGpuData()));

  nvcv::TensorDataStridedCuda in_data(
    nvcv::TensorShape{in_reqs.shape, in_reqs.rank, in_reqs.layout},
    nvcv::DataType{in_reqs.dtype}, input_image_buffer
  );
  input_image_tensor = nvcv::TensorWrapData(in_data);

  // Allocate and wrap the Output buffer.
  checkCudaErrors(
    cudaMallocAsync(&output_image_buffer.basePtr, output_image_buffer.strides[0], stream_)
  );
  nvcv::TensorDataStridedCuda out_data(
    nvcv::TensorShape{out_reqs.shape, out_reqs.rank, out_reqs.layout},
    nvcv::DataType{out_reqs.dtype}, output_image_buffer
  );
  output_image_tensor = nvcv::TensorWrapData(out_data);

  if (padding_type_val_ == PaddingType::kCenter) {
    int top = (output_image_height_ - input_height) / 2;
    int left = (output_image_width_ - input_width) / 2;

    make_border_op_(
      stream_, input_image_tensor,
      output_image_tensor, top, left, border_type_val_,
      {border_values_float_[0], border_values_float_[1],
        border_values_float_[2], border_values_float_[3]}
    );
  } else {
    // Initialize to 0 image.
    checkCudaErrors(
      cudaMemsetAsync(output_image_buffer.basePtr, 0, output_image_buffer.strides[0], stream_)
    );
    // Calculate offset for the specified type of padding
    uint32_t offset = CalculateOffset(
      input_width, input_height, output_image_width_, output_image_height_,
      padding_type_val_, output_image_buffer
    );
    // Copy input image to to the corner.
    checkCudaErrors(
      cudaMemcpy2DAsync(
        output_image_buffer.basePtr + offset,
        output_image_buffer.strides[1],
        input_image_buffer.basePtr,
        input_image_buffer.strides[1],
        input_image_buffer.strides[1],
        input_height,
        cudaMemcpyDefault,
        stream_
    ));
  }

  std_msgs::msg::Header header;
  header.stamp.sec = view.GetTimestampSeconds();
  header.stamp.nanosec = view.GetTimestampNanoseconds();
  header.frame_id = view.GetFrameId();

  checkCudaErrors(cudaStreamSynchronize(stream_));

  nvidia::isaac_ros::nitros::NitrosImage nitros_image =
    nvidia::isaac_ros::nitros::NitrosImageBuilder()
    .WithHeader(header)
    .WithEncoding(view.GetEncoding())
    .WithDimensions(output_image_height_, output_image_width_)
    .WithGpuData(output_image_buffer.basePtr)
    .Build();

  nitros_pub_->publish(nitros_image);
}
}  // namespace image_proc
}  // namespace isaac_ros
}  // namespace nvidia
// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::image_proc::PadNode)
