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

#include "isaac_ros_depth_image_proc/convert_metric_node.hpp"

#include <climits>

#include "isaac_ros_nitros_image_type/nitros_image_builder.hpp"
#include "sensor_msgs/image_encodings.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace depth_image_proc
{

namespace
{

inline void CheckCudaErrors(cudaError_t code, const char * file, const int line)
{
  if (code != cudaSuccess) {
    const std::string message = "CUDA error returned at " + std::string(file) + ":" +
      std::to_string(line) + ", Error code: " + std::to_string(code) +
      " (" + std::string(cudaGetErrorString(code)) + ")";
    throw std::runtime_error(message);
  }
}

constexpr size_t kBatchSize{1};
constexpr float kMillimetresToMetres = 0.001f;
constexpr float kConvertOpBeta = 0.0f;

}  // namespace

ConvertMetricNode::ConvertMetricNode(const rclcpp::NodeOptions options)
: rclcpp::Node("convert_metric_node", options),
  input_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos")},
  output_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos")},
  nitros_img_sub_{std::make_shared<::nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
        ::nvidia::isaac_ros::nitros::NitrosImageView>>(
      this, "image_raw", ::nvidia::isaac_ros::nitros::nitros_image_mono16_t::supported_type_name,
      std::bind(&ConvertMetricNode::DepthCallback, this,
      std::placeholders::_1), nvidia::isaac_ros::nitros::NitrosStatisticsConfig{},
      input_qos_)},
  nitros_img_pub_{std::make_shared<
      nvidia::isaac_ros::nitros::ManagedNitrosPublisher<nvidia::isaac_ros::nitros::NitrosImage>>(
      this, "image",
      nvidia::isaac_ros::nitros::nitros_image_32FC1_t::supported_type_name,
      nvidia::isaac_ros::nitros::NitrosStatisticsConfig{}, output_qos_)}
{
  CheckCudaErrors(cudaStreamCreate(&stream_), __FILE__, __LINE__);
}

void ConvertMetricNode::DepthCallback(
  const ::nvidia::isaac_ros::nitros::NitrosImageView & img_msg)
{
  if (img_msg.GetEncoding() != sensor_msgs::image_encodings::MONO16) {
    RCLCPP_ERROR(
      get_logger(),
      "Input image format is not MONO16 image. This node only supports MONO16 image."
      "The current image input is %s", img_msg.GetEncoding().c_str());
    return;
  }

  const uint32_t img_width{img_msg.GetWidth()};
  const uint32_t img_height{img_msg.GetHeight()};
  const int img_channels{sensor_msgs::image_encodings::numChannels(img_msg.GetEncoding())};

  nvcv::TensorDataStridedCuda::Buffer input_buffer;
  input_buffer.strides[3] =
    sensor_msgs::image_encodings::bitDepth(img_msg.GetEncoding()) / CHAR_BIT;
  input_buffer.strides[2] = img_channels * input_buffer.strides[3];
  input_buffer.strides[1] = img_msg.GetStride();
  input_buffer.strides[0] = img_msg.GetHeight() * input_buffer.strides[1];

  input_buffer.basePtr =
    const_cast<NVCVByte *>(reinterpret_cast<const NVCVByte *>(img_msg.GetGpuData()));

  nvcv::Tensor::Requirements input_reqs{nvcv::Tensor::CalcRequirements(
      kBatchSize, {static_cast<int32_t>(img_msg.GetWidth()),
        static_cast<int32_t>(img_msg.GetHeight())}, nvcv::FMT_U16)};

  nvcv::TensorDataStridedCuda input_data{
    nvcv::TensorShape{input_reqs.shape, input_reqs.rank, input_reqs.layout},
    nvcv::DataType{input_reqs.dtype}, input_buffer};

  nvcv::Tensor input_tensor{nvcv::TensorWrapData(input_data)};

  // Allocate the memory buffer ourselves rather than letting CV-CUDA allocate it
  float * raw_output_buffer{nullptr};
  const size_t output_buffer_size{img_width * img_height * img_channels * sizeof(float)};
  CheckCudaErrors(
    cudaMalloc(&raw_output_buffer, output_buffer_size), __FILE__, __LINE__);

  nvcv::TensorDataStridedCuda::Buffer output_buffer;
  output_buffer.strides[3] = sizeof(float);
  output_buffer.strides[2] = img_channels * output_buffer.strides[3];
  output_buffer.strides[1] = img_msg.GetWidth() * output_buffer.strides[2];
  output_buffer.strides[0] = img_msg.GetHeight() * output_buffer.strides[1];

  output_buffer.basePtr = reinterpret_cast<NVCVByte *>(raw_output_buffer);

  nvcv::Tensor::Requirements output_reqs{nvcv::Tensor::CalcRequirements(
      kBatchSize,
      {static_cast<int32_t>(img_msg.GetWidth()),
        static_cast<int32_t>(img_msg.GetHeight())}, nvcv::FMT_F32)};

  nvcv::TensorDataStridedCuda output_data{
    nvcv::TensorShape{output_reqs.shape, output_reqs.rank, output_reqs.layout},
    nvcv::DataType{output_reqs.dtype}, output_buffer};
  nvcv::Tensor output_tensor{nvcv::TensorWrapData(output_data)};

  // Convert from uint16_t -> float32.
  // And divide by 1000 to convert from millimeters -> meters
  convert_op_(stream_, input_tensor, output_tensor, kMillimetresToMetres, kConvertOpBeta);

  CheckCudaErrors(cudaStreamSynchronize(stream_), __FILE__, __LINE__);

  std_msgs::msg::Header header;
  header.frame_id = img_msg.GetFrameId();
  header.stamp.sec = img_msg.GetTimestampSeconds();
  header.stamp.nanosec = img_msg.GetTimestampNanoseconds();

  nvidia::isaac_ros::nitros::NitrosImage float_depth_image =
    nvidia::isaac_ros::nitros::NitrosImageBuilder()
    .WithHeader(header)
    .WithDimensions(img_height, img_width)
    .WithEncoding(sensor_msgs::image_encodings::TYPE_32FC1)
    .WithGpuData(raw_output_buffer)
    .Build();

  nitros_img_pub_->publish(float_depth_image);
}

ConvertMetricNode::~ConvertMetricNode()
{
  CheckCudaErrors(cudaStreamDestroy(stream_), __FILE__, __LINE__);
}

}  // namespace depth_image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::depth_image_proc::ConvertMetricNode)
