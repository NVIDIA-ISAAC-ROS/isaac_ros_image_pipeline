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

#include "isaac_ros_image_proc/image_normalize_node.hpp"

#include <climits>

#include "isaac_ros_nitros_image_type/nitros_image_builder.hpp"
#include "nvcv/TensorDataAccess.hpp"
#include "sensor_msgs/image_encodings.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace image_proc
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

struct NVCVImageFormat
{
  nvcv::ImageFormat interleaved_format;
  nvcv::ImageFormat interleaved_float_format;
  std::string image_encoding_float;
};

NVCVImageFormat ToNVCVFormat(const std::string & image_encoding)
{
  static const std::unordered_map<std::string, NVCVImageFormat> str_to_nvcv_format({
            {sensor_msgs::image_encodings::RGB8,
              NVCVImageFormat{nvcv::FMT_RGB8, nvcv::FMT_RGBf32,
                sensor_msgs::image_encodings::TYPE_32FC3}},
            {sensor_msgs::image_encodings::BGR8,
              NVCVImageFormat{nvcv::FMT_BGR8, nvcv::FMT_BGRf32,
                sensor_msgs::image_encodings::TYPE_32FC3}},
            {sensor_msgs::image_encodings::RGBA8,
              NVCVImageFormat{nvcv::FMT_RGBA8, nvcv::FMT_RGBAf32,
                sensor_msgs::image_encodings::TYPE_32FC4}},
            {sensor_msgs::image_encodings::BGRA8,
              NVCVImageFormat{nvcv::FMT_BGRA8, nvcv::FMT_BGRAf32,
                sensor_msgs::image_encodings::TYPE_32FC4}},
            {sensor_msgs::image_encodings::MONO8,
              NVCVImageFormat{nvcv::FMT_U8, nvcv::FMT_F32,
                sensor_msgs::image_encodings::TYPE_32FC1}},
            {sensor_msgs::image_encodings::TYPE_32FC3,
              NVCVImageFormat{nvcv::FMT_RGBf32, nvcv::FMT_RGBf32,
                sensor_msgs::image_encodings::TYPE_32FC3}}
          });
  return str_to_nvcv_format.at(image_encoding);
}

constexpr size_t kBatchSize{1};

}  // namespace

ImageNormalizeNode::ImageNormalizeNode(const rclcpp::NodeOptions options)
: rclcpp::Node("image_normalize_node", options),
  input_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos")},
  output_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos")},
  nitros_img_sub_{std::make_shared<::nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
        ::nvidia::isaac_ros::nitros::NitrosImageView>>(
      this, "image", ::nvidia::isaac_ros::nitros::nitros_image_rgb8_t::supported_type_name,
      std::bind(&ImageNormalizeNode::ImageNormalizeCallback, this,
      std::placeholders::_1), nvidia::isaac_ros::nitros::NitrosDiagnosticsConfig{},
      input_qos_)},
  nitros_img_pub_{std::make_shared<
      nvidia::isaac_ros::nitros::ManagedNitrosPublisher<nvidia::isaac_ros::nitros::NitrosImage>>(
      this, "normalized_image",
      nvidia::isaac_ros::nitros::nitros_image_32FC1_t::supported_type_name,
      nvidia::isaac_ros::nitros::NitrosDiagnosticsConfig{}, output_qos_)},
  mean_param_{declare_parameter<std::vector<double>>("mean", {0.5, 0.5, 0.5})},
  stddev_param_(declare_parameter<std::vector<double>>("stddev", {0.5, 0.5, 0.5}))
{
  CheckCudaErrors(cudaStreamCreate(&stream_), __FILE__, __LINE__);
  std::vector<float> mean_float(mean_param_.begin(), mean_param_.end());
  std::vector<float> stddev_float(stddev_param_.begin(), stddev_param_.end());
  nvcv::TensorShape::ShapeType shape{nvcv::TensorShape::ShapeType{1, 1, 1,
      static_cast<int64_t>(mean_param_.size())}};

  nvcv::TensorShape tensor_shape{shape, nvcv::TENSOR_NHWC};

  mean_ = nvcv::Tensor(tensor_shape, nvcv::TYPE_F32);
  stddev_ = nvcv::Tensor(tensor_shape, nvcv::TYPE_F32);

  auto mean_data = mean_.exportData<nvcv::TensorDataStridedCuda>();
  auto mean_access = nvcv::TensorDataAccessStridedImagePlanar::Create(*mean_data);
  auto stddev_data = stddev_.exportData<nvcv::TensorDataStridedCuda>();
  auto stddev_access = nvcv::TensorDataAccessStridedImagePlanar::Create(*stddev_data);

  CheckCudaErrors(
    cudaMemcpy2D(
      mean_access->sampleData(0), mean_access->rowStride(), mean_float.data(),
      mean_float.size() * sizeof(float), mean_float.size() * sizeof(float), 1,
      cudaMemcpyHostToDevice),
    __FILE__, __LINE__);
  CheckCudaErrors(
    cudaMemcpy2D(
      stddev_access->sampleData(0), stddev_access->rowStride(), stddev_float.data(),
      stddev_float.size() * sizeof(float), stddev_float.size() * sizeof(float), 1,
      cudaMemcpyHostToDevice),
    __FILE__, __LINE__);
}

void ImageNormalizeNode::ImageNormalizeCallback(
  const ::nvidia::isaac_ros::nitros::NitrosImageView & img_msg)
{
  const uint32_t img_width{img_msg.GetWidth()};
  const uint32_t img_height{img_msg.GetHeight()};
  const int img_channels{sensor_msgs::image_encodings::numChannels(img_msg.GetEncoding())};
  const NVCVImageFormat format = ToNVCVFormat(img_msg.GetEncoding());

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
        static_cast<int32_t>(img_msg.GetHeight())}, format.interleaved_format)};

  nvcv::TensorDataStridedCuda input_data{
    nvcv::TensorShape{input_reqs.shape, input_reqs.rank, input_reqs.layout},
    nvcv::DataType{input_reqs.dtype}, input_buffer};

  nvcv::Tensor input_tensor{nvcv::TensorWrapData(input_data)};

  // Allocate the memory buffer ourselves rather than letting CV-CUDA allocate it
  float * raw_output_buffer{nullptr};
  const size_t output_buffer_size{img_width * img_height * img_channels * sizeof(float)};
  CheckCudaErrors(
    cudaMallocAsync(&raw_output_buffer, output_buffer_size, stream_), __FILE__, __LINE__);

  nvcv::TensorDataStridedCuda::Buffer output_buffer;
  output_buffer.strides[3] = sizeof(float);
  output_buffer.strides[2] = img_channels * output_buffer.strides[3];
  output_buffer.strides[1] = img_msg.GetWidth() * output_buffer.strides[2];
  output_buffer.strides[0] = img_msg.GetHeight() * output_buffer.strides[1];

  output_buffer.basePtr = reinterpret_cast<NVCVByte *>(raw_output_buffer);

  nvcv::Tensor::Requirements output_reqs{nvcv::Tensor::CalcRequirements(
      kBatchSize,
      {static_cast<int32_t>(img_msg.GetWidth()),
        static_cast<int32_t>(img_msg.GetHeight())}, format.interleaved_float_format)};

  nvcv::TensorDataStridedCuda output_data{
    nvcv::TensorShape{output_reqs.shape, output_reqs.rank, output_reqs.layout},
    nvcv::DataType{output_reqs.dtype}, output_buffer};
  nvcv::Tensor output_tensor{nvcv::TensorWrapData(output_data)};

  // Convert from uint8_t -> float32 and then normalize
  convert_op_(stream_, input_tensor, output_tensor, 1.0f, 0.0f);
  norm_op_(
    stream_, output_tensor, mean_, stddev_, output_tensor, 1.0f, 0.0f, 0.0f,
    CVCUDA_NORMALIZE_SCALE_IS_STDDEV);

  CheckCudaErrors(cudaStreamSynchronize(stream_), __FILE__, __LINE__);

  std_msgs::msg::Header header;
  header.frame_id = img_msg.GetFrameId();
  header.stamp.sec = img_msg.GetTimestampSeconds();
  header.stamp.nanosec = img_msg.GetTimestampNanoseconds();

  nvidia::isaac_ros::nitros::NitrosImage normalized_image =
    nvidia::isaac_ros::nitros::NitrosImageBuilder()
    .WithHeader(header)
    .WithDimensions(img_height, img_width)
    .WithEncoding(format.image_encoding_float)
    .WithGpuData(raw_output_buffer)
    .Build();

  nitros_img_pub_->publish(normalized_image);
}

ImageNormalizeNode::~ImageNormalizeNode()
{
  CheckCudaErrors(cudaStreamDestroy(stream_), __FILE__, __LINE__);
}

}  // namespace image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::image_proc::ImageNormalizeNode)
