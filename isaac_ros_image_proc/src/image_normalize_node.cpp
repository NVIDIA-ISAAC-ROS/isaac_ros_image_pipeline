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

#include "isaac_ros_image_proc/image_normalize_node.hpp"

#include <climits>

#include "isaac_ros_cvcuda_utils/cvcuda_utilities.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_cvcuda_utils/cvcuda_handle.hpp"
#include "nvcv/TensorDataAccess.hpp"
#include "sensor_msgs/image_encodings.hpp"

using nvidia::isaac_ros::nitros::NitrosImage;
using nvidia::isaac_ros::nitros::CUDAMemoryPool;

namespace nvidia
{
namespace isaac_ros
{
namespace image_proc
{
ImageNormalizeNode::ImageNormalizeNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("image_normalize_node", options),
  input_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos")},
  output_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos")},
  mean_param_{declare_parameter<std::vector<double>>("mean", {0.5, 0.5, 0.5})},
  stddev_param_(declare_parameter<std::vector<double>>("stddev", {0.5, 0.5, 0.5})),
  // Default assumes 3-channel RGB at 1920x1200.
  // Pool will auto-resize if actual image requires more.
  memory_pool_block_size_(declare_parameter<int64_t>(
    "memory_pool_block_size", 1920 * 1200 * 3 * static_cast<int64_t>(sizeof(float)))),
  memory_pool_num_blocks_(declare_parameter<int64_t>("memory_pool_num_blocks", 40))
{
  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("ImageNormalizeNode");

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

  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  // Create subscribers and publishers
  image_sub_ = create_subscription<NitrosImage>(
    "image", input_qos_,
    std::bind(&ImageNormalizeNode::imageSubCallback, this, std::placeholders::_1),
    sub_options);
  image_pub_ = create_publisher<NitrosImage>(
    "normalized_image", output_qos_, pub_options);
  CHECK_CUDA_ERROR(
    cudaMemcpy2D(
      mean_access->sampleData(0), mean_access->rowStride(), mean_float.data(),
      mean_float.size() * sizeof(float), mean_float.size() * sizeof(float), 1,
      cudaMemcpyHostToDevice),
    "cudaMemcpy2D failed");
  CHECK_CUDA_ERROR(
    cudaMemcpy2D(
      stddev_access->sampleData(0), stddev_access->rowStride(), stddev_float.data(),
      stddev_float.size() * sizeof(float), stddev_float.size() * sizeof(float), 1,
      cudaMemcpyHostToDevice),
    "cudaMemcpy2D failed");
}

void ImageNormalizeNode::imageSubCallback(const NitrosImage::SharedPtr msg)
{
  auto encoding = msg->encoding;
  int num_channels{sensor_msgs::image_encodings::numChannels(encoding)};
  int bytes_per_channel = sensor_msgs::image_encodings::bitDepth(encoding) / CHAR_BIT;
  const cvcuda_utils::NVCVImageFormat format = cvcuda_utils::ToNVCVFormat(encoding);

  auto input_handle = cvcuda_utils::WrapCVCUDATensor(
    *msg, msg->get_read_handle(*cuda_stream_),
    format.format, num_channels, bytes_per_channel);

  const uint32_t output_step = msg->width * num_channels * sizeof(float);

  const size_t required_size =
    static_cast<size_t>(output_step) * static_cast<size_t>(msg->height);
  if (!pool_.initialized() || required_size > pool_.block_size()) {
    pool_.destroy();
    const int64_t actual_block_size = std::max(
      memory_pool_block_size_, static_cast<int64_t>(required_size));
    CHECK_CUDA_ERROR(pool_.create(
      static_cast<size_t>(actual_block_size),
      static_cast<size_t>(memory_pool_num_blocks_),
      CUDAMemoryPool::MemoryType::Device),
      "Failed to create CUDA memory pool");
  }

  auto output_msg = std::make_unique<NitrosImage>();
  auto output_write_handle = output_msg->from_pool(
    pool_, msg->width, msg->height, output_step,
    format.float_encoding, *cuda_stream_);
  num_channels = sensor_msgs::image_encodings::numChannels(format.float_encoding);
  bytes_per_channel = sensor_msgs::image_encodings::bitDepth(format.float_encoding) / CHAR_BIT;
  auto output_handle = cvcuda_utils::WrapCVCUDATensor(
    *output_msg, std::move(output_write_handle),
    format.float_format, num_channels, bytes_per_channel);

  convert_op_(*cuda_stream_, input_handle.get_tensor(), output_handle.get_tensor(), 1.0f, 0.0f);
  norm_op_(*cuda_stream_, output_handle.get_tensor(), mean_, stddev_, output_handle.get_tensor(),
    1.0f, 0.0f, 0.0f, CVCUDA_NORMALIZE_SCALE_IS_STDDEV);

  output_msg->timestamp_sec = msg->timestamp_sec;
  output_msg->timestamp_nsec = msg->timestamp_nsec;
  output_msg->frame_id = msg->frame_id;

  image_pub_->publish(std::move(output_msg));
}

ImageNormalizeNode::~ImageNormalizeNode() {}

}  // namespace image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::image_proc::ImageNormalizeNode)
