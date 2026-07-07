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

#include "isaac_ros_image_proc/rectify_node.hpp"

#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_cvcuda_utils/cvcuda_utilities.hpp"
#include "opencv2/calib3d.hpp"
#include <opencv2/opencv.hpp>

namespace nvidia
{
namespace isaac_ros
{
namespace image_proc
{
RectifyNode::RectifyNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("rectify_node", options),
  output_width_(declare_parameter<int16_t>("output_width", 1280)),
  output_height_(declare_parameter<int16_t>("output_height", 800)),
  interpolation_(declare_parameter<std::string>("interpolation", "cubic")),
  border_type_(declare_parameter<std::string>("border_type", "CONSTANT")),
  border_constant_(declare_parameter<std::vector<double>>(
    "border_constant",
      {0.0, 0.0, 0.0, 0.0})),
  align_corners_(declare_parameter<bool>("align_corners", true)),
  map_value_type_(declare_parameter<std::string>("map_value_type", "ABSOLUTE")),
  map_interpolation_type_(declare_parameter<std::string>("map_interpolation_type",
    "cubic")),
  memory_pool_block_size_(declare_parameter<int64_t>("memory_pool_block_size",
    output_width_ * output_height_ * 4)),
  memory_pool_num_blocks_(declare_parameter<int64_t>("memory_pool_num_blocks", 40)),
  input_queue_size_(declare_parameter<int64_t>("input_queue_size", 10)),
  output_queue_size_(declare_parameter<int64_t>("output_queue_size", 10)),
  image_sub_{},
  camera_info_sub_{},
  exact_sync_{ExactPolicy(input_queue_size_), image_sub_, camera_info_sub_}
{
  RCLCPP_DEBUG(get_logger(), "[RectifyNode] Constructor");

  border_ = cvcuda_utils::ToNVCVBorderType(border_type_);
  borderValue_ = {static_cast<float>(border_constant_[0]), static_cast<float>(border_constant_[1]),
    static_cast<float>(border_constant_[2]), static_cast<float>(border_constant_[3])};
  inInterp_ = cvcuda_utils::ToNVCVInterpolationType(interpolation_);
  mapInterp_ = cvcuda_utils::ToNVCVInterpolationType(map_interpolation_type_);
  mapValueType_ = cvcuda_utils::ToNVCVRemapMapValueType(map_value_type_);

  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("RectifyNode");

  // Create CUDA memory pool
  cudaError_t err = pool_.create(
    static_cast<size_t>(memory_pool_block_size_),
    static_cast<size_t>(memory_pool_num_blocks_),
    nvidia::isaac_ros::nitros::CUDAMemoryPool::MemoryType::Device);
  CHECK_CUDA_ERROR(err, "[RectifyNode] Failed to create CUDA memory pool");

  // This function sets the QoS parameter for publishers and subscribers setup by this NITROS node
  const rclcpp::QoS input_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "input_qos").keep_last(input_queue_size_);
  const rclcpp::QoS output_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "output_qos").keep_last(output_queue_size_);
  const rmw_qos_profile_t rmw_qos_profile = input_qos.get_rmw_qos_profile();

  // Subscription options (can be used for callback groups, etc.)
  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  // Publisher options
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  // Create subscribers
  exact_sync_.registerCallback(
    std::bind(
      &RectifyNode::InputCallback, this,
      std::placeholders::_1, std::placeholders::_2));
  image_sub_.subscribe(this, "image_raw", rmw_qos_profile, sub_options);
  camera_info_sub_.subscribe(this, "camera_info", rmw_qos_profile, sub_options);

  image_pub_ = create_publisher<nvidia::isaac_ros::nitros::NitrosImage>(
    "image_rect", output_qos, pub_options);
  camera_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(
    "camera_info_rect", output_qos, pub_options);
}

RectifyNode::~RectifyNode()
{
  RCLCPP_DEBUG(get_logger(), "[RectifyNode] Destructor");
  if (map_buffer_ != nullptr) {
    CHECK_CUDA_ERROR(cudaFreeAsync(map_buffer_, *cuda_stream_),
      "[RectifyNode] cudaFreeAsync for remap map buffer failed");
  }
}

nvcv::Tensor RectifyNode::WrapOpencvCVMapToCVCUDATensor(const cv::Mat & mat, void * map_buffer)
{
  RCLCPP_DEBUG(get_logger(), "[RectifyNode] WrapOpencvCVMapToCVCUDATensor");
  nvcv::TensorDataStridedCuda::Buffer tensor_buffer;
  tensor_buffer.strides[3] = sizeof(float);
  tensor_buffer.strides[2] = tensor_buffer.strides[3] * 2;  // FMT_2F32
  tensor_buffer.strides[1] = mat.cols * tensor_buffer.strides[2];
  tensor_buffer.strides[0] = mat.rows * tensor_buffer.strides[1];
  tensor_buffer.basePtr = reinterpret_cast<NVCVByte *>(map_buffer);

  constexpr size_t kBatchSize{1};
  nvcv::Tensor::Requirements reqs{nvcv::Tensor::CalcRequirements(
      kBatchSize, {static_cast<int32_t>(mat.cols),
        static_cast<int32_t>(mat.rows)}, nvcv::FMT_2F32)};
  nvcv::TensorDataStridedCuda tensor_data{
    nvcv::TensorShape{reqs.shape, reqs.rank, reqs.layout},
    nvcv::DataType{reqs.dtype}, tensor_buffer};
  return nvcv::TensorWrapData(tensor_data);
}

void RectifyNode::InitRemapMapAndOutputCameraInfo(
  const sensor_msgs::msg::CameraInfo & camera_info)
{
  RCLCPP_DEBUG(get_logger(), "[RectifyNode] InitRemapMapAndOutputCameraInfo");
  if (camera_info.k == cached_input_camera_info_.k &&
    camera_info.p == cached_input_camera_info_.p &&
    camera_info.r == cached_input_camera_info_.r &&
    camera_info.d == cached_input_camera_info_.d &&
    camera_info.distortion_model == cached_input_camera_info_.distortion_model)
  {
    return;
  }
  cached_input_camera_info_ = camera_info;

  cv::Size imageSize(camera_info.width,
    camera_info.height);
  cv::Matx33d intrinsics = {camera_info.k[0], 0.0, camera_info.k[2],
    0.0, camera_info.k[4], camera_info.k[5],
    0.0, 0.0, 1.0};

  cv::Matx33d new_intrinsics = {camera_info.p[0], 0.0, camera_info.p[2],
    0.0, camera_info.p[5], camera_info.p[6],
    0.0, 0.0, 1.0};

  cv::Matx<double, 3, 3> rotation = {camera_info.r[0], camera_info.r[1], camera_info.r[2],
    camera_info.r[3], camera_info.r[4], camera_info.r[5],
    camera_info.r[6], camera_info.r[7], camera_info.r[8]};

  // Local variable for remap map (CV_32FC2 format: interleaved x,y coordinates)
  cv::Mat remap_map;
  cv::Mat unused;
  if (camera_info.distortion_model == "equidistant") {
    // Only first 4 coefficients are needed for OpenCV fisheye model
    cv::Matx<double, 1, 4> coefficients = {camera_info.d[0], camera_info.d[1],
      camera_info.d[2], camera_info.d[3]};
    // Use fisheye::initUndistortRectifyMap for equidistant (fisheye) distortion model
    // Note: fisheye version only supports CV_32FC1 (separate x,y maps), so we merge them
    cv::Mat map_x, map_y;
    cv::fisheye::initUndistortRectifyMap(
      intrinsics, coefficients, rotation, new_intrinsics,
      imageSize, CV_32FC1, map_x, map_y);
    // Merge separate x,y maps into CV_32FC2 format for downstream compatibility
    cv::Mat maps[] = {map_x, map_y};
    cv::merge(maps, 2, remap_map);
  } else if (camera_info.distortion_model == "plumb_bob") {
    cv::Matx<double, 1, 5> coefficients = {camera_info.d[0], camera_info.d[1],
      camera_info.d[2], camera_info.d[3], camera_info.d[4]};
    cv::initUndistortRectifyMap(intrinsics, coefficients, rotation, new_intrinsics,
      imageSize, CV_32FC2, remap_map, unused);
  } else if (camera_info.distortion_model == "rational_polynomial") {
    cv::Matx<double, 1, 8> coefficients = {camera_info.d[0], camera_info.d[1], camera_info.d[2],
      camera_info.d[3], camera_info.d[4], camera_info.d[5],
      camera_info.d[6], camera_info.d[7]};
    cv::initUndistortRectifyMap(intrinsics, coefficients, rotation, new_intrinsics,
      imageSize, CV_32FC2, remap_map, unused);
  } else {
    RCLCPP_ERROR(get_logger(), "[RectifyNode] Unsupported distortion model: %s",
      camera_info.distortion_model.c_str());
    throw std::runtime_error("[RectifyNode] Unsupported distortion model: " +
      camera_info.distortion_model);
    return;
  }

  size_t map_buffer_size = remap_map.cols * remap_map.rows * sizeof(float) * 2;
  if (map_buffer_ == nullptr) {
    auto err = cudaMallocAsync(&map_buffer_, map_buffer_size, *cuda_stream_);
    CHECK_CUDA_ERROR(err, "[RectifyNode] cudaMalloc for remap map buffer failed");
  }
  CHECK_CUDA_ERROR(cudaMemcpyAsync(map_buffer_, remap_map.data, map_buffer_size,
    cudaMemcpyHostToDevice, *cuda_stream_),
    "[RectifyNode] cudaMemcpy map failed");
  CHECK_CUDA_ERROR(cudaStreamSynchronize(*cuda_stream_),
    "[RectifyNode] cudaStreamSynchronize for remap map failed");
  remap_map_ = WrapOpencvCVMapToCVCUDATensor(remap_map, map_buffer_);

  // Save output camera info
  cached_output_camera_info_ = camera_info;
  cached_output_camera_info_.d = {0.0, 0.0, 0.0, 0.0, 0.0};
  cached_output_camera_info_.k[0] = new_intrinsics(0, 0);  // fx
  cached_output_camera_info_.k[2] = new_intrinsics(0, 2);  // cx
  cached_output_camera_info_.k[4] = new_intrinsics(1, 1);  // fy
  cached_output_camera_info_.k[5] = new_intrinsics(1, 2);  // cy
}

void RectifyNode::InputCallback(
  const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & image,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info)
{
  RCLCPP_DEBUG(get_logger(), "[RectifyNode] InputCallback");

  InitRemapMapAndOutputCameraInfo(*camera_info);
  const int num_channels{sensor_msgs::image_encodings::numChannels(image->encoding)};
  const int bytes_per_element =
    sensor_msgs::image_encodings::bitDepth(image->encoding) / CHAR_BIT;
  const cvcuda_utils::NVCVImageFormat format = cvcuda_utils::ToNVCVFormat(image->encoding);
  auto input_handle = cvcuda_utils::WrapCVCUDATensor(
    *image, image->get_read_handle(*cuda_stream_),
    format.format, num_channels, bytes_per_element);

  // Create output images
  auto output_image = std::make_unique<nvidia::isaac_ros::nitros::NitrosImage>();
  size_t output_step = output_width_ * num_channels * bytes_per_element;
  auto output_write_handle = output_image->from_pool(
    pool_, output_width_, output_height_, output_step, image->encoding, *cuda_stream_);

  auto output_handle = cvcuda_utils::WrapCVCUDATensor(
    *output_image, std::move(output_write_handle),
    format.format, num_channels, bytes_per_element);

  // Remap
  remap_op_(*cuda_stream_, input_handle.get_tensor(), output_handle.get_tensor(),
    remap_map_, inInterp_, mapInterp_, mapValueType_, align_corners_, border_, borderValue_);

  output_image->timestamp_sec = image->timestamp_sec;
  output_image->timestamp_nsec = image->timestamp_nsec;
  output_image->frame_id = image->frame_id;

  auto camera_info_msg = std::make_unique<sensor_msgs::msg::CameraInfo>();
  *camera_info_msg = cached_output_camera_info_;
  camera_info_msg->header = camera_info->header;
  image_pub_->publish(std::move(output_image));
  camera_info_pub_->publish(std::move(camera_info_msg));
}

}  // namespace image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::image_proc::RectifyNode)
