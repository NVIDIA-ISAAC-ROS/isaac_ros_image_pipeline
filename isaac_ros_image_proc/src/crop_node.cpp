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

#include "isaac_ros_image_proc/crop_node.hpp"

#include <cstdio>
#include <memory>
#include <string>
#include <utility>

#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_cvcuda_utils/cvcuda_utilities.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace image_proc
{
using nvidia::isaac_ros::nitros::NitrosImage;
namespace img_encodings = sensor_msgs::image_encodings;

// User string to CROP mode
const std::unordered_map<std::string, CropMode> CROP_MODE_MAP({
        {"CENTER", CropMode::kCenter},
        {"LEFT", CropMode::kLeft},
        {"RIGHT", CropMode::kRight},
        {"TOP", CropMode::kTop},
        {"BOTTOM", CropMode::kBottom},
        {"TOPLEFT", CropMode::kTopLeft},
        {"TOPRIGHT", CropMode::kTopRight},
        {"BOTTOMLEFT", CropMode::kBottomLeft},
        {"BOTTOMRIGHT", CropMode::kBottomRight},
        {"BBOX", CropMode::kBBox}
      });

CropNode::CropNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("crop_node", options),
  input_width_(declare_parameter<int64_t>("input_width", 0)),
  input_height_(declare_parameter<int64_t>("input_height", 0)),
  crop_width_(declare_parameter<int64_t>("crop_width", 0)),
  crop_height_(declare_parameter<int64_t>("crop_height", 0)),
  roi_top_left_x_(declare_parameter<int64_t>("roi_top_left_x", 0)),
  roi_top_left_y_(declare_parameter<int64_t>("roi_top_left_y", 0)),
  crop_mode_(declare_parameter<std::string>("crop_mode", "")),
  memory_pool_block_size_(declare_parameter<int64_t>("memory_pool_block_size", 1920 * 1200 * 4)),
  memory_pool_num_blocks_(declare_parameter<int64_t>("memory_pool_num_blocks", 40)),
  input_queue_size_(declare_parameter<int64_t>("input_queue_size", 10)),
  output_queue_size_(declare_parameter<int64_t>("output_queue_size", 10)),
  image_sub_{},
  camera_info_sub_{},
  exact_sync_{ExactPolicy{input_queue_size_}, image_sub_, camera_info_sub_}
{
  RCLCPP_DEBUG(get_logger(), "[CropNode] Constructor");

  roi_ = {static_cast<int32_t>(roi_top_left_x_),
    static_cast<int32_t>(roi_top_left_y_),
    static_cast<int32_t>(crop_width_),
    static_cast<int32_t>(crop_height_)};
  if (roi_.x < 0 || roi_.y < 0 || roi_.width < 0 || roi_.height < 0) {
    RCLCPP_ERROR(get_logger(), "[CropNode] Invalid ROI. Please select the valid value.");
    throw std::invalid_argument("[CropNode] Invalid ROI. Please select the valid value.");
  }

  if (input_width_ <= 0 || input_height_ <= 0 || crop_width_ <= 0 || crop_height_ <= 0) {
    RCLCPP_ERROR(get_logger(), "[CropNode] Width and height need to be non-zero positive number");
    throw std::invalid_argument(
            "[CropNode] Invalid output dimension "
            "Width and height need to be non-zero positive number.");
  }

  if (crop_mode_.empty()) {
    RCLCPP_ERROR(get_logger(), "[CropNode] Crop Mode is not set. Please select the valid value.");
    throw std::invalid_argument("[CropNode] Crop Mode is not set. Please select the valid value.");
  }

  const auto crop_mode = CROP_MODE_MAP.find(crop_mode_);
  if (crop_mode == std::end(CROP_MODE_MAP)) {
    RCLCPP_ERROR(get_logger(), "[CropNode] Unsupported crop mode: [%s]", crop_mode_.c_str());
    throw std::invalid_argument("[CropNode] Unsupported crop mode.");
  } else {
    CalculateResizeAndCropParams(crop_mode->second);
  }

  // Create CUDA stream
  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("crop_node");

  // Create CUDA memory pool
  cudaError_t err = pool_.create(
    static_cast<size_t>(memory_pool_block_size_),
    static_cast<size_t>(memory_pool_num_blocks_),
    nvidia::isaac_ros::nitros::CUDAMemoryPool::MemoryType::Device);
  CHECK_CUDA_ERROR(err, "Failed to create CUDA memory pool");
  const rclcpp::QoS input_qos =
    ::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos")
    .keep_last(input_queue_size_);
  const rclcpp::QoS output_qos =
    ::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos")
    .keep_last(output_queue_size_);
  const rmw_qos_profile_t input_qos_profile = input_qos.get_rmw_qos_profile();

  // Subscription options
  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  // Publisher options
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  // Create subscribers
  exact_sync_.registerCallback(
    std::bind(
      &CropNode::InputCallback, this,
      std::placeholders::_1, std::placeholders::_2));
  image_sub_.subscribe(this, "image", input_qos_profile, sub_options);
  camera_info_sub_.subscribe(this, "camera_info", input_qos_profile, sub_options);
  RCLCPP_DEBUG(get_logger(), "[CropNode] subscribers created");

  // Create publishers
  image_pub_ = create_publisher<NitrosImage>(
    "crop/image", output_qos, pub_options);
  camera_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(
    "crop/camera_info", output_qos);

  RCLCPP_DEBUG(get_logger(), "[CropNode] publishers created");
}


CropNode::~CropNode() {}

void CropNode::InputCallback(
  const NitrosImage::ConstSharedPtr & nitros_image,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info
)
{
  RCLCPP_DEBUG(get_logger(), "[CropNode] InputCallback - SYNCHRONIZED!");

  if (!nitros_image || !camera_info) {
    throw std::runtime_error("[CropNode] No inputs received");
  }

  auto input_encoding = nitros_image->encoding;
  cvcuda_utils::NVCVImageFormat format = cvcuda_utils::ToNVCVFormat(input_encoding);

  int num_channels{sensor_msgs::image_encodings::numChannels(input_encoding)};
  int bytes_per_channel = sensor_msgs::image_encodings::bitDepth(input_encoding) / CHAR_BIT;
  auto input_handle = cvcuda_utils::WrapCVCUDATensor(
    *nitros_image, nitros_image->get_read_handle(*cuda_stream_), format.format, num_channels,
    bytes_per_channel);

  auto output_msg = std::make_unique<NitrosImage>();
  size_t output_step = crop_width_ * num_channels * bytes_per_channel;
  auto output_write_handle = output_msg->from_pool(
    pool_, crop_width_, crop_height_, output_step, input_encoding, *cuda_stream_);

  auto output_handle = cvcuda_utils::WrapCVCUDATensor(
    *output_msg, std::move(output_write_handle), format.format, num_channels,
    bytes_per_channel);
  crop_op_(*cuda_stream_, input_handle.get_tensor(), output_handle.get_tensor(),
    roi_);

  output_msg->timestamp_sec = nitros_image->timestamp_sec;
  output_msg->timestamp_nsec = nitros_image->timestamp_nsec;
  output_msg->frame_id = nitros_image->frame_id;

  auto camera_info_output = std::make_unique<sensor_msgs::msg::CameraInfo>();
  UpdateCameraInfo(*camera_info, *camera_info_output);

  image_pub_->publish(std::move(output_msg));
  camera_info_pub_->publish(std::move(camera_info_output));
}

void CropNode::UpdateCameraInfo(
  const sensor_msgs::msg::CameraInfo & input_camera_info,
  sensor_msgs::msg::CameraInfo & output_camera_info)
{
  output_camera_info = input_camera_info;

  output_camera_info.width = crop_width_;
  output_camera_info.height = crop_height_;
  const float scaler_x = static_cast<float>(crop_width_) / input_camera_info.width;
  const float scaler_y = static_cast<float>(crop_height_) / input_camera_info.height;
  const float pixel_center = 0.5f;
  // Update the focal length
  output_camera_info.k[0] = input_camera_info.k[0] * scaler_x;
  output_camera_info.k[4] = input_camera_info.k[4] * scaler_y;
  // Update the principal point
  output_camera_info.k[2] = (input_camera_info.k[2] + pixel_center) * scaler_x - pixel_center;
  output_camera_info.k[5] = (input_camera_info.k[5] + pixel_center) * scaler_y - pixel_center;

  output_camera_info.p[0] = output_camera_info.k[0];
  output_camera_info.p[1] = 0;
  output_camera_info.p[2] = output_camera_info.k[2];
  output_camera_info.p[3] = input_camera_info.p[3] * output_camera_info.p[0];
  output_camera_info.p[4] = 0;
  output_camera_info.p[5] = output_camera_info.k[4];
  output_camera_info.p[6] = output_camera_info.k[5];
  output_camera_info.p[7] = input_camera_info.p[7];
  output_camera_info.p[8] = 0;
  output_camera_info.p[9] = 0;
  output_camera_info.p[10] = 1;
  output_camera_info.p[11] = input_camera_info.p[11];

  output_camera_info.roi.height = crop_height_;
  output_camera_info.roi.width = crop_width_;
}

void CropNode::CalculateResizeAndCropParams(const CropMode & crop_mode)
{
  roi_.width = static_cast<int32_t>(crop_width_);
  roi_.height = static_cast<int32_t>(crop_height_);
  switch (crop_mode) {
    case CropMode::kCenter: {
        roi_.x = (input_width_ - crop_width_) / 2;
        roi_.y = (input_height_ - crop_height_) / 2;
        break;
      }
    case CropMode::kLeft: {
        roi_.x = 0;
        roi_.y = (input_height_ - crop_height_) / 2;
        break;
      }
    case CropMode::kRight: {
        roi_.x = (input_width_ - crop_width_);
        roi_.y = (input_height_ - crop_height_) / 2;
        break;
      }
    case CropMode::kTop: {
        roi_.x = (input_width_ - crop_width_) / 2;
        roi_.y = 0;
        break;
      }
    case CropMode::kBottom: {
        roi_.x = (input_width_ - crop_width_) / 2;
        roi_.y = (input_height_ - crop_height_);
        break;
      }
    case CropMode::kTopLeft: {
        roi_.x = 0;
        roi_.y = 0;
        break;
      }
    case CropMode::kTopRight: {
        roi_.x = (input_width_ - crop_width_);
        roi_.y = 0;
        break;
      }
    case CropMode::kBottomLeft: {
        roi_.x = 0;
        roi_.y = (input_height_ - crop_height_);
        break;
      }
    case CropMode::kBottomRight: {
        roi_.x = (input_width_ - crop_width_);
        roi_.y = (input_height_ - crop_height_);
        break;
      }
    case CropMode::kBBox: {
        // Use user provided roi values.
        break;
      }
    default: {
        RCLCPP_ERROR(get_logger(), "Unsupported CropMode.");
        throw std::runtime_error("Unsupported CropMode.");
      }
  }
}

}  // namespace image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::image_proc::CropNode)
