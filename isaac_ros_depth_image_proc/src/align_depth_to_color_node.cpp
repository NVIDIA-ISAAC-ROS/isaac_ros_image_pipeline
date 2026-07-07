// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_depth_image_proc/align_depth_to_color_node.hpp"

#include <cmath>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <vector>

#include <opencv2/opencv.hpp>

#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_depth_image_proc/align_depth_to_color_node.cu.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace depth_image_proc
{

namespace
{
constexpr const char kDefaultQoS[] = "DEFAULT";

bool LookupTransformMatrix(
  tf2_ros::Buffer & tf_buffer,
  const std::string & target_frame,
  const std::string & source_frame,
  Eigen::Matrix4d & target_pose_source)
{
  try {
    const geometry_msgs::msg::TransformStamped stamped =
      tf_buffer.lookupTransform(target_frame, source_frame, tf2::TimePointZero);
    const Eigen::Isometry3d eig_transform = tf2::transformToEigen(stamped);
    target_pose_source = eig_transform.matrix();
    return true;
  } catch (const tf2::TransformException & ex) {
    return false;
  }
}
}  // namespace

AlignDepthToColorNode::AlignDepthToColorNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("align_depth_to_color_node", options),
  memory_pool_block_size_(declare_parameter<int64_t>("memory_pool_block_size", 1920 * 1200 * 4)),
  memory_pool_num_blocks_(declare_parameter<int64_t>("memory_pool_num_blocks", 40)),
  input_qos_size_(declare_parameter<uint16_t>("input_qos_size", 10)),
  output_qos_size_(declare_parameter<uint16_t>("output_qos_size", 10)),
  use_cached_camera_info_(declare_parameter<bool>("use_cached_camera_info", false)),
  enable_performance_logging_(declare_parameter<bool>("enable_performance_logging", false)),
  depth_sub_{},
  depth_info_sub_{},
  color_info_sub_{},
  aligned_depth_pub_{},
  exact_sync_{ExactPolicy(static_cast<int>(input_qos_size_)), depth_sub_, depth_info_sub_,
    color_info_sub_}
{
  // create CUDA stream
  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("AlignDepthToColorNode");

  // create CUDA memory pool
  CHECK_CUDA_ERROR(pool_.create(
    static_cast<size_t>(memory_pool_block_size_),
    static_cast<size_t>(memory_pool_num_blocks_),
    nvidia::isaac_ros::nitros::CUDAMemoryPool::MemoryType::Device),
    "Failed to create CUDA memory pool");

  const rclcpp::QoS input_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "input_qos").keep_last(input_qos_size_);
  const rclcpp::QoS output_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "output_qos").keep_last(output_qos_size_);
  const rmw_qos_profile_t rmw_qos_profile = input_qos.get_rmw_qos_profile();

  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  depth_sub_.subscribe(this, "depth_image", rmw_qos_profile, sub_options);
  depth_info_sub_.subscribe(this, "camera_info_depth", rmw_qos_profile, sub_options);
  color_info_sub_.subscribe(this, "camera_info_color", rmw_qos_profile, sub_options);
  depth_sub_.registerCallback(
    std::bind(&AlignDepthToColorNode::DepthCallback, this, std::placeholders::_1));
  depth_info_sub_.registerCallback(
    std::bind(&AlignDepthToColorNode::DepthCameraInfoCallback, this, std::placeholders::_1));
  color_info_sub_.registerCallback(
    std::bind(&AlignDepthToColorNode::ColorCameraInfoCallback, this, std::placeholders::_1));

  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // Subscribers (synchronized path only when not using the cached camera info)
  if (!use_cached_camera_info_) {
    depth_sync_sub_.subscribe(this, "depth_image", rmw_qos_profile, sub_options);
    depth_info_sync_sub_.subscribe(this, "camera_info_depth", rmw_qos_profile, sub_options);
    color_info_sync_sub_.subscribe(this, "camera_info_color", rmw_qos_profile, sub_options);
    exact_sync_.registerCallback(
      std::bind(
        &AlignDepthToColorNode::OnSynchronizedInputs, this,
        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
  }

  // Publisher
  aligned_depth_pub_ = create_publisher<nvidia::isaac_ros::nitros::NitrosImage>(
    "aligned_depth", output_qos, pub_options);
}

AlignDepthToColorNode::~AlignDepthToColorNode() {}

void AlignDepthToColorNode::ComputeAndPublishAlignedDepth(
  const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr msg,
  const sensor_msgs::msg::CameraInfo & depth_camera_info,
  const sensor_msgs::msg::CameraInfo & color_camera_info)
{
  // Performance timing
  auto start_time = std::chrono::high_resolution_clock::now();

  // Validate encodings
  if (msg->encoding != sensor_msgs::image_encodings::TYPE_32FC1) {
    RCLCPP_ERROR(get_logger(), "Depth image must be TYPE_32FC1 (meters)");
    throw std::runtime_error("Invalid depth image encoding");
  }

  const auto depth_w = static_cast<int>(msg->width);
  const auto depth_h = static_cast<int>(msg->height);
  const auto color_w = static_cast<int>(color_camera_info.width);
  const auto color_h = static_cast<int>(color_camera_info.height);
  // Make sure the depth and width of depth camera info and nitros image is the same.
  if (depth_w != static_cast<int>(depth_camera_info.width) ||
    depth_h != static_cast<int>(depth_camera_info.height))
  {
    RCLCPP_ERROR(get_logger(), "Depth image dimensions do not match depth camera info dimensions");
    throw std::runtime_error("Invalid depth image dimensions");
  }

  // Create read_handle
  auto read_handle = msg->get_read_handle(*cuda_stream_);
  const auto depth_gpu_ptr = reinterpret_cast<const float *>(read_handle.get_ptr());

  // Lookup 4x4 transform depth->color from TF
  const std::string depth_frame = depth_camera_info.header.frame_id;
  const std::string color_frame = color_camera_info.header.frame_id;
  if (!color_pose_depth_.has_value()) {
    Eigen::Matrix4d color_pose_depth;
    if (!LookupTransformMatrix(*tf_buffer_, color_frame, depth_frame, color_pose_depth)) {
      RCLCPP_WARN(
        get_logger(), "Could not transform %s to %s",
        color_frame.c_str(), depth_frame.c_str());
      return;
    }
    color_pose_depth_ = color_pose_depth;
  }

  // Allocate output image from pool (from_pool sets dimensions and encoding)
  auto aligned_depth_msg = std::make_unique<Nitros::NitrosImage>();
  auto write_handle = aligned_depth_msg->from_pool(
    pool_, color_w, color_h, color_w * sizeof(float),
    sensor_msgs::image_encodings::TYPE_32FC1, *cuda_stream_);
  auto gpu_aligned = reinterpret_cast<float *>(write_handle.get_ptr());

  // Copy header from input
  aligned_depth_msg->timestamp_sec = msg->get_timestamp_sec();
  aligned_depth_msg->timestamp_nsec = msg->get_timestamp_nsec();
  aligned_depth_msg->frame_id = msg->get_frame_id();

  // Launch optimized GPU-only depth alignment
  float gpu_time_ms = 0.0f;
  CHECK_CUDA_ERROR(AlignDepthToColor(depth_gpu_ptr, gpu_aligned, depth_camera_info,
      color_camera_info, color_pose_depth_.value().cast<double>().data(), *cuda_stream_,
      &gpu_time_ms),
    "Error aligning depth to color");

  if (enable_performance_logging_) {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();

    // Log performance metrics
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "Depth alignment performance: Total=%.2fms, GPU=%.2fms, Input=%dx%d, Output=%dx%d",
      total_time_ms, gpu_time_ms, depth_w, depth_h, color_w, color_h);
  }

  // Publish Nitros image (GPU buffer owned by aligned_depth_msg)
  aligned_depth_pub_->publish(std::move(aligned_depth_msg));
}

void AlignDepthToColorNode::OnSynchronizedInputs(
  const Nitros::NitrosImage::ConstSharedPtr & depth_msg,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & depth_info_msg,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & color_info_msg)
{
  ComputeAndPublishAlignedDepth(depth_msg, *depth_info_msg, *color_info_msg);
}

void AlignDepthToColorNode::DepthCallback(
  const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & msg
)
{
  if (!use_cached_camera_info_) {
    return;
  }
  std::lock_guard<std::mutex> lock(camera_info_mutex_);

  if (!depth_camera_info_.has_value() || !color_camera_info_.has_value()) {
    RCLCPP_DEBUG(get_logger(), "Received depth image but don't have depth or color camera info !");
    // Save msg to a buffer so that if camera info comes in, we can compute the aligned depth
    depth_image_buffer_.emplace(*msg);
    return;
  } else {
    ComputeAndPublishAlignedDepth(msg, depth_camera_info_.value(), color_camera_info_.value());
    depth_image_buffer_.reset();
  }
}

void AlignDepthToColorNode::DepthCameraInfoCallback(
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg)
{
  if (!use_cached_camera_info_) {
    return;
  }
  std::lock_guard<std::mutex> lock(camera_info_mutex_);

  if (depth_camera_info_.has_value()) {
    RCLCPP_DEBUG(get_logger(), "Received depth camera info but already have it !");
    return;
  }

  depth_camera_info_ = *msg;

  if (depth_image_buffer_.has_value() && color_camera_info_.has_value()) {
    ComputeAndPublishAlignedDepth(
      std::make_shared<Nitros::NitrosImage>(depth_image_buffer_.value()),
      depth_camera_info_.value(), color_camera_info_.value());
    depth_image_buffer_.reset();
  }
}

void AlignDepthToColorNode::ColorCameraInfoCallback(
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg)
{
  if (!use_cached_camera_info_) {
    return;
  }
  std::lock_guard<std::mutex> lock(camera_info_mutex_);

  if (color_camera_info_.has_value()) {
    RCLCPP_DEBUG(get_logger(), "Received color camera info but already have it !");
    return;
  }

  color_camera_info_ = *msg;

  if (depth_image_buffer_.has_value() && depth_camera_info_.has_value()) {
    ComputeAndPublishAlignedDepth(
      std::make_shared<Nitros::NitrosImage>(depth_image_buffer_.value()),
      depth_camera_info_.value(), color_camera_info_.value());
    depth_image_buffer_.reset();
  }
}

}  // namespace depth_image_proc
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::depth_image_proc::AlignDepthToColorNode)
