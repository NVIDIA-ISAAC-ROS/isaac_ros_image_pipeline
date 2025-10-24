// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  sync_queue_size_(declare_parameter<int>("sync_queue_size", 10)),
  input_qos_{::isaac_ros::common::AddQosParameter(
      *this, kDefaultQoS, "input_qos")},
  output_qos_{::isaac_ros::common::AddQosParameter(
      *this, kDefaultQoS, "output_qos")},
  depth_sub_{std::make_shared<Nitros::ManagedNitrosSubscriber<Nitros::NitrosImageView>>(
      this,
      "depth_image",
      Nitros::nitros_image_32FC1_t::supported_type_name,
      std::bind(&AlignDepthToColorNode::DepthCallback, this, std::placeholders::_1),
      Nitros::NitrosDiagnosticsConfig{}, input_qos_)},
  depth_info_sub_{std::make_shared<Nitros::ManagedNitrosSubscriber<Nitros::NitrosCameraInfoView>>(
      this, "camera_info_depth", Nitros::nitros_camera_info_t::supported_type_name,
      std::bind(&AlignDepthToColorNode::DepthCameraInfoCallback, this, std::placeholders::_1),
      Nitros::NitrosDiagnosticsConfig{}, input_qos_)},
  color_info_sub_{std::make_shared<Nitros::ManagedNitrosSubscriber<Nitros::NitrosCameraInfoView>>(
      this, "camera_info_color", Nitros::nitros_camera_info_t::supported_type_name,
      std::bind(&AlignDepthToColorNode::ColorCameraInfoCallback, this, std::placeholders::_1),
      Nitros::NitrosDiagnosticsConfig{}, input_qos_)},
  use_cached_camera_info_(declare_parameter<bool>("use_cached_camera_info", false)),
  enable_performance_logging_(declare_parameter<bool>("enable_performance_logging", false))
{
  CHECK_CUDA_ERROR(
    ::nvidia::isaac_ros::common::initNamedCudaStream(
      cuda_stream_, "isaac_ros_align_depth_to_color_node"),
    "Error initializing CUDA stream");

  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // Subscribers (synchronized path only when not using the cached camera info)
  if (!use_cached_camera_info_) {
    depth_sync_sub_.subscribe(this, "depth_image", input_qos_.get_rmw_qos_profile());
    depth_info_sync_sub_.subscribe(this, "camera_info_depth", input_qos_.get_rmw_qos_profile());
    color_info_sync_sub_.subscribe(this, "camera_info_color", input_qos_.get_rmw_qos_profile());

    // Sync policy and callback
    exact_sync_ = std::make_shared<ExactSync>(
      ExactPolicy(sync_queue_size_), depth_sync_sub_, depth_info_sync_sub_, color_info_sync_sub_);
    exact_sync_->registerCallback(
      std::bind(
        &AlignDepthToColorNode::OnSynchronizedInputs, this,
        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
  }

  // Publisher
  aligned_depth_pub_ = std::make_shared<
    Nitros::ManagedNitrosPublisher<Nitros::NitrosImage>>(
    this, "aligned_depth",
    Nitros::nitros_image_32FC1_t::supported_type_name,
    Nitros::NitrosDiagnosticsConfig{}, output_qos_);
}

AlignDepthToColorNode::~AlignDepthToColorNode()
{
  CHECK_CUDA_ERROR(cudaStreamDestroy(cuda_stream_), "Error destroying CUDA stream");
}

void AlignDepthToColorNode::ComputeAndPublishAlignedDepth(
  const Nitros::NitrosImageView & depth_view,
  const sensor_msgs::msg::CameraInfo & depth_camera_info,
  const sensor_msgs::msg::CameraInfo & color_camera_info)
{
  // Performance timing
  auto start_time = std::chrono::high_resolution_clock::now();

  // Validate encodings
  if (depth_view.GetEncoding() != sensor_msgs::image_encodings::TYPE_32FC1) {
    RCLCPP_ERROR(get_logger(), "Depth image must be TYPE_32FC1 (meters)");
    throw std::runtime_error("Invalid depth image encoding");
  }

  const int depth_w = static_cast<int>(depth_view.GetWidth());
  const int depth_h = static_cast<int>(depth_view.GetHeight());
  const int color_w = static_cast<int>(color_camera_info.width);
  const int color_h = static_cast<int>(color_camera_info.height);
  // Make sure the depth and width of depth camera info and nitros image is the same.
  if (depth_w != static_cast<int>(depth_camera_info.width) ||
    depth_h != static_cast<int>(depth_camera_info.height))
  {
    RCLCPP_ERROR(get_logger(), "Depth image dimensions do not match depth camera info dimensions");
    throw std::runtime_error("Invalid depth image dimensions");
  }

  // Get GPU depth pointer directly - no host copies!
  const float * depth_gpu_ptr = reinterpret_cast<const float *>(depth_view.GetGpuData());

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

  // Allocate GPU memory for aligned depth output
  const size_t aligned_bytes = static_cast<size_t>(
    color_camera_info.width) * color_camera_info.height * sizeof(float);
  float * gpu_aligned = nullptr;
  CHECK_CUDA_ERROR(
    cudaMallocAsync(&gpu_aligned, aligned_bytes, cuda_stream_),
    "Error allocating GPU memory for aligned depth output");

  // Launch optimized GPU-only depth alignment
  float gpu_time_ms = 0.0f;
  CHECK_CUDA_ERROR(
    AlignDepthToColor(
      depth_gpu_ptr,
      gpu_aligned,
      depth_camera_info,
      color_camera_info,
      color_pose_depth_.value().data(),
      cuda_stream_,
      &gpu_time_ms  // Get GPU timing
    ), "Error aligning depth to color");

  CHECK_CUDA_ERROR(cudaStreamSynchronize(cuda_stream_), "Error synchronizing CUDA stream");

  if (enable_performance_logging_) {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();

    // Log performance metrics
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "Depth alignment performance: Total=%.2fms, GPU=%.2fms, Input=%dx%d, Output=%dx%d",
      total_time_ms, gpu_time_ms, depth_w, depth_h, color_w, color_h);
  }

  // Build and publish Nitros image (gpu_aligned stays on GPU)
  std_msgs::msg::Header out_header;
  out_header.frame_id = color_camera_info.header.frame_id;
  out_header.stamp.sec = depth_view.GetTimestampSeconds();
  out_header.stamp.nanosec = depth_view.GetTimestampNanoseconds();

  Nitros::NitrosImage out_img =
    Nitros::NitrosImageBuilder()
    .WithHeader(out_header)
    .WithDimensions(color_h, color_w)
    .WithEncoding(sensor_msgs::image_encodings::TYPE_32FC1)
    .WithGpuData(reinterpret_cast<uint8_t *>(gpu_aligned))
    .Build();

  aligned_depth_pub_->publish(out_img);
}

void AlignDepthToColorNode::OnSynchronizedInputs(
  const Nitros::NitrosImage::ConstSharedPtr & depth_msg,
  const Nitros::NitrosCameraInfo::ConstSharedPtr & depth_info_msg,
  const Nitros::NitrosCameraInfo::ConstSharedPtr & color_info_msg)
{
  // Views
  const Nitros::NitrosImageView depth_view{*depth_msg};

  // Convert NitrosCameraInfo to ROS CameraInfo for both cameras
  sensor_msgs::msg::CameraInfo depth_camera_info;
  sensor_msgs::msg::CameraInfo color_camera_info;
  try {
    rclcpp::TypeAdapter<Nitros::NitrosCameraInfo, sensor_msgs::msg::CameraInfo>
    ::convert_to_ros_message(*depth_info_msg, depth_camera_info);
    rclcpp::TypeAdapter<Nitros::NitrosCameraInfo, sensor_msgs::msg::CameraInfo>
    ::convert_to_ros_message(*color_info_msg, color_camera_info);
  } catch (const std::runtime_error & e) {
    RCLCPP_ERROR(get_logger(), "Failed to convert NitrosCameraInfo: %s", e.what());
    return;
  }

  ComputeAndPublishAlignedDepth(depth_view, depth_camera_info, color_camera_info);
}

void AlignDepthToColorNode::DepthCallback(const Nitros::NitrosImageView & msg)
{
  if (!use_cached_camera_info_) {
    return;
  }
  std::lock_guard<std::mutex> lock(camera_info_mutex_);

  if (!depth_camera_info_.has_value() || !color_camera_info_.has_value()) {
    RCLCPP_DEBUG(get_logger(), "Received depth image but don't have depth or color camera info !");
    // Save msg to a buffer so that if camera info comes in, we can compute the aligned depth
    depth_image_buffer_.emplace(msg);
    return;
  } else {
    ComputeAndPublishAlignedDepth(msg, depth_camera_info_.value(), color_camera_info_.value());
    depth_image_buffer_.reset();
  }
}

void AlignDepthToColorNode::DepthCameraInfoCallback(
  const Nitros::NitrosCameraInfoView & msg)
{
  if (!use_cached_camera_info_) {
    return;
  }
  std::lock_guard<std::mutex> lock(camera_info_mutex_);

  if (depth_camera_info_.has_value()) {
    RCLCPP_DEBUG(get_logger(), "Received depth camera info but already have it !");
    return;
  }

  // Convert NitrosCameraInfoView to ROS CameraInfo
  sensor_msgs::msg::CameraInfo depth_camera_info;
  try {
    rclcpp::TypeAdapter<Nitros::NitrosCameraInfo, sensor_msgs::msg::CameraInfo>
    ::convert_to_ros_message(msg.GetMessage(), depth_camera_info);
  } catch (const std::runtime_error & e) {
    RCLCPP_ERROR(get_logger(), "Failed to convert depth NitrosCameraInfo: %s", e.what());
    return;
  }

  depth_camera_info_ = depth_camera_info;

  if (depth_image_buffer_.has_value() && color_camera_info_.has_value()) {
    ComputeAndPublishAlignedDepth(
      depth_image_buffer_.value(), depth_camera_info_.value(), color_camera_info_.value());
    depth_image_buffer_.reset();
  }
}

void AlignDepthToColorNode::ColorCameraInfoCallback(
  const Nitros::NitrosCameraInfoView & msg)
{
  if (!use_cached_camera_info_) {
    return;
  }
  std::lock_guard<std::mutex> lock(camera_info_mutex_);

  if (color_camera_info_.has_value()) {
    RCLCPP_DEBUG(get_logger(), "Received color camera info but already have it !");
    return;
  }

  // Convert NitrosCameraInfoView to ROS CameraInfo
  sensor_msgs::msg::CameraInfo color_camera_info;
  try {
    rclcpp::TypeAdapter<Nitros::NitrosCameraInfo, sensor_msgs::msg::CameraInfo>
    ::convert_to_ros_message(msg.GetMessage(), color_camera_info);
  } catch (const std::runtime_error & e) {
    RCLCPP_ERROR(get_logger(), "Failed to convert color NitrosCameraInfo: %s", e.what());
    return;
  }

  color_camera_info_ = color_camera_info;

  if (depth_image_buffer_.has_value() && depth_camera_info_.has_value()) {
    ComputeAndPublishAlignedDepth(
      depth_image_buffer_.value(), depth_camera_info_.value(), color_camera_info_.value());
    depth_image_buffer_.reset();
  }
}

}  // namespace depth_image_proc
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::depth_image_proc::AlignDepthToColorNode)
