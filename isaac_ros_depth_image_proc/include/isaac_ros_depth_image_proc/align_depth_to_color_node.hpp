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

#ifndef ISAAC_ROS_DEPTH_IMAGE_PROC__ALIGN_DEPTH_TO_COLOR_NODE_HPP_
#define ISAAC_ROS_DEPTH_IMAGE_PROC__ALIGN_DEPTH_TO_COLOR_NODE_HPP_

#include <Eigen/Dense>
#include <memory>
#include <optional>
#include <mutex>

#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/exact_time.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_nitros/types/nitros_type_message_filter_traits.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "tf2_eigen/tf2_eigen.hpp"


namespace nvidia
{
namespace isaac_ros
{
namespace depth_image_proc
{

namespace Nitros = nvidia::isaac_ros::nitros;

class AlignDepthToColorNode : public rclcpp::Node
{
public:
  explicit AlignDepthToColorNode(const rclcpp::NodeOptions & options);
  ~AlignDepthToColorNode();

private:
  // QoS parameters
  const int64_t memory_pool_block_size_;
  const int64_t memory_pool_num_blocks_;
  const uint16_t input_qos_size_;
  const uint16_t output_qos_size_;
  // If true, using cached camera info and remove synchronization.
  bool use_cached_camera_info_;
  // Performance logging flag
  bool enable_performance_logging_;
  // Synchronized subscribers
  ::message_filters::Subscriber<Nitros::NitrosImage> depth_sync_sub_;
  ::message_filters::Subscriber<sensor_msgs::msg::CameraInfo> depth_info_sync_sub_;
  ::message_filters::Subscriber<sensor_msgs::msg::CameraInfo> color_info_sync_sub_;

  // Subscribers for individual callbacks
  message_filters::Subscriber<nvidia::isaac_ros::nitros::NitrosImage> depth_sub_;
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> depth_info_sub_;
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> color_info_sub_;

  // Publisher for aligned depth
  rclcpp::Publisher<nvidia::isaac_ros::nitros::NitrosImage>::SharedPtr aligned_depth_pub_;

  // Exact message sync policy for the depth image, depth camera info, and color camera info.
  using ExactPolicy = ::message_filters::sync_policies::ExactTime<
    Nitros::NitrosImage, sensor_msgs::msg::CameraInfo, sensor_msgs::msg::CameraInfo>;
  using ExactSync = ::message_filters::Synchronizer<ExactPolicy>;
  message_filters::Synchronizer<ExactPolicy> exact_sync_;

  // TF buffer and listener for receiving extrinsics
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // 4x4 transformation matrix from depth to color frame
  std::optional<Eigen::Matrix4d> color_pose_depth_;

  // Mutex for camera info
  mutable std::mutex camera_info_mutex_;

  // Cached camera infos used when use_cached_camera_info_ is true
  std::optional<sensor_msgs::msg::CameraInfo> depth_camera_info_;
  std::optional<sensor_msgs::msg::CameraInfo> color_camera_info_;
  // Buffer for depth image
  std::optional<Nitros::NitrosImage> depth_image_buffer_;

  // Callback
  void OnSynchronizedInputs(
    const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & depth_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & depth_info_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & color_info_msg);

  void DepthCallback(
    const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & msg);
  void DepthCameraInfoCallback(
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg);
  void ColorCameraInfoCallback(
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg);

  // Shared computation used by both synchronized and individual callbacks
  void ComputeAndPublishAlignedDepth(
    const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr msg,
    const sensor_msgs::msg::CameraInfo & depth_camera_info,
    const sensor_msgs::msg::CameraInfo & color_camera_info);

  // CUDA Resources
  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;
  nvidia::isaac_ros::nitros::CUDAMemoryPool pool_;
};

}  // namespace depth_image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_DEPTH_IMAGE_PROC__ALIGN_DEPTH_TO_COLOR_NODE_HPP_
