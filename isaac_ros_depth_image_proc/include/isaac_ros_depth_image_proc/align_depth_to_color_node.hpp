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

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_message_filters_subscriber.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_publisher.hpp"
#include "isaac_ros_nitros_camera_info_type/nitros_camera_info.hpp"
#include "isaac_ros_nitros_camera_info_type/nitros_camera_info_view.hpp"
#include "isaac_ros_nitros_image_type/nitros_image_builder.hpp"
#include "isaac_ros_nitros_image_type/nitros_image_view.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/exact_time.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "tf2_eigen/tf2_eigen.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"


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
  explicit AlignDepthToColorNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~AlignDepthToColorNode();

private:
  // CUDA stream
  cudaStream_t cuda_stream_;

  // QoS parameters
  int sync_queue_size_;
  rclcpp::QoS input_qos_;
  rclcpp::QoS output_qos_;

  // Synchronized subscribers
  Nitros::message_filters::Subscriber<Nitros::NitrosImageView> depth_sync_sub_;
  ::message_filters::Subscriber<Nitros::NitrosCameraInfo> depth_info_sync_sub_;
  ::message_filters::Subscriber<Nitros::NitrosCameraInfo> color_info_sync_sub_;

  // Subscribers for individual callbacks
  std::shared_ptr<Nitros::ManagedNitrosSubscriber<Nitros::NitrosImageView>> depth_sub_;
  std::shared_ptr<Nitros::ManagedNitrosSubscriber<Nitros::NitrosCameraInfoView>> depth_info_sub_;
  std::shared_ptr<Nitros::ManagedNitrosSubscriber<Nitros::NitrosCameraInfoView>> color_info_sub_;

  // Publisher for aligned depth
  std::shared_ptr<Nitros::ManagedNitrosPublisher<Nitros::NitrosImage>> aligned_depth_pub_;

  // Exact message sync policy for the depth image, depth camera info, and color camera info.
  using ExactPolicy = ::message_filters::sync_policies::ExactTime<
    Nitros::NitrosImage, Nitros::NitrosCameraInfo, Nitros::NitrosCameraInfo>;
  using ExactSync = ::message_filters::Synchronizer<ExactPolicy>;
  std::shared_ptr<ExactSync> exact_sync_;

  // TF buffer and listener for receiving extrinsics
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // 4x4 transformation matrix from depth to color frame
  std::optional<Eigen::Matrix4d> color_pose_depth_;

  // If true, using cached camera info and remove synchronization.
  bool use_cached_camera_info_;

  // Performance logging flag
  bool enable_performance_logging_;

  // Mutex for camera info
  mutable std::mutex camera_info_mutex_;

  // Cached camera infos used when use_cached_camera_info_ is true
  std::optional<sensor_msgs::msg::CameraInfo> depth_camera_info_;
  std::optional<sensor_msgs::msg::CameraInfo> color_camera_info_;
  // Buffer for depth image
  std::optional<Nitros::NitrosImageView> depth_image_buffer_;

  // Callback
  void OnSynchronizedInputs(
    const Nitros::NitrosImage::ConstSharedPtr & depth_msg,
    const Nitros::NitrosCameraInfo::ConstSharedPtr & depth_info_msg,
    const Nitros::NitrosCameraInfo::ConstSharedPtr & color_info_msg);

  void DepthCallback(const Nitros::NitrosImageView & msg);
  void DepthCameraInfoCallback(const Nitros::NitrosCameraInfoView & msg);
  void ColorCameraInfoCallback(const Nitros::NitrosCameraInfoView & msg);

  // Shared computation used by both synchronized and individual callbacks
  void ComputeAndPublishAlignedDepth(
    const Nitros::NitrosImageView & depth_view,
    const sensor_msgs::msg::CameraInfo & depth_camera_info,
    const sensor_msgs::msg::CameraInfo & color_camera_info);
};

}  // namespace depth_image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_DEPTH_IMAGE_PROC__ALIGN_DEPTH_TO_COLOR_NODE_HPP_
