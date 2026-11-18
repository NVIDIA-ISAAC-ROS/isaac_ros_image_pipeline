// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef ISAAC_ROS_DEPTH_IMAGE_PROC__POINT_CLOUD_XYZ_MASKED_NODE_HPP_
#define ISAAC_ROS_DEPTH_IMAGE_PROC__POINT_CLOUD_XYZ_MASKED_NODE_HPP_

#include <memory>
#include <string>

#include "isaac_ros_depth_image_proc/depth_mask_to_point_cloud.hpp"
#include "message_filters/subscriber.hpp"
#include "message_filters/synchronizer.hpp"
#include "message_filters/sync_policies/approximate_time.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace depth_image_proc
{

//! Publish CUDA-backed compact XYZ clouds selected by segmentation masks.
//!
//! The node caches the latest camera calibration and synchronizes depth and mask images with ROS
//! message filters. Each synchronized pair triggers a build when calibration is available. The mask
//! is applied to the depth image, and valid points are published in depth-image raster order.
//!
//! Topic defaults are `depth/image`, `depth/camera_info`, `segmentation_mask`, and
//! `segmented_point_cloud`. Inputs use SensorDataQoS, and output uses reliable keep-last-10 QoS.
//! `min_depth_m` defaults to 0.0 and must be finite and nonnegative; `max_depth_m` defaults to
//! positive infinity and must be larger. `sync_slop_sec` defaults to 0.5 and bounds approximate
//! synchronization. When `benchmark_output_path` is nonempty, each successful publication attempts
//! to append one JSON Lines timing record.
class PointCloudXYZMaskedNode : public rclcpp::Node
{
public:
  explicit PointCloudXYZMaskedNode(const rclcpp::NodeOptions & options);

private:
  void OnCameraInfo(sensor_msgs::msg::CameraInfo::ConstSharedPtr message);

  void OnSynchronizedInputs(
    const sensor_msgs::msg::Image::ConstSharedPtr & depth,
    const sensor_msgs::msg::Image::ConstSharedPtr & mask);

  // The default mutually exclusive callback group serializes calibration reads and updates.
  sensor_msgs::msg::CameraInfo::ConstSharedPtr latest_camera_info_;

  // Inclusive depth interval in meters, validated as representable by builder floats.
  double min_depth_m_;
  double max_depth_m_;

  // Finite nonnegative maximum depth-to-mask separation for approximate synchronization.
  double sync_slop_sec_;

  // Empty disables benchmarking; otherwise names the append-only JSON Lines output file.
  std::string benchmark_output_path_;

  // CUDA-backed converter with reusable stream and compaction workspace state.
  std::unique_ptr<DepthMaskToPointCloudBuilder> builder_;

  // Sensor-data subscriptions for synchronized depth/mask images and cached calibration.
  message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> mask_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;

  using ApproximateSyncPolicy = message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
  std::unique_ptr<message_filters::Synchronizer<ApproximateSyncPolicy>> approximate_sync_;

  // Reliable publisher for one-row, CUDA-backed compact XYZ clouds.
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
};

}  // namespace depth_image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_DEPTH_IMAGE_PROC__POINT_CLOUD_XYZ_MASKED_NODE_HPP_
