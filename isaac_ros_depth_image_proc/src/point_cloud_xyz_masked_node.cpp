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

#include "isaac_ros_depth_image_proc/point_cloud_xyz_masked_node.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#include <rclcpp_components/register_node_macro.hpp>

#include "isaac_ros_common/qos.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace depth_image_proc
{
namespace
{

constexpr uint32_t kSyncQueueSize = 10U;

//! Calculate elapsed ROS time in milliseconds.
//!
//! `start` and `end` are samples from the same ROS clock.
//!
//! Returns the floating-point duration from `start` to `end`.
double MillisecondsBetween(const rclcpp::Time & start, const rclcpp::Time & end)
{
  return (end - start).seconds() * 1000.0;
}

//! Append one masked-cloud timing record to a JSON Lines file.
//!
//! `output_path` names the destination file; empty disables output and succeeds without file I/O.
//! `point_count` is the number of packed XYZ points in that publication.
//! `conversion_ms` is the builder call duration, including completion of the final output copy.
//! `total_ms` ends when publish() returns. It excludes benchmark I/O, transport, subscriber
//! receipt, and processing.
//!
//! Returns `true` when output is disabled or the complete flushed record was written. Returns
//! `false` when the file cannot be opened or flushed. A process-wide mutex prevents records from
//! concurrent node instances from interleaving.
bool WriteCloudBenchmark(
  const std::string & output_path,
  std::size_t point_count,
  double conversion_ms,
  double total_ms)
{
  if (output_path.empty()) {
    return true;
  }

  static std::mutex output_mutex;
  std::lock_guard<std::mutex> lock(output_mutex);
  std::ofstream output(output_path, std::ios::app);
  if (!output) {
    return false;
  }

  output << std::fixed << std::setprecision(6) << "{\"point_count\":" << point_count
         << ",\"conversion_ms\":" << conversion_ms
         << ",\"mask_to_cloud_total_ms\":" << total_ms << "}\n";
  output.flush();
  return output.good();
}

}  // namespace

PointCloudXYZMaskedNode::PointCloudXYZMaskedNode(const rclcpp::NodeOptions & options)
: Node("point_cloud_xyz_masked", options),
  min_depth_m_(declare_parameter<double>("min_depth_m", 0.0)),
  max_depth_m_(declare_parameter<double>(
      "max_depth_m", std::numeric_limits<double>::infinity())),
  sync_slop_sec_(declare_parameter<double>("sync_slop_sec", 0.5)),
  benchmark_output_path_(declare_parameter<std::string>("benchmark_output_path", ""))
{
  constexpr double kFloatMax = std::numeric_limits<float>::max();
  if (!std::isfinite(min_depth_m_) || std::isnan(max_depth_m_) || min_depth_m_ < 0.0 ||
    max_depth_m_ <= min_depth_m_ || min_depth_m_ > kFloatMax ||
    (std::isfinite(max_depth_m_) && max_depth_m_ > kFloatMax))
  {
    throw std::invalid_argument("min_depth_m and max_depth_m must define a valid float range");
  }
  if (!std::isfinite(sync_slop_sec_) || sync_slop_sec_ < 0.0) {
    throw std::invalid_argument("sync_slop_sec must be finite and nonnegative");
  }
  // Prior range checks guarantee valid float results, including positive infinity for max depth.
  // Finite configured doubles may round when narrowed to the builder's float32 depth domain.
  builder_ = std::make_unique<DepthMaskToPointCloudBuilder>(
    static_cast<float>(min_depth_m_), static_cast<float>(max_depth_m_));

  // Relative topic defaults preserve namespace and remapping behavior.
  const auto depth_topic = declare_parameter<std::string>("depth_topic", "depth/image");
  const auto camera_info_topic =
    declare_parameter<std::string>("camera_info_topic", "depth/camera_info");
  const auto mask_topic =
    declare_parameter<std::string>("segmentation_mask_topic", "segmentation_mask");
  const auto cloud_topic = declare_parameter<std::string>("output_topic", "segmented_point_cloud");

  // Accept every backend during transport negotiation. The builder classifies CPU and CUDA ingress
  // explicitly and throws before accessing storage from any other backend.
  rclcpp::SubscriptionOptions depth_options;
  depth_options.acceptable_buffer_backends = "any";
  rclcpp::SubscriptionOptions mask_options;
  mask_options.acceptable_buffer_backends = "any";

  const int default_input_qos_depth =
    static_cast<int>(rclcpp::SensorDataQoS().get_rmw_qos_profile().depth);
  const auto input_qos = ::isaac_ros::common::AddQosParameter(
    *this, "SENSOR_DATA", "input_qos", default_input_qos_depth);
  camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
    camera_info_topic,
    input_qos,
    std::bind(&PointCloudXYZMaskedNode::OnCameraInfo, this, std::placeholders::_1));
  depth_sub_.subscribe(
    this, depth_topic, input_qos, depth_options);
  mask_sub_.subscribe(
    this, mask_topic, input_qos, mask_options);

  approximate_sync_ =
    std::make_unique<message_filters::Synchronizer<ApproximateSyncPolicy>>(
    ApproximateSyncPolicy(kSyncQueueSize), depth_sub_, mask_sub_);
  approximate_sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(sync_slop_sec_));
  approximate_sync_->registerCallback(
    std::bind(
      &PointCloudXYZMaskedNode::OnSynchronizedInputs, this,
      std::placeholders::_1, std::placeholders::_2));
  const auto output_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "output_qos", 10);
  cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
    cloud_topic, output_qos);
}

//! Cache the most recently received depth-camera calibration.
//!
//! `message` is the non-null immutable message delivered by the camera-info subscription. Ownership
//! is shared with the cache until newer calibration replaces it.
//!
//! Performs no conversion or publication.
void PointCloudXYZMaskedNode::OnCameraInfo(sensor_msgs::msg::CameraInfo::ConstSharedPtr message)
{
  latest_camera_info_ = std::move(message);
}

//! Build and publish a cloud using synchronized depth and mask images plus cached calibration.
//!
//! `depth` and `mask` are the immutable pair emitted by the selected message-filter policy.
//!
//! Missing calibration and empty results are logged and skipped. Successful conversion publishes
//! one CUDA-backed cloud and optionally attempts one benchmark append. Exceptions are caught and
//! logged so later synchronized pairs can still be processed.
void PointCloudXYZMaskedNode::OnSynchronizedInputs(
  const sensor_msgs::msg::Image::ConstSharedPtr & depth,
  const sensor_msgs::msg::Image::ConstSharedPtr & mask)
{
  const auto build_start = now();

  const auto camera_info = latest_camera_info_;
  if (!camera_info) {
    RCLCPP_WARN(get_logger(), "Skipping cloud build: camera info is unavailable");
    return;
  }

  try {
    const auto conversion_start = now();
    auto cloud = builder_->Build(*depth, *camera_info, *mask);
    const double conversion_ms = MillisecondsBetween(conversion_start, now());

    if (!cloud) {
      RCLCPP_DEBUG(get_logger(), "Skipping cloud publication: no valid masked depth points");
      return;
    }

    // Widen width before multiplication so large organized-cloud metadata cannot overflow uint32.
    const std::size_t point_count =
      static_cast<std::size_t>(cloud->width) * cloud->height;
    cloud_pub_->publish(std::move(cloud));

    // This total ends after publish() returns; transport and GPU completion remain asynchronous.
    const double total_ms = MillisecondsBetween(build_start, now());
    if (!WriteCloudBenchmark(
        benchmark_output_path_, point_count, conversion_ms, total_ms))
    {
      RCLCPP_WARN_ONCE(
        get_logger(),
        "Unable to write masked point-cloud benchmark file: '%s'",
        benchmark_output_path_.c_str());
    }
    RCLCPP_DEBUG(
      get_logger(),
      "Published CUDA-backed masked XYZ cloud with %zu points in %.3f ms",
      point_count,
      total_ms);
  } catch (const std::exception & error) {
    RCLCPP_ERROR(get_logger(), "Masked point-cloud build failed: %s", error.what());
  }
}

}  // namespace depth_image_proc
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::depth_image_proc::PointCloudXYZMaskedNode)
