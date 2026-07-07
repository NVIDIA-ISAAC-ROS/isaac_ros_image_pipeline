// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_depth_image_proc/point_cloud_xyzrgb_node.hpp"

#include <cstdio>
#include <memory>
#include <string>
#include <utility>

#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "isaac_ros_nitros_point_cloud_type/nitros_point_cloud.hpp"
#include "rclcpp/rclcpp.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace depth_image_proc
{

PointCloudXyzrgbNode::PointCloudXyzrgbNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("PointCloudXyzrgbNode", options),
  skip_(declare_parameter<int>("skip", 1)),
  output_height_(declare_parameter<uint16_t>("output_height", 1200)),
  output_width_(declare_parameter<uint16_t>("output_width", 1920)),
  memory_pool_block_size_(declare_parameter<int64_t>("memory_pool_block_size", 1920 * 1200 * 4)),
  memory_pool_num_blocks_(declare_parameter<int64_t>("memory_pool_num_blocks", 40)),
  input_qos_size_(declare_parameter<int32_t>("input_qos_size", 10)),
  output_qos_size_(declare_parameter<int32_t>("output_qos_size", 10)),
  depth_sub_{},
  rgb_sub_{},
  camera_info_sub_{},
  exact_sync_{ExactSyncPolicy(static_cast<int>(input_qos_size_)), depth_sub_,
    rgb_sub_, camera_info_sub_}
{
  RCLCPP_DEBUG(get_logger(), "[PointCloudXyzrgbNode] Constructor");

  // Create CUDA stream
  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("PointCloudXyzrgbNode");

  CHECK_CUDA_ERROR(pool_.create(
    static_cast<size_t>(memory_pool_block_size_),
    static_cast<size_t>(memory_pool_num_blocks_),
    nvidia::isaac_ros::nitros::CUDAMemoryPool::MemoryType::Device),
    "[PointCloudXyzNode] Failed to create CUDA memory pool");

  if (skip_ < 1) {
    RCLCPP_ERROR(get_logger(), "skip must be strictly positive, %d was provided", skip_);
    throw std::invalid_argument("skip must be strictly positive");
  }

  // Subscribers
  const rclcpp::QoS input_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "input_qos").keep_last(input_qos_size_);
  const rclcpp::QoS output_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "output_qos").keep_last(output_qos_size_);
  const rmw_qos_profile_t rmw_qos_profile = input_qos.get_rmw_qos_profile();

  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  exact_sync_.registerCallback(std::bind(&PointCloudXyzrgbNode::OnSynchronizedInputs, this,
    std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

  depth_sub_.subscribe(this, "depth_registered/image_rect", rmw_qos_profile, sub_options);
  rgb_sub_.subscribe(this, "rgb/image_rect_color", rmw_qos_profile, sub_options);
  camera_info_sub_.subscribe(this, "rgb/camera_info", rmw_qos_profile, sub_options);

  // Publisher
  point_cloud_pub_ = create_publisher<nvidia::isaac_ros::nitros::NitrosPointCloud>(
    "points", output_qos, pub_options);

  camera_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(
    "camera_info", output_qos, pub_options);

  RCLCPP_DEBUG(get_logger(), "[PointCloudXyzrgbNode] Setup complete");
}

PointCloudXyzrgbNode::~PointCloudXyzrgbNode() {}


PointCloudProperties PointCloudXyzrgbNode::CreatePointCloudProperties(
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & depth_info_msg,
  const int skip)
{
  PointCloudProperties point_cloud_properties;

  const int point_step = 4;

  point_cloud_properties.n_points = depth_info_msg->height *
    depth_info_msg->width / skip;
  point_cloud_properties.point_step = point_step;
  point_cloud_properties.x_offset = 0;
  point_cloud_properties.y_offset = 1;
  point_cloud_properties.z_offset = 2;
  point_cloud_properties.rgb_offset = 3;
  point_cloud_properties.bad_point = std::numeric_limits<float>::quiet_NaN();

  return point_cloud_properties;
}

DepthProperties PointCloudXyzrgbNode::CreateDepthProperties(
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & depth_info_msg)
{
  DepthProperties depth_properties;

  depth_properties.width = depth_info_msg->width;
  depth_properties.height = depth_info_msg->height;
  depth_properties.f_x = depth_info_msg->k[0];
  depth_properties.f_y = depth_info_msg->k[4];
  depth_properties.c_x = depth_info_msg->k[2];
  depth_properties.c_y = depth_info_msg->k[5];

  depth_properties.red_offset = 0;
  depth_properties.green_offset = 1;
  depth_properties.blue_offset = 2;

  return depth_properties;
}

void PointCloudXyzrgbNode::OnSynchronizedInputs(
  const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & depth_msg,
  const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & rgb_msg,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_msg)
{
  RCLCPP_DEBUG(get_logger(), "[PointCloudXyzrgbNode] OnSynchronizedInputs");

  // Get read handles from input messages
  auto depth_read_handle = depth_msg->get_read_handle(*cuda_stream_);
  const float * depth_ptr = reinterpret_cast<const float *>(depth_read_handle.get_ptr());

  auto rgb_read_handle = rgb_msg->get_read_handle(*cuda_stream_);
  const uint8_t * rgb_ptr = reinterpret_cast<const uint8_t *>(rgb_read_handle.get_ptr());

  // create output point cloud message
  PointCloudProperties point_cloud_properties = CreatePointCloudProperties(camera_info_msg,
    skip_);
  DepthProperties depth_properties = CreateDepthProperties(camera_info_msg);

  nvidia::isaac_ros::nitros::NitrosPointCloud point_cloud_msg;
  uint32_t width = point_cloud_properties.n_points;
  uint32_t height = 1;
  uint32_t point_step = point_cloud_properties.point_step * sizeof(float);
  uint32_t row_step = width * point_step;
  auto point_cloud_write_handle = point_cloud_msg.from_pool(
    pool_, width, height, point_step, row_step, false, true, *cuda_stream_);
  float * point_cloud_ptr = reinterpret_cast<float *>(point_cloud_write_handle.get_ptr());

  // Compute the point cloud
  cloud_compute_.DepthToPointCloudCuda(
    depth_ptr,
    rgb_ptr,
    point_cloud_ptr,
    point_cloud_properties,
    depth_properties,
    true,
    skip_,
    *cuda_stream_);

  point_cloud_msg.width = width;
  point_cloud_msg.height = height;
  point_cloud_msg.point_step = point_step;
  point_cloud_msg.row_step = row_step;
  point_cloud_msg.is_bigendian = false;
  point_cloud_msg.frame_id = depth_msg->frame_id;
  point_cloud_msg.timestamp_sec = depth_msg->timestamp_sec;
  point_cloud_msg.timestamp_nsec = depth_msg->timestamp_nsec;

  // Publish the point cloud message
  point_cloud_pub_->publish(point_cloud_msg);
  camera_info_pub_->publish(*camera_info_msg);
}

}  // namespace depth_image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::depth_image_proc::PointCloudXyzrgbNode)
