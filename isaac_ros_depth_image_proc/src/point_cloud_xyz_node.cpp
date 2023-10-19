// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_depth_image_proc/point_cloud_xyz_node.hpp"

#include <cstdio>
#include <memory>
#include <string>
#include <utility>

#include "isaac_ros_nitros_camera_info_type/nitros_camera_info.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "isaac_ros_nitros_point_cloud_type/nitros_point_cloud.hpp"

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace depth_image_proc
{

using nvidia::gxf::optimizer::GraphIOGroupSupportedDataTypesInfoList;

constexpr char INPUT_DEPTH_COMPONENT_KEY[] = "sync/depth_image_receiver";
constexpr char INPUT_DEPTH_TENSOR_FORMAT[] = "nitros_image_32FC1";
constexpr char INPUT_DEPTH_TOPIC_NAME[] = "image_rect";

constexpr char INPUT_DEPTH_CAMERA_INFO_COMPONENT_KEY[] = "sync/depth_cam_info_receiver";
constexpr char INPUT_DEPTH_CAMERA_INFO_FORMAT[] = "nitros_camera_info";
constexpr char INPUT_DEPTH_CAMERA_INFO_TOPIC_NAME[] = "camera_info";

constexpr char OUTPUT_COMPONENT_KEY[] = "sink/sink";
constexpr char OUTPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_point_cloud";
constexpr char OUTPUT_TOPIC_NAME[] = "points";

constexpr char APP_YAML_FILENAME[] = "config/nitros_point_cloud_xyz_node.yaml";
constexpr char PACKAGE_NAME[] = "isaac_ros_depth_image_proc";

const std::vector<std::pair<std::string, std::string>> EXTENSIONS = {
  {"isaac_ros_gxf", "gxf/lib/std/libgxf_std.so"},
  {"isaac_ros_gxf", "gxf/lib/multimedia/libgxf_multimedia.so"},
  {"isaac_ros_gxf", "gxf/lib/cuda/libgxf_cuda.so"},
  {"isaac_ros_stereo_image_proc", "gxf/lib/sgm_disparity/libgxf_sgm.so"},
  {"isaac_ros_gxf", "gxf/lib/serialization/libgxf_serialization.so"},
  {"isaac_ros_gxf", "gxf/lib/libgxf_synchronization.so"},
  {"isaac_ros_depth_image_proc", "gxf/lib/depth_image_proc/libgxf_depth_image_proc.so"}
};
const std::vector<std::string> PRESET_EXTENSION_SPEC_NAMES = {
  "isaac_ros_point_cloud_xyz",
};
const std::vector<std::string> EXTENSION_SPEC_FILENAMES = {};
const std::vector<std::string> GENERATOR_RULE_FILENAMES = {};
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
const nitros::NitrosPublisherSubscriberConfigMap CONFIG_MAP = {
  {INPUT_DEPTH_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = INPUT_DEPTH_TENSOR_FORMAT,
      .topic_name = INPUT_DEPTH_TOPIC_NAME,
    }
  },
  {INPUT_DEPTH_CAMERA_INFO_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = INPUT_DEPTH_CAMERA_INFO_FORMAT,
      .topic_name = INPUT_DEPTH_CAMERA_INFO_TOPIC_NAME,
    }
  },
  {OUTPUT_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = OUTPUT_DEFAULT_TENSOR_FORMAT,
      .topic_name = OUTPUT_TOPIC_NAME,
      .frame_id_source_key = INPUT_DEPTH_COMPONENT_KEY
    }
  }
};
#pragma GCC diagnostic pop

PointCloudXyzNode::PointCloudXyzNode(const rclcpp::NodeOptions & options)
: nitros::NitrosNode(options,
    APP_YAML_FILENAME,
    CONFIG_MAP,
    PRESET_EXTENSION_SPEC_NAMES,
    EXTENSION_SPEC_FILENAMES,
    GENERATOR_RULE_FILENAMES,
    EXTENSIONS,
    PACKAGE_NAME),
  skip_(declare_parameter<int>("skip", 1)),
  output_height_(declare_parameter<uint16_t>("output_height", 1200)),
  output_width_(declare_parameter<uint16_t>("output_width", 1920))
{
  RCLCPP_DEBUG(get_logger(), "[PointCloudXyzNode] Constructor");

  registerSupportedType<nvidia::isaac_ros::nitros::NitrosCameraInfo>();
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosImage>();
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosPointCloud>();

  startNitrosNode();
}

void PointCloudXyzNode::postLoadGraphCallback()
{
  RCLCPP_DEBUG(get_logger(), "[PointCloudXyzNode] postLoadGraphCallback().");

  // Update depth_to_pointcloud parameters
  getNitrosContext().setParameterInt32(
    "depth_to_pointcloud", "nvidia::isaac_ros::depth_image_proc::DepthToPointCloud", "skip",
    skip_);

  // Make allocator size variable depending on output size
  getNitrosContext().setParameterUInt64(
    "depth_to_pointcloud", "nvidia::gxf::BlockMemoryPool", "block_size",
    4 * 4 * output_width_ * output_height_);  // 4 bytes for each x, y, z, alpha (hence 4 * 4)
}

PointCloudXyzNode::~PointCloudXyzNode() {}

}  // namespace depth_image_proc
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::depth_image_proc::PointCloudXyzNode)
