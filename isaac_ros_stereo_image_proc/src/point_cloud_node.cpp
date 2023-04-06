// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_stereo_image_proc/point_cloud_node.hpp"

#include <cstdio>
#include <memory>
#include <string>
#include <utility>

#include "isaac_ros_nitros_camera_info_type/nitros_camera_info.hpp"
#include "isaac_ros_nitros_disparity_image_type/nitros_disparity_image.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "isaac_ros_nitros_point_cloud_type/nitros_point_cloud.hpp"

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace stereo_image_proc
{

using nvidia::gxf::optimizer::GraphIOGroupSupportedDataTypesInfoList;

constexpr char INPUT_RGB_COMPONENT_KEY[] = "sync/rgb_image_receiver";
constexpr char INPUT_RGB_TENSOR_FORMAT[] = "nitros_image_bgr8";
constexpr char INPUT_RGB_TOPIC_NAME[] = "left/image_rect_color";

constexpr char INPUT_DISPARITY_COMPONENT_KEY[] = "sync/disparity_image_receiver";
constexpr char INPUT_DISPARITY_TENSOR_FORMAT[] = "nitros_disparity_image_32FC1";
constexpr char INPUT_DISPARITY_TOPIC_NAME[] = "disparity";

constexpr char INPUT_LEFT_CAM_COMPONENT_KEY[] = "sync/left_cam_receiver";
constexpr char INPUT_CAMERA_INFO_FORMAT[] = "nitros_camera_info";
constexpr char INPUT_LEFT_CAMERA_TOPIC_NAME[] = "left/camera_info";

constexpr char INPUT_RIGHT_CAM_COMPONENT_KEY[] = "sync/right_cam_receiver";
constexpr char INPUT_RIGHT_CAMERA_TOPIC_NAME[] = "right/camera_info";

constexpr char OUTPUT_COMPONENT_KEY[] = "vault/vault";
constexpr char OUTPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_point_cloud";
constexpr char OUTPUT_TOPIC_NAME[] = "points2";

constexpr char APP_YAML_FILENAME[] = "config/nitros_point_cloud_node.yaml";
constexpr char PACKAGE_NAME[] = "isaac_ros_stereo_image_proc";

const std::vector<std::pair<std::string, std::string>> EXTENSIONS = {
  {"isaac_ros_gxf", "gxf/lib/std/libgxf_std.so"},
  {"isaac_ros_gxf", "gxf/lib/multimedia/libgxf_multimedia.so"},
  {"isaac_ros_gxf", "gxf/lib/cuda/libgxf_cuda.so"},
  {"isaac_ros_gxf", "gxf/lib/serialization/libgxf_serialization.so"},
  {"isaac_ros_gxf", "gxf/lib/libgxf_synchronization.so"},
  {"isaac_ros_gxf", "gxf/lib/libgxf_point_cloud.so"}
};
const std::vector<std::string> PRESET_EXTENSION_SPEC_NAMES = {
  "isaac_ros_stereo_point_cloud",
};
const std::vector<std::string> EXTENSION_SPEC_FILENAMES = {};
const std::vector<std::string> GENERATOR_RULE_FILENAMES = {};
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
const nitros::NitrosPublisherSubscriberConfigMap CONFIG_MAP = {
  {INPUT_RGB_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = INPUT_RGB_TENSOR_FORMAT,
      .topic_name = INPUT_RGB_TOPIC_NAME,
    }
  },
  {INPUT_DISPARITY_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = INPUT_DISPARITY_TENSOR_FORMAT,
      .topic_name = INPUT_DISPARITY_TOPIC_NAME,
    }
  },
  {INPUT_LEFT_CAM_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = INPUT_CAMERA_INFO_FORMAT,
      .topic_name = INPUT_LEFT_CAMERA_TOPIC_NAME,
    }
  },
  {INPUT_RIGHT_CAM_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = INPUT_CAMERA_INFO_FORMAT,
      .topic_name = INPUT_RIGHT_CAMERA_TOPIC_NAME,
    }
  },
  {OUTPUT_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = OUTPUT_DEFAULT_TENSOR_FORMAT,
      .topic_name = OUTPUT_TOPIC_NAME,
      .frame_id_source_key = INPUT_RGB_COMPONENT_KEY
    }
  }
};
#pragma GCC diagnostic pop

PointCloudNode::PointCloudNode(const rclcpp::NodeOptions & options)
: nitros::NitrosNode(options,
    APP_YAML_FILENAME,
    CONFIG_MAP,
    PRESET_EXTENSION_SPEC_NAMES,
    EXTENSION_SPEC_FILENAMES,
    GENERATOR_RULE_FILENAMES,
    EXTENSIONS,
    PACKAGE_NAME),
  use_color_(declare_parameter<bool>("use_color", false)),
  unit_scaling_(declare_parameter<float>("unit_scaling", 1.0))
{
  RCLCPP_DEBUG(get_logger(), "[PointCloudNode] Constructor");

  registerSupportedType<nvidia::isaac_ros::nitros::NitrosCameraInfo>();
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosDisparityImage>();
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosImage>();
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosPointCloud>();

  startNitrosNode();
}

void PointCloudNode::preLoadGraphCallback()
{
  RCLCPP_INFO(get_logger(), "[PointCloudNode] preLoadGraphCallback().");

  NitrosNode::preLoadGraphSetParameter(
    "point_cloud",
    "nvidia::isaac_ros::point_cloud::PointCloud",
    "unit_scaling",
    std::to_string(unit_scaling_));
}

void PointCloudNode::postLoadGraphCallback()
{
  RCLCPP_DEBUG(get_logger(), "[PointCloudNode] postLoadGraphCallback().");

  // Update point cloud parameters
  getNitrosContext().setParameterBool(
    "point_cloud", "nvidia::isaac_ros::point_cloud::PointCloud", "use_color",
    use_color_);
}

PointCloudNode::~PointCloudNode() {}

}  // namespace stereo_image_proc
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::stereo_image_proc::PointCloudNode)
