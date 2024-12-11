// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_image_proc/rectify_node.hpp"

#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "isaac_ros_common/qos.hpp"

#include "isaac_ros_nitros_camera_info_type/nitros_camera_info.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace image_proc
{

using nvidia::gxf::optimizer::GraphIOGroupSupportedDataTypesInfoList;

constexpr char INPUT_CAM_COMPONENT_KEY[] = "sync/camera_info_in";
constexpr char INPUT_DEFAULT_CAM_INFO_FORMAT[] = "nitros_camera_info";
constexpr char INPUT_CAM_TOPIC_NAME[] = "camera_info";

constexpr char INPUT_COMPONENT_KEY[] = "sync/image_in";
constexpr char INPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_image_bgr8";
constexpr char INPUT_TOPIC_NAME[] = "image_raw";

constexpr char OUTPUT_COMPONENT_KEY[] = "image_sink/sink";
constexpr char OUTPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_image_bgr8";
constexpr char OUTPUT_TOPIC_NAME[] = "image_rect";

constexpr char OUTPUT_CAM_COMPONENT_KEY[] = "camera_info_sink/sink";
constexpr char OUTPUT_DEFAULT_CAM_INFO_FORMAT[] = "nitros_camera_info";
constexpr char OUTPUT_CAM_TOPIC_NAME[] = "camera_info_rect";

constexpr char APP_YAML_FILENAME[] = "config/nitros_rectify_node.yaml";
constexpr char PACKAGE_NAME[] = "isaac_ros_image_proc";

const std::vector<std::pair<std::string, std::string>> EXTENSIONS = {
  {"isaac_ros_gxf", "gxf/lib/std/libgxf_std.so"},
  {"isaac_ros_gxf", "gxf/lib/cuda/libgxf_cuda.so"},
  {"gxf_isaac_message_compositor", "gxf/lib/libgxf_isaac_message_compositor.so"},
  {"gxf_isaac_tensorops", "gxf/lib/libgxf_isaac_tensorops.so"},
};
const std::vector<std::string> PRESET_EXTENSION_SPEC_NAMES = {
  "isaac_ros_image_proc",
};
const std::vector<std::string> EXTENSION_SPEC_FILENAMES = {};
const std::vector<std::string> GENERATOR_RULE_FILENAMES = {
  "config/namespace_injector_rule_rectify.yaml",
};
const std::map<gxf::optimizer::ComponentKey, std::string> COMPATIBLE_DATA_FORMAT_MAP = {
  {INPUT_COMPONENT_KEY, INPUT_DEFAULT_TENSOR_FORMAT},
  {OUTPUT_COMPONENT_KEY, OUTPUT_DEFAULT_TENSOR_FORMAT}
};
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
const nitros::NitrosPublisherSubscriberConfigMap CONFIG_MAP = {
  {INPUT_CAM_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = INPUT_DEFAULT_CAM_INFO_FORMAT,
      .topic_name = INPUT_CAM_TOPIC_NAME,
    }
  },
  {INPUT_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = INPUT_DEFAULT_TENSOR_FORMAT,
      .topic_name = INPUT_TOPIC_NAME,
    }
  },
  {OUTPUT_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = OUTPUT_DEFAULT_TENSOR_FORMAT,
      .topic_name = OUTPUT_TOPIC_NAME,
      .frame_id_source_key = INPUT_COMPONENT_KEY,
    }
  },
  {OUTPUT_CAM_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = OUTPUT_DEFAULT_CAM_INFO_FORMAT,
      .topic_name = OUTPUT_CAM_TOPIC_NAME,
      .frame_id_source_key = INPUT_CAM_COMPONENT_KEY,
    }
  }
};
#pragma GCC diagnostic pop

RectifyNode::RectifyNode(const rclcpp::NodeOptions & options)
: nitros::NitrosNode(options,
    APP_YAML_FILENAME,
    CONFIG_MAP,
    PRESET_EXTENSION_SPEC_NAMES,
    EXTENSION_SPEC_FILENAMES,
    GENERATOR_RULE_FILENAMES,
    EXTENSIONS,
    PACKAGE_NAME),
  output_width_(declare_parameter<int16_t>("output_width", 1280)),
  output_height_(declare_parameter<int16_t>("output_height", 800)),
  num_blocks_(declare_parameter<int64_t>("num_blocks", 40)),
  horizontal_interval_(declare_parameter<int16_t>("horizontal_interval", 1)),
  vertical_interval_(declare_parameter<int16_t>("vertical_interval", 1)),
  interpolation_(declare_parameter<std::string>("interpolation", "cubic_catmullrom")),
  border_type_(declare_parameter<std::string>("border_type", "zero"))
{
  RCLCPP_DEBUG(get_logger(), "[RectifyNode] Constructor");

  // This function sets the QoS parameter for publishers and subscribers setup by this NITROS node
  rclcpp::QoS input_qos_ = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "input_qos");
  rclcpp::QoS output_qos_ = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "output_qos");
  for (auto & config : config_map_) {
    if (config.second.topic_name == INPUT_CAM_TOPIC_NAME ||
      config.second.topic_name == INPUT_TOPIC_NAME)
    {
      config.second.qos = input_qos_;
    } else {
      config.second.qos = output_qos_;
    }
  }

  registerSupportedType<nvidia::isaac_ros::nitros::NitrosCameraInfo>();
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosImage>();

  startNitrosNode();
}

void RectifyNode::preLoadGraphCallback()
{
  RCLCPP_INFO(get_logger(), "[RectifyNode] preLoadGraphCallback().");

  std::string w = "[" + std::to_string(output_width_) + "]";
  NitrosNode::preLoadGraphSetParameter(
    "rectifier",
    "nvidia::isaac::tensor_ops::StreamUndistort",
    "regions_width",
    w);

  std::string h = "[" + std::to_string(output_height_) + "]";
  NitrosNode::preLoadGraphSetParameter(
    "rectifier",
    "nvidia::isaac::tensor_ops::StreamUndistort",
    "regions_height",
    h);

  std::string hz_interval = "[" + std::to_string(horizontal_interval_) + "]";
  NitrosNode::preLoadGraphSetParameter(
    "rectifier",
    "nvidia::isaac::tensor_ops::StreamUndistort",
    "horizontal_intervals",
    hz_interval);

  std::string vt_interval = "[" + std::to_string(vertical_interval_) + "]";
  NitrosNode::preLoadGraphSetParameter(
    "rectifier",
    "nvidia::isaac::tensor_ops::StreamUndistort",
    "vertical_intervals",
    vt_interval);

  NitrosNode::preLoadGraphSetParameter(
    "rectifier",
    "nvidia::isaac::tensor_ops::StreamUndistort",
    "interp_type",
    interpolation_);

  NitrosNode::preLoadGraphSetParameter(
    "rectifier",
    "nvidia::isaac::tensor_ops::StreamUndistort",
    "border_type",
    border_type_);

  RCLCPP_INFO(
    get_logger(),
    "[RectifyNode] preLoadGraphCallback() with image (%s x %s)."
    "[RectifyNode] preLoadGraphCallback() with interval (%s x %s)."
    "[RectifyNode] preLoadGraphCallback() with interpolation %s and border type %s.",
    w.c_str(), h.c_str(), hz_interval.c_str(), vt_interval.c_str(),
    interpolation_.c_str(), border_type_.c_str());
}

void RectifyNode::postLoadGraphCallback()
{
  RCLCPP_INFO(
    get_logger(),
    "[RectifyNode] postLoadGraphCallback().");

  const gxf::optimizer::ComponentInfo component = {
    "nvidia::isaac_ros::MessageRelay",  // component_type_name
    "sink",                             // component_name
    "image_sink"                        // entity_name
  };
  std::string image_format = getFinalDataFormat(component);
  uint64_t block_size = calculate_image_size(image_format, output_width_, output_height_);
  RCLCPP_DEBUG(
    get_logger(),
    "[RectifyNode] postLoadGraphCallback() block_size = %ld.",
    block_size);
  getNitrosContext().setParameterUInt64(
    "rectifier", "nvidia::gxf::BlockMemoryPool", "block_size",
    block_size);

  uint64_t num_blocks = std::max(static_cast<int>(num_blocks_), 40);
  getNitrosContext().setParameterUInt64(
    "rectifier", "nvidia::gxf::BlockMemoryPool", "num_blocks",
    num_blocks);
}

RectifyNode::~RectifyNode() {}

}  // namespace image_proc
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::image_proc::RectifyNode)
