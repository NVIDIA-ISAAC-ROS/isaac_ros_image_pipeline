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

#include "isaac_ros_image_proc/image_format_converter_node.hpp"

#include <cstdio>
#include <memory>
#include <string>
#include <utility>

#include "isaac_ros_nitros_image_type/nitros_image.hpp"

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/image_encodings.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace image_proc
{

using nvidia::gxf::optimizer::GraphIOGroupSupportedDataTypesInfoList;

constexpr char INPUT_COMPONENT_KEY[] = "imageConverter/data_receiver";
constexpr char INPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_image_rgb8";
constexpr char INPUT_TOPIC_NAME[] = "image_raw";

constexpr char OUTPUT_COMPONENT_KEY[] = "vault/vault";
constexpr char OUTPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_image_bgr8";
constexpr char OUTPUT_TOPIC_NAME[] = "image";

constexpr char APP_YAML_FILENAME[] = "config/nitros_image_format_converter_node.yaml";
constexpr char PACKAGE_NAME[] = "isaac_ros_image_proc";

const std::vector<std::pair<std::string, std::string>> EXTENSIONS = {
  {"isaac_ros_gxf", "gxf/lib/std/libgxf_std.so"},
  {"isaac_ros_gxf", "gxf/lib/cuda/libgxf_cuda.so"},
  {"isaac_ros_image_proc", "gxf/lib/image_proc/libgxf_tensorops.so"},
};
const std::vector<std::string> PRESET_EXTENSION_SPEC_NAMES = {
  "isaac_ros_image_proc",
};
const std::vector<std::string> EXTENSION_SPEC_FILENAMES = {};
const std::vector<std::string> GENERATOR_RULE_FILENAMES = {
  "config/namespace_injector_rule_image_format_converter.yaml",
  "config/image_format_converter_substitution.yaml"
};
const std::map<gxf::optimizer::ComponentKey, std::string> COMPATIBLE_DATA_FORMAT_MAP = {
  {INPUT_COMPONENT_KEY, INPUT_DEFAULT_TENSOR_FORMAT},
  {OUTPUT_COMPONENT_KEY, OUTPUT_DEFAULT_TENSOR_FORMAT}
};
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
const nitros::NitrosPublisherSubscriberConfigMap CONFIG_MAP = {
  {INPUT_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = INPUT_DEFAULT_TENSOR_FORMAT,
      .topic_name = INPUT_TOPIC_NAME,
      .use_compatible_format_only = false,
    }
  },
  {OUTPUT_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = OUTPUT_DEFAULT_TENSOR_FORMAT,
      .topic_name = OUTPUT_TOPIC_NAME,
      .use_compatible_format_only = false,
      .frame_id_source_key = INPUT_COMPONENT_KEY
    }
  }
};
#pragma GCC diagnostic pop

// ROS image type to Nitros image type mapping
namespace img_encodings = sensor_msgs::image_encodings;
const std::unordered_map<std::string, std::string> ROS_2_NITROS_FORMAT_MAP({
        {img_encodings::RGB8, nitros::nitros_image_rgb8_t::supported_type_name},
        {img_encodings::RGB16, nitros::nitros_image_rgb16_t::supported_type_name},
        {img_encodings::BGR8, nitros::nitros_image_bgr8_t::supported_type_name},
        {img_encodings::BGR16, nitros::nitros_image_bgr16_t::supported_type_name},
        {img_encodings::MONO8, nitros::nitros_image_mono8_t::supported_type_name},
        {img_encodings::MONO16, nitros::nitros_image_mono16_t::supported_type_name},
        {img_encodings::NV24, nitros::nitros_image_nv24_t::supported_type_name},
        {"nv12", nitros::nitros_image_nv12_t::supported_type_name},
      });

ImageFormatConverterNode::ImageFormatConverterNode(const rclcpp::NodeOptions & options)
: nitros::NitrosNode(options,
    APP_YAML_FILENAME,
    CONFIG_MAP,
    PRESET_EXTENSION_SPEC_NAMES,
    EXTENSION_SPEC_FILENAMES,
    GENERATOR_RULE_FILENAMES,
    EXTENSIONS,
    PACKAGE_NAME),
  encoding_desired_(declare_parameter<std::string>("encoding_desired", ""))
{
  RCLCPP_DEBUG(get_logger(), "[ImageFormatConverterNode] Constructor");

  if (!encoding_desired_.empty()) {
    auto nitros_format = ROS_2_NITROS_FORMAT_MAP.find(encoding_desired_);
    if (nitros_format == std::end(ROS_2_NITROS_FORMAT_MAP)) {
      RCLCPP_ERROR(
        get_logger(), "[ImageFormatConverterNode] Unsupported encoding[%s]",
        encoding_desired_.c_str());
      throw std::invalid_argument("[ImageFormatConverterNode] Unsupported encoding.");

    } else {
      config_map_[OUTPUT_COMPONENT_KEY].compatible_data_format = nitros_format->second;
      config_map_[OUTPUT_COMPONENT_KEY].use_compatible_format_only = true;

      RCLCPP_INFO(
        get_logger(),
        "[ImageFormatConverterNode] Set output data format to: \"%s\"",
        nitros_format->second.c_str());
    }
  }

  registerSupportedType<nvidia::isaac_ros::nitros::NitrosImage>();

  startNitrosNode();
}

void ImageFormatConverterNode::postLoadGraphCallback()
{
  RCLCPP_INFO(get_logger(), "[ImageFormatConverterNode] postLoadGraphCallback().");
}

ImageFormatConverterNode::~ImageFormatConverterNode() {}

}  // namespace image_proc
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::image_proc::ImageFormatConverterNode)
