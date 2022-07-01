/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "isaac_ros_image_proc/image_format_converter_node.hpp"

#include <cstdio>
#include <memory>
#include <string>
#include <utility>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"

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
  {"isaac_ros_nitros", "gxf/std/libgxf_std.so"},
  {"isaac_ros_nitros", "gxf/cuda/libgxf_cuda.so"},
  {"isaac_ros_nitros", "gxf/tensorops/libgxf_tensorops.so"},
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
      .qos = rclcpp::QoS(1),
      .compatible_data_format = INPUT_DEFAULT_TENSOR_FORMAT,
      .topic_name = INPUT_TOPIC_NAME,
      .use_compatible_format_only = false,
    }
  },
  {OUTPUT_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(1),
      .compatible_data_format = OUTPUT_DEFAULT_TENSOR_FORMAT,
      .topic_name = OUTPUT_TOPIC_NAME,
      .use_compatible_format_only = false,
      .frame_id_source_key = INPUT_COMPONENT_KEY
    }
  }
};
#pragma GCC diagnostic pop

// Ros image type to Nitros image type mapping
const std::unordered_map<std::string, std::string> ROS_2_NITROS_FORMAT_MAP({
        {"rgb8", "nitros_image_rgb8"},
        {"rgb16", "nitros_image_rgb16"},
        {"bgr8", "nitros_image_bgr8"},
        {"bgr16", "nitros_image_bgr16"},
        {"mono8", "nitros_image_mono8"},
        {"mono16", "nitros_image_mono16"},
        {"nv24", "nitros_image_nv24"}
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
      throw std::invalid_argument(
              "[ImageFormatConverterNode] Unsupported encoding.");

    } else {
      config_map_[OUTPUT_COMPONENT_KEY].compatible_data_format = nitros_format->second;
      config_map_[OUTPUT_COMPONENT_KEY].use_compatible_format_only = true;

      RCLCPP_INFO(
        get_logger(),
        "[ImageFormatConverterNode] Set output data format to: \"%s\"",
        nitros_format->second.c_str());
    }
  }

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
