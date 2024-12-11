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

#include "isaac_ros_image_proc/resize_node.hpp"

#include <cstdio>
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
constexpr char INPUT_TOPIC_NAME[] = "image";

constexpr char OUTPUT_COMPONENT_KEY[] = "image_sink/sink";
constexpr char OUTPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_image_bgr8";
constexpr char OUTPUT_TOPIC_NAME[] = "resize/image";

constexpr char OUTPUT_CAM_COMPONENT_KEY[] = "camera_info_sink/sink";
constexpr char OUTPUT_DEFAULT_CAM_INFO_FORMAT[] = "nitros_camera_info";
constexpr char OUTPUT_CAM_TOPIC_NAME[] = "resize/camera_info";

constexpr char APP_YAML_FILENAME[] = "config/nitros_resize_node.yaml";
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
  "config/isaac_ros_image_proc_namespace_injector_rule.yaml",
  "config/resize_substitution_rule.yaml",
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
      .frame_id_source_key = INPUT_COMPONENT_KEY
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

// ROS image type to Nitros image type mapping
namespace img_encodings = sensor_msgs::image_encodings;
const std::unordered_map<std::string, std::string> ROS_2_NITROS_FORMAT_MAP({
        {img_encodings::RGB8, nitros::nitros_image_rgb8_t::supported_type_name},
        {img_encodings::RGBA8, nitros::nitros_image_rgba8_t::supported_type_name},
        {img_encodings::RGB16, nitros::nitros_image_rgb16_t::supported_type_name},
        {img_encodings::BGR8, nitros::nitros_image_bgr8_t::supported_type_name},
        {img_encodings::BGRA8, nitros::nitros_image_bgra8_t::supported_type_name},
        {img_encodings::BGR16, nitros::nitros_image_bgr16_t::supported_type_name},
        {img_encodings::MONO8, nitros::nitros_image_mono8_t::supported_type_name},
        {img_encodings::MONO16, nitros::nitros_image_mono16_t::supported_type_name},
        {img_encodings::NV24, nitros::nitros_image_nv24_t::supported_type_name},
        {"nv12", nitros::nitros_image_nv12_t::supported_type_name},
      });

ResizeNode::ResizeNode(const rclcpp::NodeOptions & options)
: nitros::NitrosNode(options,
    APP_YAML_FILENAME,
    CONFIG_MAP,
    PRESET_EXTENSION_SPEC_NAMES,
    EXTENSION_SPEC_FILENAMES,
    GENERATOR_RULE_FILENAMES,
    EXTENSIONS,
    PACKAGE_NAME),
  output_width_(declare_parameter<int64_t>("output_width", 1080)),
  output_height_(declare_parameter<int64_t>("output_height", 720)),
  num_blocks_(declare_parameter<int64_t>("num_blocks", 40)),
  keep_aspect_ratio_(static_cast<bool>(declare_parameter<bool>("keep_aspect_ratio", false))),
  encoding_desired_(declare_parameter<std::string>("encoding_desired", "")),
  disable_padding_(static_cast<bool>(declare_parameter<bool>("disable_padding", false))),
  input_width_(declare_parameter<int64_t>("input_width", 0)),
  input_height_(declare_parameter<int64_t>("input_height", 0))
{
  RCLCPP_DEBUG(get_logger(), "[ResizeNode] Constructor");

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

  if (output_width_ <= 0 || output_height_ <= 0) {
    RCLCPP_ERROR(
      get_logger(),
      "[ResizeNode] Width and height need to be non-zero positive number");
    throw std::invalid_argument(
            "[ResizeNode] Invalid output dimension "
            "Width and height need to be non-zero positive number.");
  }

  if (!encoding_desired_.empty()) {
    auto nitros_format = ROS_2_NITROS_FORMAT_MAP.find(encoding_desired_);
    if (nitros_format == std::end(ROS_2_NITROS_FORMAT_MAP)) {
      RCLCPP_ERROR(
        get_logger(), "[ResizeNode] Unsupported encoding[%s]",
        encoding_desired_.c_str());
      throw std::invalid_argument("[ResizeNode] Unsupported encoding.");
    } else {
      config_map_[INPUT_COMPONENT_KEY].compatible_data_format = nitros_format->second;
      config_map_[OUTPUT_COMPONENT_KEY].compatible_data_format = nitros_format->second;
      config_map_[OUTPUT_COMPONENT_KEY].use_compatible_format_only = true;

      RCLCPP_INFO(
        get_logger(),
        "[ResizeNode] Set output data format to: \"%s\"",
        nitros_format->second.c_str());
    }
  }

  if (keep_aspect_ratio_ && disable_padding_) {
    if (input_width_ <= 0 || input_height_ <= 0) {
      RCLCPP_ERROR(
        get_logger(),
        "[ResizeNode] Input Width and height need to be non-zero positive number");
      throw std::invalid_argument(
              "[ResizeNode] Invalid input dimension "
              "Width and height need to be non-zero positive number.");
    }
    calculateOutputDims();
  }

  registerSupportedType<nvidia::isaac_ros::nitros::NitrosCameraInfo>();
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosImage>();

  startNitrosNode();
}

void ResizeNode::postLoadGraphCallback()
{
  RCLCPP_INFO(get_logger(), "[ResizeNode] postLoadGraphCallback().");

  const gxf::optimizer::ComponentInfo component = {
    "nvidia::isaac_ros::MessageRelay",  // component_type_name
    "sink",                             // component_name
    "image_sink"                        // entity_name
  };
  std::string image_format = getFinalDataFormat(component);

  std::string resize_comp_type_name = "nvidia::isaac::tensor_ops::Resize";
  if (image_format == "nitros_image_nv12" || image_format == "nitros_image_bgra8" ||
    image_format == "nitros_image_bgra8")
  {
    resize_comp_type_name = "nvidia::isaac::tensor_ops::StreamResize";
  }

  // Update resize parameters
  getNitrosContext().setParameterUInt64(
    "imageResizer", resize_comp_type_name, "output_width",
    (uint64_t)output_width_);

  getNitrosContext().setParameterUInt64(
    "imageResizer", resize_comp_type_name, "output_height",
    (uint64_t)output_height_);

  bool keep_aspect_ratio = keep_aspect_ratio_;
  // If keep_aspect_ratio is true and padding is disabled then
  // cvcore doesn't have to do output dims calculation i.e pass
  // keep_aspect_ratio as false.
  if (keep_aspect_ratio && disable_padding_) {
    keep_aspect_ratio = false;
  }
  getNitrosContext().setParameterBool(
    "imageResizer", resize_comp_type_name, "keep_aspect_ratio",
    keep_aspect_ratio);

  // The minimum number of memory blocks is set based on the receiver queue capacity
  uint64_t num_blocks = std::max(static_cast<int>(num_blocks_), 40);
  getNitrosContext().setParameterUInt64(
    "imageResizer", "nvidia::gxf::BlockMemoryPool", "num_blocks",
    num_blocks);

  RCLCPP_DEBUG(
    get_logger(),
    "[ResizeNode] postLoadGraphCallback() with image [%ld x %ld], keep_aspect_ratio: %s.",
    output_width_, output_height_, keep_aspect_ratio_ ? "true" : "false");

  uint64_t block_size = calculate_image_size(image_format, output_width_, output_height_);
  RCLCPP_DEBUG(
    get_logger(),
    "[ResizeNode] postLoadGraphCallback() block_size = %ld.",
    block_size);
  getNitrosContext().setParameterUInt64(
    "imageResizer", "nvidia::gxf::BlockMemoryPool", "block_size",
    block_size);
}

void ResizeNode::calculateOutputDims()
{
  float height_factor = static_cast<float>(output_height_) / input_height_;
  float width_factor = static_cast<float>(output_width_) / input_width_;
  if (height_factor < width_factor) {
    output_width_ = input_width_ * height_factor;
    // To make sure output width is even
    if (output_width_ % 2 != 0) {output_width_++;}
  } else if (width_factor < height_factor) {
    output_height_ = input_height_ * width_factor;
    if (output_height_ % 2 != 0) {output_height_++;}
  }
}

ResizeNode::~ResizeNode() {}

}  // namespace image_proc
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::image_proc::ResizeNode)
