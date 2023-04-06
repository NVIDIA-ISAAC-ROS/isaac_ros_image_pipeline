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

#include "isaac_ros_image_proc/resize_node.hpp"

#include <cstdio>
#include <memory>
#include <string>
#include <utility>

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

constexpr char INPUT_CAM_COMPONENT_KEY[] = "input_compositor/cam_info_in";
constexpr char INPUT_DEFAULT_CAM_INFO_FORMAT[] = "nitros_camera_info";
constexpr char INPUT_CAM_TOPIC_NAME[] = "camera_info";

constexpr char INPUT_COMPONENT_KEY[] = "input_compositor/image_in";
constexpr char INPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_image_bgr8";
constexpr char INPUT_TOPIC_NAME[] = "image";

constexpr char OUTPUT_COMPONENT_KEY[] = "image_vault/vault";
constexpr char OUTPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_image_bgr8";
constexpr char OUTPUT_TOPIC_NAME[] = "resize/image";

constexpr char OUTPUT_CAM_COMPONENT_KEY[] = "camerainfo_vault/vault";
constexpr char OUTPUT_DEFAULT_CAM_INFO_FORMAT[] = "nitros_camera_info";
constexpr char OUTPUT_CAM_TOPIC_NAME[] = "resize/camera_info";

constexpr char APP_YAML_FILENAME[] = "config/nitros_resize_node.yaml";
constexpr char PACKAGE_NAME[] = "isaac_ros_image_proc";

const std::vector<std::pair<std::string, std::string>> EXTENSIONS = {
  {"isaac_ros_gxf", "gxf/lib/std/libgxf_std.so"},
  {"isaac_ros_gxf", "gxf/lib/cuda/libgxf_cuda.so"},
  {"isaac_ros_gxf", "gxf/lib/libgxf_message_compositor.so"},
  {"isaac_ros_image_proc", "gxf/lib/image_proc/libgxf_tensorops.so"},
};
const std::vector<std::string> PRESET_EXTENSION_SPEC_NAMES = {
  "isaac_ros_image_proc",
};
const std::vector<std::string> EXTENSION_SPEC_FILENAMES = {};
const std::vector<std::string> GENERATOR_RULE_FILENAMES = {
  "config/namespace_injector_rule_resize.yaml",
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
  keep_aspect_ratio_(static_cast<bool>(declare_parameter<bool>("keep_aspect_ratio", false)))
{
  RCLCPP_DEBUG(get_logger(), "[ResizeNode] Constructor");
  if (output_width_ <= 0 || output_height_ <= 0) {
    RCLCPP_ERROR(
      get_logger(),
      "[ResizeNode] Width and height need to be non-zero positive number");
    throw std::invalid_argument(
            "[ResizeNode] Invalid output dimension "
            "Width and height need to be non-zero positive number.");
  }

  registerSupportedType<nvidia::isaac_ros::nitros::NitrosCameraInfo>();
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosImage>();

  startNitrosNode();
}

void ResizeNode::postLoadGraphCallback()
{
  RCLCPP_INFO(get_logger(), "[ResizeNode] postLoadGraphCallback().");

  // Update resize parameters
  getNitrosContext().setParameterUInt64(
    "imageResizer", "nvidia::cvcore::tensor_ops::Resize", "output_width",
    (uint64_t)output_width_);

  getNitrosContext().setParameterUInt64(
    "imageResizer", "nvidia::cvcore::tensor_ops::Resize", "output_height",
    (uint64_t)output_height_);

  getNitrosContext().setParameterBool(
    "imageResizer", "nvidia::cvcore::tensor_ops::Resize", "keep_aspect_ratio",
    keep_aspect_ratio_);

  // The minimum number of memory blocks is set based on the receiver queue capacity
  uint64_t num_blocks = std::max(static_cast<int>(num_blocks_), 40);
  getNitrosContext().setParameterUInt64(
    "imageResizer", "nvidia::gxf::BlockMemoryPool", "num_blocks",
    num_blocks);

  RCLCPP_DEBUG(
    get_logger(),
    "[ResizeNode] postLoadGraphCallback() with image [%ld x %ld], keep_aspect_ratio: %s.",
    output_width_, output_height_, keep_aspect_ratio_ ? "true" : "false");

  const gxf::optimizer::ComponentInfo component = {
    "nvidia::gxf::Vault",  // component_type_name
    "vault",               // component_name
    "image_vault"          // entity_name
  };
  std::string image_format = getFinalDataFormat(component);
  uint64_t block_size = calculate_image_size(image_format, output_width_, output_height_);
  RCLCPP_DEBUG(
    get_logger(),
    "[ResizeNode] postLoadGraphCallback() block_size = %ld.",
    block_size);
  getNitrosContext().setParameterUInt64(
    "imageResizer", "nvidia::gxf::BlockMemoryPool", "block_size",
    block_size);
}

ResizeNode::~ResizeNode() {}

}  // namespace image_proc
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::image_proc::ResizeNode)
