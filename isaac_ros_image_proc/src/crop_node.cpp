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

#include "isaac_ros_image_proc/crop_node.hpp"

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

constexpr char INPUT_CAM_COMPONENT_KEY[] = "input_compositor/cam_info_in";
constexpr char INPUT_DEFAULT_CAM_INFO_FORMAT[] = "nitros_camera_info";
constexpr char INPUT_CAM_TOPIC_NAME[] = "camera_info";

constexpr char INPUT_COMPONENT_KEY[] = "input_compositor/image_in";
constexpr char INPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_image_bgr8";
constexpr char INPUT_TOPIC_NAME[] = "image";

constexpr char OUTPUT_COMPONENT_KEY[] = "image_sink/sink";
constexpr char OUTPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_image_bgr8";
constexpr char OUTPUT_TOPIC_NAME[] = "crop/image";

constexpr char OUTPUT_CAM_COMPONENT_KEY[] = "camera_info_sink/sink";
constexpr char OUTPUT_DEFAULT_CAM_INFO_FORMAT[] = "nitros_camera_info";
constexpr char OUTPUT_CAM_TOPIC_NAME[] = "crop/camera_info";

constexpr char APP_YAML_FILENAME[] = "config/nitros_crop_node.yaml";
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
  "config/namespace_injector_rule_crop.yaml",
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
        {img_encodings::RGB16, nitros::nitros_image_rgb16_t::supported_type_name},
        {img_encodings::BGR8, nitros::nitros_image_bgr8_t::supported_type_name},
        {img_encodings::BGR16, nitros::nitros_image_bgr16_t::supported_type_name},
        {img_encodings::MONO8, nitros::nitros_image_mono8_t::supported_type_name},
        {img_encodings::MONO16, nitros::nitros_image_mono16_t::supported_type_name},
        {img_encodings::NV24, nitros::nitros_image_nv24_t::supported_type_name},
        {"nv12", nitros::nitros_image_nv12_t::supported_type_name},
      });

// User string to CROP mode
const std::unordered_map<std::string, CropMode> CROP_MODE_MAP({
        {"CENTER", CropMode::kCenter},
        {"LEFT", CropMode::kLeft},
        {"RIGHT", CropMode::kRight},
        {"TOP", CropMode::kTop},
        {"BOTTOM", CropMode::kBottom},
        {"TOPLEFT", CropMode::kTopLeft},
        {"TOPRIGHT", CropMode::kTopRight},
        {"BOTTOMLEFT", CropMode::kBottomLeft},
        {"BOTTOMRIGHT", CropMode::kBottomRight},
        {"BBOX", CropMode::kBBox}
      });

CropNode::CropNode(const rclcpp::NodeOptions & options)
: nitros::NitrosNode(options,
    APP_YAML_FILENAME,
    CONFIG_MAP,
    PRESET_EXTENSION_SPEC_NAMES,
    EXTENSION_SPEC_FILENAMES,
    GENERATOR_RULE_FILENAMES,
    EXTENSIONS,
    PACKAGE_NAME),
  input_width_(declare_parameter<int64_t>("input_width", 0)),
  input_height_(declare_parameter<int64_t>("input_height", 0)),
  crop_width_(declare_parameter<int64_t>("crop_width", 0)),
  crop_height_(declare_parameter<int64_t>("crop_height", 0)),
  num_blocks_(declare_parameter<int64_t>("num_blocks", 40)),
  crop_mode_(declare_parameter<std::string>("crop_mode", "")),
  roi_{static_cast<size_t>(declare_parameter<int64_t>("roi_top_left_x", 0)),
    static_cast<size_t>(declare_parameter<int64_t>("roi_top_left_y", 0)),
    static_cast<size_t>(crop_width_),
    static_cast<size_t>(crop_height_),
  }
{
  RCLCPP_DEBUG(get_logger(), "[CropNode] Constructor");

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

  if (input_width_ <= 0 || input_height_ <= 0 || crop_width_ <= 0 || crop_height_ <= 0) {
    RCLCPP_ERROR(get_logger(), "[CropNode] Width and height need to be non-zero positive number");
    throw std::invalid_argument(
            "[CropNode] Invalid output dimension "
            "Width and height need to be non-zero positive number.");
  }

  if (crop_mode_.empty()) {
    RCLCPP_ERROR(get_logger(), "[CropNode] Crop Mode is not set. Please select the valid value.");
    throw std::invalid_argument("[CropNode] Crop Mode is not set. Please select the valid value.");
  }

  if (!crop_mode_.empty()) {
    const auto crop_mode = CROP_MODE_MAP.find(crop_mode_);
    if (crop_mode == std::end(CROP_MODE_MAP)) {
      RCLCPP_ERROR(get_logger(), "[CropNode] Unsupported crop mode: [%s]", crop_mode_.c_str());
      throw std::invalid_argument("[CropNode] Unsupported crop mode.");
    } else {
      CalculateResizeAndCropParams(crop_mode->second);
    }
  }

  registerSupportedType<nvidia::isaac_ros::nitros::NitrosCameraInfo>();
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosImage>();

  startNitrosNode();
}

void CropNode::CalculateResizeAndCropParams(const CropMode & crop_mode)
{
  // Only in CropMode::kBBox user provided roi values are used.
  switch (crop_mode) {
    case CropMode::kCenter: {
        roi_.top_left_x = (input_width_ - crop_width_) / 2;
        roi_.top_left_y = (input_height_ - crop_height_) / 2;
        roi_.width = crop_width_;
        roi_.height = crop_height_;
        break;
      }
    case CropMode::kLeft: {
        roi_.top_left_x = 0;
        roi_.top_left_y = (input_height_ - crop_height_) / 2;
        roi_.width = crop_width_;
        roi_.height = crop_height_;
        break;
      }
    case CropMode::kRight: {
        roi_.top_left_x = (input_width_ - crop_width_);
        roi_.top_left_y = (input_height_ - crop_height_) / 2;
        roi_.width = crop_width_;
        roi_.height = crop_height_;
        break;
      }
    case CropMode::kTop: {
        roi_.top_left_x = (input_width_ - crop_width_) / 2;
        roi_.top_left_y = 0;
        roi_.width = crop_width_;
        roi_.height = crop_height_;
        break;
      }
    case CropMode::kBottom: {
        roi_.top_left_x = (input_width_ - crop_width_) / 2;
        roi_.top_left_y = (input_height_ - crop_height_);
        roi_.width = crop_width_;
        roi_.height = crop_height_;
        break;
      }
    case CropMode::kTopLeft: {
        roi_.top_left_x = 0;
        roi_.top_left_y = 0;
        roi_.width = crop_width_;
        roi_.height = crop_height_;
        break;
      }
    case CropMode::kTopRight: {
        roi_.top_left_x = (input_width_ - crop_width_);
        roi_.top_left_y = 0;
        roi_.width = crop_width_;
        roi_.height = crop_height_;
        break;
      }
    case CropMode::kBottomLeft: {
        roi_.top_left_x = 0;
        roi_.top_left_y = (input_height_ - crop_height_);
        roi_.width = crop_width_;
        roi_.height = crop_height_;
        break;
      }
    case CropMode::kBottomRight: {
        roi_.top_left_x = (input_width_ - crop_width_);
        roi_.top_left_y = (input_height_ - crop_height_);
        roi_.width = crop_width_;
        roi_.height = crop_height_;
        break;
      }
    case CropMode::kBBox: {break;}   // Use the one set by the user
    default: {
        RCLCPP_ERROR(get_logger(), "Unsupported CropMode.");
        throw std::runtime_error("Unsupported CropMode.");
      }
  }
}

void CropNode::postLoadGraphCallback()
{
  RCLCPP_INFO(get_logger(), "[CropNode] postLoadGraphCallback().");

  // Update resize parameters
  getNitrosContext().setParameterUInt64(
    "crop_and_resizer", "nvidia::isaac::tensor_ops::CropAndResize", "output_width",
    (uint64_t)crop_width_);

  getNitrosContext().setParameterUInt64(
    "crop_and_resizer", "nvidia::isaac::tensor_ops::CropAndResize", "output_height",
    (uint64_t)crop_height_);

  // The minimum number of memory blocks is set based on the receiver queue capacity
  uint64_t num_blocks = std::max(static_cast<int>(num_blocks_), 40);
  getNitrosContext().setParameterUInt64(
    "crop_and_resizer", "nvidia::gxf::BlockMemoryPool", "num_blocks",
    num_blocks);

  const gxf::optimizer::ComponentInfo component = {
    "nvidia::isaac_ros::MessageRelay",  // component_type_name
    "sink",                             // component_name
    "image_sink"                        // entity_name
  };
  std::string image_format = getFinalDataFormat(component);
  uint64_t block_size = calculate_image_size(image_format, crop_width_, crop_height_);

  getNitrosContext().setParameterUInt64(
    "bbox", "nvidia::isaac::tensor_ops::BBoxGenerator", "image_width", crop_width_);
  getNitrosContext().setParameterUInt64(
    "bbox", "nvidia::isaac::tensor_ops::BBoxGenerator", "image_height", crop_height_);
  getNitrosContext().setParameterUInt64(
    "bbox", "nvidia::isaac::tensor_ops::BBoxGenerator", "bbox_width", roi_.width);
  getNitrosContext().setParameterUInt64(
    "bbox", "nvidia::isaac::tensor_ops::BBoxGenerator", "bbox_height", roi_.height);
  getNitrosContext().setParameterUInt64(
    "bbox", "nvidia::isaac::tensor_ops::BBoxGenerator", "bbox_loc_x", roi_.top_left_x);
  getNitrosContext().setParameterUInt64(
    "bbox", "nvidia::isaac::tensor_ops::BBoxGenerator", "bbox_loc_y", roi_.top_left_y);

  getNitrosContext().setParameterUInt64(
    "crop_and_resizer", "nvidia::gxf::BlockMemoryPool", "block_size",
    block_size);
}

CropNode::~CropNode() {}

}  // namespace image_proc
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::image_proc::CropNode)
