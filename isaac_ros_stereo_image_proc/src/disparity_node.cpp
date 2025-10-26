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

#include "isaac_ros_stereo_image_proc/disparity_node.hpp"

#include <cstdio>
#include <memory>
#include <string>
#include <utility>

#include "isaac_ros_common/qos.hpp"

#include "isaac_ros_nitros_camera_info_type/nitros_camera_info.hpp"
#include "isaac_ros_nitros_disparity_image_type/nitros_disparity_image.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace stereo_image_proc
{

using nvidia::gxf::optimizer::GraphIOGroupSupportedDataTypesInfoList;

constexpr char INPUT_LEFT_COMPONENT_KEY[] = "sync/left_image_receiver";
constexpr char INPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_image_bgr8";
constexpr char INPUT_LEFT_TOPIC_NAME[] = "left/image_rect";

constexpr char INPUT_RIGHT_COMPONENT_KEY[] = "sync/right_image_receiver";
constexpr char INPUT_RIGHT_TOPIC_NAME[] = "right/image_rect";

constexpr char INPUT_LEFT_CAM_COMPONENT_KEY[] = "sync/left_cam_receiver";
constexpr char INPUT_CAMERA_INFO_FORMAT[] = "nitros_camera_info";
constexpr char INPUT_LEFT_CAMERA_TOPIC_NAME[] = "left/camera_info";

constexpr char INPUT_RIGHT_CAM_COMPONENT_KEY[] = "sync/right_cam_receiver";
constexpr char INPUT_RIGHT_CAMERA_TOPIC_NAME[] = "right/camera_info";

constexpr char OUTPUT_COMPONENT_KEY[] = "sink/sink";
constexpr char OUTPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_disparity_image_32FC1";
constexpr char OUTPUT_TOPIC_NAME[] = "disparity";

constexpr char APP_YAML_FILENAME[] = "config/nitros_disparity_node.yaml";
constexpr char PACKAGE_NAME[] = "isaac_ros_stereo_image_proc";

const std::vector<std::pair<std::string, std::string>> EXTENSIONS = {
  {"isaac_ros_gxf", "gxf/lib/std/libgxf_std.so"},
  {"isaac_ros_gxf", "gxf/lib/multimedia/libgxf_multimedia.so"},
  {"isaac_ros_gxf", "gxf/lib/cuda/libgxf_cuda.so"},
  {"isaac_ros_gxf", "gxf/lib/serialization/libgxf_serialization.so"},
  {"gxf_isaac_sgm", "gxf/lib/libgxf_isaac_sgm.so"}
};
const std::vector<std::string> PRESET_EXTENSION_SPEC_NAMES = {
  "isaac_ros_stereo_disparity",
};
const std::vector<std::string> EXTENSION_SPEC_FILENAMES = {};
const std::vector<std::string> GENERATOR_RULE_FILENAMES = {};
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
const nitros::NitrosPublisherSubscriberConfigMap CONFIG_MAP = {
  {INPUT_LEFT_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = INPUT_DEFAULT_TENSOR_FORMAT,
      .topic_name = INPUT_LEFT_TOPIC_NAME,
    }
  },
  {INPUT_RIGHT_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = INPUT_DEFAULT_TENSOR_FORMAT,
      .topic_name = INPUT_RIGHT_TOPIC_NAME,
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
      .frame_id_source_key = INPUT_LEFT_COMPONENT_KEY
    }
  }
};
#pragma GCC diagnostic pop

DisparityNode::DisparityNode(const rclcpp::NodeOptions & options)
: nitros::NitrosNode(options,
    APP_YAML_FILENAME,
    CONFIG_MAP,
    PRESET_EXTENSION_SPEC_NAMES,
    EXTENSION_SPEC_FILENAMES,
    GENERATOR_RULE_FILENAMES,
    EXTENSIONS,
    PACKAGE_NAME),
  vpi_backend_(declare_parameter<std::string>("backend", "CUDA")),
  max_disparity_(declare_parameter<float>("max_disparity", 256)),
  confidence_threshold_(declare_parameter<int>("confidence_threshold", 60000)),
  confidence_type_(declare_parameter<int>("confidence_type", 0)),
  window_size_(declare_parameter<int>("window_size", 7)),
  num_passes_(declare_parameter<int>("num_passes", 2)),
  p1_(declare_parameter<int>("p1", 8)),
  p2_(declare_parameter<int>("p2", 120)),
  p2_alpha_(declare_parameter<int>("p2_alpha", 1)),
  quality_(declare_parameter<int>("quality", 1))
{
  RCLCPP_DEBUG(get_logger(), "[DisparityNode] Constructor");

  // This function sets the QoS parameter for publishers and subscribers setup by this NITROS node
  rclcpp::QoS input_qos_ = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "input_qos");
  rclcpp::QoS output_qos_ = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "output_qos");
  for (auto & config : config_map_) {
    if (config.second.topic_name == INPUT_LEFT_TOPIC_NAME ||
      config.second.topic_name == INPUT_RIGHT_TOPIC_NAME ||
      config.second.topic_name == INPUT_LEFT_CAMERA_TOPIC_NAME ||
      config.second.topic_name == INPUT_RIGHT_CAMERA_TOPIC_NAME)
    {
      config.second.qos = input_qos_;
    } else {
      config.second.qos = output_qos_;
    }
  }

  registerSupportedType<nvidia::isaac_ros::nitros::NitrosCameraInfo>();
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosDisparityImage>();
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosImage>();

  startNitrosNode();
}

void DisparityNode::preLoadGraphCallback()
{
  RCLCPP_INFO(get_logger(), "[DisparityNode] preLoadGraphCallback().");

  NitrosNode::preLoadGraphSetParameter(
    "disparity",
    "nvidia::isaac::SGMDisparity",
    "max_disparity",
    std::to_string(max_disparity_));
}

void DisparityNode::postLoadGraphCallback()
{
  RCLCPP_INFO(get_logger(), "[DisparityNode] postLoadGraphCallback().");

  // Supported backend: CUDA, XAVIER, ORIN.
  getNitrosContext().setParameterStr(
    "disparity", "nvidia::isaac::SGMDisparity", "backend",
    vpi_backend_);
  getNitrosContext().setParameterInt32(
    "disparity", "nvidia::isaac::SGMDisparity", "confidence_threshold",
    confidence_threshold_);
  getNitrosContext().setParameterInt32(
    "disparity", "nvidia::isaac::SGMDisparity", "confidence_type",
    confidence_type_);
  getNitrosContext().setParameterInt32(
    "disparity", "nvidia::isaac::SGMDisparity", "window_size",
    window_size_);
  getNitrosContext().setParameterInt32(
    "disparity", "nvidia::isaac::SGMDisparity", "num_passes",
    num_passes_);
  getNitrosContext().setParameterInt32(
    "disparity", "nvidia::isaac::SGMDisparity", "p1",
    p1_);
  getNitrosContext().setParameterInt32(
    "disparity", "nvidia::isaac::SGMDisparity", "p2",
    p2_);
  getNitrosContext().setParameterInt32(
    "disparity", "nvidia::isaac::SGMDisparity", "p2_alpha",
    p2_alpha_);
  getNitrosContext().setParameterInt32(
    "disparity", "nvidia::isaac::SGMDisparity", "quality",
    quality_);
}

DisparityNode::~DisparityNode() {}

}  // namespace stereo_image_proc
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::stereo_image_proc::DisparityNode)
