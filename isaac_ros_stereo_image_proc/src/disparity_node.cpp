/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "isaac_ros_stereo_image_proc/disparity_node.hpp"

#include <cstdio>
#include <memory>
#include <string>
#include <utility>

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

constexpr char OUTPUT_COMPONENT_KEY[] = "vault/vault";
constexpr char OUTPUT_DEFAULT_TENSOR_FORMAT[] = "nitros_disparity_image_32FC1";
constexpr char OUTPUT_TOPIC_NAME[] = "disparity";

constexpr char APP_YAML_FILENAME[] = "config/nitros_disparity_node.yaml";
constexpr char PACKAGE_NAME[] = "isaac_ros_stereo_image_proc";

const std::vector<std::pair<std::string, std::string>> EXTENSIONS = {
  {"isaac_ros_nitros", "gxf/std/libgxf_std.so"},
  {"isaac_ros_nitros", "gxf/multimedia/libgxf_multimedia.so"},
  {"isaac_ros_nitros", "gxf/cuda/libgxf_cuda.so"},
  {"isaac_ros_nitros", "gxf/serialization/libgxf_serialization.so"},
  {"isaac_ros_stereo_image_proc", "lib/libgxf_disparity_extension.so"},
  {"isaac_ros_stereo_image_proc", "lib/libgxf_synchronization.so"},
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
      .qos = rclcpp::QoS(1),
      .compatible_data_format = INPUT_DEFAULT_TENSOR_FORMAT,
      .topic_name = INPUT_LEFT_TOPIC_NAME,
    }
  },
  {INPUT_RIGHT_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(1),
      .compatible_data_format = INPUT_DEFAULT_TENSOR_FORMAT,
      .topic_name = INPUT_RIGHT_TOPIC_NAME,
    }
  },
  {INPUT_LEFT_CAM_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(1),
      .compatible_data_format = INPUT_CAMERA_INFO_FORMAT,
      .topic_name = INPUT_LEFT_CAMERA_TOPIC_NAME,
    }
  },
  {INPUT_RIGHT_CAM_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(1),
      .compatible_data_format = INPUT_CAMERA_INFO_FORMAT,
      .topic_name = INPUT_RIGHT_CAMERA_TOPIC_NAME,
    }
  },
  {OUTPUT_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(1),
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
  vpi_backends_(declare_parameter<std::string>("backends", "CUDA")),
  max_disparity_(declare_parameter<float>("max_disparity", 64))
{
  RCLCPP_DEBUG(get_logger(), "[DisparityNode] Constructor");

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
    "nvidia::isaac_ros::SGMDisparity",
    "max_disparity",
    std::to_string(max_disparity_));
}

void DisparityNode::postLoadGraphCallback()
{
  RCLCPP_INFO(get_logger(), "[DisparityNode] postLoadGraphCallback().");

  // Supported backends: CUDA, XAVIER, ORIN.
  getNitrosContext().setParameterStr(
    "disparity", "nvidia::isaac_ros::SGMDisparity", "backends",
    vpi_backends_);
}

DisparityNode::~DisparityNode() {}

}  // namespace stereo_image_proc
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::stereo_image_proc::DisparityNode)
