// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_vpi_utils/vpi_utilities.hpp"
#include "vpi_conversions/vpi_conversions.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace stereo_image_proc
{
namespace
{
constexpr int32_t kTegraSupportedScale = 256;
}  // namespace

DisparityNode::DisparityNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("disparity_node", options),
  backend_(declare_parameter<std::string>("backend", "CUDA")),
  max_disparity_(declare_parameter<float>("max_disparity", 256.0)),
  confidence_threshold_(declare_parameter<int>("confidence_threshold", 60000)),
  confidence_type_(declare_parameter<int>("confidence_type", 0)),
  window_size_(declare_parameter<int>("window_size", 7)),
  num_passes_(declare_parameter<int>("num_passes", 2)),
  p1_(declare_parameter<int>("p1", 8)),
  p2_(declare_parameter<int>("p2", 120)),
  p2_alpha_(declare_parameter<int>("p2_alpha", 1)),
  quality_(declare_parameter<int>("quality", 1)),
  input_queue_size_(declare_parameter<uint16_t>("input_queue_size", 10)),
  output_queue_size_(declare_parameter<uint16_t>("output_queue_size", 10)),
  left_image_sub_{},
  right_image_sub_{},
  left_camera_info_sub_{},
  right_camera_info_sub_{},
  exact_sync_{ExactPolicy{input_queue_size_}, left_image_sub_, right_image_sub_,
    left_camera_info_sub_, right_camera_info_sub_}
{
  RCLCPP_DEBUG(get_logger(), "[DisparityNode] Constructor");
  impl_.vpi_backends = vpi_utils::ToVPIBackend(backend_);

  // Initialize VPI stream
  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("DisparityNode");
  CHECK_VPI_STATUS(vpiStreamCreateWrapperCUDA(*cuda_stream_, impl_.vpi_backends, &impl_.stream));

  if (!initialize()) {
    RCLCPP_ERROR(get_logger(), "[DisparityNode] Failed to initialize");
    throw std::runtime_error("[DisparityNode] Failed to initialize");
  }

  const rclcpp::QoS input_qos =
    ::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos")
    .keep_last(input_queue_size_);
  const rclcpp::QoS output_qos =
    ::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos")
    .keep_last(output_queue_size_);

  // Register synchronized callback
  exact_sync_.registerCallback(
    std::bind(
      &DisparityNode::InputCallback, this,
      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));

  // Subscription options (can be used for callback groups, etc.)
  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  sub_options.acceptable_buffer_backends = "any";

  left_image_sub_.subscribe(this, "left/image_rect", input_qos, sub_options);
  right_image_sub_.subscribe(this, "right/image_rect", input_qos, sub_options);
  left_camera_info_sub_.subscribe(this, "left/camera_info", input_qos, sub_options);
  right_camera_info_sub_.subscribe(this, "right/camera_info", input_qos, sub_options);

  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  disparity_pub_ = create_publisher<stereo_msgs::msg::DisparityImage>(
    "disparity", output_qos, pub_options);

  RCLCPP_DEBUG(get_logger(), "[DisparityNode] Setup complete");
}

bool DisparityNode::initialize()
{
  if (max_disparity_ < 1.0) {
    std::ostringstream ss;
    ss << __FILE__ << ":" << __LINE__ << ": " << "max_disparity" <<
      " must be greater than or equal to 1.0";
    RCLCPP_ERROR(get_logger(), "%s", ss.str().c_str());
    return false;
  }

  // Set and print out backend used
  if (impl_.vpi_backends == VPI_BACKEND_CPU) {
    RCLCPP_ERROR(get_logger(), "CPU backend is not supported for SGM");
    throw std::invalid_argument("[DisparityNode] CPU backend is not supported for SGM");
  }
  RCLCPP_DEBUG(get_logger(), "Using VPI backend: %s",
    nvidia::isaac_ros::vpi_utils::VPIBackendToString(
        static_cast<VPIBackend>(impl_.vpi_backends)).c_str());

  // Image conversions use CUDA even when stereo estimation runs on Jetson accelerators.
  impl_.vpi_flags = impl_.vpi_backends | VPI_BACKEND_CUDA;
  CHECK_VPI_STATUS(vpiInitStereoDisparityEstimatorCreationParams(&impl_.disparity_params));
  impl_.disparity_params.maxDisparity = static_cast<int32_t>(max_disparity_);

  // Fill it up with the best known good confidence threshold params
  CHECK_VPI_STATUS(vpiInitStereoDisparityEstimatorParams(&impl_.disparity_context_params));
  impl_.disparity_context_params.confidenceThreshold = confidence_threshold_;
  // Map confidence_type_ parameter directly to VPI enum, supporting all values
  // 0 = VPI_STEREO_CONFIDENCE_ABSOLUTE
  // 1 = VPI_STEREO_CONFIDENCE_RELATIVE
  // 2 = VPI_STEREO_CONFIDENCE_INFERENCE
  switch (static_cast<VPIStereoDisparityConfidenceType>(confidence_type_)) {
    case VPI_STEREO_CONFIDENCE_ABSOLUTE:
    case VPI_STEREO_CONFIDENCE_RELATIVE:
    case VPI_STEREO_CONFIDENCE_INFERENCE:
      impl_.disparity_context_params.confidenceType =
        static_cast<VPIStereoDisparityConfidenceType>(confidence_type_);
      break;
    default:
      RCLCPP_WARN(get_logger(),
        "[DisparityNode] Invalid confidence_type=%d, using RELATIVE (1)", confidence_type_);
      impl_.disparity_context_params.confidenceType = VPI_STEREO_CONFIDENCE_RELATIVE;
      break;
  }

  impl_.disparity_context_params.windowSize = window_size_;
  impl_.disparity_context_params.numPasses = num_passes_;
  impl_.disparity_context_params.maxDisparity = max_disparity_;
  impl_.disparity_context_params.p1 = p1_;
  impl_.disparity_context_params.p2 = p2_;
  impl_.disparity_context_params.p2Alpha = p2_alpha_;
  impl_.disparity_context_params.quality = quality_;

  // Convert Format Parameters
  CHECK_VPI_STATUS(vpiInitConvertImageFormatParams(&impl_.stereo_input_scale_params));
  if (impl_.vpi_backends == VPI_BACKEND_JETSON) {
    impl_.stereo_input_scale_params.scale = kTegraSupportedScale;
  }

  CHECK_VPI_STATUS(vpiInitConvertImageFormatParams(&impl_.disparity_scale_params));
  // Scale the per-pixel disparity output using the default value.
  impl_.disparity_scale_params.scale = 1.0 / 32.0;
  return true;
}

void DisparityNode::deinitialize()
{
  if (impl_.stream) {
    VPIStatus status = vpiStreamSync(impl_.stream);
    if (status != VPI_SUCCESS) {
      RCLCPP_WARN(get_logger(), "[DisparityNode] vpiStreamSync failed during cleanup: %d", status);
    }
    vpiStreamDestroy(impl_.stream);
    impl_.stream = nullptr;
  }

  vpiImageDestroy(impl_.left_formatted);
  vpiImageDestroy(impl_.right_formatted);
  vpiPayloadDestroy(impl_.stereo_payload);
  vpiImageDestroy(impl_.disparity_raw);
  vpiImageDestroy(impl_.confidence_map);
  impl_.left_formatted = nullptr;
  impl_.right_formatted = nullptr;
  impl_.stereo_payload = nullptr;
  impl_.disparity_raw = nullptr;
  impl_.confidence_map = nullptr;
}

DisparityNode::~DisparityNode()
{
  RCLCPP_DEBUG(get_logger(), "[DisparityNode] Destructor");
  deinitialize();
}

bool DisparityNode::InitVPIImages(
  const sensor_msgs::msg::Image & left_image_msg,
  const sensor_msgs::msg::Image & right_image_msg)
{
  // Check encoding of left and right images
  const bool left_ok = (left_image_msg.encoding == "bgr8" || left_image_msg.encoding == "rgb8");
  const bool right_ok = (right_image_msg.encoding == "bgr8" || right_image_msg.encoding == "rgb8");
  if (!left_ok || !right_ok) {
    RCLCPP_ERROR(get_logger(),
      "[DisparityNode] Left and right images must be in BGR8 or RGB8 format");
    return false;
  }

  if (left_image_msg.width <= 0 || left_image_msg.height <= 0 ||
    right_image_msg.width <= 0 || right_image_msg.height <= 0)
  {
    RCLCPP_ERROR(get_logger(),
      "[DisparityNode] Left and right images must have non-zero width and height");
    return false;
  }

  const int32_t width = static_cast<int32_t>(left_image_msg.width);
  const int32_t height = static_cast<int32_t>(left_image_msg.height);

  if (left_image_msg.width != right_image_msg.width ||
    left_image_msg.height != right_image_msg.height)
  {
    RCLCPP_ERROR(get_logger(),
      "[DisparityNode] Left and right images must have the same width and height");
    return false;
  }

  const bool cache_valid = (height == impl_.prev_height) && (width == impl_.prev_width);
  if (cache_valid) {
    return true;
  }

  if (impl_.vpi_backends & VPI_BACKEND_JETSON) {
    // VPI_BACKEND_JETSON uses a block-linear Y8 format.
    impl_.stereo_format = VPI_IMAGE_FORMAT_Y8_ER_BL;
  } else if (impl_.vpi_backends & VPI_BACKEND_CUDA) {
    // The CUDA x86 pipeline uses a Y16 format.
    impl_.stereo_format = VPI_IMAGE_FORMAT_Y16_ER;
  } else {
    RCLCPP_ERROR(get_logger(), "[DisparityNode] SGM on CPU not supported");
    throw std::runtime_error("[DisparityNode] SGM on CPU not supported");
  }

  const VPIImageFormat disparity_format = VPI_IMAGE_FORMAT_S16;
  const VPIImageFormat confidence_format = VPI_IMAGE_FORMAT_U16;

  // Recreate left and right input VPI images
  vpiImageDestroy(impl_.left_formatted);
  vpiImageDestroy(impl_.right_formatted);
  CHECK_VPI_STATUS(vpiImageCreate(
      width, height, impl_.stereo_format, impl_.vpi_flags, &impl_.left_formatted));
  CHECK_VPI_STATUS(vpiImageCreate(
      width, height, impl_.stereo_format, impl_.vpi_flags, &impl_.right_formatted));

  // Recreate temporaries for changing format from input to stereo format
  vpiImageDestroy(impl_.disparity_raw);
  CHECK_VPI_STATUS(vpiImageCreate(
      width, height, disparity_format, impl_.vpi_flags, &impl_.disparity_raw));

  // Use a confidence map on both backends.
  vpiImageDestroy(impl_.confidence_map);
  CHECK_VPI_STATUS(vpiImageCreate(
      width, height, confidence_format, impl_.vpi_flags, &impl_.confidence_map));

  // Recreate stereo payload with parameters for stereo disparity algorithm
  vpiPayloadDestroy(impl_.stereo_payload);
  CHECK_VPI_STATUS(vpiCreateStereoDisparityEstimator(
      impl_.vpi_backends, width, height, impl_.stereo_format, &impl_.disparity_params,
      &impl_.stereo_payload));

  // Update cached dimensions
  impl_.prev_width = width;
  impl_.prev_height = height;
  return true;
}

void DisparityNode::InputCallback(
  const sensor_msgs::msg::Image::ConstSharedPtr & left_image_msg,
  const sensor_msgs::msg::Image::ConstSharedPtr & right_image_msg,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & left_camera_info_msg,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & right_camera_info_msg)
{
  RCLCPP_DEBUG(get_logger(), "[DisparityNode] InputCallback");

  if (left_image_msg == nullptr || right_image_msg == nullptr ||
    left_camera_info_msg == nullptr || right_camera_info_msg == nullptr)
  {
    RCLCPP_ERROR(get_logger(), "[DisparityNode] InputCallback: Missing input data");
    return;
  }

  if (left_image_msg->width != right_image_msg->width ||
    left_image_msg->height != right_image_msg->height)
  {
    RCLCPP_ERROR(get_logger(),
      "[DisparityNode] InputCallback: Left and right images must have the same width and height");
    return;
  }

  // Update VPI images
  if (!InitVPIImages(*left_image_msg, *right_image_msg)) {
    RCLCPP_ERROR(get_logger(), "[DisparityNode] Failed to initialize VPI images");
    return;
  }

  auto disparity_msg =
    vpi_conversions::allocate_disparity_image_msg_from_input(left_image_msg, right_image_msg);

  {
    // Get read handles for input images and a write handle for the disparity output
    auto left_input = vpi_conversions::from_input_image_msg(
      *left_image_msg, *cuda_stream_, impl_.vpi_flags);
    auto right_input = vpi_conversions::from_input_image_msg(
      *right_image_msg, *cuda_stream_, impl_.vpi_flags);
    auto disparity_output = vpi_conversions::from_output_disparity_image_msg(
      *disparity_msg, *cuda_stream_, impl_.vpi_flags);

    // Convert input-format images to stereo-format images
    CHECK_VPI_STATUS(vpiSubmitConvertImageFormat(
      impl_.stream, VPI_BACKEND_CUDA, left_input.image(), impl_.left_formatted, NULL));
    CHECK_VPI_STATUS(vpiSubmitConvertImageFormat(
      impl_.stream, VPI_BACKEND_CUDA, right_input.image(), impl_.right_formatted, NULL));

    // There's possibly a bug in VPI, without this sync there's an error trying to
    // lock a container for shared access while it's already locked for exclusive access.
    CHECK_VPI_STATUS(vpiStreamSync(impl_.stream));

    // Calculate raw disparity and confidence map
    CHECK_VPI_STATUS(vpiSubmitStereoDisparityEstimator(
        impl_.stream, impl_.vpi_backends, impl_.stereo_payload, impl_.left_formatted,
        impl_.right_formatted, impl_.disparity_raw, impl_.confidence_map,
        &impl_.disparity_context_params));

    // Sync before using disparity output: avoid VPI_ERROR_BUFFER_LOCKED when
    // converting into the wrapped ROS buffer (exclusive write vs shared lock).
    CHECK_VPI_STATUS(vpiStreamSync(impl_.stream));

    // Convert to ROS 2 standard 32-bit float format
    CHECK_VPI_STATUS(vpiSubmitConvertImageFormat(
        impl_.stream, VPI_BACKEND_CUDA, impl_.disparity_raw,
        disparity_output.image(), &impl_.disparity_scale_params));

    // Wait for operations to complete
    CHECK_VPI_STATUS(vpiStreamSync(impl_.stream));
    // Ensure CUDA stream is fully complete so published buffer is visible to consumers
    CHECK_CUDA_ERROR(cudaStreamSynchronize(*cuda_stream_), "[DisparityNode] cudaStreamSynchronize");
  }

  // Pass the updated max_disparity to compositor
  disparity_msg->t = -right_camera_info_msg->p[3];
  disparity_msg->f = left_camera_info_msg->p[0];
  disparity_msg->min_disparity = 0.0f;
  disparity_msg->max_disparity = max_disparity_;

  // Publish the message
  disparity_pub_->publish(std::move(disparity_msg));
}

}  // namespace stereo_image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::stereo_image_proc::DisparityNode)
