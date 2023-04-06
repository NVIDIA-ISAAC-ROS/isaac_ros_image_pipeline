// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "disparity_compositor.hpp"

#include <string>
#include <climits>

#include "gxf/multimedia/camera.hpp"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/parameter_parser_std.hpp"
#include "gxf/std/timestamp.hpp"


namespace nvidia
{
namespace isaac_ros
{

namespace
{
constexpr const char kNameFrame[] = "frame";
}  // namespace


gxf_result_t DisparityCompositor::registerInterface(gxf::Registrar * registrar) noexcept
{
  gxf::Expected<void> result;

  result &= registrar->parameter(
    disparity_receiver_, "disparity_receiver", "Image Input",
    "The image input frame");

  result &= registrar->parameter(
    left_camera_model_receiver_, "left_camera_model_receiver",
    "Camera Message Input", "The camera message input frame");

  result &= registrar->parameter(
    right_camera_model_receiver_, "right_camera_model_receiver",
    "Camera Message Input", "The camera message input frame");

  result &= registrar->parameter(
    output_transmitter_, "output_transmitter", "Combined output",
    "The image and cameara model output frames");

  result &= registrar->parameter(
    min_disparity_, "min_disparity",
    "The min value of disparity search range", "");

  result &= registrar->parameter(
    max_disparity_, "max_disparity",
    "The maximum value of disparity search range, has to be positive",
    "");

  result &= registrar->parameter(
    debug_, "debug", "Toggle debug messages",
    "True for enabling debug messages", false);

  return gxf::ToResultCode(result);
}

gxf_result_t DisparityCompositor::start() noexcept
{
  if (debug_) {
    SetSeverity(Severity::DEBUG);
  } else {
    SetSeverity(Severity::ERROR);
  }

  return GXF_SUCCESS;
}

gxf_result_t DisparityCompositor::tick() noexcept
{

  gxf::Expected<void> result;

  // Receive disparity image and left/right camera info
  auto maybe_disparity_message = disparity_receiver_->receive();
  if (!maybe_disparity_message) {
    return gxf::ToResultCode(maybe_disparity_message);
  }

  auto maybe_left_camera_model_message = left_camera_model_receiver_->receive();
  if (!maybe_left_camera_model_message) {
    return gxf::ToResultCode(maybe_left_camera_model_message);
  }

  auto maybe_right_camera_model_message = right_camera_model_receiver_->receive();
  if (!maybe_right_camera_model_message) {
    return gxf::ToResultCode(maybe_right_camera_model_message);
  }

  // Add timestamps
  std::string timestamp_name{"timestamp"};

  auto maybe_disparity_timestamp = maybe_disparity_message->get<nvidia::gxf::Timestamp>("timestamp");
  if (!maybe_disparity_timestamp) {    // Fallback to any 'timestamp'
    maybe_disparity_timestamp = maybe_disparity_message->get<nvidia::gxf::Timestamp>();
  }
  if(!maybe_disparity_timestamp) {
    auto maybe_input_timestamp = maybe_left_camera_model_message->get<nvidia::gxf::Timestamp>("timestamp");
    if (!maybe_input_timestamp) {
      timestamp_name = std::string{""};
      maybe_input_timestamp = maybe_left_camera_model_message.value().get<gxf::Timestamp>(timestamp_name.c_str());
    }
    if (!maybe_input_timestamp) {return GXF_FAILURE;}
    auto out_timestamp = maybe_disparity_message.value().add<gxf::Timestamp>(timestamp_name.c_str());
    if (!out_timestamp) {return GXF_FAILURE;}
    *out_timestamp.value() = *maybe_input_timestamp.value();
  }

  auto right_gxf_camera_model = maybe_right_camera_model_message.value().get<gxf::CameraModel>();
  if (!right_gxf_camera_model) {
    GXF_LOG_ERROR("Failed to get right camera model");
    return right_gxf_camera_model.error();
  }

  auto right_gxf_pose_3d = maybe_right_camera_model_message.value().get<gxf::Pose3D>();
  if (!right_gxf_pose_3d) {
    GXF_LOG_ERROR("Failed to get right pose 3D");
    return right_gxf_pose_3d.error();
  }

  auto disparity_image = maybe_disparity_message.value().get<gxf::VideoBuffer>();
  if (!disparity_image) {
    GXF_LOG_ERROR("Failed to get disparity image from message");
    return disparity_image.error();
  }

  GXF_LOG_DEBUG("[Disparity Compositor] Fulfill the disparity message headers");
  // Add focal length into output messages
  auto gxf_t = maybe_disparity_message.value().add<float>("t");
  if (!gxf_t) {return GXF_FAILURE;}
  *gxf_t->get() = right_gxf_camera_model.value()->focal_length.x;

  // Add Baseline in world units into output messages
  auto gxf_f = maybe_disparity_message.value().add<float>("f");
  if (!gxf_f) {return GXF_FAILURE;}
  *gxf_f->get() = -right_gxf_pose_3d.value()->translation[0];

  // Add min_disparity into output message
  auto gxf_min_disparity = maybe_disparity_message.value().add<float>("min_disparity");
  if (!gxf_min_disparity) {return GXF_FAILURE;}
  *gxf_min_disparity->get() = min_disparity_;

  // First try to get max_disparity from disparity message.
  // If not found, add max_disparity into output message, -1 represents unbounded value
  auto maybe_max_disparity = maybe_disparity_message->get<float>("max_disparity");
  if(!maybe_max_disparity){
    auto gxf_max_disparity = maybe_disparity_message.value().add<float>("max_disparity");
    if (!gxf_max_disparity) {return GXF_FAILURE;}
    *gxf_max_disparity->get() = (max_disparity_ == -1) ? INT_MAX : max_disparity_;
  }

  // Publish the message
  GXF_LOG_DEBUG("[Disparity Compositor] Publishing the fused message");
  result = output_transmitter_->publish(maybe_disparity_message.value());
  if (!result) {
    GXF_LOG_ERROR("Failed to send a message to the transmitter");
    return gxf::ToResultCode(result);
  }

  return GXF_SUCCESS;
}

}  // namespace isaac_ros
}  // namespace nvidia
