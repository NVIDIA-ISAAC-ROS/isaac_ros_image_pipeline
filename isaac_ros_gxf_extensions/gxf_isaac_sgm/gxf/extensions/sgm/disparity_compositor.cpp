// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "extensions/sgm/disparity_compositor.hpp"

#include <climits>
#include <string>
#include <utility>

#include "gxf/core/parameter_parser_std.hpp"
#include "gxf/multimedia/camera.hpp"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/timestamp.hpp"


namespace nvidia {
namespace isaac {

namespace {
constexpr const char kNameFrame[] = "frame";
constexpr const char kNameIntrinsics[] = "intrinsics";
constexpr const char kNameExtrinsics[] = "extrinsics";
constexpr const char kNameSequenceNumber[] = "sequence_number";
}  // namespace


gxf_result_t DisparityCompositor::registerInterface(gxf::Registrar * registrar) noexcept {
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

  return gxf::ToResultCode(result);
}

gxf_result_t DisparityCompositor::start() noexcept {
  return GXF_SUCCESS;
}

gxf_result_t DisparityCompositor::tick() noexcept {
  gxf::Expected<void> result;

  // Receive disparity image and left/right camera info
  auto maybe_input_disparity_message = disparity_receiver_->receive();
  if (!maybe_input_disparity_message) {
    GXF_LOG_ERROR("[Disparity Compositor] Failed to received input disparity");
    return gxf::ToResultCode(maybe_input_disparity_message);
  }

  auto maybe_left_camera_model_message = left_camera_model_receiver_->receive();
  if (!maybe_left_camera_model_message) {
    GXF_LOG_ERROR("[Disparity Compositor] Failed to receive left camera model");
    return gxf::ToResultCode(maybe_left_camera_model_message);
  }

  auto maybe_right_camera_model_message = right_camera_model_receiver_->receive();
  if (!maybe_right_camera_model_message) {
    GXF_LOG_ERROR("[Disparity Compositor] Failed to receive right camera model");
    return gxf::ToResultCode(maybe_right_camera_model_message);
  }

  // Make sure a timestamp is present in the input disparity message
  auto maybe_disparity_timestamp = maybe_input_disparity_message->get<nvidia::gxf::Timestamp>();
  if (!maybe_disparity_timestamp) {
    GXF_LOG_ERROR("[Disparity Compositor] Input disparity image msg does not contain a timestamp");
    return GXF_FAILURE;
  }

  // Check if the input disparity image has the camera message fields
  // If it doesn't, manually add it
  auto maybe_output_disparity_image =
    maybe_input_disparity_message->get<gxf::VideoBuffer>(kNameFrame);
  if (!maybe_output_disparity_image) {
    // Rename the unamed gxf::VideoBuffer as a named gxf::VideoBuffer
    // only if the input entity does NOT contain a named gxf::VideoBuffer already.
    // This also assumes that only one VideoBuffer component is present in the messgae entity

    gxf_tid_t video_buffer_tid;
    const auto result_1 = GxfComponentTypeId(
      context(), "nvidia::gxf::VideoBuffer", &video_buffer_tid);

    if (result_1 != GXF_SUCCESS) {
      GXF_LOG_ERROR("Failed to get tid of nvidia::gxf::VideoBuffer");
      return result_1;
    }

    gxf_uid_t video_buffer_cid;
    const auto result_2 = GxfComponentFind(
      context(), maybe_input_disparity_message->eid(),
      video_buffer_tid, nullptr, nullptr, &video_buffer_cid);

    if (result_2 != GXF_SUCCESS) {
      GXF_LOG_ERROR("Failed to get cid of the VideoBuffer component");
      return result_2;
    }
    GxfParameterSetStr(context(), video_buffer_cid, kInternalNameParameterKey, kNameFrame);
    GXF_LOG_DEBUG("Override the VideoBuffer component name to %s", kNameFrame);
  }

  auto maybe_output_intrinsics =
    maybe_input_disparity_message->get<gxf::CameraModel>(kNameIntrinsics);
  if (!maybe_output_intrinsics) {
    maybe_output_intrinsics = maybe_input_disparity_message->add<gxf::CameraModel>(kNameIntrinsics);
    if (!maybe_output_intrinsics) {
      GXF_LOG_ERROR("[Disparity Compositor] Unable to add intrinsics to disparity message");
      return maybe_output_intrinsics.error();
    }
  }
  auto output_intrinsics = maybe_output_intrinsics.value();

  auto maybe_output_extrinsics = maybe_input_disparity_message->get<gxf::Pose3D>(kNameExtrinsics);
  if (!maybe_output_extrinsics) {
    maybe_output_extrinsics = maybe_input_disparity_message->add<gxf::Pose3D>(kNameExtrinsics);
    if (!maybe_output_extrinsics) {
      GXF_LOG_ERROR("[Disparity Compositor] Unable to add extrinsics to disparity message");
      return maybe_output_extrinsics.error();
    }
  }
  auto output_extrinsics = maybe_output_extrinsics.value();

  auto maybe_output_seq_num = maybe_input_disparity_message->get<int64_t>(kNameSequenceNumber);
  if (!maybe_output_seq_num) {
    maybe_output_seq_num = maybe_input_disparity_message->add<int64_t>(kNameSequenceNumber);
    if (!maybe_output_seq_num) {
      GXF_LOG_ERROR("[Disparity Compositor] Unable to add sequence number to disparity message");
      return maybe_output_seq_num.error();
    }
  }

  // Forward data from camera model receivers to the disparity message

  // Copy intrinsics from left camera model to the disparity message
  auto left_gxf_camera_model =
    maybe_left_camera_model_message.value().get<gxf::CameraModel>(kNameIntrinsics);
  if (!left_gxf_camera_model) {
    GXF_LOG_ERROR("[Disparity Compositor] Failed to get instrinsics from the left camera model");
    return left_gxf_camera_model.error();
  }
  *output_intrinsics = *left_gxf_camera_model.value();

  // Adjust instrinsics if needed
  const auto disp_video_buff = maybe_input_disparity_message->get<gxf::VideoBuffer>(kNameFrame);
  const gxf::VideoBufferInfo video_info = disp_video_buff.value()->video_frame_info();
  if (video_info.width != output_intrinsics->dimensions.x ||
    video_info.height != output_intrinsics->dimensions.y)
  {

    const float scaler_x = static_cast<float>(video_info.width) / output_intrinsics->dimensions.x;
    const float scaler_y = static_cast<float>(video_info.height) / output_intrinsics->dimensions.y;
    const float min_scaler = std::min(scaler_x, scaler_y);
    output_intrinsics->dimensions.x = video_info.width;
    output_intrinsics->dimensions.y = video_info.height;
    output_intrinsics->focal_length.x *= scaler_x;
    output_intrinsics->focal_length.y *= scaler_y;

    const float pixel_center = 0.5f;
    output_intrinsics->principal_point.x =
      scaler_x * (output_intrinsics->principal_point.x + pixel_center) - pixel_center;
    output_intrinsics->principal_point.y =
      scaler_y * (output_intrinsics->principal_point.y + pixel_center) - pixel_center;
  }

  // Forward extrinsics from right camera model to the disparity message
  auto right_gxf_pose_3d = maybe_right_camera_model_message->get<gxf::Pose3D>(kNameExtrinsics);
  if (!right_gxf_pose_3d) {
    GXF_LOG_ERROR("[Disparity Compositor] Failed to get extrinsics from the right camera model");
    return right_gxf_pose_3d.error();
  }
  *output_extrinsics = *right_gxf_pose_3d.value();

  GXF_LOG_DEBUG("[Disparity Compositor] Fulfill the disparity message headers");
  // Add focal length into output messages
  auto gxf_t = maybe_input_disparity_message->add<float>("t");
  if (!gxf_t) {
    return GXF_FAILURE;}
  *gxf_t->get() = -right_gxf_pose_3d.value()->translation[0];

  // Add Baseline in world units into output messages
  auto gxf_f = maybe_input_disparity_message->add<float>("f");
  if (!gxf_f) {
    return GXF_FAILURE;
  }
  *gxf_f->get() = output_intrinsics->focal_length.x;

  // Add min_disparity into output message
  auto gxf_min_disparity = maybe_input_disparity_message->add<float>("min_disparity");
  if (!gxf_min_disparity) {return GXF_FAILURE;}
  *gxf_min_disparity->get() = min_disparity_;

  // First try to get max_disparity from disparity message.
  // If not found, add max_disparity into output message, -1 represents unbounded value
  auto maybe_max_disparity = maybe_input_disparity_message->get<float>("max_disparity");
  if (!maybe_max_disparity) {
    auto gxf_max_disparity = maybe_input_disparity_message->add<float>("max_disparity");
    if (!gxf_max_disparity) {return GXF_FAILURE;}
    *gxf_max_disparity->get() = (max_disparity_ == -1) ? INT_MAX : max_disparity_;
  }

  // Publish the message
  GXF_LOG_DEBUG("[Disparity Compositor] Publishing the fused message");
  result = output_transmitter_->publish(maybe_input_disparity_message.value());
  if (!result) {
    GXF_LOG_ERROR("Failed to send a message to the transmitter");
    return gxf::ToResultCode(result);
  }

  return GXF_SUCCESS;
}

}  // namespace isaac
}  // namespace nvidia
