// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "extensions/sgm/camera_message_compositor.hpp"

#include <utility>

#include "extensions/messages/camera_message.hpp"

namespace nvidia
{
namespace isaac
{

namespace
{
constexpr const char kNameFrame[] = "frame";
constexpr const char kNameIntrinsics[] = "intrinsics";
constexpr const char kNameExtrinsics[] = "extrinsics";
constexpr const char kNameSequenceNumber[] = "sequence_number";
}  // namespace

gxf_result_t CameraMessageCompositor::registerInterface(gxf::Registrar * registrar)
{
  gxf::Expected<void> result;
  result &= registrar->parameter(
    video_buffer_input_, "video_buffer_input", "Video Buffer Input",
    "Input for messages containing video buffer component");
  result &= registrar->parameter(
    camera_model_input_, "camera_model_input",
    "Intrinsics Camera Model Input",
    "Input for messages containing camera model component");
  result &= registrar->parameter(
    camera_message_output_, "camera_message_output", "Camera Message Output",
    "Output CameraMessage with components combined from inputs");
  return gxf::ToResultCode(result);
}

gxf_result_t CameraMessageCompositor::tick()
{
  // Receive inputs
  auto maybe_input_video_buffer_entity = video_buffer_input_->receive();
  if (!maybe_input_video_buffer_entity) {
    GXF_LOG_ERROR("[CameraMessageCompositor] Failed to received input video_buffer");
    return gxf::ToResultCode(maybe_input_video_buffer_entity);
  }

  auto maybe_camera_model_entity = camera_model_input_->receive();
  if (!maybe_camera_model_entity) {
    GXF_LOG_ERROR("[CameraMessageCompositor] Failed to receive camera model");
    return gxf::ToResultCode(maybe_camera_model_entity);
  }

  // Check if the input message entity has the camera message fields. If it doesn't, add it
  // Make sure a timestamp is present in the input videobuff entity
  // This timestamp is passedthrough since the input videobuff entity is being passedthrough
  auto maybe_timestamp =
    maybe_input_video_buffer_entity->get<nvidia::gxf::Timestamp>();
  if (!maybe_timestamp) {
    GXF_LOG_ERROR("[CameraMessageCompositor] Input video buffer does not contain a timestamp");
    return GXF_FAILURE;
  }

  auto maybe_output_video_buffer =
    maybe_input_video_buffer_entity->get<gxf::VideoBuffer>(kNameFrame);
  if (!maybe_output_video_buffer) {
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
      context(), maybe_input_video_buffer_entity->eid(),
      video_buffer_tid, nullptr, nullptr, &video_buffer_cid);

    if (result_2 != GXF_SUCCESS) {
      GXF_LOG_ERROR("Failed to get cid of the VideoBuffer component");
      return result_2;
    }
    GxfParameterSetStr(context(), video_buffer_cid, kInternalNameParameterKey, kNameFrame);
    GXF_LOG_DEBUG("Override the VideoBuffer component name to %s", kNameFrame);
  }

  auto maybe_output_intrinsics =
    maybe_input_video_buffer_entity->get<gxf::CameraModel>(kNameIntrinsics);
  if (!maybe_output_intrinsics) {
    maybe_output_intrinsics =
      maybe_input_video_buffer_entity->add<gxf::CameraModel>(kNameIntrinsics);
    if (!maybe_output_intrinsics) {
      GXF_LOG_ERROR("[CameraMessageCompositor] Unable to add intrinsics to output message");
      return maybe_output_intrinsics.error();
    }
  }
  auto output_intrinsics = maybe_output_intrinsics.value();

  auto maybe_output_extrinsics =
    maybe_input_video_buffer_entity->get<gxf::Pose3D>(kNameExtrinsics);
  if (!maybe_output_extrinsics) {
    maybe_output_extrinsics = maybe_input_video_buffer_entity->add<gxf::Pose3D>(kNameExtrinsics);
    if (!maybe_output_extrinsics) {
      GXF_LOG_ERROR("[CameraMessageCompositor] Unable to add extrinsics to output message");
      return maybe_output_extrinsics.error();
    }
  }
  auto output_extrinsics = maybe_output_extrinsics.value();

  auto maybe_output_seq_num = maybe_input_video_buffer_entity->get<int64_t>(kNameSequenceNumber);
  if (!maybe_output_seq_num) {
    maybe_output_seq_num = maybe_input_video_buffer_entity->add<int64_t>(kNameSequenceNumber);
    if (!maybe_output_seq_num) {
      GXF_LOG_ERROR("[CameraMessageCompositor] Unable to add sequence number to output message");
      return maybe_output_seq_num.error();
    }
  }

  // Populate the output entity with extrinsics and intrinsics
  // Copy intrinsics from left camera model to the disparity message
  auto maybe_input_camera_model =
    maybe_camera_model_entity->get<gxf::CameraModel>(kNameIntrinsics);
  if (!maybe_input_camera_model) {
    GXF_LOG_ERROR("[CameraMessageCompositor] Failed to get instrinsics from the input entity");
    return maybe_input_camera_model.error();
  }
  *output_intrinsics = *maybe_input_camera_model.value();

  // Forward extrinsics from right camera model to the video_buffer message
  auto maybe_extrinsics_pose = maybe_camera_model_entity->get<gxf::Pose3D>(kNameExtrinsics);
  if (!maybe_extrinsics_pose) {
    GXF_LOG_ERROR("[CameraMessageCompositor] Failed to get extrinsics from the input entity");
    return maybe_extrinsics_pose.error();
  }
  *output_extrinsics = *maybe_extrinsics_pose.value();

  return gxf::ToResultCode(
    camera_message_output_->publish(maybe_input_video_buffer_entity.value()));
}

}  // namespace isaac
}  // namespace nvidia
