// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "depth_to_point_cloud.hpp"

#include <limits>

#include "gems/gxf_helpers/expected_macro_gxf.hpp"

namespace nvidia {
namespace isaac_ros {
namespace depth_image_proc {

gxf_result_t DepthToPointCloud::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(
      allocator_, "allocator", "Memory allocator",
      "Handle to the memory allocator pool used for data generation.");
  result &= registrar->parameter(
      depth_receiver_, "depth_receiver", "Depth map input",
      "Name of incoming 'depth_receiver' channel.");
  result &= registrar->parameter(
      image_receiver_, "image_receiver", "Image input aligned with depth input",
      "Name of incoming 'image_receiver' channel."
      "Must be aligned at a pixel level with depth image.",
      gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(
    point_cloud_transmitter_, "point_cloud_transmitter",
    "DepthToPointCloud Transmitter to send the data", "");
  result &= registrar->parameter(
      skip_, "skip", "Skip",
      "If set to a value greater than 1 depth pixels will be skipped. For example `skip` = 2 wil "
      "skip half of the pixels. Use this value to limit the number of pixels converted to points.",
      1, GXF_PARAMETER_FLAGS_DYNAMIC);

  GXF_LOG_DEBUG("[DepthToPointCloud] Register Interfaces Finish.");
  return gxf::ToResultCode(result);
}

gxf_result_t DepthToPointCloud::start() {
  colorize_point_cloud_ = static_cast<bool>(image_receiver_.try_get());
  if (skip_ < 1) {
    GXF_LOG_ERROR("skip must be strictly positive, %d was provided", skip_.get());
    return GXF_ARGUMENT_OUT_OF_RANGE;
  }
  return GXF_SUCCESS;
}

gxf_result_t DepthToPointCloud::tick() {
  // Read input message(s) and validate them
  gxf::Entity depth_entity = UNWRAP_OR_RETURN(depth_receiver_->receive());
  nvidia::isaac::CameraMessageParts depth_message = UNWRAP_OR_RETURN(
    nvidia::isaac::GetCameraMessage(depth_entity));
  RETURN_IF_ERROR(validateDepthMessage(depth_message));

  nvidia::isaac::CameraMessageParts image_message;
  if (colorize_point_cloud_) {
    gxf::Entity image_entity = UNWRAP_OR_RETURN(image_receiver_.try_get().value()->receive());
    image_message = UNWRAP_OR_RETURN(nvidia::isaac::GetCameraMessage(image_entity));
    RETURN_IF_ERROR(validateImageMessage(depth_message, image_message));
  }

  // Create new message
  PointCloudProperties point_cloud_properties = createPointCloudProperties(
    depth_message, skip_.get());
  nvidia::isaac_ros::messages::PointCloudMessageParts point_cloud_message = UNWRAP_OR_RETURN(
    nvidia::isaac_ros::messages::CreatePointCloudMessage(
      context(), allocator_, point_cloud_properties.n_points, colorize_point_cloud_));

  point_cloud_message.info->use_color = colorize_point_cloud_;

  DepthProperties depth_properties = createDepthProperties(depth_message);
  cloud_compute_.DepthToPointCloudCuda(depth_message, image_message, point_cloud_message,
    point_cloud_properties, depth_properties, colorize_point_cloud_,
  skip_.get());

  // Forward the timestamp from the depth message
  *point_cloud_message.timestamp = *depth_message.timestamp;

  return gxf::ToResultCode(point_cloud_transmitter_->publish(point_cloud_message.message));
}

gxf_result_t DepthToPointCloud::stop() {
  return GXF_SUCCESS;
}

gxf::Expected<void> DepthToPointCloud::validateImageMessage(
  const nvidia::isaac::CameraMessageParts& depth_message,
  const nvidia::isaac::CameraMessageParts& image_message) {
  if (depth_message.frame->storage_type() != image_message.frame->storage_type()) {
    GXF_LOG_ERROR("Input image image must be stored in the same type as input depth");
    return gxf::Unexpected{GXF_INVALID_DATA_FORMAT};
  }

  gxf::VideoBufferInfo image_info = image_message.frame->video_frame_info();
  if (image_info.color_format != gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB) {
    GXF_LOG_ERROR("Input image must be of type "
                  "gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB ");
    return gxf::Unexpected{GXF_INVALID_DATA_FORMAT};
  }

  if (image_info.width != image_message.intrinsics->dimensions.x ||
      image_info.height != image_message.intrinsics->dimensions.y) {
    GXF_LOG_ERROR("Input image dimensions must match its camera model dimensions");
    return gxf::Unexpected{GXF_INVALID_DATA_FORMAT};
  }

  gxf::VideoBufferInfo depth_info = depth_message.frame->video_frame_info();
  if (depth_info.width != image_info.width ||
      depth_info.height != image_info.height) {
    GXF_LOG_ERROR("Input image dimensions must match input depth dimensions");
    return gxf::Unexpected{GXF_INVALID_DATA_FORMAT};
  }
  return gxf::Expected<void>{};
}

gxf::Expected<void> DepthToPointCloud::validateDepthMessage(
  const nvidia::isaac::CameraMessageParts& depth_message) {
  if (depth_message.frame->storage_type() != gxf::MemoryStorageType::kDevice) {
    GXF_LOG_ERROR("Input depth image must be stored in "
                  "gxf::MemoryStorageType::kDevice");
    return gxf::Unexpected{GXF_INVALID_DATA_FORMAT};
  }

  gxf::VideoBufferInfo depth_info = depth_message.frame->video_frame_info();
  if (depth_info.color_format != gxf::VideoFormat::GXF_VIDEO_FORMAT_D32F &&
      depth_info.color_format != gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY32) {
    GXF_LOG_ERROR("Input depth must be of type "
                  "gxf::VideoFormat::GXF_VIDEO_FORMAT_D32F %i",
                  static_cast<int>(gxf::VideoFormat::GXF_VIDEO_FORMAT_D32F));
    GXF_LOG_ERROR("Input depth must be of type "
                  "depth_info.color_format %i", static_cast<int>(depth_info.color_format));
    return gxf::Unexpected{GXF_INVALID_DATA_FORMAT};
  }

  if (depth_info.width != depth_message.intrinsics->dimensions.x ||
      depth_info.height != depth_message.intrinsics->dimensions.y) {
    GXF_LOG_ERROR("Input depth dimensions must match its camera model dimensions");
    return gxf::Unexpected{GXF_INVALID_DATA_FORMAT};
  }
  return gxf::Expected<void>{};
}

PointCloudProperties DepthToPointCloud::createPointCloudProperties(
  const nvidia::isaac::CameraMessageParts & depth_img,
  const int skip) {
  PointCloudProperties point_cloud_properties;

  const int point_step = colorize_point_cloud_ ? 4 : 3;

  point_cloud_properties.n_points = depth_img.intrinsics->dimensions.y *
                                    depth_img.intrinsics->dimensions.x / skip;
  point_cloud_properties.point_step = point_step;
  point_cloud_properties.x_offset = 0;
  point_cloud_properties.y_offset = 1;
  point_cloud_properties.z_offset = 2;
  point_cloud_properties.bad_point = std::numeric_limits<float>::quiet_NaN();
  if (colorize_point_cloud_) {
    point_cloud_properties.rgb_offset = 3;
  }
  return point_cloud_properties;
}

DepthProperties DepthToPointCloud::createDepthProperties(
  const nvidia::isaac::CameraMessageParts & depth_img) {
  DepthProperties depth_properties;

  depth_properties.width = depth_img.intrinsics->dimensions.x;
  depth_properties.height = depth_img.intrinsics->dimensions.y;
  depth_properties.f_x = depth_img.intrinsics->focal_length.x;
  depth_properties.f_y = depth_img.intrinsics->focal_length.y;
  depth_properties.c_x = depth_img.intrinsics->principal_point.x;
  depth_properties.c_y = depth_img.intrinsics->principal_point.y;

  depth_properties.red_offset = 0;  // Red bytes offset
  depth_properties.green_offset = 1;  // Red bytes offset
  depth_properties.blue_offset = 2;  // Red bytes offset

  return depth_properties;
}

}  // namespace depth_image_proc
}  // namespace isaac_ros
}  // namespace nvidia
