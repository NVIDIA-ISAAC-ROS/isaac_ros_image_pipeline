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

#include "image_flip.hpp"

#include <vector>

#include "gxf/multimedia/video.hpp"
#include "gxf/std/timestamp.hpp"

namespace nvidia
{
namespace isaac_ros
{

#define CHECK_VPI_STATUS(STMT) \
  do { \
    VPIStatus status = (STMT); \
    if (status != VPI_SUCCESS) { \
      char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH]; \
      vpiGetLastStatusMessage(buffer, sizeof(buffer)); \
      std::ostringstream ss; \
      ss << __FILE__ << ":" << __LINE__ << ": " << vpiStatusGetName(status) << ": " << buffer; \
      GXF_LOG_ERROR(ss.str().c_str()); \
      return GXF_FAILURE; \
    } \
  } while (0);

using VideoFormat = nvidia::gxf::VideoFormat;

const std::unordered_map<std::string, uint32_t> kStrToVpiBackend({{"CPU", VPI_BACKEND_CPU},
    {"CUDA", VPI_BACKEND_CUDA},
    {"VIC", VPI_BACKEND_VIC},
    {"ALL", VPI_BACKEND_ALL}});

const std::unordered_map<std::string, VPIFlipMode> kStrToVpiFlipMode(
  {{"HORIZONTAL", VPI_FLIP_HORIZ}, {"VERTICAL", VPI_FLIP_VERT}, {"BOTH", VPI_FLIP_BOTH}});

struct VPIFormat
{
  VPIImageFormat image_format;
  std::vector<VPIPixelType> pixel_type;
};

VPIFormat ToVpiFormat(VideoFormat value)
{
  switch (value) {
    case VideoFormat::GXF_VIDEO_FORMAT_NV12:
      return VPIFormat{.image_format = VPI_IMAGE_FORMAT_NV12,
        .pixel_type = {VPI_PIXEL_TYPE_U8, VPI_PIXEL_TYPE_2U8}};
    case VideoFormat::GXF_VIDEO_FORMAT_NV12_ER:
      return VPIFormat{.image_format = VPI_IMAGE_FORMAT_NV12_ER,
        .pixel_type = {VPI_PIXEL_TYPE_U8, VPI_PIXEL_TYPE_2U8}};
    case VideoFormat::GXF_VIDEO_FORMAT_RGBA:
      return VPIFormat{.image_format = VPI_IMAGE_FORMAT_RGBA8, .pixel_type = {VPI_PIXEL_TYPE_4U8}};
    case VideoFormat::GXF_VIDEO_FORMAT_BGRA:
      return VPIFormat{.image_format = VPI_IMAGE_FORMAT_BGRA8, .pixel_type = {VPI_PIXEL_TYPE_4U8}};
    case VideoFormat::GXF_VIDEO_FORMAT_RGB:
      return VPIFormat{.image_format = VPI_IMAGE_FORMAT_RGB8, .pixel_type = {VPI_PIXEL_TYPE_3U8}};
    case VideoFormat::GXF_VIDEO_FORMAT_BGR:
      return VPIFormat{.image_format = VPI_IMAGE_FORMAT_BGR8, .pixel_type = {VPI_PIXEL_TYPE_3U8}};
    case VideoFormat::GXF_VIDEO_FORMAT_GRAY:
      return VPIFormat{.image_format = VPI_IMAGE_FORMAT_U8, .pixel_type = {VPI_PIXEL_TYPE_U8}};
    case VideoFormat::GXF_VIDEO_FORMAT_GRAY16:
      return VPIFormat{.image_format = VPI_IMAGE_FORMAT_U16, .pixel_type = {VPI_PIXEL_TYPE_U16}};
    case VideoFormat::GXF_VIDEO_FORMAT_GRAY32:
      return VPIFormat{.image_format = VPI_IMAGE_FORMAT_U32, .pixel_type = {VPI_PIXEL_TYPE_U32}};
    case VideoFormat::GXF_VIDEO_FORMAT_NV24:
      return VPIFormat{.image_format = VPI_IMAGE_FORMAT_NV24,
        .pixel_type = {VPI_PIXEL_TYPE_U8, VPI_PIXEL_TYPE_2U8}};
    case VideoFormat::GXF_VIDEO_FORMAT_NV24_ER:
      return VPIFormat{.image_format = VPI_IMAGE_FORMAT_NV24_ER,
        .pixel_type = {VPI_PIXEL_TYPE_U8, VPI_PIXEL_TYPE_2U8}};
    default:
      return VPIFormat{.image_format = VPI_IMAGE_FORMAT_RGB8, .pixel_type = {VPI_PIXEL_TYPE_3U8}};
  }
}

VPIStatus CreateVPIImageWrapper(
  VPIImage & vpi_image, VPIImageData & img_data, uint64_t flags,
  const nvidia::gxf::Handle<nvidia::gxf::VideoBuffer> & video_buff)
{
  nvidia::gxf::VideoBufferInfo image_info = video_buff->video_frame_info();
  VPIFormat vpi_format = ToVpiFormat(image_info.color_format);
  img_data.bufferType = VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;
  img_data.buffer.pitch.format = vpi_format.image_format;
  img_data.buffer.pitch.numPlanes = image_info.color_planes.size();
  auto data_ptr_offset = 0;
  for (size_t i = 0; i < image_info.color_planes.size(); ++i) {
    img_data.buffer.pitch.planes[i].data = video_buff->pointer() + data_ptr_offset;
    img_data.buffer.pitch.planes[i].height = image_info.color_planes[i].height;
    img_data.buffer.pitch.planes[i].width = image_info.color_planes[i].width;
    img_data.buffer.pitch.planes[i].pixelType = vpi_format.pixel_type[i];
    img_data.buffer.pitch.planes[i].pitchBytes = image_info.color_planes[i].stride;

    data_ptr_offset = image_info.color_planes[i].size;
  }
  return vpiImageCreateWrapper(&img_data, nullptr, flags, &vpi_image);
}

VPIStatus UpdateVPIImageWrapper(
  VPIImage & image, VPIImageData & imageWrap,
  const nvidia::gxf::Handle<nvidia::gxf::VideoBuffer> & video_buff)
{
  nvidia::gxf::VideoBufferInfo image_info = video_buff->video_frame_info();
  auto data_ptr_offset = 0;
  for (size_t i = 0; i < image_info.color_planes.size(); ++i) {
    imageWrap.buffer.pitch.planes[i].data = video_buff->pointer() + data_ptr_offset;
    data_ptr_offset = image_info.color_planes[i].size;
  }
  return vpiImageSetWrapper(image, &imageWrap);
}

gxf_result_t ImageFlip::registerInterface(gxf::Registrar * registrar)
{
  gxf::Expected<void> result;

  result &= registrar->parameter(
    image_name_, "image_name", "Image Name",
    "The name of image to be received", std::string("input"));
  result &= registrar->parameter(
    output_name_, "output_name", "Output Image Name",
    "The name of the videobuffer to be passed to next node",
    std::string("flip"));
  result &= registrar->parameter(pool_, "pool", "Pool", "Memory pool for allocating output data");
  result &= registrar->parameter(
    image_receiver_, "image_receiver", "Input Receiver",
    "Receiver to get the image");
  result &= registrar->parameter(
    output_transmitter_, "output_transmitter", "Output Transmitter",
    "Transmitter to send the data");
  result &= registrar->parameter(
    vpi_backends_param_, "backends", "Backends for VPI",
    "The flip computation backend", std::string("ALL"));
  result &= registrar->parameter(
    vpi_flip_mode_param_, "mode", "Mode",
    "Flip Mode: BOTH, VERTICAL, or HORIZONTAL", std::string("BOTH"));

  GXF_LOG_DEBUG("[Image Flip] Register Interfaces Finish");
  return gxf::ToResultCode(result);
}

gxf_result_t ImageFlip::start()
{
  // Set and print out backend used
  auto backend_it = kStrToVpiBackend.find(vpi_backends_param_.get());
  if (backend_it != kStrToVpiBackend.end()) {
    vpi_backends_ = backend_it->second;
    GXF_LOG_DEBUG(
      "[Image Flip] Found given backend, using backend: %s",
      vpi_backends_param_.get().c_str());
  } else {
    vpi_backends_ = VPI_BACKEND_CUDA;
    GXF_LOG_WARNING("[Image Flip] Can't find given backend, using backend CUDA");
  }

  // Set and print out mode used
  auto mode_it = kStrToVpiFlipMode.find(vpi_flip_mode_param_.get());
  if (mode_it != kStrToVpiFlipMode.end()) {
    vpi_flip_mode_ = mode_it->second;
    GXF_LOG_DEBUG("[Image Flip] Using flip mode: %s", vpi_flip_mode_param_.get().c_str());
  } else {
    GXF_LOG_ERROR("[Image Flip] Unsupported mode: %s", vpi_flip_mode_param_.get().c_str());
    return GXF_FAILURE;
  }

  // Initialize VPI stream
  vpi_flags_ = vpi_backends_;
  CHECK_VPI_STATUS(vpiStreamCreate(vpi_flags_, &vpi_stream_));

  return GXF_SUCCESS;
}

gxf_result_t ImageFlip::tick()
{
  auto maybe_input_message = image_receiver_->receive();
  if (!maybe_input_message) {
    GXF_LOG_ERROR("Failed to receive image message");
    return maybe_input_message.error();
  }

  auto maybe_input_image = maybe_input_message.value().get<gxf::VideoBuffer>();
  if (!maybe_input_image) {
    GXF_LOG_ERROR("Failed to get image from message");
    return maybe_input_image.error();
  }

  auto input_img_handle = maybe_input_image.value();
  auto input_info = input_img_handle->video_frame_info();

  gxf::Expected<gxf::Entity> output_message = gxf::Entity::New(context());
  if (!output_message) {
    GXF_LOG_ERROR("Failed to create out message");
    return output_message.error();
  }

  auto output_image = output_message->add<gxf::VideoBuffer>(output_name_.get().c_str());
  if (!output_image) {
    GXF_LOG_ERROR("Failed to add flip image");
    return output_image.error();
  }

  constexpr auto kStorageType = nvidia::gxf::MemoryStorageType::kDevice;

  const uint32_t width = input_info.width;
  const uint32_t height = input_info.height;

  auto output_img_handle = output_image.value();
  auto output_info = output_img_handle->video_frame_info();
  auto resize_result = output_img_handle->resizeCustom(
    input_info, input_img_handle->size(),
    kStorageType, pool_.get());
  if (!resize_result) {
    GXF_LOG_ERROR("[Image Flip] Failed to create output buffer");
    return resize_result.error();
  }

  const bool cache_valid = (height == prev_height_) && (width == prev_width_) &&
    (input_info.color_format == prev_color_format_);

  if (cache_valid) {
    // The image dimensions are the same, so rewrap using existing VPI images
    CHECK_VPI_STATUS(UpdateVPIImageWrapper(input_, input_data_, input_img_handle));
    CHECK_VPI_STATUS(UpdateVPIImageWrapper(flipped_, flipped_data_, output_img_handle));
  } else {
    // Recreate input VPI image
    vpiImageDestroy(input_);
    CHECK_VPI_STATUS(CreateVPIImageWrapper(input_, input_data_, vpi_flags_, input_img_handle));
    GXF_LOG_DEBUG("[Image Flip] Created input VPI Image");

    // Recreate the final image
    vpiImageDestroy(flipped_);
    CHECK_VPI_STATUS(CreateVPIImageWrapper(flipped_, flipped_data_, vpi_flags_, output_img_handle));
    GXF_LOG_DEBUG("[Image Flip] Created output VPI Image");

    // Update cached dimensions
    prev_height_ = height;
    prev_width_ = width;
  }

  GXF_LOG_DEBUG("[Image Flip] Convert flip format");
  CHECK_VPI_STATUS(
    vpiSubmitImageFlip(vpi_stream_, vpi_backends_, input_, flipped_, vpi_flip_mode_));

  // Wait for operations to complete
  CHECK_VPI_STATUS(vpiStreamSync(vpi_stream_));

  // Add timestamps
  GXF_LOG_DEBUG("[Image Flip] Add timestamp");
  std::string timestamp_name{"timestamp"};
  auto maybe_timestamp = maybe_input_message.value().get<gxf::Timestamp>(timestamp_name.c_str());
  if (!maybe_timestamp) {
    timestamp_name = std::string{""};
    maybe_timestamp = maybe_input_message.value().get<gxf::Timestamp>(timestamp_name.c_str());
  }
  if (!maybe_timestamp) {
    GXF_LOG_ERROR("[Image Flip] Failed to get timestamp");
    return maybe_timestamp.error();
  }
  auto out_timestamp = output_message.value().add<gxf::Timestamp>(timestamp_name.c_str());
  if (!out_timestamp) {
    GXF_LOG_ERROR("[Image Flip] Failed to add timestamp");
    return out_timestamp.error();
  }

  *out_timestamp.value() = *maybe_timestamp.value();

  GXF_LOG_DEBUG("[Image Flip] Publish results");
  auto maybe_published = output_transmitter_->publish(output_message.value());
  if (!maybe_published) {
    GXF_LOG_ERROR("[Image Flip] Failed to publish.");
    return maybe_published.error();
  }
  return GXF_SUCCESS;
}

gxf_result_t ImageFlip::stop()
{
  if (!vpi_stream_) {CHECK_VPI_STATUS(vpiStreamSync(vpi_stream_));}
  vpiStreamDestroy(vpi_stream_);
  vpiImageDestroy(input_);
  vpiImageDestroy(flipped_);
  return GXF_SUCCESS;
}

}  // namespace isaac_ros
}  // namespace nvidia
