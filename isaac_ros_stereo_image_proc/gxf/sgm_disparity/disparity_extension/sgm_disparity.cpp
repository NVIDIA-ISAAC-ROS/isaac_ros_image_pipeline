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

#include "sgm_disparity.hpp"

#include <array>
#include <functional>
#include <iostream>

#include "gxf/std/timestamp.hpp"

#include "gxf/multimedia/video.hpp"
#include "gxf/std/tensor.hpp"

namespace nvidia
{
namespace isaac_ros
{

using VideoFormat = nvidia::gxf::VideoFormat;


const std::unordered_map<std::string, uint32_t> kStrToVpiBackend({
      {"CPU", VPI_BACKEND_CPU},
      {"CUDA", VPI_BACKEND_CUDA},
      {"XAVIER", VPI_BACKEND_XAVIER},
      {"ORIN", VPI_BACKEND_ORIN},
      {"PVA", VPI_BACKEND_PVA},
      {"ALL", VPI_BACKEND_ALL},
    });

constexpr VPIImageFormat ToVpiImageFormat(VideoFormat value)
{
  VPIImageFormat result = VPI_IMAGE_FORMAT_Y8_ER;

  switch (value) {
    case VideoFormat::GXF_VIDEO_FORMAT_GRAY32:
      result = VPI_IMAGE_FORMAT_F32;
      break;
    case VideoFormat::GXF_VIDEO_FORMAT_RGB:
      result = VPI_IMAGE_FORMAT_RGB8;
      break;
    case VideoFormat::GXF_VIDEO_FORMAT_BGR:
      result = VPI_IMAGE_FORMAT_BGR8;
      break;
    default:
      break;
  }
  return result;
}

constexpr VPIPixelType ToVpiPixelType(nvidia::gxf::VideoFormat value)
{
  VPIPixelType result = VPI_PIXEL_TYPE_U8;

  switch (value) {
    case VideoFormat::GXF_VIDEO_FORMAT_GRAY32:
      result = VPI_PIXEL_TYPE_F32;
      break;
    case VideoFormat::GXF_VIDEO_FORMAT_RGB:
      result = VPI_PIXEL_TYPE_3U8;
      break;
    case VideoFormat::GXF_VIDEO_FORMAT_BGR:
      result = VPI_PIXEL_TYPE_3U8;
      break;
    default:
      break;
  }
  return result;
}

VPIStatus CreateVPIImageWrapper(
  VPIImage & vpi_image, VPIImageData & img_data, uint64_t flags,
  const nvidia::gxf::Handle<nvidia::gxf::VideoBuffer> & video_buff)
{
  std::memset(reinterpret_cast<void *>(&img_data), 0, sizeof(VPIImageData));
  nvidia::gxf::VideoBufferInfo image_info = video_buff->video_frame_info();
  img_data.bufferType = VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;
  img_data.buffer.pitch.format = ToVpiImageFormat(image_info.color_format);
  img_data.buffer.pitch.numPlanes = 1;
  img_data.buffer.pitch.planes[0].data = video_buff->pointer();
  img_data.buffer.pitch.planes[0].height = image_info.height;
  img_data.buffer.pitch.planes[0].width = image_info.width;
  img_data.buffer.pitch.planes[0].pixelType = ToVpiPixelType(image_info.color_format);
  img_data.buffer.pitch.planes[0].pitchBytes = image_info.color_planes[0].stride;
  return vpiImageCreateWrapper(&img_data, nullptr, flags, &vpi_image);
}

VPIStatus UpdateVPIImageWrapper(
  VPIImage & image, VPIImageData & imageWrap,
  const nvidia::gxf::Handle<nvidia::gxf::VideoBuffer> & video_buff)
{
  imageWrap.buffer.pitch.planes[0].data = video_buff->pointer();
  return vpiImageSetWrapper(image, &imageWrap);
}

gxf_result_t SGMDisparity::registerInterface(gxf::Registrar * registrar)
{
  gxf::Expected<void> result;

  result &= registrar->parameter(
    min_disparity_, "min_disparity",
    "The min value of disparity search range", "");
  result &= registrar->parameter(
    max_disparity_, "max_disparity",
    "The maximum value of disparity search range, has to be positive",
    "");
  result &= registrar->parameter(
    left_image_name_, "left_image_name",
    "The name of the left image to be received", "");
  result &= registrar->parameter(
    right_image_name_, "right_image_name",
    "The name of the right image to be received", "");
  result &= registrar->parameter(
    output_name_, "output_name",
    "The name of the tensor to be passed to next node", "");
  result &= registrar->parameter(pool_, "pool", "Memory pool for allocating output data", "");
  result &= registrar->parameter(
    left_image_receiver_, "left_image_receiver",
    "Receiver to get the left image", "");
  result &= registrar->parameter(
    right_image_receiver_, "right_image_receiver",
    "Receiver to get the right image", "");
  result &= registrar->parameter(
    output_transmitter_, "output_transmitter",
    "Transmitter to send the data", "");
  result &= registrar->parameter(
    vpi_backends_string_, "backends",
    "The disparity computation backend", "");
  result &= registrar->parameter(
    debug_, "debug", "Toggle debug messages",
    "True for enabling debug messages", false);

  GXF_LOG_DEBUG("[SGM Disparity] Register Interfaces Finish");
  return gxf::ToResultCode(result);
}

gxf_result_t SGMDisparity::start()
{
  if (debug_) {
    SetSeverity(Severity::DEBUG);
  } else {
    SetSeverity(Severity::ERROR);
  }

  if (max_disparity_ <= 0) {
    GXF_LOG_ERROR(
      "Error: max disparity must be strictly positive, but received %zu",
      max_disparity_);
    return GXF_FAILURE;
  }

  // Set and print out backend used
  auto backend_it = kStrToVpiBackend.find(vpi_backends_string_.get());
  if (backend_it != kStrToVpiBackend.end()) {
    vpi_backends_ = backend_it->second;
    GXF_LOG_DEBUG(
      "[SGM Disparity] Found given backend, using backend: %s",
      vpi_backends_string_.get().c_str());
  } else {
    vpi_backends_ = VPI_BACKEND_CUDA;
    GXF_LOG_WARNING("[SGM Disparity] Can't find give backend, using backend CUDA");
  }

  CHECK_STATUS(vpiInitStereoDisparityEstimatorCreationParams(&disparity_params_));
  disparity_params_.maxDisparity = max_disparity_;

  // VPI_BACKEND_XAVIER backend only accepts downscaleFactor = 4
  if (vpi_backends_ == VPI_BACKEND_XAVIER || vpi_backends_ == VPI_BACKEND_ORIN) {
    disparity_params_.downscaleFactor = 4;
  }

  // Convert Format Parameters
  CHECK_STATUS(vpiInitConvertImageFormatParams(&stereo_input_scale_params_));
  if (vpi_backends_ == VPI_BACKEND_XAVIER || vpi_backends_ == VPI_BACKEND_ORIN) {
    stereo_input_scale_params_.scale = kTegraSupportedScale;
  }

  CHECK_STATUS(vpiInitConvertImageFormatParams(&disparity_scale_params_));
  // Scale the per-pixel disparity output using the default value.
  disparity_scale_params_.scale = 1 / 32.0;

  // Initialize VPI stream
  vpi_flags = vpi_backends_;
  if(vpi_backends_ != VPI_BACKEND_CUDA){
    vpi_flags = VPI_BACKEND_CUDA | vpi_backends_;
  }
  CHECK_STATUS(vpiStreamCreate(vpi_flags, &vpi_stream_));

  return GXF_SUCCESS;
}

gxf_result_t SGMDisparity::tick()
{
  auto maybe_input_left_message = left_image_receiver_->receive();
  if (!maybe_input_left_message) {
    GXF_LOG_ERROR("Failed to receive left image message");
    return maybe_input_left_message.error();
  }

  auto maybe_input_right_message = right_image_receiver_->receive();
  if (!maybe_input_right_message) {
    GXF_LOG_ERROR("Failed to receive right image message");
    return maybe_input_right_message.error();
  }

  auto input_left_image = maybe_input_left_message.value().get<gxf::VideoBuffer>();
  if (!input_left_image) {
    GXF_LOG_ERROR("Failed to get left image from message");
    return input_left_image.error();
  }
  auto input_left_info = input_left_image.value()->video_frame_info();

  auto input_right_image = maybe_input_right_message.value().get<gxf::VideoBuffer>();
  if (!input_right_image) {
    GXF_LOG_ERROR("Failed to get right image from message");
    return input_right_image.error();
  }
  auto input_right_info = input_right_image.value()->video_frame_info();

  gxf::Expected<gxf::Entity> output_message = gxf::Entity::New(context());
  if (!output_message) {
    GXF_LOG_ERROR("Failed to create out message");
    return output_message.error();
  }

  auto output_image = output_message->add<gxf::VideoBuffer>("disparity");
  if (!output_image) {
    GXF_LOG_ERROR("Failed to add disparity image");
    return output_image.error();
  }

  constexpr auto surface_layout = nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR;
  constexpr auto storage_type = nvidia::gxf::MemoryStorageType::kDevice;

  const int width = input_left_info.width;
  const int height = input_left_info.height;

  auto resize_result = output_image.value()->resize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY32>(
    width, height,
    surface_layout,
    storage_type,
    pool_.get());
  if (!resize_result) {
    GXF_LOG_ERROR("[SGM Disparity] Failed to create output buffer");
    return resize_result.error();
  }

  const bool cache_valid = (height == prev_height_) && (width == prev_width_);

  if (cache_valid) {
    // The image dimensions are the same, so rewrap using existing VPI images
    CHECK_STATUS(UpdateVPIImageWrapper(left_input_, left_input_data_, input_left_image.value()));
    CHECK_STATUS(UpdateVPIImageWrapper(right_input_, right_input_data_, input_right_image.value()));
    CHECK_STATUS(UpdateVPIImageWrapper(disparity_, disparity_data_, output_image.value()));
  } else {
    // Recreate left and right input VPI images
    vpiImageDestroy(left_input_);
    vpiImageDestroy(right_input_);

    CHECK_STATUS(CreateVPIImageWrapper(left_input_, left_input_data_, vpi_backends_, input_left_image.value()));
    CHECK_STATUS(CreateVPIImageWrapper(right_input_, right_input_data_, vpi_backends_, input_right_image.value()));

    if (vpi_backends_ == VPI_BACKEND_XAVIER) {
      // VPI_BACKEND_XAVIER backend only accepts 1920x1080 images and Y16 Block linear format.
      // VPI_BACKEND_ORIN backend has no limitation on input size, but the format has to be block linear.
      stereo_format_ = VPI_IMAGE_FORMAT_Y16_ER_BL;
      stereo_width_ = kTegraSupportedStereoWidth;
      stereo_height_ = kTegraSupportedStereoHeight;
    } else if (vpi_backends_ == VPI_BACKEND_ORIN) {
      stereo_format_ = VPI_IMAGE_FORMAT_Y16_ER_BL;
      stereo_width_ = width;
      stereo_height_ = height;
    } else {
      stereo_format_ = VPI_IMAGE_FORMAT_Y16_ER;
      stereo_width_ = width;
      stereo_height_ = height;
    }

    // Recreate temporaries for changing format from input to stereo format
    GXF_LOG_DEBUG("[SGM Disparity] Create left and right formatted image");
    vpiImageDestroy(left_formatted_);
    vpiImageDestroy(right_formatted_);

    // VPI_BACKEND_XAVIER backend only accepts 1920x1080 images and Y16 Block linear format.
    CHECK_STATUS(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_Y16_ER, vpi_flags, &left_formatted_));
    CHECK_STATUS(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_Y16_ER, vpi_flags, &right_formatted_));

    // Recreate left and right Tegra-specific resized VPI images
    vpiImageDestroy(left_stereo_);
    vpiImageDestroy(right_stereo_);
    CHECK_STATUS(vpiImageCreate(stereo_width_, stereo_height_, stereo_format_, vpi_backends_, &left_stereo_));
    CHECK_STATUS(vpiImageCreate(stereo_width_, stereo_height_, stereo_format_, vpi_backends_, &right_stereo_));

    // Recreate disparity VPI image with original input dimensions
    GXF_LOG_DEBUG("[SGM Disparity] Create resized disparity image");
    vpiImageDestroy(disparity_resized_);
    CHECK_STATUS(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_S16, vpi_flags, &disparity_resized_));

    // Recreate confidence map, which is used only on Tegra backend
    if (vpi_backends_ == VPI_BACKEND_XAVIER || vpi_backends_ == VPI_BACKEND_ORIN) {
      vpiImageDestroy(confidence_map_);
      CHECK_STATUS(
        vpiImageCreate(
          stereo_width_ / disparity_params_.downscaleFactor,
          stereo_height_ / disparity_params_.downscaleFactor,
          VPI_IMAGE_FORMAT_U16, vpi_backends_, &confidence_map_));
    } else {
      confidence_map_ = nullptr;
    }

    // Recreate stereo payload with parameters for stereo disparity algorithm
    vpiPayloadDestroy(stereo_payload_);
    CHECK_STATUS(
      vpiCreateStereoDisparityEstimator(
        vpi_backends_, stereo_width_, stereo_height_,
        stereo_format_, &disparity_params_, &stereo_payload_));

    // Recreate raw disparity VPI image
    vpiImageDestroy(disparity_raw_);
    CHECK_STATUS(
      vpiImageCreate(
        stereo_width_ / disparity_params_.downscaleFactor,
        stereo_height_ / disparity_params_.downscaleFactor,
        VPI_IMAGE_FORMAT_S16, vpi_flags, &disparity_raw_));

    // Recreate the final disparity
    GXF_LOG_DEBUG("[SGM Disparity] Create disparity wrapper");
    vpiImageDestroy(disparity_);
    CHECK_STATUS(CreateVPIImageWrapper(disparity_, disparity_data_, vpi_flags, output_image.value()));

    // Update cached dimensions
    prev_height_ = height;
    prev_width_ = width;
  }


  // Convert input-format images to stereo-format images
  GXF_LOG_DEBUG("[SGM Disparity] Convert input image format");
  CHECK_STATUS(
    vpiSubmitConvertImageFormat(
      vpi_stream_, VPI_BACKEND_CUDA,
      left_input_, left_formatted_, &stereo_input_scale_params_));
  CHECK_STATUS(
    vpiSubmitConvertImageFormat(
      vpi_stream_, VPI_BACKEND_CUDA,
      right_input_, right_formatted_, &stereo_input_scale_params_));

  // Resize whiling using PVA-NVENC-VIC backend, this backend only accepts 1920x1080 images.
  GXF_LOG_DEBUG("[SGM Disparity] Resize input images");
  if (vpi_backends_ == VPI_BACKEND_XAVIER || vpi_backends_ == VPI_BACKEND_ORIN) {
    CHECK_STATUS(
      vpiSubmitRescale(
        vpi_stream_, VPI_BACKEND_VIC, left_formatted_, left_stereo_,
        VPI_INTERP_LINEAR, VPI_BORDER_CLAMP, 0));
    CHECK_STATUS(
      vpiSubmitRescale(
        vpi_stream_, VPI_BACKEND_VIC, right_formatted_, right_stereo_,
        VPI_INTERP_LINEAR, VPI_BORDER_CLAMP, 0));
  } else {
    left_stereo_ = left_formatted_;
    right_stereo_ = right_formatted_;
  }

  // Calculate raw disparity and confidence map
  GXF_LOG_DEBUG("[SGM Disparity] Calculate disparity");
  CHECK_STATUS(
    vpiSubmitStereoDisparityEstimator(
      vpi_stream_, vpi_backends_,
      stereo_payload_, left_stereo_, right_stereo_,
      disparity_raw_, confidence_map_, nullptr));

  // Submit resize operation from Tegra-specific stereo dimensions back to input dimensions
  GXF_LOG_DEBUG("[SGM Disparity] Resize disparity");
  if (vpi_backends_ == VPI_BACKEND_XAVIER || vpi_backends_ == VPI_BACKEND_ORIN) {
    CHECK_STATUS(
      vpiSubmitRescale(
        vpi_stream_, VPI_BACKEND_CUDA,
        disparity_raw_, disparity_resized_,
        VPI_INTERP_LINEAR, VPI_BORDER_CLAMP, 0));
  } else {
    disparity_resized_ = disparity_raw_;
  }

  GXF_LOG_DEBUG("[SGM Disparity] Convert disparity format");
  // Convert to ROS 2 standard 32-bit float format
  CHECK_STATUS(
    vpiSubmitConvertImageFormat(
      vpi_stream_, VPI_BACKEND_CUDA,
      disparity_resized_, disparity_,
      &disparity_scale_params_));

  // Wait for operations to complete
  CHECK_STATUS(vpiStreamSync(vpi_stream_));

  // Add timestamps
  GXF_LOG_DEBUG("[SGM Disparity] Add timestamp");
  std::string timestamp_name{"timestamp"};
  auto maybe_timestamp = maybe_input_left_message.value().get<gxf::Timestamp>(timestamp_name.c_str());
  if (!maybe_timestamp) {
    timestamp_name = std::string{""};
    maybe_timestamp = maybe_input_left_message.value().get<gxf::Timestamp>(timestamp_name.c_str());
  }
  if (!maybe_timestamp) {return maybe_timestamp.error();}
  auto out_timestamp = output_message.value().add<gxf::Timestamp>(timestamp_name.c_str());
  if (!out_timestamp) {return GXF_FAILURE;}

  *out_timestamp.value() = *maybe_timestamp.value();

  // Pass the updated max_disparity to compositor
  auto gxf_max_disparity = output_message.value().add<float>("max_disparity");
  if (!gxf_max_disparity) {return GXF_FAILURE;}
  *gxf_max_disparity->get() = max_disparity_;

  GXF_LOG_DEBUG("[SGM Disparity] Publish results");
  output_transmitter_->publish(output_message.value());

  return GXF_SUCCESS;
}

gxf_result_t SGMDisparity::stop()
{

  if (!vpi_stream_) {
    vpiStreamSync(vpi_stream_);
  }
  vpiStreamDestroy(vpi_stream_);

  vpiImageDestroy(left_input_);
  vpiImageDestroy(right_input_);
  vpiImageDestroy(left_formatted_);
  vpiImageDestroy(right_formatted_);
  vpiImageDestroy(left_stereo_);
  vpiImageDestroy(right_stereo_);
  vpiPayloadDestroy(stereo_payload_);
  vpiImageDestroy(disparity_raw_);
  vpiImageDestroy(confidence_map_);
  vpiImageDestroy(disparity_resized_);
  vpiImageDestroy(disparity_);
  return GXF_SUCCESS;
}

} // namespace isaac_ros
} // namespace nvidia
