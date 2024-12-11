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

#include "extensions/sgm/sgm_disparity.hpp"

#include <array>
#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>

#include "extensions/messages/camera_message.hpp"
#include "extensions/tensorops/components/ImageUtils.hpp"
#include "gems/gxf_helpers/expected_macro_gxf.hpp"
#include "gems/vpi/constants.hpp"
#include "gems/vpi/image_wrapper.hpp"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/timestamp.hpp"
#include "vpi/algo/ConvertImageFormat.h"
#include "vpi/algo/Rescale.h"
#include "vpi/algo/StereoDisparity.h"
#include "vpi/CUDAInterop.h"
#include "vpi/VPI.h"

namespace nvidia {
namespace isaac {

namespace {

// When using PVA+NVENC+VIC AKA Xavier Backend, input dimensions must be 1920x1080
// https://docs.nvidia.com/vpi/group__VPI__StereoDisparityEstimator.html#ga54b192495300259dd7b94410ca86655c
constexpr int32_t kXavierStereoWidth = 1920;
constexpr int32_t kXavierStereoHeight = 1080;
constexpr int32_t kTegraSupportedScale = 256;

template <typename FAction>
gxf::Expected<void> CheckStatus(const FAction& action) {
  VPIStatus status = (action);
  if (status != VPI_SUCCESS) {
    char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];
    vpiGetLastStatusMessage(buffer, sizeof(buffer));
    GXF_LOG_ERROR("%s: %s", vpiStatusGetName(status), buffer);
    return gxf::Unexpected{GXF_FAILURE};
  }
  return gxf::Success;
}

}  // namespace

struct SGMDisparity::Impl {
  // Shared VPI stream for submitting all operations
  VPIStream stream;
  // Left and right VPI images in original input dimensions
  vpi::ImageWrapper left_input;
  vpi::ImageWrapper right_input;
  // Left and right VPI images in stereo algorithm-specific format
  VPIImage left_formatted;
  VPIImage right_formatted;
  // Left and right VPI images in stereo algorithm-specific format and size
  VPIImage left_stereo_resized;
  VPIImage right_stereo_resized;
  // Raw disparity, resized disparity, and confidence map in VPI-specific format
  VPIImage disparity_raw;
  VPIImage disparity_resized;
  VPIImage confidence_map;
  // Final disparity output in display-friendly format
  vpi::ImageWrapper disparity;
  // VPI algorithm parameters
  VPIConvertImageFormatParams stereo_input_scale_params;
  VPIStereoDisparityEstimatorCreationParams disparity_params;
  VPIStereoDisparityEstimatorParams disparity_context_params;
  VPIStereoDisparityConfidenceType disparity_confidence_type;
  VPIConvertImageFormatParams disparity_scale_params;
  // VPI stereo calculation parameters
  VPIPayload stereo_payload;
  VPIImageFormat stereo_format;
  // VPI backends
  uint64_t vpi_backends;
  uint64_t vpi_flags;
  // Cached values from previous iteration to compare against
  int32_t prev_height;
  int32_t prev_width;
};

SGMDisparity::SGMDisparity() {}

SGMDisparity::~SGMDisparity() {}

gxf_result_t SGMDisparity::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(
      left_image_receiver_, "left_image_receiver", "Left Image Receiver",
      "Receiver to get the left image");
  result &= registrar->parameter(
      right_image_receiver_, "right_image_receiver", "Right Image Receiver",
      "Receiver to get the right image");
  result &= registrar->parameter(
      output_transmitter_, "output_transmitter", "Output Transmitter",
      "Transmitter to send the data");
  result &= registrar->parameter(pool_, "pool", "Pool", "Memory pool for allocating output data");
  result &= registrar->parameter(
      backend_, "backend", "Backend", "The disparity computation backend", std::string("CUDA"));
  result &= registrar->parameter(
      max_disparity_, "max_disparity", "Max Disparity",
      "The maximum value of disparity search range, has to be positive", 256.0f);
  result &= registrar->parameter(
      confidence_threshold_, "confidence_threshold", "Confidence Threshold",
      "The confidence threshold for VPI SGM algorithm", 65023);
  result &= registrar->parameter(
      confidence_type_, "confidence_type", "Confidence Type",
      "Defines the way the confidence values are computed", 0);  // VPI_STEREO_CONFIDENCE_ABSOLUTE
  result &= registrar->parameter(
      window_size_, "window_size", "Window Size", "The window size for SGM disparity calculation",
      7);
  result &= registrar->parameter(
      num_passes_, "num_passes", "Num Passes", "The number of passes SGM takes to compute result",
      2);
  result &= registrar->parameter(
      p1_, "p1_", "P1", "Penalty on disparity changes of +/- 1 between neighbor pixels.", 8);
  result &= registrar->parameter(
      p2_, "p2_", "P2", "Penalty on disparity changes of more than 1 between neighbor pixels.",
      120);
  result &= registrar->parameter(p2_alpha_, "p2_alpha", "P2 Alpha", "Alpha for P2", 1);
  result &= registrar->parameter(quality_, "quality", "Quality", "Quality of disparity output", 1);
  return gxf::ToResultCode(result);
}

gxf_result_t SGMDisparity::initialize() {
  impl_ = MakeUniqueNoThrow<Impl>();
  return impl_ != nullptr ? GXF_SUCCESS : GXF_OUT_OF_MEMORY;
}

gxf_result_t SGMDisparity::deinitialize() {
  impl_.reset();
  return GXF_SUCCESS;
}

gxf_result_t SGMDisparity::start() {
  if (max_disparity_ <= 0.0) {
    GXF_LOG_ERROR("%s (%f) must be positive", max_disparity_.key(), max_disparity_.get());
    return GXF_FAILURE;
  }

  // Set and print out backend used
  impl_->vpi_backends = UNWRAP_OR_RETURN(vpi::StringToBackend(backend_));
  if (impl_->vpi_backends == VPI_BACKEND_CPU) {
    GXF_LOG_ERROR("CPU backend is not supported for SGM");
    return GXF_FAILURE;
  }
  GXF_LOG_DEBUG("Using backend: %s", backend_.get().c_str());

  RETURN_IF_ERROR(
      CheckStatus(vpiInitStereoDisparityEstimatorCreationParams(&impl_->disparity_params)));
  impl_->disparity_params.maxDisparity = max_disparity_;
  // VPI_BACKEND_XAVIER backend only accepts downscaleFactor = 4
  // for all other backends default value of 1 is used
  // https://docs.nvidia.com/vpi/group__VPI__StereoDisparityEstimator.html#ga54b192495300259dd7b94410ca86655c
  if (impl_->vpi_backends == vpi::VPI_BACKEND_XAVIER) {
    impl_->disparity_params.downscaleFactor = 4;
  }

  // Fill it up with the best known good confidence threshold params
  RETURN_IF_ERROR(
      CheckStatus(vpiInitStereoDisparityEstimatorParams(&impl_->disparity_context_params)));
  impl_->disparity_context_params.confidenceThreshold = confidence_threshold_;
  // https://docs.nvidia.com/vpi/StereoDisparity_8h_source.html#l00203
  // 0 = VPI_STEREO_CONFIDENCE_ABSOLUTE
  // 1 = VPI_STEREO_CONFIDENCE_RELATIVE
  impl_->disparity_context_params.confidenceType = (confidence_type_ == 0) ?
    VPI_STEREO_CONFIDENCE_ABSOLUTE : VPI_STEREO_CONFIDENCE_RELATIVE;

  impl_->disparity_context_params.windowSize = window_size_;
  impl_->disparity_context_params.numPasses = num_passes_;
  impl_->disparity_context_params.maxDisparity = max_disparity_;
  impl_->disparity_context_params.p1 = p1_;
  impl_->disparity_context_params.p2 = p2_;
  impl_->disparity_context_params.p2Alpha = p2_alpha_;
  impl_->disparity_context_params.quality = quality_;

  // Convert Format Parameters
  RETURN_IF_ERROR(
      CheckStatus(vpiInitConvertImageFormatParams(&impl_->stereo_input_scale_params)));
  if (impl_->vpi_backends == vpi::VPI_BACKEND_XAVIER ||
      impl_->vpi_backends == vpi::VPI_BACKEND_ORIN) {
    impl_->stereo_input_scale_params.scale = kTegraSupportedScale;
  }

  RETURN_IF_ERROR(CheckStatus(vpiInitConvertImageFormatParams(&impl_->disparity_scale_params)));
  // Scale the per-pixel disparity output using the default value.
  impl_->disparity_scale_params.scale = 1.0 / 32.0;

  // Initialize VPI stream
  impl_->vpi_flags = impl_->vpi_backends;
  if (impl_->vpi_backends != VPI_BACKEND_CUDA) {
    impl_->vpi_flags = VPI_BACKEND_CUDA | impl_->vpi_backends;
  }
  RETURN_IF_ERROR(CheckStatus(vpiStreamCreate(impl_->vpi_flags, &impl_->stream)));

  return GXF_SUCCESS;
}

gxf_result_t SGMDisparity::tick() {
  // Get left and right images, this codelet assumes that images are received in NV12 format on GPU
  CameraMessageParts left_image =
      UNWRAP_OR_RETURN(left_image_receiver_->receive().map(GetCameraMessage));
  CameraMessageParts right_image =
      UNWRAP_OR_RETURN(right_image_receiver_->receive().map(GetCameraMessage));

  // Get input image width and height
  const int width = left_image.frame->video_frame_info().width;
  const int height = left_image.frame->video_frame_info().height;

  if (impl_->vpi_backends == vpi::VPI_BACKEND_XAVIER &&
        (width != kXavierStereoWidth || height != kXavierStereoHeight)) {
    GXF_LOG_WARNING(
        "The input images are not %d x %d. They are %d x %d. SGM on Xavier assumes this",
        kXavierStereoWidth, kXavierStereoHeight, width, height);
    if (impl_->vpi_backends == vpi::VPI_BACKEND_ORIN ||
        impl_->vpi_backends == vpi::VPI_BACKEND_XAVIER) {
      GXF_LOG_ERROR(
          "SGM needs images in %d x %d for Xavier backend", kXavierStereoWidth,
          kXavierStereoHeight);
      return GXF_FAILURE;
    }
  }

  // Images from HAWK are in `GXF_SURFACE_LAYOUT_PITCH_LINEAR` which equate to
  // VPI_IMAGE_FORMAT_Y16_ER for CUDA and VPI_IMAGE_FORMAT_Y16_ER_BL for Tegra based devices
  const gxf::SurfaceLayout surface_layout = gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR;
  const gxf::MemoryStorageType storage_type = gxf::MemoryStorageType::kDevice;

  int disparity_width = impl_->vpi_backends == vpi::VPI_BACKEND_XAVIER ? kXavierStereoWidth
                                                                : width;
  int disparity_height = impl_->vpi_backends == vpi::VPI_BACKEND_XAVIER ? kXavierStereoHeight
                                                                 : height;
  // Create a new camera message for the disparity output.
  CameraMessageParts disparity =
      UNWRAP_OR_RETURN(CreateCameraMessage<gxf::VideoFormat::GXF_VIDEO_FORMAT_D32F>(
          context(), disparity_width, disparity_height, surface_layout, storage_type, pool_));

  const bool cache_valid = (height == impl_->prev_height) && (width == impl_->prev_width);
  if (cache_valid) {
    // The image dimensions are the same, so rewrap using existing VPI images
    RETURN_IF_ERROR(impl_->left_input.update(*left_image.frame));
    RETURN_IF_ERROR(impl_->right_input.update(*right_image.frame));
    RETURN_IF_ERROR(impl_->disparity.update(*disparity.frame));
  } else {
    uint64_t image_creation_flags = impl_->vpi_flags;
    if (impl_->vpi_backends == vpi::VPI_BACKEND_ORIN) {
      image_creation_flags |= VPI_RESTRICT_MEM_USAGE;
    }
    // Recreate left and right input VPI images
    RETURN_IF_ERROR(
        impl_->left_input.createFromVideoBuffer(*left_image.frame, image_creation_flags));
    RETURN_IF_ERROR(
        impl_->right_input.createFromVideoBuffer(*right_image.frame, image_creation_flags));
    // Recreate the final disparity
    RETURN_IF_ERROR(impl_->disparity.createFromVideoBuffer(*disparity.frame, impl_->vpi_flags));
    VPIImageFormat disparityFormat = VPI_IMAGE_FORMAT_S16;
    VPIImageFormat confidenceFormat = VPI_IMAGE_FORMAT_U16;
    VPIImageFormat stereoPayloadFormat = VPI_IMAGE_FORMAT_Y16_ER;
    if (impl_->vpi_backends == vpi::VPI_BACKEND_XAVIER ||
        impl_->vpi_backends == vpi::VPI_BACKEND_ORIN) {
      stereoPayloadFormat = VPI_IMAGE_FORMAT_Y8_ER_BL;
    }

    if (impl_->vpi_backends == vpi::VPI_BACKEND_XAVIER ||
          impl_->vpi_backends == vpi::VPI_BACKEND_ORIN) {
      // VPI_BACKEND_XAVIER and VPI_BACKEND_ORIN backend Y16 Block linear format.
      // TODO(kchahal): Not tested on Xavier
      impl_->stereo_format = VPI_IMAGE_FORMAT_Y8_ER_BL;
    } else if (impl_->vpi_flags == VPI_BACKEND_CUDA) {
      // CUDA x86 pipeline
      impl_->stereo_format = VPI_IMAGE_FORMAT_Y16_ER;
    } else {
      GXF_LOG_ERROR("SGM on CPU not supported");
      return GXF_FAILURE;
    }

    // Recreate temporaries for changing format from input to stereo format
    vpiImageDestroy(impl_->left_formatted);
    vpiImageDestroy(impl_->right_formatted);

    RETURN_IF_ERROR(CheckStatus(vpiImageCreate(
        disparity_width, disparity_height, impl_->stereo_format, impl_->vpi_flags,
        &impl_->left_formatted)));
    RETURN_IF_ERROR(CheckStatus(vpiImageCreate(
        disparity_width, disparity_height, impl_->stereo_format, impl_->vpi_flags,
        &impl_->right_formatted)));

    // VPI_BACKEND_XAVIER backend only accepts 1920 x 1080 images.
    // Create VPI images for left and right images of size 1920 x 1080
    if (impl_->vpi_backends == vpi::VPI_BACKEND_XAVIER) {
      vpiImageDestroy(impl_->left_stereo_resized);
      vpiImageDestroy(impl_->right_stereo_resized);

      RETURN_IF_ERROR(CheckStatus(vpiImageCreate(
          kXavierStereoWidth, kXavierStereoHeight, impl_->stereo_format,
          impl_->vpi_backends, &impl_->left_stereo_resized)));
      RETURN_IF_ERROR(CheckStatus(vpiImageCreate(
          kXavierStereoWidth, kXavierStereoHeight, impl_->stereo_format,
          impl_->vpi_backends, &impl_->right_stereo_resized)));
    }

    vpiImageDestroy(impl_->disparity_raw);
    RETURN_IF_ERROR(CheckStatus(vpiImageCreate(
        disparity_width, disparity_height, disparityFormat, impl_->vpi_flags,
        &impl_->disparity_raw)));

    // Use confidence map on both backends of size 1280 x 800
    vpiImageDestroy(impl_->confidence_map);
    RETURN_IF_ERROR(CheckStatus(vpiImageCreate(
        disparity_width, disparity_height, confidenceFormat, impl_->vpi_flags,
        &impl_->confidence_map)));

    // Recreate stereo payload with parameters for stereo disparity algorithm
    // for 1280 x 800
    vpiPayloadDestroy(impl_->stereo_payload);
    RETURN_IF_ERROR(CheckStatus(vpiCreateStereoDisparityEstimator(
        impl_->vpi_backends, disparity_width, disparity_height,
        stereoPayloadFormat, &impl_->disparity_params, &impl_->stereo_payload)));

    // Update cached dimensions
    impl_->prev_width = width;
    impl_->prev_height = height;
  }

  // Resize for all backends
  auto backend_for_rescale = impl_->vpi_flags;
  if (impl_->vpi_backends == vpi::VPI_BACKEND_XAVIER ||
      impl_->vpi_backends == vpi::VPI_BACKEND_ORIN) {
    backend_for_rescale = VPI_BACKEND_VIC;
  }

  // Convert input-format images to stereo-format images
  RETURN_IF_ERROR(CheckStatus(vpiSubmitConvertImageFormat(
      impl_->stream, VPI_BACKEND_CUDA, impl_->left_input.getImage(), impl_->left_formatted,
      NULL)));
  RETURN_IF_ERROR(CheckStatus(vpiSubmitConvertImageFormat(
      impl_->stream, VPI_BACKEND_CUDA, impl_->right_input.getImage(), impl_->right_formatted,
      NULL)));

  // There's possibly a bug in VPI, without this sync there's a error trying to
  // lock a container for shared access while it's already locked for exclusive access.
  RETURN_IF_ERROR(CheckStatus(vpiStreamSync(impl_->stream)));

  // VPI_BACKEND_XAVIER backend only accepts 1920 x 1080 images.
  if (impl_->vpi_backends == vpi::VPI_BACKEND_XAVIER) {
    RETURN_IF_ERROR(CheckStatus(vpiSubmitRescale(
        impl_->stream, backend_for_rescale, impl_->left_formatted, impl_->left_stereo_resized,
        VPI_INTERP_LINEAR, VPI_BORDER_CLAMP, 0)));
    RETURN_IF_ERROR(CheckStatus(vpiSubmitRescale(
        impl_->stream, backend_for_rescale, impl_->right_formatted, impl_->right_stereo_resized,
        VPI_INTERP_LINEAR, VPI_BORDER_CLAMP, 0)));
    // There's possibly a bug in VPI, without this sync there's a error trying to
    // lock a container for shared access while it's already locked for exclusive access.
    RETURN_IF_ERROR(CheckStatus(vpiStreamSync(impl_->stream)));

    // Calculate raw disparity and confidence map
    RETURN_IF_ERROR(CheckStatus(vpiSubmitStereoDisparityEstimator(
        impl_->stream, 0, impl_->stereo_payload, impl_->left_stereo_resized,
        impl_->right_stereo_resized, impl_->disparity_raw, impl_->confidence_map,
        &impl_->disparity_context_params)));
  } else {
    // Calculate raw disparity and confidence map
    RETURN_IF_ERROR(CheckStatus(vpiSubmitStereoDisparityEstimator(
        impl_->stream, 0, impl_->stereo_payload, impl_->left_formatted, impl_->right_formatted,
        impl_->disparity_raw, impl_->confidence_map, &impl_->disparity_context_params)));
  }

  // Convert to ROS2 standard 32-bit float format
  RETURN_IF_ERROR(CheckStatus(vpiSubmitConvertImageFormat(
      impl_->stream, VPI_BACKEND_CUDA, impl_->disparity_raw, impl_->disparity.getImage(),
      &impl_->disparity_scale_params)));

  // Wait for operations to complete
  RETURN_IF_ERROR(CheckStatus(vpiStreamSync(impl_->stream)));

  // Pass the updated max_disparity to compositor
  gxf::Handle<float> max_disparity =
      UNWRAP_OR_RETURN(disparity.entity.add<float>("max_disparity"));
  *max_disparity = max_disparity_;

  // Pass through from right because left is zero
  *disparity.extrinsics = *right_image.extrinsics;

  if (impl_->vpi_backends == vpi::VPI_BACKEND_XAVIER) {
    // For Xavier as we rescale the disparity output to 1920 x 1080, we need to rescale the focal
    // length as well.
    gxf::Expected<gxf::CameraModel> maybe_scaled_model = tensor_ops::GetScaledCameraModel(
        *left_image.intrinsics, disparity_width, disparity_height, false);
    if (!maybe_scaled_model) {
      GXF_LOG_ERROR("Intrinsics for disparity output could not be scaled");
      return GXF_FAILURE;
    }
    *disparity.intrinsics = maybe_scaled_model.value();
  } else {
    *disparity.intrinsics = *left_image.intrinsics;
  }

  // Forward other components as is
  *disparity.sequence_number = *left_image.sequence_number;
  *disparity.timestamp = *left_image.timestamp;

  return gxf::ToResultCode(output_transmitter_->publish(disparity.entity));
}

gxf_result_t SGMDisparity::stop() {
  if (!impl_->stream) {
    CheckStatus(vpiStreamSync(impl_->stream));
  }
  vpiStreamDestroy(impl_->stream);

  impl_->left_input.release();
  impl_->right_input.release();

  vpiImageDestroy(impl_->left_formatted);
  vpiImageDestroy(impl_->right_formatted);
  if (impl_->vpi_backends == vpi::VPI_BACKEND_XAVIER) {
    vpiImageDestroy(impl_->left_stereo_resized);
    vpiImageDestroy(impl_->right_stereo_resized);
  }
  vpiPayloadDestroy(impl_->stereo_payload);
  vpiImageDestroy(impl_->disparity_raw);
  vpiImageDestroy(impl_->confidence_map);
  vpiImageDestroy(impl_->disparity_resized);

  impl_->disparity.release();
  return GXF_SUCCESS;
}

}  // namespace isaac
}  // namespace nvidia
