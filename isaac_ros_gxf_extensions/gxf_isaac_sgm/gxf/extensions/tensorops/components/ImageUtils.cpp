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
#include "ImageUtils.hpp"

#include <algorithm>
#include <iterator>
#include <string>

namespace nvidia {
namespace isaac {
namespace tensor_ops {

// helper function to match input image type string to cvcore::ImageType
gxf::Expected<cvcore::tensor_ops::ImageType> GetImageTypeFromString(const std::string& type) {
  if (type == "Y_U8") {
    return cvcore::tensor_ops::ImageType::Y_U8;
  } else if (type == "Y_U16") {
    return cvcore::tensor_ops::ImageType::Y_U16;
  } else if (type == "Y_F32") {
    return cvcore::tensor_ops::ImageType::Y_F32;
  } else if (type == "RGB_U8") {
    return cvcore::tensor_ops::ImageType::RGB_U8;
  } else if (type == "RGB_U16") {
    return cvcore::tensor_ops::ImageType::RGB_U16;
  } else if (type == "RGB_F32") {
    return cvcore::tensor_ops::ImageType::RGB_F32;
  } else if (type == "BGR_U8") {
    return cvcore::tensor_ops::ImageType::BGR_U8;
  } else if (type == "BGR_U16") {
    return cvcore::tensor_ops::ImageType::BGR_U16;
  } else if (type == "BGR_F32") {
    return cvcore::tensor_ops::ImageType::BGR_F32;
  } else if (type == "RGBA_U8") {
    return cvcore::tensor_ops::ImageType::RGBA_U8;
  } else if (type == "RGBA_U16") {
    return cvcore::tensor_ops::ImageType::RGBA_U16;
  } else if (type == "RGBA_F16") {
    return cvcore::tensor_ops::ImageType::RGBA_F16;
  } else if (type == "RGBA_F32") {
    return cvcore::tensor_ops::ImageType::RGBA_F32;
  } else if (type == "BGRA_U8") {
    return cvcore::tensor_ops::ImageType::BGRA_U8;
  } else if (type == "BGRA_U16") {
    return cvcore::tensor_ops::ImageType::BGRA_U16;
  } else if (type == "BGRA_F16") {
    return cvcore::tensor_ops::ImageType::BGRA_F16;
  } else if (type == "BGRA_F32") {
    return cvcore::tensor_ops::ImageType::BGRA_F32;
  } else if (type == "PLANAR_RGB_U8") {
    return cvcore::tensor_ops::ImageType::PLANAR_RGB_U8;
  } else if (type == "PLANAR_RGB_U16") {
    return cvcore::tensor_ops::ImageType::PLANAR_RGB_U16;
  } else if (type == "PLANAR_RGB_F32") {
    return cvcore::tensor_ops::ImageType::PLANAR_RGB_F32;
  } else if (type == "PLANAR_BGR_U8") {
    return cvcore::tensor_ops::ImageType::PLANAR_BGR_U8;
  } else if (type == "PLANAR_BGR_U16") {
    return cvcore::tensor_ops::ImageType::PLANAR_BGR_U16;
  } else if (type == "PLANAR_BGR_F32") {
    return cvcore::tensor_ops::ImageType::PLANAR_BGR_F32;
  } else if (type == "NV12") {
    return cvcore::tensor_ops::ImageType::NV12;
  } else if (type == "NV24") {
    return cvcore::tensor_ops::ImageType::NV24;
  } else {
    GXF_LOG_ERROR("invalid image type.");
    return gxf::Unexpected{GXF_FAILURE};
  }
}

gxf::Expected<cvcore::tensor_ops::InterpolationType> GetInterpolationType(
    const std::string& type) {
  if (type == "nearest") {
    return cvcore::tensor_ops::InterpolationType::INTERP_NEAREST;
  } else if (type == "linear") {
    return cvcore::tensor_ops::InterpolationType::INTERP_LINEAR;
  } else if (type == "cubic_bspline") {
    return cvcore::tensor_ops::InterpolationType::INTERP_CUBIC_BSPLINE;
  } else if (type == "cubic_catmullrom") {
    return cvcore::tensor_ops::InterpolationType::INTERP_CUBIC_CATMULLROM;
  } else {
    GXF_LOG_ERROR("invalid interpolation type.");
    return gxf::Unexpected{GXF_FAILURE};
  }
}

gxf::Expected<cvcore::tensor_ops::BorderType> GetBorderType(const std::string& type) {
  if (type == "zero") {
    return cvcore::tensor_ops::BorderType::BORDER_ZERO;
  } else if (type == "repeat") {
    return cvcore::tensor_ops::BorderType::BORDER_REPEAT;
  } else if (type == "reverse") {
    return cvcore::tensor_ops::BorderType::BORDER_REVERSE;
  } else if (type == "mirror") {
    return cvcore::tensor_ops::BorderType::BORDER_MIRROR;
  } else {
    GXF_LOG_ERROR("invalid border type.");
    return gxf::Unexpected{GXF_FAILURE};
  }
}

gxf::Expected<cvcore::tensor_ops::CameraDistortionType> GetCameraDistortionType(
    gxf::DistortionType type) {
  switch (type) {
  case gxf::DistortionType::Perspective:
    return cvcore::tensor_ops::CameraDistortionType::NONE;
  case gxf::DistortionType::Brown:
  case gxf::DistortionType::Polynomial:
    return cvcore::tensor_ops::CameraDistortionType::Polynomial;
  case gxf::DistortionType::FisheyeEquidistant:
    return cvcore::tensor_ops::CameraDistortionType::FisheyeEquidistant;
  case gxf::DistortionType::FisheyeEquisolid:
    return cvcore::tensor_ops::CameraDistortionType::FisheyeEquisolid;
  case gxf::DistortionType::FisheyeOrthoGraphic:
    return cvcore::tensor_ops::CameraDistortionType::FisheyeOrthoGraphic;
  case gxf::DistortionType::FisheyeStereographic:
    return cvcore::tensor_ops::CameraDistortionType::FisheyeStereographic;
  default:
    GXF_LOG_ERROR("invalid distortion type.");
    return gxf::Unexpected{GXF_FAILURE};
  }
}

gxf::Expected<gxf::DistortionType> GetDistortionType(
    cvcore::tensor_ops::CameraDistortionType type) {
  switch (type) {
  case cvcore::tensor_ops::CameraDistortionType::Polynomial:
    return gxf::DistortionType::Polynomial;
  case cvcore::tensor_ops::CameraDistortionType::FisheyeEquidistant:
    return gxf::DistortionType::FisheyeEquidistant;
  case cvcore::tensor_ops::CameraDistortionType::FisheyeEquisolid:
    return gxf::DistortionType::FisheyeEquisolid;
  case cvcore::tensor_ops::CameraDistortionType::FisheyeOrthoGraphic:
    return gxf::DistortionType::FisheyeOrthoGraphic;
  case cvcore::tensor_ops::CameraDistortionType::FisheyeStereographic:
    return gxf::DistortionType::FisheyeStereographic;
  default:
    GXF_LOG_ERROR("invalid distortion type.");
    return gxf::Unexpected{GXF_FAILURE};
  }
}

gxf::Expected<gxf::CameraModel> GetCroppedCameraModel(
    const gxf::CameraModel& input,
    const cvcore::tensor_ops::BBox& roi) {
  if (!roi.isValid()) {
    return gxf::Unexpected{GXF_FAILURE};
  }
  gxf::CameraModel camera;
  const size_t output_width  = roi.xmax - roi.xmin;
  const size_t output_height = roi.ymax - roi.ymin;
  camera.dimensions          = {static_cast<uint32_t>(output_width),
                                static_cast<uint32_t>(output_height)};
  camera.focal_length        = input.focal_length;
  // We will keep the relative principal point location unchanged for cropping;
  camera.principal_point = {input.principal_point.x / input.dimensions.x * output_width,
                            input.principal_point.y / input.dimensions.y * output_height},
  camera.skew_value      = input.skew_value;
  camera.distortion_type = input.distortion_type;
  std::copy(std::begin(input.distortion_coefficients), std::end(input.distortion_coefficients),
            std::begin(camera.distortion_coefficients));
  return camera;
}

gxf::Expected<gxf::CameraModel> GetScaledCameraModel(
    const gxf::CameraModel& input, size_t output_width,
     size_t output_height, bool keep_aspect_ratio) {
  gxf::CameraModel camera;
  const float scaler_x   = static_cast<float>(output_width) / input.dimensions.x;
  const float scaler_y   = static_cast<float>(output_height) / input.dimensions.y;
  const float min_scaler = std::min(scaler_x, scaler_y);
  camera.dimensions      = {static_cast<uint32_t>(output_width),
                            static_cast<uint32_t>(output_height)};
  camera.focal_length    = keep_aspect_ratio
      ? nvidia::gxf::Vector2f{min_scaler * input.focal_length.x, min_scaler * input.focal_length.y}
      : nvidia::gxf::Vector2f{scaler_x * input.focal_length.x, scaler_y * input.focal_length.y};
  const float pixel_center = 0.5f;
  camera.principal_point = {scaler_x * (input.principal_point.x + pixel_center) - pixel_center,
                            scaler_y * (input.principal_point.y + pixel_center) - pixel_center},
  camera.skew_value      = input.skew_value;
  camera.distortion_type = input.distortion_type;
  std::copy(std::begin(input.distortion_coefficients), std::end(input.distortion_coefficients),
            std::begin(camera.distortion_coefficients));
  return camera;
}

}  // namespace tensor_ops
}  // namespace isaac
}  // namespace nvidia
