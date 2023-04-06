// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NVIDIA_CVCORE_IMAGE_UTILS_HPP
#define NVIDIA_CVCORE_IMAGE_UTILS_HPP

#include "cv/core/BBox.h"
#include "cv/core/CameraModel.h"
#include "cv/core/Image.h"
#include "cv/tensor_ops/ImageUtils.h"
#include "gxf/core/expected.hpp"
#include "gxf/multimedia/camera.hpp"

namespace nvidia {
namespace cvcore {
namespace tensor_ops {

// Description of Image
struct ImageInfo {
  ::cvcore::ImageType type;
  size_t width;
  size_t height;
  bool is_cpu;
};

// helper function to match input image type string to cvcore::ImageType
gxf::Expected<::cvcore::ImageType> GetImageTypeFromString(const std::string& type);

// Helper function to get the interpolation type
gxf::Expected<::cvcore::tensor_ops::InterpolationType> GetInterpolationType(const std::string& type);

// Helper function to get the border type
gxf::Expected<::cvcore::tensor_ops::BorderType> GetBorderType(const std::string& type);

// Helper function to get the cvcore camera distortion type
gxf::Expected<::cvcore::CameraDistortionType> GetCameraDistortionType(gxf::DistortionType type);

// Helper function to get the gxf distortion type
gxf::Expected<gxf::DistortionType> GetDistortionType(::cvcore::CameraDistortionType type);

// Helper function to get the new camera model after applying crop operation
gxf::Expected<gxf::CameraModel> GetCroppedCameraModel(const gxf::CameraModel& input, const ::cvcore::BBox& roi);

// Helper function to get the new camera model after applying scale operation
gxf::Expected<gxf::CameraModel> GetScaledCameraModel(const gxf::CameraModel& input, size_t output_width,
                                                     size_t output_height, bool keep_aspect_ratio);

} // namespace tensor_ops
} // namespace cvcore
} // namespace nvidia

#endif
