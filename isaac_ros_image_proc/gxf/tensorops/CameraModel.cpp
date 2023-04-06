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

#include "CameraModel.hpp"

namespace nvidia {
namespace cvcore {
namespace tensor_ops {

namespace detail {

gxf::Expected<::cvcore::CameraDistortionType> GetCameraDistortionType(const std::string& type) {
  if (type == "Perspective") {
    return ::cvcore::CameraDistortionType::NONE;
  } else if (type == "Polynomial") {
    return ::cvcore::CameraDistortionType::Polynomial;
  } else if (type == "FisheyeEquidistant") {
    return ::cvcore::CameraDistortionType::FisheyeEquidistant;
  } else if (type == "FisheyeEquisolid") {
    return ::cvcore::CameraDistortionType::FisheyeEquisolid;
  } else if (type == "FisheyeOrthoGraphic") {
    return ::cvcore::CameraDistortionType::FisheyeOrthoGraphic;
  } else if (type == "FisheyeStereographic") {
    return ::cvcore::CameraDistortionType::FisheyeStereographic;
  } else {
    return gxf::Unexpected{GXF_FAILURE};
  }
}

} // namespace detail

gxf_result_t CameraModel::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;

  result &= registrar->parameter(distortion_type_, "distortion_type");
  result &= registrar->parameter(distortion_coefficients_, "distortion_coefficients");
  result &= registrar->parameter(focal_length_, "focal_length");
  result &= registrar->parameter(principle_point_, "principle_point");
  result &= registrar->parameter(skew_value_, "skew_value");

  return gxf::ToResultCode(result);
}

gxf_result_t CameraModel::initialize() {
  // Construct distortion model
  auto type = detail::GetCameraDistortionType(distortion_type_.get());
  if (!type) {
    return GXF_FAILURE;
  }
  if (distortion_coefficients_.get().size() != 8) {
    GXF_LOG_ERROR("size of distortion coefficients must be 8.");
    return GXF_FAILURE;
  }
  for (size_t i = 0; i < 8; i++) {
    distortions_.coefficients[i] = distortion_coefficients_.get()[i];
  }
  distortions_.type = type.value();

  // Construct intrinsic model
  if (focal_length_.get().size() != 2 || principle_point_.get().size() != 2) {
    GXF_LOG_ERROR("focal length and principle point must be 2-element array.");
    return GXF_FAILURE;
  }
  intrinsics_ = ::cvcore::CameraIntrinsics(focal_length_.get()[0], focal_length_.get()[1], principle_point_.get()[0],
                                           principle_point_.get()[1], skew_value_.get());

  return GXF_SUCCESS;
}

} // namespace tensor_ops
} // namespace cvcore
} // namespace nvidia
