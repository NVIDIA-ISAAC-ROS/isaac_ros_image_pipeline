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
#include "extensions/rectify/components/undistort_params_generator.hpp"

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <string>
#include <utility>

#include "extensions/messages/camera_message.hpp"
#include "extensions/rectify/utils/camera_utils.hpp"
#include "gems/gxf_helpers/expected_macro_gxf.hpp"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/timestamp.hpp"

namespace nvidia {
namespace isaac {
namespace {
using DistortionCoefficients = cv::Matx<double, 1, 4>;
using CameraMatrix = cv::Matx33d;

gxf::Expected<CameraMatrix> InitCameraMatrixFromModel(
  const gxf::CameraModel& intrinsics) {
    CameraMatrix camera_matrix = {
      intrinsics.focal_length.x, 0.0, intrinsics.principal_point.x,
      0.0, intrinsics.focal_length.y, intrinsics.principal_point.y,
      0.0, 0.0, 1.0};
    return camera_matrix;
}

gxf::Expected<DistortionCoefficients> InitFisheyeDistortionCoefficients(
  const gxf::CameraModel& intrinsics) {
    DistortionCoefficients coefficents = {
      intrinsics.distortion_coefficients[0],
      intrinsics.distortion_coefficients[1],
      intrinsics.distortion_coefficients[2],
      intrinsics.distortion_coefficients[3]
    };
    return coefficents;
}

bool IsFisheyeCamera(const gxf::CameraModel& intrinsics) {
  switch (intrinsics.distortion_type) {
    case gxf::DistortionType::FisheyeEquidistant:
    case gxf::DistortionType::FisheyeEquisolid:
    case gxf::DistortionType::FisheyeOrthoGraphic:
    case gxf::DistortionType::FisheyeStereographic:
      return true;
    case gxf::DistortionType::Polynomial:
    case gxf::DistortionType::Perspective:
    case gxf::DistortionType::Brown:
      return false;
    default:
      GXF_LOG_ERROR("Distortion type is NOT supported!");
      return false;
  }
  return false;
}

}  // namespace

gxf_result_t UndistortParamsGenerator::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(
      camera_input_, "camera_input", "Camera Model input",
      "Camera model input for camera");
  result &= registrar->parameter(
      camera_output_, "camera_output", "Camera Model output",
      "Camera model output for camera undistortion");
  result &= registrar->parameter(
      alpha_, "alpha", "Alpha",
      "Free scaling parameter between 0 and 1", 0.0);
  result &= registrar->parameter(
      bypass_estimation_, "bypass_estimation",
      "Use same intrinsics as in input camera message",
      "Use same intrinsics as in input camera message", false);
  return gxf::ToResultCode(result);
}

gxf_result_t UndistortParamsGenerator::start() {
  return GXF_SUCCESS;
}

gxf_result_t UndistortParamsGenerator::tick() {
  gxf::Entity entity = UNWRAP_OR_RETURN(camera_input_->receive());

  CameraMessageParts input = UNWRAP_OR_RETURN(GetCameraMessage(entity));
  RETURN_IF_ERROR(computeUndistortParams(*input.intrinsics, *input.extrinsics));

  gxf::Handle<gxf::CameraModel> target_camera =
      UNWRAP_OR_RETURN(entity.add<gxf::CameraModel>("target_camera"));
  gxf::Handle<gxf::Pose3D> target_extrinsics =
      UNWRAP_OR_RETURN(entity.add<gxf::Pose3D>("target_extrinsics_delta"));

  *target_camera = undistort_intrinsics_;
  *target_extrinsics = undistort_extrinsics_;

  RETURN_IF_ERROR(camera_output_->publish(entity, input.timestamp->acqtime));

  return GXF_SUCCESS;
}

gxf::Expected<void> UndistortParamsGenerator::computeUndistortParams(
  const gxf::CameraModel& camera_intrinsics,
  const gxf::Pose3D& camera_extrinsics) {
  if (IsFisheyeCamera(camera_intrinsics) && !bypass_estimation_.get()) {
    CameraMatrix intrinsics = UNWRAP_OR_RETURN(
      InitCameraMatrixFromModel(camera_intrinsics));
    DistortionCoefficients coefficients = UNWRAP_OR_RETURN(
      InitFisheyeDistortionCoefficients(camera_intrinsics));

    cv::Matx33d R = {
      camera_extrinsics.rotation[0],
      camera_extrinsics.rotation[1],
      camera_extrinsics.rotation[2],
      camera_extrinsics.rotation[3],
      camera_extrinsics.rotation[4],
      camera_extrinsics.rotation[5],
      camera_extrinsics.rotation[6],
      camera_extrinsics.rotation[7],
      camera_extrinsics.rotation[8]};
    cv::Size imagesize(camera_intrinsics.dimensions.x,
                     camera_intrinsics.dimensions.y);
    CameraMatrix new_intrinsics;
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(intrinsics, coefficients,
        imagesize, R, new_intrinsics, alpha_);

    undistort_intrinsics_ = camera_intrinsics;
    undistort_intrinsics_.focal_length.x = new_intrinsics(0, 0);
    undistort_intrinsics_.focal_length.y = new_intrinsics(0, 4);
    undistort_intrinsics_.principal_point.x = new_intrinsics(0, 2);
    undistort_intrinsics_.principal_point.y = new_intrinsics(0, 5);
    undistort_intrinsics_.skew_value = 0;

    undistort_intrinsics_.distortion_type = gxf::DistortionType::Perspective;
    std::fill(std::begin(undistort_intrinsics_.distortion_coefficients),
        std::end(undistort_intrinsics_.distortion_coefficients), 0);
  } else {
    GXF_LOG_DEBUG("Using the source camera model for target model");
    undistort_intrinsics_ = camera_intrinsics;
  }
  undistort_extrinsics_ = camera_extrinsics;

  return gxf::Success;
}

}  // namespace isaac
}  // namespace nvidia
