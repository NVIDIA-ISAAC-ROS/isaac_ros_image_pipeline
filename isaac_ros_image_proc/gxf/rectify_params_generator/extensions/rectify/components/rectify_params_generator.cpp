// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "extensions/rectify/components/rectify_params_generator.hpp"

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <string>
#include <utility>

#include "extensions/messages/camera_message.hpp"
#include "extensions/rectify/utils/camera_utils.hpp"
#include "gems/gxf_helpers/expected_macro.hpp"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/timestamp.hpp"

namespace nvidia {
namespace isaac {
namespace {

void PrintRectifyParameters(cv::Matx33d R1, cv::Matx33d R2,
  cv::Matx34d P1, cv::Matx34d P2, cv::Matx44d Q) {
  GXF_LOG_DEBUG("left camera Rotation: \n[%g, %g, %g, \n%g, %g, %g, \n%g, %g, %g]\n",
    R1(0, 0), R1(0, 1), R1(0, 2), R1(0, 3), R1(0, 4), R1(0, 5), R1(0, 6), R1(0, 7), R1(0, 8));

  GXF_LOG_DEBUG("right camera Rotation: \n[%g, %g, %g, \n%g, %g, %g, \n%g, %g, %g]\n",
    R2(0, 0), R2(0, 1), R2(0, 2), R2(0, 3), R2(0, 4), R2(0, 5), R2(0, 6), R2(0, 7), R2(0, 8));

  GXF_LOG_DEBUG("left camera projection: \n[%g, %g, %g, %g, \n%g, %g, %g, %g,"
    "\n%g, %g, %g, %g]\n",
    P1(0, 0), P1(0, 1), P1(0, 2), P1(0, 3),
    P1(0, 4), P1(0, 5), P1(0, 6), P1(0, 7),
    P1(0, 8), P1(0, 9), P1(0, 10), P1(0, 11));

  GXF_LOG_DEBUG("right camera projection: \n[%g, %g, %g, %g, \n%g, %g, %g, %g,"
    "\n%g, %g, %g, %g]\n",
    P2(0, 0), P2(0, 1), P2(0, 2), P2(0, 3),
    P2(0, 4), P2(0, 5), P2(0, 6), P2(0, 7),
    P2(0, 8), P2(0, 9), P2(0, 10), P2(0, 11));

  GXF_LOG_DEBUG("diaparity-to-depth: \n[%g, %g, %g, %g, \n%g, %g, %g, %g,"
    "\n%g, %g, %g, %g, \n%g, %g, %g, %g]\n",
    Q(0, 0), Q(0, 1), Q(0, 2), Q(0, 3),
    Q(0, 4), Q(0, 5), Q(0, 6), Q(0, 7),
    Q(0, 8), Q(0, 9), Q(0, 10), Q(0, 11),
    Q(0, 12), Q(0, 13), Q(0, 14), Q(0, 15));
}

void InitCameraMatrixFromModel(
  const gxf::CameraModel& intrinsics,
  cv::Matx33d& camera_matrix) {
    camera_matrix = {intrinsics.focal_length.x, 0.0, intrinsics.principal_point.x, \
                     0.0, intrinsics.focal_length.y, intrinsics.principal_point.y, \
                     0.0, 0.0, 1.0};
}

void InitDistortionCoefficients(
  const gxf::CameraModel& intrinsics,
  cv::Matx<double, 1, 8>& coefficents) {
    coefficents = {
      intrinsics.distortion_coefficients[0],
      intrinsics.distortion_coefficients[1],
      intrinsics.distortion_coefficients[6],
      intrinsics.distortion_coefficients[7],
      intrinsics.distortion_coefficients[2],
      intrinsics.distortion_coefficients[3],
      intrinsics.distortion_coefficients[4],
      intrinsics.distortion_coefficients[5]
    };
}

gxf::Expected<void> CopyVideoBuffer(
    gxf::Handle<gxf::VideoBuffer> input,
    gxf::Handle<gxf::VideoBuffer> output) {
  if (!input || !output) {
    return gxf::Unexpected{GXF_ARGUMENT_NULL};
  }
  // Copy image frame using same storage type
  cudaMemcpyKind operation;
  switch (input->storage_type()) {
    case gxf::MemoryStorageType::kHost:
    case gxf::MemoryStorageType::kSystem: {
      operation = cudaMemcpyHostToHost;
    } break;
    case gxf::MemoryStorageType::kDevice: {
      operation = cudaMemcpyDeviceToDevice;
    } break;
    default:
      return gxf::Unexpected{GXF_MEMORY_INVALID_STORAGE_MODE};
  }

  // TODO(tazhang) zero-copy forwarding when GXF implements GXF-490
  const cudaError_t error = cudaMemcpy(
      output->pointer(),
      input->pointer(),
      input->size(),
      operation);
  if (error != cudaSuccess) {
    GXF_LOG_ERROR("%s", cudaGetErrorString(error));
    return gxf::Unexpected{GXF_FAILURE};
  }

  return gxf::Success;
}

}  // namespace

RectifyParamsGenerator::RectifyParamsGenerator() {}

RectifyParamsGenerator::~RectifyParamsGenerator() {}

gxf_result_t RectifyParamsGenerator::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(
      left_camera_input_, "left_camera_input", "Left Camera Model input",
      "Camera model input for left camera");
  result &= registrar->parameter(
      left_camera_output_, "left_camera_output", "Left Camera Model output",
      "Camera model output for left camera rectification");
  result &= registrar->parameter(
      right_camera_input_, "right_camera_input", "Right Camera Model input",
      "Camera model input for left camera");
  result &= registrar->parameter(
      right_camera_output_, "right_camera_output", "Right Camera Model output",
      "Camera model output for right camera rectification");
  result &= registrar->parameter(
      allocator_, "allocator", "Allocator",
      "Memory allocator");
  result &= registrar->parameter(
      alpha_, "alpha", "Alpha",
      "Free scaling parameter between 0 and 1", 0.0);
  result &= registrar->parameter(
      left_camera_model_file_, "left_camera_model_file", "File name for left camera model",
      "File that contains paramemters that defines a camera model",
      gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(
      right_camera_model_file_, "right_camera_model_file", "File name for right camera model",
      "File that contains paramemters that defines a camera model",
      gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  return gxf::ToResultCode(result);
}

gxf_result_t RectifyParamsGenerator::start() {
  use_camera_model_file_ = false;
  std::string filename[2];
  filename[0] = left_camera_model_file_.try_get().value_or("");
  if (filename[0].empty()) {
    return GXF_SUCCESS;
  }
  filename[1] = right_camera_model_file_.try_get().value_or("");
  if (filename[1].empty()) {
    return GXF_SUCCESS;
  }

  for (auto idx = 0; idx < 2; idx++) {
    const auto maybe_json = ::isaac::serialization::TryLoadJsonFromFile(filename[idx]);
    if (!maybe_json) {
      GXF_LOG_ERROR("Failed to read json from file '%s'", filename[idx].c_str());
      return GXF_ARGUMENT_NULL;
    }
    const nlohmann::json& json = *maybe_json;
    if (!cameraUtils::importCameraIntrinsics(json["model"], camera_intrinsics_[idx])) {
      GXF_LOG_ERROR("Can't read model from json");
      return GXF_ARGUMENT_NULL;
    }
    camera_intrinsics_[idx].distortion_type = gxf::DistortionType::Polynomial;
    if (!cameraUtils::importCameraExtrinsics(json["model"], camera_extrinsics_[idx], true)) {
      GXF_LOG_ERROR("Can't read camera extrinsics from json");
      return GXF_ARGUMENT_NULL;
    }
    cameraUtils::printCameraInfo(camera_intrinsics_[idx], camera_extrinsics_[idx]);
  }

  use_camera_model_file_ = true;
  return GXF_SUCCESS;
}

gxf_result_t RectifyParamsGenerator::tick() {
  // Receive messages
  CameraMessageParts left_input;
  CameraMessageParts right_input;

  if (use_camera_model_file_) {
    GXF_RETURN_IF_ERROR(left_camera_input_->receive().assign_to(left_input.entity));
    left_input.frame = GXF_UNWRAP_OR_RETURN(left_input.entity.get<gxf::VideoBuffer>());
    left_input.sequence_number =
        GXF_UNWRAP_OR_RETURN(left_input.entity.get<int64_t>("sequence_number"));
    left_input.timestamp = GXF_UNWRAP_OR_RETURN(left_input.entity.get<gxf::Timestamp>());

    GXF_RETURN_IF_ERROR(right_camera_input_->receive().assign_to(right_input.entity));
    right_input.frame = GXF_UNWRAP_OR_RETURN(right_input.entity.get<gxf::VideoBuffer>());
    right_input.sequence_number =
        GXF_UNWRAP_OR_RETURN(right_input.entity.get<int64_t>("sequence_number"));
    right_input.timestamp = GXF_UNWRAP_OR_RETURN(right_input.entity.get<gxf::Timestamp>());

    GXF_RETURN_IF_ERROR(computeRectifyParams(camera_intrinsics_[0], camera_intrinsics_[1],
                                             camera_extrinsics_[1]));

  } else {
    left_input = GXF_UNWRAP_OR_RETURN(left_camera_input_->receive().map(GetCameraMessage));
    right_input = GXF_UNWRAP_OR_RETURN(right_camera_input_->receive().map(GetCameraMessage));
    GXF_RETURN_IF_ERROR(computeRectifyParams(*left_input.intrinsics, *right_input.intrinsics,
                                             *right_input.extrinsics));
  }

  // Create left output message
  CameraMessageParts left_output = GXF_UNWRAP_OR_RETURN(CreateCameraMessage(
      context(), left_input.frame->video_frame_info(), left_input.frame->size(),
      left_input.frame->storage_type(), allocator_));

  gxf::Handle<gxf::CameraModel> left_target_camera =
      GXF_UNWRAP_OR_RETURN(left_output.entity.add<gxf::CameraModel>("target_camera"));
  gxf::Handle<gxf::Pose3D> left_target_extrinsics =
      GXF_UNWRAP_OR_RETURN(left_output.entity.add<gxf::Pose3D>("target_extrinsics_delta"));
  // We put rectified_camera_rotation_raw_camera here
  *left_target_extrinsics = rectify_extrinsics_[0];
  GXF_RETURN_IF_ERROR(CopyVideoBuffer(left_input.frame, left_output.frame));
  *left_target_camera = rectify_intrinsics_[0];
  *left_output.intrinsics = use_camera_model_file_ ? camera_intrinsics_[0] : *left_input.intrinsics;
  // This puts stereo extrinsics in
  *left_output.extrinsics = use_camera_model_file_ ? camera_extrinsics_[0] : *left_input.extrinsics;
  *left_output.sequence_number = *left_input.sequence_number;
  *left_output.timestamp = *left_input.timestamp;

  // Create right output message
  CameraMessageParts right_output = GXF_UNWRAP_OR_RETURN(CreateCameraMessage(
      context(), right_input.frame->video_frame_info(), right_input.frame->size(),
      right_input.frame->storage_type(), allocator_));
  // We put rectified_camera_rotation_raw_camera here
  gxf::Handle<gxf::Pose3D> right_target_extrinsics =
      GXF_UNWRAP_OR_RETURN(right_output.entity.add<gxf::Pose3D>("target_extrinsics_delta"));
  *right_target_extrinsics = rectify_extrinsics_[1];
  gxf::Handle<gxf::CameraModel> right_target_camera =
      GXF_UNWRAP_OR_RETURN(right_output.entity.add<gxf::CameraModel>("target_camera"));
  GXF_RETURN_IF_ERROR(CopyVideoBuffer(right_input.frame, right_output.frame));
  *right_target_camera = rectify_intrinsics_[1];
  *right_output.intrinsics = use_camera_model_file_ ? camera_intrinsics_[1] :
    *right_input.intrinsics;
  *right_output.extrinsics =
      use_camera_model_file_ ? camera_extrinsics_[1] : *right_input.extrinsics;
  *right_output.sequence_number = *right_input.sequence_number;
  *right_output.timestamp = *right_input.timestamp;

  GXF_RETURN_IF_ERROR(left_camera_output_->publish(left_output.entity,
    left_input.timestamp->acqtime));
  GXF_RETURN_IF_ERROR(right_camera_output_->publish(right_output.entity,
    right_input.timestamp->acqtime));

  return GXF_SUCCESS;
}

gxf::Expected<void> RectifyParamsGenerator::computeRectifyParams(
  const gxf::CameraModel& left_camera_intrinsics,
  const gxf::CameraModel& right_camera_intrinsics,
  const gxf::Pose3D& right_extrinsics) {
  cv::Size imagesize(left_camera_intrinsics.dimensions.x,
                     left_camera_intrinsics.dimensions.y);

  cv::Matx33d left_intrinsics;
  InitCameraMatrixFromModel(left_camera_intrinsics, left_intrinsics);
  cv::Matx<double, 1, 8> left_coefficients;
  InitDistortionCoefficients(left_camera_intrinsics, left_coefficients);

  cv::Matx33d right_intrinsics;
  InitCameraMatrixFromModel(right_camera_intrinsics, right_intrinsics);
  cv::Matx<double, 1, 8> right_coefficients;
  InitDistortionCoefficients(right_camera_intrinsics, right_coefficients);

  cv::Matx31d Tlr = { right_extrinsics.translation[0],
    right_extrinsics.translation[1],
    right_extrinsics.translation[2]};

  cv::Matx33d Rlr = { right_extrinsics.rotation[0],
    right_extrinsics.rotation[1],
    right_extrinsics.rotation[2],
    right_extrinsics.rotation[3],
    right_extrinsics.rotation[4],
    right_extrinsics.rotation[5],
    right_extrinsics.rotation[6],
    right_extrinsics.rotation[7],
    right_extrinsics.rotation[8]};

  cv::Matx33d R[2];
  cv::Matx34d P[2];
  cv::Matx44d Q;
  cv::stereoRectify(
      left_intrinsics, left_coefficients,
      right_intrinsics, right_coefficients,
      imagesize, Rlr, Tlr,
      R[0], R[1],
      P[0], P[1],
      Q, cv::CALIB_ZERO_DISPARITY, alpha_);

  PrintRectifyParameters(R[0], R[1], P[0], P[1], Q);

  for (int i = 0; i < 2; i++) {
    rectify_intrinsics_[i].dimensions = left_camera_intrinsics.dimensions;

    rectify_intrinsics_[i].distortion_type = gxf::DistortionType::Perspective;
    memset(rectify_intrinsics_[i].distortion_coefficients, 0,
        rectify_intrinsics_[i].kMaxDistortionCoefficients * sizeof(float));

    rectify_intrinsics_[i].focal_length.x = P[i](0, 0);
    rectify_intrinsics_[i].focal_length.y = P[i](0, 5);
    rectify_intrinsics_[i].principal_point.x = P[i](0, 2);
    rectify_intrinsics_[i].principal_point.y = P[i](0, 6);
    rectify_intrinsics_[i].skew_value = 0;

    rectify_extrinsics_[i].translation = {0, 0, 0};
    rectify_extrinsics_[i].rotation = {
      static_cast<float>(R[i](0, 0)),
      static_cast<float>(R[i](0, 1)),
      static_cast<float>(R[i](0, 2)),
      static_cast<float>(R[i](0, 3)),
      static_cast<float>(R[i](0, 4)),
      static_cast<float>(R[i](0, 5)),
      static_cast<float>(R[i](0, 6)),
      static_cast<float>(R[i](0, 7)),
      static_cast<float>(R[i](0, 8))};

    cameraUtils::printCameraInfo(rectify_intrinsics_[i], rectify_extrinsics_[i]);
  }

  return gxf::Success;
}

}  // namespace isaac
}  // namespace nvidia
