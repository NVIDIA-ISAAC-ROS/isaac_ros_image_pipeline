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
#include "extensions/rectify/components/stereo_extrinsics_normalizer.hpp"

#include "extensions/rectify/utils/utils.hpp"
#include "gems/gxf_helpers/expected_macro_gxf.hpp"
#include "gxf/multimedia/camera.hpp"

namespace nvidia {
namespace isaac {

gxf_result_t StereoExtrinsicsNormalizer::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(
      left_camera_input_, "left_camera_input", "Left camera input",
      "Left camera input containing extrinsics from StreamUndistort module");
  result &= registrar->parameter(
      right_camera_input_, "right_camera_input", "Right camera input",
      "Right camera input containing extrinsics from StreamUndistort module");
  result &= registrar->parameter(
      left_camera_output_, "left_camera_output", "Left camera output",
      "Left camera output containing extrinsics from StreamUndistort module");
  result &= registrar->parameter(
      right_camera_output_, "right_camera_output", "Right camera output",
      "Right camera output containing extrinsics from StreamUndistort module");
  return gxf::ToResultCode(result);
}

gxf::Expected<void> StereoExtrinsicsNormalizer::postProcessTransformation(
    gxf::Entity& input_left_message, gxf::Entity& input_right_message) {
  // Modify extrinsics information for left camera
  gxf::Handle<gxf::Pose3D> left_camera_extrinsics =
      UNWRAP_OR_RETURN(input_left_message.get<gxf::Pose3D>("extrinsics"));
  // Modify extrinsics information for right camera
  gxf::Handle<gxf::Pose3D> right_camera_extrinsics =
      UNWRAP_OR_RETURN(input_right_message.get<gxf::Pose3D>("extrinsics"));
  // Modify target extrinsics information for left camera
  gxf::Handle<gxf::Pose3D> left_camera_target_extrinsics =
      UNWRAP_OR_RETURN(input_left_message.get<gxf::Pose3D>("target_extrinsics_delta"));
  // Modify target extrinsics information for right camera
  gxf::Handle<gxf::Pose3D> right_camera_target_extrinsics =
      UNWRAP_OR_RETURN(input_right_message.get<gxf::Pose3D>("target_extrinsics_delta"));

  // Do post processing of extrinsic data
  ::nvidia::isaac::Pose3d left_T_origin =
      ConvertToIsaacPose(*left_camera_extrinsics.get());
  ::nvidia::isaac::Pose3d right_T_left =
      ConvertToIsaacPose(*right_camera_extrinsics.get());
  ::nvidia::isaac::Pose3d left_rectified_T_left =
      ConvertToIsaacPose(*left_camera_target_extrinsics.get());
  ::nvidia::isaac::Pose3d right_rectified_T_right =
      ConvertToIsaacPose(*right_camera_target_extrinsics.get());
  ::nvidia::isaac::Pose3d rectified_right_T_rectified_left =
      right_rectified_T_right * right_T_left * left_rectified_T_left.inverse();

  gxf::Pose3D processed_right_extrinsics = ConvertToGxfPose(
      rectified_right_T_rectified_left);
  ::nvidia::isaac::Pose3d origin_T_rectified_left =
      (left_rectified_T_left * left_T_origin).inverse();

  gxf::Pose3D processed_left_extrinsics = ConvertToGxfPose(
      origin_T_rectified_left);

  *left_camera_extrinsics.get() = processed_left_extrinsics;
  *right_camera_extrinsics.get() = processed_right_extrinsics;

  // Publish messages to left camera output and right camera output
  return left_camera_output_->publish(input_left_message).and_then([&] {
    return right_camera_output_->publish(input_right_message);
  });
}

gxf_result_t StereoExtrinsicsNormalizer::tick() {
  // Receiving the left data
  gxf::Entity input_left_message = UNWRAP_OR_RETURN(left_camera_input_->receive());
  // Receiving the right camera data
  gxf::Entity input_right_message = UNWRAP_OR_RETURN(right_camera_input_->receive());
  return gxf::ToResultCode(postProcessTransformation(input_left_message, input_right_message));
}

}  // namespace isaac
}  // namespace nvidia
