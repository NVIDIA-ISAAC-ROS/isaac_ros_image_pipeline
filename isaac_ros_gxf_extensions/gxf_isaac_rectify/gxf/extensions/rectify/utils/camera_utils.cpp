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
#include "extensions/rectify/utils/camera_utils.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <string>

#include "gems/serialization/json.hpp"
#include "gems/serialization/json_formatter.hpp"
#include "gxf/multimedia/camera.hpp"
#include "gxf/std/codelet.hpp"

namespace nvidia {

namespace isaac {

namespace cameraUtils {
using Json = nlohmann::json;

bool importCameraIntrinsics(const Json& json, gxf::CameraModel& intrinsics) {
  try {
    intrinsics.dimensions.x = json["dimensions"].at("width");
    intrinsics.dimensions.y = json["dimensions"].at("height");
    intrinsics.focal_length.x = json["focal_length"].at("fx");
    intrinsics.focal_length.y = json["focal_length"].at("fy");
    intrinsics.principal_point.x = json["principal_point"].at("cx");
    intrinsics.principal_point.y = json["principal_point"].at("cy");
    intrinsics.skew_value = json.at("skew");

    std::string type = json.at("distortion_type");
    if (!type.compare("Polynomial"))
        intrinsics.distortion_type = gxf::DistortionType::Polynomial;
    else if (!type.compare("Perspective"))
        intrinsics.distortion_type = gxf::DistortionType::Perspective;
    else if (!type.compare("Equidistant"))
        intrinsics.distortion_type = gxf::DistortionType::FisheyeEquidistant;
    else if (!type.compare("Equisolid"))
        intrinsics.distortion_type = gxf::DistortionType::FisheyeEquisolid;
    else if (!type.compare("OrthoGraphic"))
        intrinsics.distortion_type = gxf::DistortionType::FisheyeOrthoGraphic;
    else if (!type.compare("Stereographic"))
        intrinsics.distortion_type = gxf::DistortionType::FisheyeStereographic;
    else if (!type.compare("Brown"))
        intrinsics.distortion_type = gxf::DistortionType::Brown;
    for (auto idx = 0; idx < intrinsics.kMaxDistortionCoefficients; idx++) {
        intrinsics.distortion_coefficients[idx] = json.at("distortion_coefficients")[idx];
    }
  } catch (Json::out_of_range& e) {
    LOG_WARNING(e.what());
    return false;
  }
  return true;
}

bool importCameraExtrinsics(const Json& json,
  gxf::Pose3D& extrinsics, bool use_rodrigez) {
  try {
    for (auto idx = 0; idx < 3; idx++) {
        extrinsics.translation[idx] = json.at("translation")[idx];
    }
    if (use_rodrigez) {
        cv::Matx33d Rlr;
        cv::Matx31d aux(json.at("rotation_rodrigez")[0],
                        json.at("rotation_rodrigez")[1],
                        json.at("rotation_rodrigez")[2]);
        cv::Rodrigues(aux, Rlr);
        for (auto j = 0; j < 9; j++) {
            extrinsics.rotation[j] = Rlr(j / 3, j % 3);
        }
    } else {
        for (auto idx = 0; idx < 9; idx++) {
            extrinsics.rotation[idx] = json.at("rotation")[idx];
        }
    }
  } catch (Json::out_of_range& e) {
    LOG_WARNING(e.what());
    return false;
  }
  return true;
}

void printCameraInfo(const gxf::CameraModel& intrinsics,
  const gxf::Pose3D& extrinsics) {
  GXF_LOG_DEBUG("intrinsics dimension: (%d, %d)", intrinsics.dimensions.x,
    intrinsics.dimensions.y);
  GXF_LOG_DEBUG("intrinsics principal_point: (%4.6f, %4.6f)", intrinsics.principal_point.x,
    intrinsics.principal_point.y);
  GXF_LOG_DEBUG("intrinsics focal_length: (%4.6f, %4.6f)", intrinsics.focal_length.x,
    intrinsics.focal_length.y);
  GXF_LOG_DEBUG("skew value: %g", intrinsics.skew_value);
  GXF_LOG_DEBUG("distortion coeffs: [%g, %g, %g, %g, %g, %g, %g, %g]",
    intrinsics.distortion_coefficients[0],
    intrinsics.distortion_coefficients[1],
    intrinsics.distortion_coefficients[2],
    intrinsics.distortion_coefficients[3],
    intrinsics.distortion_coefficients[4],
    intrinsics.distortion_coefficients[5],
    intrinsics.distortion_coefficients[6],
    intrinsics.distortion_coefficients[7]);

  GXF_LOG_DEBUG("translation matrix:[%g, %g, %g]",
    extrinsics.translation[0],
    extrinsics.translation[1],
    extrinsics.translation[2]);
  GXF_LOG_DEBUG("rotation matrix:[%g, %g, %g, %g, %g, %g, %g, %g, %g]",
    extrinsics.rotation[0],
    extrinsics.rotation[1],
    extrinsics.rotation[2],
    extrinsics.rotation[3],
    extrinsics.rotation[4],
    extrinsics.rotation[5],
    extrinsics.rotation[6],
    extrinsics.rotation[7],
    extrinsics.rotation[8]);
}

void printCameraInfo(const gxf::CameraModel& source_intrinsics,
  const gxf::CameraModel& target_intrinsics,
  const gxf::Pose3D& extrinsics) {
  GXF_LOG_DEBUG("=========SOURCE CAMERA===========");
  GXF_LOG_DEBUG("intrinsics dimension: (%d, %d)", source_intrinsics.dimensions.x,
               source_intrinsics.dimensions.y);
  GXF_LOG_DEBUG("intrinsics principal_point: (%4.6f, %4.6f)", source_intrinsics.principal_point.x,
               source_intrinsics.principal_point.y);
  GXF_LOG_DEBUG("intrinsics focal_length: (%4.6f, %4.6f)", source_intrinsics.focal_length.x,
               source_intrinsics.focal_length.y);
  GXF_LOG_DEBUG("skew value: %g", source_intrinsics.skew_value);
  GXF_LOG_DEBUG("distortion coeffs: [%g, %g, %g, %g, %g, %g, %g, %g]",
                source_intrinsics.distortion_coefficients[0],
                source_intrinsics.distortion_coefficients[1],
                source_intrinsics.distortion_coefficients[2],
                source_intrinsics.distortion_coefficients[3],
                source_intrinsics.distortion_coefficients[4],
                source_intrinsics.distortion_coefficients[5],
                source_intrinsics.distortion_coefficients[6],
                source_intrinsics.distortion_coefficients[7]);

  GXF_LOG_DEBUG("=========TARGET CAMERA===========");
  GXF_LOG_DEBUG("intrinsics dimension: (%d, %d)", target_intrinsics.dimensions.x,
               target_intrinsics.dimensions.y);
  GXF_LOG_DEBUG("intrinsics principal_point: (%4.6f, %4.6f)", target_intrinsics.principal_point.x,
               target_intrinsics.principal_point.y);
  GXF_LOG_DEBUG("intrinsics focal_length: (%4.6f, %4.6f)", target_intrinsics.focal_length.x,
               target_intrinsics.focal_length.y);
  GXF_LOG_DEBUG("skew value: %g", target_intrinsics.skew_value);
  GXF_LOG_DEBUG("distortion coeffs: [%g, %g, %g, %g, %g, %g, %g, %g]",
                target_intrinsics.distortion_coefficients[0],
                target_intrinsics.distortion_coefficients[1],
                target_intrinsics.distortion_coefficients[2],
                target_intrinsics.distortion_coefficients[3],
                target_intrinsics.distortion_coefficients[4],
                target_intrinsics.distortion_coefficients[5],
                target_intrinsics.distortion_coefficients[6],
                target_intrinsics.distortion_coefficients[7]);

  GXF_LOG_DEBUG("=========EXTRINSICS===========");
  GXF_LOG_DEBUG("translation matrix: [%g, %g, %g]",
    extrinsics.translation[0],
    extrinsics.translation[1],
    extrinsics.translation[2]);
  GXF_LOG_DEBUG("rotation matrix: [%g, %g, %g, %g, %g, %g, %g, %g, %g]",
    extrinsics.rotation[0],
    extrinsics.rotation[1],
    extrinsics.rotation[2],
    extrinsics.rotation[3],
    extrinsics.rotation[4],
    extrinsics.rotation[5],
    extrinsics.rotation[6],
    extrinsics.rotation[7],
    extrinsics.rotation[8]);
}

}  // namespace cameraUtils
}  // namespace isaac
}  // namespace nvidia
