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
#include "extensions/rectify/utils/utils.hpp"

#include "gems/core/math/pose3.hpp"
#include "gxf/multimedia/camera.hpp"

namespace nvidia {
namespace isaac {

// TODO(kchahal) remove copy of ConvertToGxfPose and ConvertToIsaacPose in
// `calibration/file/utils.cpp`
gxf::Pose3D ConvertToGxfPose(const ::nvidia::isaac::Pose3d& pose) {
  gxf::Pose3D res;
  auto matrix = pose.rotation.matrix();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      res.rotation[(i * 3) + j] = matrix(i, j);
    }
  }
  res.translation = {
      static_cast<float>(pose.translation.x()), static_cast<float>(pose.translation.y()),
      static_cast<float>(pose.translation.z())};
  return res;
}

::nvidia::isaac::Pose3d ConvertToIsaacPose(const gxf::Pose3D& pose) {
  ::nvidia::isaac::Matrix3d matrix;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      matrix(i, j) = pose.rotation[(i * 3) + j];
    }
  }
  auto quaternion = ::nvidia::isaac::SO3d::FromQuaternion(::nvidia::isaac::Quaterniond(matrix));
  auto translation =
      ::nvidia::isaac::Vector3d(pose.translation[0], pose.translation[1], pose.translation[2]);
  return ::nvidia::isaac::Pose3d{quaternion, translation};
}

}  // namespace isaac
}  // namespace nvidia
