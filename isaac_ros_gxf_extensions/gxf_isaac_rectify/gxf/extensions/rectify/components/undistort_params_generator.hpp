// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#pragma once

#include <opencv2/core.hpp>
#include <memory>
#include <string>

#include "gxf/multimedia/camera.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace isaac {

// Takes an camera model as input and generate undistort parameters.
class UndistortParamsGenerator : public gxf::Codelet {
 public:
  gxf_result_t registerInterface(gxf::Registrar* registrar) override;
  gxf_result_t initialize() { return GXF_SUCCESS; }
  gxf_result_t deinitialize() { return GXF_SUCCESS; }

  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override { return GXF_SUCCESS; }

 private:
  gxf::Expected<void> computeUndistortParams(
    const gxf::CameraModel& camera_intrinsics,
    const gxf::Pose3D& camera_extrinsics);

  gxf::Parameter<gxf::Handle<gxf::Receiver>> camera_input_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> camera_output_;
  gxf::Parameter<double> alpha_;
  gxf::Parameter<bool> bypass_estimation_;

  gxf::CameraModel undistort_intrinsics_;
  gxf::Pose3D undistort_extrinsics_;
  gxf::CameraModel camera_intrinsics_;
  gxf::Pose3D camera_extrinsics_;
};

}  // namespace isaac
}  // namespace nvidia
