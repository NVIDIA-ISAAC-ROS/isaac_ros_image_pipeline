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
#pragma once

#include <memory>
#include <string>

#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace isaac {

// Takes input messages from Undistort module and applies post processing to extrinsics messages.
// The post processing outputs a GXF message under "extrinsics" (gxf::Pose3D) for both left and
// right imagers such that:
// 1. rectified_right_T_rectified_left: Position of rectified left camera with respect to the
//    rectified right camera
// 2. origin_T_rectified_left: Position of the rectified left camera with respect to
//    the stereo camera origin
class StereoExtrinsicsNormalizer : public gxf::Codelet {
 public:
  gxf_result_t registerInterface(gxf::Registrar* registrar) override;
  gxf_result_t initialize() { return GXF_SUCCESS; }
  gxf_result_t deinitialize() { return GXF_SUCCESS; }

  gxf_result_t start() override { return GXF_SUCCESS; }
  gxf_result_t tick() override;
  gxf_result_t stop() override { return GXF_SUCCESS; }

 private:
  gxf::Expected<void> postProcessTransformation(
      gxf::Entity& input_left_message, gxf::Entity& input_right_message);

  gxf::Parameter<gxf::Handle<gxf::Receiver>> left_camera_input_;
  gxf::Parameter<gxf::Handle<gxf::Receiver>> right_camera_input_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> left_camera_output_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> right_camera_output_;
};

}  // namespace isaac
}  // namespace nvidia
