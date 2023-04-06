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
#ifndef NVIDIA_CVCORE_CAMERA_MODEL_HPP
#define NVIDIA_CVCORE_CAMERA_MODEL_HPP

#include "gxf/core/component.hpp"
#include "gxf/std/parameter_parser_std.hpp"

#include "cv/core/CameraModel.h"

namespace nvidia {
namespace cvcore {
namespace tensor_ops {

// Wrapper of CameraModel compatible with CVCORE
class CameraModel : public gxf::Component {
public:
  virtual ~CameraModel() = default;
  CameraModel()          = default;

  gxf_result_t registerInterface(gxf::Registrar* registrar) override;
  gxf_result_t initialize() override;

  ::cvcore::CameraDistortionModel getDistortionModel() const {
    return distortions_;
  }
  ::cvcore::CameraIntrinsics getCameraIntrinsics() const {
    return intrinsics_;
  }

private:
  gxf::Parameter<std::string> distortion_type_;
  gxf::Parameter<std::vector<float>> distortion_coefficients_;
  gxf::Parameter<std::vector<float>> focal_length_;
  gxf::Parameter<std::vector<float>> principle_point_;
  gxf::Parameter<float> skew_value_;

  ::cvcore::CameraDistortionModel distortions_;
  ::cvcore::CameraIntrinsics intrinsics_;
};

} // namespace tensor_ops
} // namespace cvcore
} // namespace nvidia

#endif
