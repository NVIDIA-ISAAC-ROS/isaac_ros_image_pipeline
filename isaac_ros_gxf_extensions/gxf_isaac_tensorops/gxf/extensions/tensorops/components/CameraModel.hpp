// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "extensions/tensorops/core/CameraModel.h"
#include <string>
#include <vector>
#include "gxf/core/component.hpp"
#include "gxf/core/parameter_parser_std.hpp"

namespace nvidia {
namespace isaac {
namespace tensor_ops {

// Wrapper of CameraModel compatible with CVCORE
class CameraModel : public gxf::Component {
 public:
  virtual ~CameraModel() = default;
  CameraModel()          = default;

  gxf_result_t registerInterface(gxf::Registrar* registrar) override;
  gxf_result_t initialize() override;

  cvcore::tensor_ops::CameraDistortionModel getDistortionModel() const {
    return distortions_;
  }
  cvcore::tensor_ops::CameraIntrinsics getCameraIntrinsics() const {
    return intrinsics_;
  }

 private:
  gxf::Parameter<std::string> distortion_type_;
  gxf::Parameter<std::vector<float>> distortion_coefficients_;
  gxf::Parameter<std::vector<float>> focal_length_;
  gxf::Parameter<std::vector<float>> principle_point_;
  gxf::Parameter<float> skew_value_;

  cvcore::tensor_ops::CameraDistortionModel distortions_;
  cvcore::tensor_ops::CameraIntrinsics intrinsics_;
};

}  // namespace tensor_ops
}  // namespace isaac
}  // namespace nvidia
