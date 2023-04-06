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

#include "Frame3D.hpp"

namespace nvidia {
namespace cvcore {
namespace tensor_ops {

gxf_result_t Frame3D::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;

  result &= registrar->parameter(rotation_, "rotation");
  result &= registrar->parameter(translation_, "translation");

  return gxf::ToResultCode(result);
}

gxf_result_t Frame3D::initialize() {
  // Construct extrinsic model
  if (rotation_.get().size() != 9) {
    GXF_LOG_ERROR("size of rotation matrix must be 9");
    return GXF_FAILURE;
  }
  if (translation_.get().size() != 3) {
    GXF_LOG_ERROR("size of translation vector must be 3");
    return GXF_FAILURE;
  }
  float raw_matrix[3][4];
  for (size_t i = 0; i < 9; i++) {
    raw_matrix[i / 3][i % 3] = rotation_.get()[i];
  }
  for (size_t i = 0; i < 3; i++) {
    raw_matrix[i][3] = translation_.get()[i];
  }
  extrinsics_ = ::cvcore::CameraExtrinsics(raw_matrix);
  return GXF_SUCCESS;
}

} // namespace tensor_ops
} // namespace cvcore
} // namespace nvidia
