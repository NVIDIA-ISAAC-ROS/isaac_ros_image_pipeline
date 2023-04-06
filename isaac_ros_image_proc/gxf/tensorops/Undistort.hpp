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
#ifndef NVIDIA_CVCORE_UNDISTORT_HPP
#define NVIDIA_CVCORE_UNDISTORT_HPP

#include "CameraModel.hpp"
#include "Frame3D.hpp"
#include "TensorOperator.hpp"

namespace nvidia {
namespace cvcore {
namespace tensor_ops {

// Undistort operator.
class UndistortBase : public TensorOperator {
public:
  virtual ~UndistortBase() {}

  gxf_result_t start() override final;
  gxf_result_t stop() override final;

  gxf_result_t registerInterface(gxf::Registrar* registrar) override;

private:
  gxf::Expected<ImageInfo> doInferOutputInfo(gxf::Entity& input) override final;
  gxf_result_t doUpdateCameraMessage(gxf::Handle<gxf::CameraModel>& output,
                                     gxf::Handle<gxf::CameraModel>& input) override final;
  gxf_result_t doExecute(gxf::Entity& output, gxf::Entity& input, cudaStream_t stream, const char* output_name,
                         const char* input_name) override final;

  gxf::Parameter<gxf::Handle<CameraModel>> input_camera_model_;
  gxf::Parameter<gxf::Handle<Frame3D>> reference_frame_;
  gxf::Parameter<gxf::Handle<CameraModel>> output_camera_model_;
  gxf::Parameter<std::vector<std::int16_t>> regions_width_;
  gxf::Parameter<std::vector<std::int16_t>> regions_height_;
  gxf::Parameter<std::vector<std::int16_t>> horizontal_intervals_;
  gxf::Parameter<std::vector<std::int16_t>> vertical_intervals_;
  gxf::Parameter<std::string> interp_type_;
  gxf::Parameter<std::string> border_type_;

  ::cvcore::tensor_ops::ImageGrid image_grid_;
  ::cvcore::tensor_ops::ImageWarp image_warp_;
  gxf::Vector2u output_shape_;

  ::cvcore::CameraModel input_camera_info_;
  ::cvcore::CameraIntrinsics output_camera_intrinsics_;
};

class StreamUndistort : public UndistortBase {};

} // namespace tensor_ops
} // namespace cvcore
} // namespace nvidia

#endif
