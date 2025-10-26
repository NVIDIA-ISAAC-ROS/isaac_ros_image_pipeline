// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string>
#include "extensions/tensorops/components/TensorOperator.hpp"

namespace nvidia {
namespace isaac {
namespace tensor_ops {

// Resizing operator.
template<bool USE_TENSOR_STREAM>
class ResizeBase : public TensorOperator {
 public:
  virtual ~ResizeBase() {}

  gxf_result_t registerInterface(gxf::Registrar* registrar) override;

 private:
  gxf::Expected<ImageInfo> doInferOutputInfo(gxf::Entity& input) override;
  gxf_result_t doUpdateCameraMessage(gxf::Handle<gxf::CameraModel>& output,
      gxf::Handle<gxf::CameraModel>& input) override;
  gxf_result_t doForwardTargetCamera(gxf::Expected<nvidia::gxf::Entity> input_message,
                                     gxf::Expected<nvidia::gxf::Entity> output_message) override;
  gxf_result_t doExecute(gxf::Entity& output, gxf::Entity& input,
      cudaStream_t stream, const char* output_name,
      const char* input_name) override;

  gxf::Parameter<size_t> output_width_;
  gxf::Parameter<size_t> output_height_;
  gxf::Parameter<std::string> interp_type_;
  gxf::Parameter<std::string> border_type_;
  gxf::Parameter<bool> keep_aspect_ratio_;
};

class Resize : public ResizeBase<false> {};
class StreamResize : public ResizeBase<true> {};

}  // namespace tensor_ops
}  // namespace isaac
}  // namespace nvidia
