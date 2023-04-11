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
#ifndef NVIDIA_CVCORE_CONVERT_COLOR_FORMAT_HPP
#define NVIDIA_CVCORE_CONVERT_COLOR_FORMAT_HPP

#include "TensorOperator.hpp"

namespace nvidia {
namespace cvcore {
namespace tensor_ops {

// ConvertColorFormat operator.
template<bool USE_TENSOR_STREAM>
class ConvertColorFormatBase : public TensorOperator {
public:
  virtual ~ConvertColorFormatBase() {}

  gxf_result_t registerInterface(gxf::Registrar* registrar) override final;

private:
  gxf::Expected<ImageInfo> doInferOutputInfo(gxf::Entity& input) override final;
  gxf_result_t doUpdateCameraMessage(gxf::Handle<gxf::CameraModel>& output,
                                     gxf::Handle<gxf::CameraModel>& input) override final;
  gxf_result_t doExecute(gxf::Entity& output, gxf::Entity& input, cudaStream_t stream, const char* output_name,
                         const char* input_name) override final;

  gxf::Parameter<std::string> output_type_;
};

class ConvertColorFormat : public ConvertColorFormatBase<false> {};
class StreamConvertColorFormat : public ConvertColorFormatBase<true> {};

} // namespace tensor_ops
} // namespace cvcore
} // namespace nvidia

#endif