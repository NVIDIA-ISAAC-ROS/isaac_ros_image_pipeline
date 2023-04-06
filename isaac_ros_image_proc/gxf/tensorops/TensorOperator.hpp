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
#ifndef NVIDIA_CVCORE_TENSOR_OPERATOR_HPP
#define NVIDIA_CVCORE_TENSOR_OPERATOR_HPP

#include "ImageAdapter.hpp"
#include "ImageUtils.hpp"
#include "TensorStream.hpp"

#include "gxf/cuda/cuda_stream.hpp"
#include "gxf/cuda/cuda_stream_id.hpp"
#include "gxf/cuda/cuda_stream_pool.hpp"
#include "gxf/std/allocator.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/parameter_parser_std.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/transmitter.hpp"

#include "cv/core/Image.h"
#include "cv/core/Tensor.h"
#include "cv/tensor_ops/ImageUtils.h"

namespace nvidia {
namespace cvcore {
namespace tensor_ops {

// Base class for all tensor_ops operators
class TensorOperator : public gxf::Codelet {
public:
  virtual ~TensorOperator() = default;

  gxf_result_t inferOutputInfo(gxf::Entity& input);

  gxf_result_t updateCameraMessage(gxf::Handle<gxf::CameraModel>& output, gxf::Handle<gxf::CameraModel>& input);

  gxf_result_t execute(gxf::Entity& output, gxf::Entity& input, cudaStream_t stream);

  gxf_result_t start() override;

  gxf_result_t tick() override;

  virtual gxf_result_t stop() override {
    return GXF_SUCCESS;
  }

protected:
  gxf::Parameter<gxf::Handle<gxf::Receiver>> receiver_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> transmitter_;
  gxf::Parameter<gxf::Handle<gxf::Allocator>> pool_;
  gxf::Parameter<gxf::Handle<TensorStream>> stream_;
  gxf::Parameter<gxf::Handle<gxf::CudaStreamPool>> stream_pool_;
  gxf::Parameter<gxf::Handle<ImageAdapter>> input_adapter_;
  gxf::Parameter<gxf::Handle<ImageAdapter>> output_adapter_;
  gxf::Parameter<std::string> input_name_;
  gxf::Parameter<std::string> output_name_;

  // Input image info
  ImageInfo input_info_;
  // Output image info
  ImageInfo output_info_;
  // Whether to skip the operation(by default is false)
  bool no_op_ = false;

private:
  virtual gxf::Expected<ImageInfo> doInferOutputInfo(gxf::Entity& input) = 0;

  virtual gxf_result_t doUpdateCameraMessage(gxf::Handle<gxf::CameraModel>& output,
                                             gxf::Handle<gxf::CameraModel>& input) = 0;

  virtual gxf_result_t doExecute(gxf::Entity& output, gxf::Entity& input, cudaStream_t stream, const char* output_name,
                                 const char* input_name) = 0;

  gxf::Handle<gxf::CudaStream> cuda_stream_ptr_ = nullptr;
};

} // namespace tensor_ops
} // namespace cvcore
} // namespace nvidia

#endif
