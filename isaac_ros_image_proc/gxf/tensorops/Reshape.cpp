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
#include "Reshape.hpp"

namespace nvidia {
namespace cvcore {
namespace tensor_ops {

gxf_result_t Reshape::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;

  result &= registrar->parameter(output_shape_, "output_shape");
  result &= registrar->parameter(receiver_, "receiver");
  result &= registrar->parameter(transmitter_, "transmitter");
  result &= registrar->parameter(pool_, "pool");
  result &= registrar->parameter(input_adapter_, "input_adapter");
  result &= registrar->parameter(output_adapter_, "output_adapter");
  result &= registrar->parameter(input_name_, "input_name", "input name", "input tensor name",
                                 gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(output_name_, "output_name", "output name", "output tensor name",
                                 gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);

  return gxf::ToResultCode(result);
}

gxf_result_t Reshape::doUpdateCameraMessage(gxf::Handle<gxf::CameraModel>& output,
                                            gxf::Handle<gxf::CameraModel>& input) {
  *output = *input;
  return GXF_SUCCESS;
}

gxf_result_t Reshape::doExecute(gxf::Entity& output, gxf::Entity& input, cudaStream_t stream, const char* output_name,
                                const char* input_name) {
  GXF_LOG_INFO("execute reshape.");

  auto input_tensor = input.get<gxf::Tensor>(input_name);
  if (!input_tensor) {
    GXF_LOG_ERROR("input message does not contain Tensor");
    return input_tensor.error();
  }

  auto output_tensor = output.add<gxf::Tensor>(output_name);
  if (!output_tensor) {
    GXF_LOG_ERROR("unable to add output Tensor");
    return output_tensor.error();
  }

  const auto& input_shape                        = input_tensor.value()->shape();
  const std::vector<int32_t>& output_shape_arr   = output_shape_;
  std::array<int32_t, gxf::Shape::kMaxRank> dims = {};
  std::copy(output_shape_arr.begin(), output_shape_arr.end(), dims.begin());
  const auto output_shape = gxf::Shape(dims, output_shape_arr.size());

  if (output_shape.size() != input_shape.size()) {
    GXF_LOG_ERROR("reshape size mismatch.");
    return GXF_FAILURE;
  }

  auto result = output_tensor.value()->reshapeCustom(
    output_shape, input_tensor.value()->element_type(), gxf::PrimitiveTypeSize(input_tensor.value()->element_type()),
    gxf::Unexpected{GXF_UNINITIALIZED_VALUE}, input_tensor.value()->storage_type(), pool_.get());

  if (!result) {
    GXF_LOG_ERROR("reshape tensor failed.");
    return result.error();
  }

  // Simply copy the memory
  if (input_tensor.value()->storage_type() == gxf::MemoryStorageType::kDevice) {
    cudaError_t error = cudaMemcpyAsync(output_tensor.value()->pointer(), input_tensor.value()->pointer(),
                                        input_tensor.value()->size(), cudaMemcpyDeviceToDevice, stream);
    if (error != cudaSuccess) {
      GXF_LOG_ERROR("cudaMemcpyAsync returned error code");
      return GXF_FAILURE;
    }
  } else {
    memcpy(output_tensor.value()->pointer(), input_tensor.value()->pointer(), input_tensor.value()->size());
  }
  return GXF_SUCCESS;
}

} // namespace tensor_ops
} // namespace cvcore
} // namespace nvidia
