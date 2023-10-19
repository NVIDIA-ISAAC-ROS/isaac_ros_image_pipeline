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
#pragma once

#include "engine/core/math/types.hpp"
#include "engine/core/tensor/tensor.hpp"
#include "gxf/core/expected.hpp"
#include "gxf/std/tensor.hpp"

namespace nvidia {
namespace isaac {

// Creates an Isaac tensor view of a GXF host tensor
template <typename K, int N>
gxf::Expected<::nvidia::isaac::CpuTensorView<K, N>> ToIsaacCpuTensorView(gxf::Tensor& tensor) {
  if ((tensor.storage_type() != gxf::MemoryStorageType::kHost) &&
      (tensor.storage_type() != gxf::MemoryStorageType::kSystem)) {
    GXF_LOG_ERROR("Tensor does not have host or system storage type");
    return gxf::Unexpected{GXF_INVALID_DATA_FORMAT};
  }
  if (tensor.rank() != N) {
    GXF_LOG_ERROR("Tensor rank mismatch. Expected: %d, actual: %d", N, tensor.rank());
    return gxf::Unexpected{GXF_INVALID_DATA_FORMAT};
  }
  ::nvidia::isaac::Vector<int, N> dimensions;
  for (int i = 0; i < N; i++) {
    dimensions[i] = tensor.shape().dimension(i);
  }

  auto maybe_ptr = tensor.data<K>();
  if (!maybe_ptr) {
    GXF_LOG_ERROR("Requested type for tensor pointer does not match tensor data type");
    return gxf::ForwardError(maybe_ptr);
  }

  return ::nvidia::isaac::CreateCpuTensorViewFromData<K, N>(maybe_ptr.value(), dimensions.prod(),
      dimensions);
}

// Creates an Isaac tensor const view of a GXF host tensor
template <typename K, int N>
gxf::Expected<::nvidia::isaac::CpuTensorConstView<K, N>> ToIsaacCpuTensorView(const gxf::Tensor& tensor) {
  if ((tensor.storage_type() != gxf::MemoryStorageType::kHost) &&
      (tensor.storage_type() != gxf::MemoryStorageType::kSystem)) {
    GXF_LOG_ERROR("Tensor does not have host or system storage type");
    return gxf::Unexpected{GXF_INVALID_DATA_FORMAT};
  }
  if (tensor.rank() != N) {
    GXF_LOG_ERROR("Tensor does not have expected rank %i", N);
    return gxf::Unexpected{GXF_INVALID_DATA_FORMAT};
  }
  ::nvidia::isaac::Vector<int, N> dimensions;
  for (int i = 0; i < N; i++) {
    dimensions[i] = tensor.shape().dimension(i);
  }

  auto maybe_ptr = tensor.data<K>();
  if (!maybe_ptr) {
    GXF_LOG_ERROR("Requested type for tensor pointer does not match tensor data type");
    return gxf::ForwardError(maybe_ptr);
  }

  return ::nvidia::isaac::CreateCpuTensorConstViewFromData, N> (maybe_ptr.value(), dimensions.prod(),
                                                      dimensions);
}

// Creates an Isaac Cuda tensor view of a GXF Cuda tensor
template <typename K, int N>
gxf::Expected<::nvidia::isaac::GpuTensorView<K, N>> ToIsaacGpuTensorView(gxf::Tensor& tensor) {
  if (tensor.storage_type() != gxf::MemoryStorageType::kDevice) {
    GXF_LOG_ERROR("Tensor does not have device storage type");
    return gxf::Unexpected{GXF_INVALID_DATA_FORMAT};
  }
  if (tensor.rank() != N) {
    GXF_LOG_ERROR("Tensor does not have expected rank %i", N);
    return gxf::Unexpected{GXF_INVALID_DATA_FORMAT};
  }
  ::nvidia::isaac::Vector<int, N> dimensions;
  for (int i = 0; i < N; i++) {
    dimensions[i] = tensor.shape().dimension(i);
  }

  auto maybe_ptr = tensor.data<K>();
  if (!maybe_ptr) {
    GXF_LOG_ERROR("Requested type for tensor pointer does not match tensor data type");
    return gxf::ForwardError(maybe_ptr);
  }

  return ::nvidia::isaac::CreateGpuTensorViewFromData<K, N>(maybe_ptr.value(), dimensions.prod(),
                                                     dimensions);
}

// Creates an Isaac Cuda tensor const view of a GXF Cuda tensor
template <typename K, int N>
gxf::Expected<::nvidia::isaac::GpuTensorConstView<K, N>> ToIsaacGpuTensorView(const gxf::Tensor& tensor) {
  if (tensor.storage_type() != gxf::MemoryStorageType::kDevice) {
    GXF_LOG_ERROR("Tensor does not have device storage type");
    return gxf::Unexpected{GXF_INVALID_DATA_FORMAT};
  }
  if (tensor.rank() != N) {
    GXF_LOG_ERROR("Tensor does not have expected rank %i", N);
    return gxf::Unexpected{GXF_INVALID_DATA_FORMAT};
  }
  ::nvidia::isaac::Vector<int, N> dimensions;
  for (int i = 0; i < N; i++) {
    dimensions[i] = tensor.shape().dimension(i);
  }

  auto maybe_ptr = tensor.data<K>();
  if (!maybe_ptr) {
    GXF_LOG_ERROR("Requested type for tensor pointer does not match tensor data type");
    return gxf::ForwardError(maybe_ptr);
  }

  return ::nvidia::isaac::CreateGpuTensorConstViewFromData<K, N>(maybe_ptr.value(), dimensions.prod(),
                                                          dimensions);
}

}  // namespace isaac
}  // namespace nvidia
