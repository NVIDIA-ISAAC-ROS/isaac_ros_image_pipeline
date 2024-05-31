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

#include <tuple>

#include "extensions/tensorops/components/ImageUtils.hpp"
#include "extensions/tensorops/core/Image.h"
#include "gxf/std/allocator.hpp"
#include "gxf/std/tensor.hpp"

namespace nvidia {
namespace isaac {
namespace tensor_ops {
namespace detail {

gxf::Expected<std::tuple<size_t, size_t, size_t>> GetHWCIndices(
    const cvcore::tensor_ops::ImageType type);

gxf::Expected<gxf::PrimitiveType> GetPrimitiveType(
    const cvcore::tensor_ops::ImageType image_type);

gxf::Expected<ImageInfo> GetTensorInfo(gxf::Handle<gxf::Tensor> tensor,
    const cvcore::tensor_ops::ImageType type);

template<cvcore::tensor_ops::ImageType T,
    typename std::enable_if<T != cvcore::tensor_ops::ImageType::NV12 &&
    T != cvcore::tensor_ops::ImageType::NV24>::type* = nullptr>
gxf::Expected<cvcore::tensor_ops::Image<T>> WrapImageFromTensor(
    gxf::Handle<gxf::Tensor> tensor) {
  const auto info = GetTensorInfo(tensor, T);
  if (!info) {
    return gxf::Unexpected{GXF_FAILURE};
  }
  using D = typename cvcore::tensor_ops::detail::
      ChannelTypeToNative<cvcore::tensor_ops::ImageTraits<T, 3>::CT>::Type;
  auto pointer = tensor->data<D>();
  if (!pointer) {
    return gxf::Unexpected{GXF_FAILURE};
  }
  const size_t stride = tensor->stride(std::get<0>(GetHWCIndices(T).value()));
  return cvcore::tensor_ops::Image<T>(info.value().width, info.value().height,
      stride, pointer.value(), info.value().is_cpu);
}

template<cvcore::tensor_ops::ImageType T,
    typename std::enable_if<T == cvcore::tensor_ops::ImageType::NV12 ||
    T == cvcore::tensor_ops::ImageType::NV24>::type* = nullptr>
gxf::Expected<cvcore::tensor_ops::Image<T>> WrapImageFromTensor(gxf::Handle<gxf::Tensor> tensor) {
  GXF_LOG_ERROR("NV12/NV24 not supported for gxf::Tensor");
  return gxf::Unexpected{GXF_FAILURE};
}

template<cvcore::tensor_ops::ImageType T,
    typename std::enable_if<T != cvcore::tensor_ops::ImageType::NV12 &&
    T != cvcore::tensor_ops::ImageType::NV24>::type* = nullptr>
gxf_result_t AllocateTensor(gxf::Handle<gxf::Tensor> tensor, size_t width, size_t height,
      gxf::Handle<gxf::Allocator> allocator, bool is_cpu, bool allocate_pitch_linear) {
  const auto primitive_type = GetPrimitiveType(T);
  if (!primitive_type) {
    return primitive_type.error();
  }

  const auto indices = GetHWCIndices(T);
  if (!indices) {
    return GXF_FAILURE;
  }
  std::array<int32_t, gxf::Shape::kMaxRank> dims;
  dims[std::get<0>(indices.value())] = height;
  dims[std::get<1>(indices.value())] = width;
  dims[std::get<2>(indices.value())] =
      cvcore::tensor_ops::detail::ChannelToCount<cvcore::tensor_ops::ImageTraits<T, 3>::CC>();
  const gxf::Shape shape(dims, 3);

  auto result =
    tensor->reshapeCustom(shape, primitive_type.value(),
        gxf::PrimitiveTypeSize(primitive_type.value()),
        gxf::Unexpected{GXF_UNINITIALIZED_VALUE},
        is_cpu ? gxf::MemoryStorageType::kHost : gxf::MemoryStorageType::kDevice, allocator);
  if (!result) {
    GXF_LOG_ERROR("reshape tensor failed.");
    return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}

template<cvcore::tensor_ops::ImageType T,
    typename std::enable_if<T == cvcore::tensor_ops::ImageType::NV12 ||
    T == cvcore::tensor_ops::ImageType::NV24>::type* = nullptr>
gxf_result_t AllocateTensor(gxf::Handle<gxf::Tensor> tensor, size_t width, size_t height,
      gxf::Handle<gxf::Allocator> allocator, bool is_cpu, bool allocate_pitch_linear) {
  GXF_LOG_ERROR("NV12/NV24 not supported for gxf::Tensor");
  return GXF_FAILURE;
}

}  // namespace detail
}  // namespace tensor_ops
}  // namespace isaac
}  // namespace nvidia
