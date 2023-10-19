// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "gems/gxf_helpers/gxf_to_stride_pointer_3d.hpp"

#include <array>

namespace nvidia {
namespace isaac {

namespace {

gxf::Expected<std::array<size_t, 3>> GetTensorStrides(
    const gxf::Tensor& tensor, ::nvidia::isaac::StridePointer3D_MemoryLayout memory_layout) {
  switch (memory_layout) {
    case ::nvidia::isaac::StridePointer3D_MemoryLayout::kHWC:
      return std::array<size_t, 3>{tensor.stride(0), tensor.stride(1), tensor.stride(2)};
    case ::nvidia::isaac::StridePointer3D_MemoryLayout::kNCHW: {
      if (tensor.shape().dimension(0) > 1) {
        GXF_LOG_ERROR("Failed to get Tensor strides: Received batch size dimension greater than 1");
        return gxf::Unexpected{GXF_FAILURE};
      }
      return std::array<size_t, 3>{tensor.stride(1), tensor.stride(2), tensor.stride(3)};
    } break;
    case ::nvidia::isaac::StridePointer3D_MemoryLayout::kNHWC: {
      if (tensor.shape().dimension(0) > 1) {
        GXF_LOG_ERROR("Failed to get Tensor strides: recieved batch size dimension greater than 1");
        return gxf::Unexpected{GXF_FAILURE};
      }
      return std::array<size_t, 3>{tensor.stride(1), tensor.stride(2), tensor.stride(3)};
    } break;
    default:
      return gxf::Unexpected{GXF_FAILURE};
  }
}

}  // namespace

gxf::Expected<::nvidia::isaac::StridePointer3D_Shape> GetTensorStridePointer3DShape(
    const gxf::Tensor& tensor, ::nvidia::isaac::StridePointer3D_MemoryLayout memory_layout) {
  ::nvidia::isaac::StridePointer3D_Shape shape;
  switch (memory_layout) {
    case ::nvidia::isaac::StridePointer3D_MemoryLayout::kHWC: {
      shape.height = tensor.shape().dimension(0);
      shape.width = tensor.shape().dimension(1);
      shape.channels = tensor.shape().dimension(2);
      return shape;
    } break;
    case ::nvidia::isaac::StridePointer3D_MemoryLayout::kNCHW: {
      if (tensor.shape().dimension(0) > 1) {
        GXF_LOG_ERROR("Failed to get Tensor shape: received batch size dimension greater than 1");
        return gxf::Unexpected{GXF_FAILURE};
      }
      shape.channels = tensor.shape().dimension(1);
      shape.height = tensor.shape().dimension(2);
      shape.width = tensor.shape().dimension(3);
      return shape;
    } break;
    case ::nvidia::isaac::StridePointer3D_MemoryLayout::kNHWC: {
      if (tensor.shape().dimension(0) > 1) {
        GXF_LOG_ERROR("Failed to get Tensor shape: received batch size dimension greater than 1");
        return gxf::Unexpected{GXF_FAILURE};
      }
      shape.height = tensor.shape().dimension(1);
      shape.width = tensor.shape().dimension(2);
      shape.channels = tensor.shape().dimension(3);
      return shape;
    } break;
    default:
      return gxf::Unexpected{GXF_FAILURE};
  }
}

gxf::Expected<::nvidia::isaac::StridePointer3D_Shape>
GetVideoBufferStridePointer3DShape(
    const gxf::VideoBuffer& video_buffer,
    ::nvidia::isaac::StridePointer3D_MemoryLayout memory_layout) {
  // TODO(kchinniah): support more memory layouts later
  if (memory_layout != ::nvidia::isaac::StridePointer3D_MemoryLayout::kHWC) {
    GXF_LOG_ERROR("Failed to get VideoBuffer shape: received unexpected memory layout!");
    return gxf::Unexpected{GXF_FAILURE};
  }

  ::nvidia::isaac::StridePointer3D_Shape shape{};
  shape.width = video_buffer.video_frame_info().width;
  shape.height = video_buffer.video_frame_info().height;
  switch (video_buffer.video_frame_info().color_format) {
    case gxf::VideoFormat::GXF_VIDEO_FORMAT_BGR:
    case gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB: {
      shape.channels = 3;
    } break;
    case gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY: {
      shape.channels = 1;
    } break;
    default:
      GXF_LOG_ERROR("Failed to get VideoBuffer shape: received unsupported color format!");
      return gxf::Unexpected{GXF_FAILURE};
  }
  return shape;
}

template <typename T>
gxf::Expected<::nvidia::isaac::StridePointer3D<T>> GXFTensorToStridePointer3D(
    gxf::Tensor& tensor, ::nvidia::isaac::StridePointer3D_MemoryLayout memory_layout) {
  ::nvidia::isaac::StridePointer3D_Shape shape;
  std::array<size_t, 3> byte_strides;
  T* tensor_data{nullptr};
  return GetTensorStridePointer3DShape(tensor, memory_layout)
      .assign_to(shape)
      .and_then([&]() { return GetTensorStrides(tensor, memory_layout); })
      .assign_to(byte_strides)
      .and_then([&]() { return tensor.data<T>(); })
      .assign_to(tensor_data)
      .and_then([&]() -> gxf::Expected<void> {
        if (!tensor_data) {
          GXF_LOG_ERROR("Failed to get tensor data!");
          return gxf::Unexpected{GXF_FAILURE};
        }
        return gxf::Success;
      })
      .and_then([&]() {
        return ::nvidia::isaac::StridePointer3D<T>(tensor_data, shape, memory_layout, byte_strides);
      });
}

template <typename T>
gxf::Expected<::nvidia::isaac::StridePointer3D<T>> GXFVideoBufferToStridePointer3D(
    gxf::VideoBuffer& video_buffer, ::nvidia::isaac::StridePointer3D_MemoryLayout memory_layout) {
  // TODO(kchinniah): add support for video buffer strides
  ::nvidia::isaac::StridePointer3D_Shape shape;
  std::array<size_t, 3> strides{};
  T* video_buffer_data{nullptr};
  return GetVideoBufferStridePointer3DShape(video_buffer, memory_layout)
      .assign_to(shape)
      .and_then([&]() -> gxf::Expected<std::array<size_t, 3>> {
        if (video_buffer.video_frame_info().color_planes.size() != 1) {
          GXF_LOG_ERROR("Error: received unsupported color plane size!");
          return gxf::Unexpected{GXF_FAILURE};
        }

        if (video_buffer.video_frame_info().color_planes[0].stride <= 0) {
          GXF_LOG_ERROR("Error: received a stride less than equal to 0!");
          return gxf::Unexpected{GXF_FAILURE};
        }

        return std::array<size_t, 3>{
            static_cast<size_t>(video_buffer.video_frame_info().color_planes[0].stride),
            video_buffer.video_frame_info().color_planes[0].bytes_per_pixel, sizeof(T)};
      })
      .assign_to(strides)
      .and_then([&]() { return reinterpret_cast<T*>(video_buffer.pointer()); })
      .assign_to(video_buffer_data)
      .and_then([&]() -> gxf::Expected<void> {
        if (!video_buffer_data) {
          GXF_LOG_ERROR("Failed to get video buffer data!");
          return gxf::Unexpected{GXF_FAILURE};
        }
        return gxf::Success;
      })
      .and_then(
          [&]() { return ::nvidia::isaac::StridePointer3D<T>(
              video_buffer_data, shape, memory_layout); });
}

#define NVIDIA_ISAAC_GEMS_HELPERS_GXF_TO_STRIDE_POINTER_3D_FUNCTION_DEFN(T)            \
  template gxf::Expected<::nvidia::isaac::StridePointer3D<T>> GXFTensorToStridePointer3D(      \
      gxf::Tensor& tensor, ::nvidia::isaac::StridePointer3D_MemoryLayout memory_layout);       \
  template gxf::Expected<::nvidia::isaac::StridePointer3D<T>> GXFVideoBufferToStridePointer3D( \
      gxf::VideoBuffer& video_buffer, ::nvidia::isaac::StridePointer3D_MemoryLayout memory_layout);

NVIDIA_ISAAC_GEMS_HELPERS_GXF_TO_STRIDE_POINTER_3D_FUNCTION_DEFN(uint8_t)
NVIDIA_ISAAC_GEMS_HELPERS_GXF_TO_STRIDE_POINTER_3D_FUNCTION_DEFN(uint16_t)
NVIDIA_ISAAC_GEMS_HELPERS_GXF_TO_STRIDE_POINTER_3D_FUNCTION_DEFN(int)
NVIDIA_ISAAC_GEMS_HELPERS_GXF_TO_STRIDE_POINTER_3D_FUNCTION_DEFN(double)
NVIDIA_ISAAC_GEMS_HELPERS_GXF_TO_STRIDE_POINTER_3D_FUNCTION_DEFN(float)

}  // namespace isaac
}  // namespace nvidia
