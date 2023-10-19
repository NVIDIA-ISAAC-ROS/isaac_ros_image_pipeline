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
#pragma once

#include "engine/gems/cuda_utils/stride_pointer_3d.hpp"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/tensor.hpp"

namespace nvidia {
namespace isaac {

gxf::Expected<::nvidia::isaac::StridePointer3D_Shape> GetTensorStridePointer3DShape(
    const gxf::Tensor& tensor, ::nvidia::isaac::StridePointer3D_MemoryLayout memory_layout);

gxf::Expected<::nvidia::isaac::StridePointer3D_Shape> GetVideoBufferStridePointer3DShape(
    const gxf::VideoBuffer& video_buffer,
    ::nvidia::isaac::StridePointer3D_MemoryLayout memory_layout);

/**
Converts a gxf::Tensor into a ::nvidia::isaac::StridePointer3D<T> type.

NOTE: the ::nvidia::isaac::StridePointer3D does not own the underlying memory, hence
the gxf::Tensor must remain valid.

@param T The type to cast the underlying data of the tensor into
@param tensor The tensor that will be converted into a StridePointer3D
@param memory_layout The memory layout of the passed in tensor

@return A StridePointer3D to the data underlying the tensor
**/
template <typename T>
gxf::Expected<::nvidia::isaac::StridePointer3D<T>> GXFTensorToStridePointer3D(
    gxf::Tensor& tensor, ::nvidia::isaac::StridePointer3D_MemoryLayout memory_layout);

/**
Converts a gxf::VideoBuffer into a ::nvidia::isaac::StridePointer3D<T> type.

NOTE: the ::nvidia::isaac::StridePointer3D does not own the underlying memory, hence
the gxf::VideoBuffer must remain valid.
WARNING: currently, the memory_layout MUST be ::nvidia::isaac::StridePointer3D_MemoryLayout::kHWC,
and the passed in video_buffer MUST be unpadded as this does not use the strides of the
video_buffer. Instead, the trivial strides are computed.

@param T The type to cast the underlying data of the video_buffer into
@param video_buffer The video_buffer that will be converted into a StridePointer3D
@param memory_layout The memory layout of the passed in video_buffer

@return A StridePointer3D to the data underlying the video_buffer
**/
template <typename T>
gxf::Expected<::nvidia::isaac::StridePointer3D<T>> GXFVideoBufferToStridePointer3D(
    gxf::VideoBuffer& video_buffer, ::nvidia::isaac::StridePointer3D_MemoryLayout memory_layout);

}  // namespace isaac
}  // namespace nvidia
