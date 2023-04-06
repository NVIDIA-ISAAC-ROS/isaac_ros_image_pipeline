// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "Filters.h"
#include "NppUtils.h"

#include "cv/core/MathTypes.h"
#include "cv/core/Memory.h"

#include <nppi_filtering_functions.h>

#include <cassert>
#include <cstdint>
#include <stdexcept>

namespace cvcore { namespace tensor_ops {

void BoxFilter(Tensor<HWC, C3, U8> &dst, const Tensor<HWC, C3, U8> &src, const Vector2i &maskSize,
               const Vector2i &anchor, cudaStream_t stream)
{
    assert(!src.isCPU() && !dst.isCPU());
    NppStatus status = nppiFilterBoxBorder_8u_C3R_Ctx(
        static_cast<const Npp8u *>(src.getData()), src.getStride(TensorDimension::HEIGHT) * sizeof(Npp8u),
        {static_cast<int>(src.getWidth()), static_cast<int>(src.getHeight())}, {0, 0},
        static_cast<Npp8u *>(dst.getData()), dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp8u),
        {static_cast<int>(src.getWidth()), static_cast<int>(src.getHeight())}, {maskSize.x, maskSize.y},
        {anchor.x, anchor.y},
        NPP_BORDER_REPLICATE, //Only Npp Replicate is supported!!!
        GetNppStreamContext(stream));
    assert(status == NPP_SUCCESS);
}

void BoxFilter(Tensor<HWC, C1, U8> &dst, const Tensor<HWC, C1, U8> &src, const Vector2i &maskSize,
               const Vector2i &anchor, cudaStream_t stream)
{
    assert(!src.isCPU() && !dst.isCPU());
    NppStatus status = nppiFilterBoxBorder_8u_C1R_Ctx(
        static_cast<const Npp8u *>(src.getData()), src.getStride(TensorDimension::HEIGHT) * sizeof(Npp8u),
        {static_cast<int>(src.getWidth()), static_cast<int>(src.getHeight())}, {0, 0},
        static_cast<Npp8u *>(dst.getData()), dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp8u),
        {static_cast<int>(src.getWidth()), static_cast<int>(src.getHeight())}, {maskSize.x, maskSize.y},
        {anchor.x, anchor.y}, NPP_BORDER_REPLICATE, GetNppStreamContext(stream));
    assert(status == NPP_SUCCESS);
}

void BoxFilter(Tensor<HWC, C1, U16> &dst, const Tensor<HWC, C1, U16> &src, const Vector2i &maskSize,
               const Vector2i &anchor, cudaStream_t stream)
{
    assert(!src.isCPU() && !dst.isCPU());
    NppStatus status = nppiFilterBoxBorder_16u_C1R_Ctx(
        static_cast<const Npp16u *>(src.getData()), src.getStride(TensorDimension::HEIGHT) * sizeof(Npp16u),
        {static_cast<int>(src.getWidth()), static_cast<int>(src.getHeight())}, {0, 0},
        static_cast<Npp16u *>(dst.getData()), dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp16u),
        {static_cast<int>(src.getWidth()), static_cast<int>(src.getHeight())}, {maskSize.x, maskSize.y},
        {anchor.x, anchor.y}, NPP_BORDER_REPLICATE, GetNppStreamContext(stream));
    assert(status == NPP_SUCCESS);
}

void BoxFilter(Tensor<HWC, C3, U16> &dst, const Tensor<HWC, C3, U16> &src, const Vector2i &maskSize,
               const Vector2i &anchor, cudaStream_t stream)
{
    assert(!src.isCPU() && !dst.isCPU());
    NppStatus status = nppiFilterBoxBorder_16u_C3R_Ctx(
        static_cast<const Npp16u *>(src.getData()), src.getStride(TensorDimension::HEIGHT) * sizeof(Npp16u),
        {static_cast<int>(src.getWidth()), static_cast<int>(src.getHeight())}, {0, 0},
        static_cast<Npp16u *>(dst.getData()), dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp16u),
        {static_cast<int>(src.getWidth()), static_cast<int>(src.getHeight())}, {maskSize.x, maskSize.y},
        {anchor.x, anchor.y}, NPP_BORDER_REPLICATE, GetNppStreamContext(stream));
    assert(status == NPP_SUCCESS);
}

void BoxFilter(Tensor<HWC, C3, F32> &dst, const Tensor<HWC, C3, F32> &src, const Vector2i &maskSize,
               const Vector2i &anchor, cudaStream_t stream)
{
    assert(!src.isCPU() && !dst.isCPU());
    NppStatus status = nppiFilterBoxBorder_32f_C3R_Ctx(
        static_cast<const Npp32f *>(src.getData()), src.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f),
        {static_cast<int>(src.getWidth()), static_cast<int>(src.getHeight())}, {0, 0},
        static_cast<Npp32f *>(dst.getData()), dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f),
        {static_cast<int>(src.getWidth()), static_cast<int>(src.getHeight())}, {maskSize.x, maskSize.y},
        {anchor.x, anchor.y}, NPP_BORDER_REPLICATE, GetNppStreamContext(stream));
    assert(status == NPP_SUCCESS);
}

void BoxFilter(Tensor<HWC, C1, F32> &dst, const Tensor<HWC, C1, F32> &src, const Vector2i &maskSize,
               const Vector2i &anchor, cudaStream_t stream)
{
    assert(!src.isCPU() && !dst.isCPU());
    NppStatus status = nppiFilterBoxBorder_32f_C1R_Ctx(
        static_cast<const Npp32f *>(src.getData()), src.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f),
        {static_cast<int>(src.getWidth()), static_cast<int>(src.getHeight())}, {0, 0},
        static_cast<Npp32f *>(dst.getData()), dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f),
        {static_cast<int>(src.getWidth()), static_cast<int>(src.getHeight())}, {maskSize.x, maskSize.y},
        {anchor.x, anchor.y}, NPP_BORDER_REPLICATE, GetNppStreamContext(stream));
    assert(status == NPP_SUCCESS);
}

}} // namespace cvcore::tensor_ops
