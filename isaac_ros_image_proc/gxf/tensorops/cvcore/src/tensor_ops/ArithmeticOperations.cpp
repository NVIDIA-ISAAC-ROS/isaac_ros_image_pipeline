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

#include "NppUtils.h"

#include "cv/tensor_ops/ImageUtils.h"

#include "cv/core/Memory.h"

#include <nppi_arithmetic_and_logical_operations.h>
#include <nppi_data_exchange_and_initialization.h>

#include <cassert>
#include <cstdint>
#include <stdexcept>

namespace cvcore { namespace tensor_ops {

namespace {

static void NormalizeTensorC3F32Inplace(Tensor<HWC, C3, F32> &src, const float scale[3], const float offset[3],
                                        NppStreamContext streamContext)
{
    const int srcW         = src.getWidth();
    const int srcH         = src.getHeight();
    const NppiSize srcSize = {srcW, srcH};

    const Npp32f offsets[3] = {static_cast<Npp32f>(offset[0]), static_cast<Npp32f>(offset[1]),
                               static_cast<Npp32f>(offset[2])};
    NppStatus status =
        nppiAddC_32f_C3IR_Ctx(offsets, static_cast<Npp32f *>(src.getData()),
                              src.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f), srcSize, streamContext);
    assert(status == NPP_SUCCESS);

    const Npp32f scales[3] = {static_cast<Npp32f>(scale[0]), static_cast<Npp32f>(scale[1]),
                              static_cast<Npp32f>(scale[2])};
    status                 = nppiMulC_32f_C3IR_Ctx(scales, static_cast<Npp32f *>(src.getData()),
                                   src.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f), srcSize, streamContext);
    assert(status == NPP_SUCCESS);
}

template<TensorLayout TL>
static void NormalizeTensorC1F32Inplace(Tensor<TL, C1, F32> &src, const float scale, const float offset,
                                        NppStreamContext streamContext)
{
    const int srcW         = src.getWidth();
    const int srcH         = src.getHeight();
    const NppiSize srcSize = {srcW, srcH};

    NppStatus status =
        nppiAddC_32f_C1IR_Ctx(static_cast<Npp32f>(offset), static_cast<Npp32f *>(src.getData()),
                              src.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f), srcSize, streamContext);
    assert(status == NPP_SUCCESS);

    status = nppiMulC_32f_C1IR_Ctx(static_cast<Npp32f>(scale), static_cast<Npp32f *>(src.getData()),
                                   src.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f), srcSize, streamContext);
    assert(status == NPP_SUCCESS);
}

template<TensorLayout TL>
void NormalizeC1U8Impl(Tensor<TL, C1, F32> &dst, const Tensor<TL, C1, U8> &src, const float scale, const float offset,
                       cudaStream_t stream)
{
    // src and dst must be GPU tensors
    assert(!src.isCPU() && !dst.isCPU());
    assert((src.getWidth() == dst.getWidth()) && (src.getHeight() == dst.getHeight()));

    NppStreamContext streamContext = GetNppStreamContext(stream);

    NppStatus status = nppiConvert_8u32f_C1R_Ctx(
        static_cast<const Npp8u *>(src.getData()), src.getStride(TensorDimension::HEIGHT) * sizeof(Npp8u),
        static_cast<Npp32f *>(dst.getData()), dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f),
        {int(src.getWidth()), int(src.getHeight())}, streamContext);
    assert(status == NPP_SUCCESS);

    NormalizeTensorC1F32Inplace(dst, scale, offset, streamContext);
}

template<TensorLayout TL>
void NormalizeC1U16Impl(Tensor<TL, C1, F32> &dst, const Tensor<TL, C1, U16> &src, const float scale, const float offset,
                        cudaStream_t stream)
{
    // src and dst must be GPU tensors
    assert(!src.isCPU() && !dst.isCPU());
    assert((src.getWidth() == dst.getWidth()) && (src.getHeight() == dst.getHeight()));

    NppStreamContext streamContext = GetNppStreamContext(stream);

    NppStatus status = nppiConvert_16u32f_C1R_Ctx(
        static_cast<const Npp16u *>(src.getData()), src.getStride(TensorDimension::HEIGHT) * sizeof(Npp16u),
        static_cast<Npp32f *>(dst.getData()), dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f),
        {int(src.getWidth()), int(src.getHeight())}, streamContext);
    assert(status == NPP_SUCCESS);

    NormalizeTensorC1F32Inplace(dst, scale, offset, streamContext);
}

template<TensorLayout TL>
void NormalizeC1F32Impl(Tensor<TL, C1, F32> &dst, const Tensor<TL, C1, F32> &src, const float scale, const float offset,
                        cudaStream_t stream)
{
    // src and dst must be GPU tensors
    assert(!src.isCPU() && !dst.isCPU());
    assert((src.getWidth() == dst.getWidth()) && (src.getHeight() == dst.getHeight()));

    Copy(dst, src, stream);
    NormalizeTensorC1F32Inplace(dst, scale, offset, GetNppStreamContext(stream));
}

template<ChannelType CT>
void NormalizeC3Batch(Tensor<NHWC, C3, F32> &dst, Tensor<NHWC, C3, CT> &src, const float scale[3],
                      const float offset[3], cudaStream_t stream)
{
    // src and dst must be GPU tensors
    assert(!src.isCPU() && !dst.isCPU());
    assert(src.getDepth() == dst.getDepth());

    for (int i = 0; i < src.getDepth(); i++)
    {
        size_t shiftSrc = i * src.getStride(TensorDimension::DEPTH);
        size_t shiftDst = i * dst.getStride(TensorDimension::DEPTH);
        Tensor<HWC, C3, CT> srcTmp(src.getWidth(), src.getHeight(),
                                   src.getStride(TensorDimension::HEIGHT) * GetChannelSize(CT),
                                   src.getData() + shiftSrc, false);
        Tensor<HWC, C3, F32> dstTmp(dst.getWidth(), dst.getHeight(),
                                    dst.getStride(TensorDimension::HEIGHT) * GetChannelSize(F32),
                                    dst.getData() + shiftDst, false);
        Normalize(dstTmp, srcTmp, scale, offset, stream);
    }
}

template<ChannelType CT>
void NormalizeC1Batch(Tensor<NHWC, C1, F32> &dst, Tensor<NHWC, C1, CT> &src, const float scale, const float offset,
                      cudaStream_t stream)
{
    // src and dst must be GPU tensors
    assert(!src.isCPU() && !dst.isCPU());
    assert(src.getDepth() == dst.getDepth());

    for (int i = 0; i < src.getDepth(); i++)
    {
        size_t shiftSrc = i * src.getStride(TensorDimension::DEPTH);
        size_t shiftDst = i * dst.getStride(TensorDimension::DEPTH);
        Tensor<HWC, C1, CT> srcTmp(src.getWidth(), src.getHeight(),
                                   src.getStride(TensorDimension::HEIGHT) * GetChannelSize(CT),
                                   src.getData() + shiftSrc, false);
        Tensor<HWC, C1, F32> dstTmp(dst.getWidth(), dst.getHeight(),
                                    dst.getStride(TensorDimension::HEIGHT) * GetChannelSize(F32),
                                    dst.getData() + shiftDst, false);
        Normalize(dstTmp, srcTmp, scale, offset, stream);
    }
}

template<ChannelType CT>
void NormalizeC1Batch(Tensor<NCHW, C1, F32> &dst, Tensor<NCHW, C1, CT> &src, const float scale, const float offset,
                      cudaStream_t stream)
{
    // src and dst must be GPU tensors
    assert(!src.isCPU() && !dst.isCPU());
    assert(src.getDepth() == dst.getDepth());

    for (int i = 0; i < src.getDepth(); i++)
    {
        size_t shiftSrc = i * src.getStride(TensorDimension::DEPTH);
        size_t shiftDst = i * dst.getStride(TensorDimension::DEPTH);
        Tensor<CHW, C1, CT> srcTmp(src.getWidth(), src.getHeight(),
                                   src.getStride(TensorDimension::HEIGHT) * GetChannelSize(CT),
                                   src.getData() + shiftSrc, false);
        Tensor<CHW, C1, F32> dstTmp(dst.getWidth(), dst.getHeight(),
                                    dst.getStride(TensorDimension::HEIGHT) * GetChannelSize(F32),
                                    dst.getData() + shiftDst, false);
        Normalize(dstTmp, srcTmp, scale, offset, stream);
    }
}

} // anonymous namespace

void Normalize(Tensor<HWC, C3, F32> &dst, const Tensor<HWC, C3, U8> &src, const float scale[3], const float offset[3],
               cudaStream_t stream)
{
    // src and dst must be GPU tensors
    assert(!src.isCPU() && !dst.isCPU());
    assert((src.getWidth() == dst.getWidth()) && (src.getHeight() == dst.getHeight()));

    NppStreamContext streamContext = GetNppStreamContext(stream);

    NppStatus status = nppiConvert_8u32f_C3R_Ctx(
        static_cast<const Npp8u *>(src.getData()), src.getStride(TensorDimension::HEIGHT) * sizeof(Npp8u),
        static_cast<Npp32f *>(dst.getData()), dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f),
        {int(src.getWidth()), int(src.getHeight())}, streamContext);
    assert(status == NPP_SUCCESS);

    NormalizeTensorC3F32Inplace(dst, scale, offset, streamContext);
}

void Normalize(Tensor<NHWC, C3, F32> &dst, const Tensor<NHWC, C3, U8> &src, const float scale[3], const float offset[3],
               cudaStream_t stream)
{
    NormalizeC3Batch(dst, const_cast<Tensor<NHWC, C3, U8> &>(src), scale, offset, stream);
}

void Normalize(Tensor<HWC, C3, F32> &dst, const Tensor<HWC, C3, U16> &src, const float scale[3], const float offset[3],
               cudaStream_t stream)
{
    // src and dst must be GPU tensors
    assert(!src.isCPU() && !dst.isCPU());
    assert((src.getWidth() == dst.getWidth()) && (src.getHeight() == dst.getHeight()));

    NppStreamContext streamContext = GetNppStreamContext(stream);

    NppStatus status = nppiConvert_16u32f_C3R_Ctx(
        static_cast<const Npp16u *>(src.getData()), src.getStride(TensorDimension::HEIGHT) * sizeof(Npp16u),
        static_cast<Npp32f *>(dst.getData()), dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f),
        {int(src.getWidth()), int(src.getHeight())}, streamContext);
    assert(status == NPP_SUCCESS);

    NormalizeTensorC3F32Inplace(dst, scale, offset, streamContext);
}

void Normalize(Tensor<NHWC, C3, F32> &dst, const Tensor<NHWC, C3, U16> &src, const float scale[3],
               const float offset[3], cudaStream_t stream)
{
    NormalizeC3Batch(dst, const_cast<Tensor<NHWC, C3, U16> &>(src), scale, offset, stream);
}

void Normalize(Tensor<HWC, C3, F32> &dst, const Tensor<HWC, C3, F32> &src, const float scale[3], const float offset[3],
               cudaStream_t stream)
{
    // src and dst must be GPU tensors
    assert(!src.isCPU() && !dst.isCPU());
    assert((src.getWidth() == dst.getWidth()) && (src.getHeight() == dst.getHeight()));

    Copy(dst, src, stream);
    NormalizeTensorC3F32Inplace(dst, scale, offset, GetNppStreamContext(stream));
}

void Normalize(Tensor<NHWC, C3, F32> &dst, const Tensor<NHWC, C3, F32> &src, const float scale[3],
               const float offset[3], cudaStream_t stream)
{
    NormalizeC3Batch(dst, const_cast<Tensor<NHWC, C3, F32> &>(src), scale, offset, stream);
}

void Normalize(Tensor<HWC, C1, F32> &dst, const Tensor<HWC, C1, U8> &src, const float scale, const float offset,
               cudaStream_t stream)
{
    NormalizeC1U8Impl(dst, src, scale, offset, stream);
}

void Normalize(Tensor<NHWC, C1, F32> &dst, const Tensor<NHWC, C1, U8> &src, const float scale, const float offset,
               cudaStream_t stream)
{
    NormalizeC1Batch(dst, const_cast<Tensor<NHWC, C1, U8> &>(src), scale, offset, stream);
}

void Normalize(Tensor<HWC, C1, F32> &dst, const Tensor<HWC, C1, U16> &src, const float scale, const float offset,
               cudaStream_t stream)
{
    NormalizeC1U16Impl(dst, src, scale, offset, stream);
}

void Normalize(Tensor<NHWC, C1, F32> &dst, const Tensor<NHWC, C1, U16> &src, const float scale, const float offset,
               cudaStream_t stream)
{
    NormalizeC1Batch(dst, const_cast<Tensor<NHWC, C1, U16> &>(src), scale, offset, stream);
}

void Normalize(Tensor<HWC, C1, F32> &dst, const Tensor<HWC, C1, F32> &src, const float scale, const float offset,
               cudaStream_t stream)
{
    NormalizeC1F32Impl(dst, src, scale, offset, stream);
}

void Normalize(Tensor<NHWC, C1, F32> &dst, const Tensor<NHWC, C1, F32> &src, const float scale, const float offset,
               cudaStream_t stream)
{
    NormalizeC1Batch(dst, const_cast<Tensor<NHWC, C1, F32> &>(src), scale, offset, stream);
}

void Normalize(Tensor<CHW, C1, F32> &dst, const Tensor<CHW, C1, U8> &src, const float scale, const float offset,
               cudaStream_t stream)
{
    NormalizeC1U8Impl(dst, src, scale, offset, stream);
}

void Normalize(Tensor<NCHW, C1, F32> &dst, const Tensor<NCHW, C1, U8> &src, const float scale, const float offset,
               cudaStream_t stream)
{
    NormalizeC1Batch(dst, const_cast<Tensor<NCHW, C1, U8> &>(src), scale, offset, stream);
}

void Normalize(Tensor<CHW, C1, F32> &dst, const Tensor<CHW, C1, U16> &src, const float scale, const float offset,
               cudaStream_t stream)
{
    NormalizeC1U16Impl(dst, src, scale, offset, stream);
}

void Normalize(Tensor<NCHW, C1, F32> &dst, const Tensor<NCHW, C1, U16> &src, const float scale, const float offset,
               cudaStream_t stream)
{
    NormalizeC1Batch(dst, const_cast<Tensor<NCHW, C1, U16> &>(src), scale, offset, stream);
}

void Normalize(Tensor<CHW, C1, F32> &dst, const Tensor<CHW, C1, F32> &src, const float scale, const float offset,
               cudaStream_t stream)
{
    NormalizeC1F32Impl(dst, src, scale, offset, stream);
}

void Normalize(Tensor<NCHW, C1, F32> &dst, const Tensor<NCHW, C1, F32> &src, const float scale, const float offset,
               cudaStream_t stream)
{
    NormalizeC1Batch(dst, const_cast<Tensor<NCHW, C1, F32> &>(src), scale, offset, stream);
}

}} // namespace cvcore::tensor_ops