// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cassert>
#include <cstdint>
#include <stdexcept>

#include "extensions/tensorops/core/ImageUtils.h"
#include "extensions/tensorops/core/Memory.h"
#include "extensions/tensorops/core/NppUtils.h"
#include "nppi_arithmetic_and_logical_operations.h"
#include "nppi_data_exchange_and_initialization.h"

namespace cvcore {
namespace tensor_ops {

namespace {

static bool NormalizeTensorC3F32Inplace(Tensor<HWC, C3, F32>& src, const float scale[3],
    const float offset[3], NppStreamContext streamContext) {
    const int srcW         = src.getWidth();
    const int srcH         = src.getHeight();
    const NppiSize srcSize = {srcW, srcH};

    const Npp32f offsets[3] = {static_cast<Npp32f>(offset[0]), static_cast<Npp32f>(offset[1]),
                               static_cast<Npp32f>(offset[2])};
    NppStatus status =
        nppiAddC_32f_C3IR_Ctx(offsets, static_cast<Npp32f *>(src.getData()),
            src.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f), srcSize, streamContext);
    if (status != NPP_SUCCESS) {
        return false;
    }

    const Npp32f scales[3] = {static_cast<Npp32f>(scale[0]), static_cast<Npp32f>(scale[1]),
                              static_cast<Npp32f>(scale[2])};
    status = nppiMulC_32f_C3IR_Ctx(scales, static_cast<Npp32f *>(src.getData()),
                 src.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f), srcSize, streamContext);
    if (status != NPP_SUCCESS) {
        return false;
    }
    return true;
}

template<TensorLayout TL>
static bool NormalizeTensorC1F32Inplace(Tensor<TL, C1, F32>& src, const float scale,
    const float offset, NppStreamContext streamContext) {
    const int srcW         = src.getWidth();
    const int srcH         = src.getHeight();
    const NppiSize srcSize = {srcW, srcH};

    NppStatus status =
        nppiAddC_32f_C1IR_Ctx(static_cast<Npp32f>(offset), static_cast<Npp32f *>(src.getData()),
            src.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f), srcSize, streamContext);
    if (status != NPP_SUCCESS) {
        return false;
    }

    status = nppiMulC_32f_C1IR_Ctx(static_cast<Npp32f>(scale), static_cast<Npp32f *>(src.getData()),
                src.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f), srcSize, streamContext);
    if (status != NPP_SUCCESS) {
        return false;
    }
    return true;
}

template<TensorLayout TL>
bool NormalizeC1U8Impl(Tensor<TL, C1, F32>& dst, const Tensor<TL, C1, U8>& src,
    const float scale, const float offset, cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }
    if ((src.getWidth() != dst.getWidth()) || (src.getHeight() != dst.getHeight())) {
        return false;
    }

    NppStreamContext streamContext = GetNppStreamContext(stream);

    NppStatus status = nppiConvert_8u32f_C1R_Ctx(
        static_cast<const Npp8u *>(src.getData()),
        src.getStride(TensorDimension::HEIGHT) * sizeof(Npp8u),
        static_cast<Npp32f *>(dst.getData()),
        dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f),
        {static_cast<int>(src.getWidth()), static_cast<int>(src.getHeight())}, streamContext);
    if (status != NPP_SUCCESS) {
        return false;
    }

    return NormalizeTensorC1F32Inplace(dst, scale, offset, streamContext);
}

template<TensorLayout TL>
bool NormalizeC1U16Impl(Tensor<TL, C1, F32>& dst, const Tensor<TL, C1, U16>& src,
    const float scale, const float offset, cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }
    if ((src.getWidth() != dst.getWidth()) || (src.getHeight() != dst.getHeight())) {
        return false;
    }
    NppStreamContext streamContext = GetNppStreamContext(stream);

    NppStatus status = nppiConvert_16u32f_C1R_Ctx(
        static_cast<const Npp16u *>(src.getData()),
        src.getStride(TensorDimension::HEIGHT) * sizeof(Npp16u),
        static_cast<Npp32f *>(dst.getData()),
        dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f),
        {static_cast<int>(src.getWidth()), static_cast<int>(src.getHeight())}, streamContext);
    if (status != NPP_SUCCESS) {
        return false;
    }

    return NormalizeTensorC1F32Inplace(dst, scale, offset, streamContext);
}

template<TensorLayout TL>
bool NormalizeC1F32Impl(Tensor<TL, C1, F32>& dst, const Tensor<TL, C1, F32>& src,
    const float scale, const float offset, cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }
    if ((src.getWidth() != dst.getWidth()) || (src.getHeight() != dst.getHeight())) {
        return false;
    }

    Copy(dst, src, stream);
    return NormalizeTensorC1F32Inplace(dst, scale, offset, GetNppStreamContext(stream));
}

template<ChannelType CT>
bool NormalizeC3Batch(Tensor<NHWC, C3, F32>& dst, Tensor<NHWC, C3, CT>& src, const float scale[3],
    const float offset[3], cudaStream_t stream) {
    // src and dst must be GPU tensors
    if ((src.isCPU() || dst.isCPU()) || (src.getDepth() != dst.getDepth())) {
        return false;
    }

    for (size_t i = 0; i < src.getDepth(); i++) {
        size_t shiftSrc = i * src.getStride(TensorDimension::DEPTH);
        size_t shiftDst = i * dst.getStride(TensorDimension::DEPTH);
        Tensor<HWC, C3, CT> srcTmp(src.getWidth(), src.getHeight(),
                                   src.getStride(TensorDimension::HEIGHT) * GetChannelSize(CT),
                                   src.getData() + shiftSrc, false);
        Tensor<HWC, C3, F32> dstTmp(dst.getWidth(), dst.getHeight(),
                                    dst.getStride(TensorDimension::HEIGHT) * GetChannelSize(F32),
                                    dst.getData() + shiftDst, false);
        bool ret = Normalize(dstTmp, srcTmp, scale, offset, stream);
        if (!ret) {
            return false;
        }
    }
    return true;
}

template<ChannelType CT>
bool NormalizeC1Batch(Tensor<NHWC, C1, F32>& dst, Tensor<NHWC, C1, CT>& src,
    const float scale, const float offset, cudaStream_t stream) {
    // src and dst must be GPU tensors
    if ((src.isCPU() || dst.isCPU()) || (src.getDepth() != dst.getDepth())) {
        return false;
    }

    for (size_t i = 0; i < src.getDepth(); i++) {
        size_t shiftSrc = i * src.getStride(TensorDimension::DEPTH);
        size_t shiftDst = i * dst.getStride(TensorDimension::DEPTH);
        Tensor<HWC, C1, CT> srcTmp(src.getWidth(), src.getHeight(),
                                   src.getStride(TensorDimension::HEIGHT) * GetChannelSize(CT),
                                   src.getData() + shiftSrc, false);
        Tensor<HWC, C1, F32> dstTmp(dst.getWidth(), dst.getHeight(),
                                    dst.getStride(TensorDimension::HEIGHT) * GetChannelSize(F32),
                                    dst.getData() + shiftDst, false);
        bool ret = Normalize(dstTmp, srcTmp, scale, offset, stream);
        if (!ret) {
            return false;
        }
    }
    return true;
}

template<ChannelType CT>
bool NormalizeC1Batch(Tensor<NCHW, C1, F32>& dst, Tensor<NCHW, C1, CT>& src,
    const float scale, const float offset, cudaStream_t stream) {
    // src and dst must be GPU tensors
    if ((src.isCPU() || dst.isCPU()) || (src.getDepth() != dst.getDepth())) {
        return false;
    }

    for (size_t i = 0; i < src.getDepth(); i++) {
        size_t shiftSrc = i * src.getStride(TensorDimension::DEPTH);
        size_t shiftDst = i * dst.getStride(TensorDimension::DEPTH);
        Tensor<CHW, C1, CT> srcTmp(src.getWidth(), src.getHeight(),
                                   src.getStride(TensorDimension::HEIGHT) * GetChannelSize(CT),
                                   src.getData() + shiftSrc, false);
        Tensor<CHW, C1, F32> dstTmp(dst.getWidth(), dst.getHeight(),
                                    dst.getStride(TensorDimension::HEIGHT) * GetChannelSize(F32),
                                    dst.getData() + shiftDst, false);
        bool ret = Normalize(dstTmp, srcTmp, scale, offset, stream);
        if (!ret) {
            return false;
        }
    }
    return true;
}

}  // anonymous namespace

bool Normalize(Tensor<HWC, C3, F32>& dst, const Tensor<HWC, C3, U8>& src,
    const float scale[3], const float offset[3], cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }
    if ((src.getWidth() != dst.getWidth()) || (src.getHeight() != dst.getHeight())) {
        return false;
    }

    NppStreamContext streamContext = GetNppStreamContext(stream);

    NppStatus status = nppiConvert_8u32f_C3R_Ctx(
        static_cast<const Npp8u *>(src.getData()),
        src.getStride(TensorDimension::HEIGHT) * sizeof(Npp8u),
        static_cast<Npp32f *>(dst.getData()),
        dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f),
        {static_cast<int>(src.getWidth()), static_cast<int>(src.getHeight())}, streamContext);
    if (status != NPP_SUCCESS) {
        return false;
    }

    return NormalizeTensorC3F32Inplace(dst, scale, offset, streamContext);
}

bool Normalize(Tensor<NHWC, C3, F32>& dst, const Tensor<NHWC, C3, U8>& src,
    const float scale[3], const float offset[3], cudaStream_t stream) {
    return NormalizeC3Batch(dst, const_cast<Tensor<NHWC, C3, U8> &>(src), scale, offset, stream);
}

bool Normalize(Tensor<HWC, C3, F32>& dst, const Tensor<HWC, C3, U16>& src,
    const float scale[3], const float offset[3], cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }
    if ((src.getWidth() != dst.getWidth()) || (src.getHeight() != dst.getHeight())) {
        return false;
    }
    NppStreamContext streamContext = GetNppStreamContext(stream);

    NppStatus status = nppiConvert_16u32f_C3R_Ctx(
        static_cast<const Npp16u *>(src.getData()),
        src.getStride(TensorDimension::HEIGHT) * sizeof(Npp16u),
        static_cast<Npp32f *>(dst.getData()),
        dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f),
        {static_cast<int>(src.getWidth()), static_cast<int>(src.getHeight())}, streamContext);
    if (status != NPP_SUCCESS) {
        return false;
    }

    return NormalizeTensorC3F32Inplace(dst, scale, offset, streamContext);
}

bool Normalize(Tensor<NHWC, C3, F32>& dst, const Tensor<NHWC, C3, U16>& src, const float scale[3],
               const float offset[3], cudaStream_t stream) {
    return (NormalizeC3Batch(dst, const_cast<Tensor<NHWC, C3, U16> &>(src), scale, offset, stream));
}

bool Normalize(Tensor<HWC, C3, F32>& dst, const Tensor<HWC, C3, F32>& src,
    const float scale[3], const float offset[3], cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }
    if ((src.getWidth() != dst.getWidth()) || (src.getHeight() != dst.getHeight())) {
        return false;
    }
    Copy(dst, src, stream);
    return NormalizeTensorC3F32Inplace(dst, scale, offset, GetNppStreamContext(stream));
}

bool Normalize(Tensor<NHWC, C3, F32> & dst, const Tensor<NHWC, C3, F32> & src, const float scale[3],
    const float offset[3], cudaStream_t stream) {
    return (NormalizeC3Batch(dst, const_cast<Tensor<NHWC, C3, F32> &>(src), scale, offset, stream));
}

bool Normalize(Tensor<HWC, C1, F32>& dst, const Tensor<HWC, C1, U8>& src,
    const float scale, const float offset, cudaStream_t stream) {
    return NormalizeC1U8Impl(dst, src, scale, offset, stream);
}

bool Normalize(Tensor<NHWC, C1, F32>& dst, const Tensor<NHWC, C1, U8>& src,
    const float scale, const float offset, cudaStream_t stream) {
    return NormalizeC1Batch(dst, const_cast<Tensor<NHWC, C1, U8> &>(src), scale, offset, stream);
}

bool Normalize(Tensor<HWC, C1, F32>& dst, const Tensor<HWC, C1, U16>& src,
    const float scale, const float offset, cudaStream_t stream) {
    return NormalizeC1U16Impl(dst, src, scale, offset, stream);
}

bool Normalize(Tensor<NHWC, C1, F32>& dst, const Tensor<NHWC, C1, U16>& src,
    const float scale, const float offset, cudaStream_t stream) {
    return NormalizeC1Batch(dst, const_cast<Tensor<NHWC, C1, U16> &>(src), scale, offset, stream);
}

bool Normalize(Tensor<HWC, C1, F32>& dst, const Tensor<HWC, C1, F32>& src,
    const float scale, const float offset, cudaStream_t stream) {
    return NormalizeC1F32Impl(dst, src, scale, offset, stream);
}

bool Normalize(Tensor<NHWC, C1, F32>& dst, const Tensor<NHWC, C1, F32>& src,
    const float scale, const float offset, cudaStream_t stream) {
    return NormalizeC1Batch(dst, const_cast<Tensor<NHWC, C1, F32> &>(src), scale, offset, stream);
}

bool Normalize(Tensor<CHW, C1, F32>& dst, const Tensor<CHW, C1, U8>& src,
    const float scale, const float offset, cudaStream_t stream) {
    return NormalizeC1U8Impl(dst, src, scale, offset, stream);
}

bool Normalize(Tensor<NCHW, C1, F32>& dst, const Tensor<NCHW, C1, U8>& src,
    const float scale, const float offset, cudaStream_t stream) {
    return NormalizeC1Batch(dst, const_cast<Tensor<NCHW, C1, U8> &>(src), scale, offset, stream);
}

bool Normalize(Tensor<CHW, C1, F32>& dst, const Tensor<CHW, C1, U16>& src,
    const float scale, const float offset, cudaStream_t stream) {
    return NormalizeC1U16Impl(dst, src, scale, offset, stream);
}

bool Normalize(Tensor<NCHW, C1, F32>& dst, const Tensor<NCHW, C1, U16>& src,
    const float scale, const float offset, cudaStream_t stream) {
    return NormalizeC1Batch(dst, const_cast<Tensor<NCHW, C1, U16> &>(src), scale, offset, stream);
}

bool Normalize(Tensor<CHW, C1, F32>& dst, const Tensor<CHW, C1, F32>& src,
    const float scale, const float offset, cudaStream_t stream) {
    return NormalizeC1F32Impl(dst, src, scale, offset, stream);
}

bool Normalize(Tensor<NCHW, C1, F32>& dst, const Tensor<NCHW, C1, F32>& src,
    const float scale, const float offset,  cudaStream_t stream) {
    return NormalizeC1Batch(dst, const_cast<Tensor<NCHW, C1, F32> &>(src), scale, offset, stream);
}

}  // namespace tensor_ops
}  // namespace cvcore
