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

#include <cstdint>
#include <stdexcept>

#include "extensions/tensorops/core/ImageUtils.h"
#include "extensions/tensorops/core/Memory.h"
#include "extensions/tensorops/core/NppUtils.h"
#include "nppi_arithmetic_and_logical_operations.h"
#include "nppi_color_conversion.h"
#include "nppi_data_exchange_and_initialization.h"
namespace cvcore {
namespace tensor_ops {

const float BGR2GRAY_COEFFS[3] = {0.114f, 0.587f, 0.299f};
const float RGB2GRAY_COEFFS[3] = {0.299f, 0.587f, 0.114f};

template<ChannelCount CC1, ChannelCount CC2, ChannelType CT>
bool ConvertColorFormatBatch(Tensor<NHWC, CC1, CT>& dst,
    Tensor<NHWC, CC2, CT>& src, ColorConversionType type,
    cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU() || (src.getDepth() != dst.getDepth())) {
        return false;
    }
    for (size_t i = 0; i < src.getDepth(); i++) {
        size_t offsetSrc = i * src.getStride(TensorDimension::DEPTH);
        size_t offsetDst = i * dst.getStride(TensorDimension::DEPTH);
        Tensor<HWC, CC1, CT> srcTmp(src.getWidth(), src.getHeight(),
                                    src.getStride(TensorDimension::HEIGHT) * GetChannelSize(CT),
                                    src.getData() + offsetSrc, false);
        Tensor<HWC, CC2, CT> dstTmp(dst.getWidth(), dst.getHeight(),
                                    dst.getStride(TensorDimension::HEIGHT) * GetChannelSize(CT),
                                    dst.getData() + offsetDst, false);
        bool ret = ConvertColorFormat(dstTmp, srcTmp, type, stream);
        if (!ret) {
            return false;
        }
    }
    return true;
}

template<ChannelCount CC, ChannelType CT>
bool InterleavedToPlanarBatch(Tensor<NCHW, CC, CT>& dst,
    Tensor<NHWC, CC, CT>& src, cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU() || (src.getDepth() != dst.getDepth())) {
        return false;
    }

    for (size_t i = 0; i < src.getDepth(); i++) {
        size_t offsetSrc = i * src.getStride(TensorDimension::DEPTH);
        size_t offsetDst = i * dst.getStride(TensorDimension::DEPTH);
        Tensor<HWC, CC, CT> srcTmp(src.getWidth(), src.getHeight(),
                                   src.getStride(TensorDimension::HEIGHT) * GetChannelSize(CT),
                                   src.getData() + offsetSrc, false);
        Tensor<CHW, CC, CT> dstTmp(dst.getWidth(), dst.getHeight(),
                                   dst.getStride(TensorDimension::HEIGHT) * GetChannelSize(CT),
                                   dst.getData() + offsetDst, false);
        bool ret = InterleavedToPlanar(dstTmp, srcTmp, stream);
        if (!ret) {
            return false;
        }
    }
    return true;
}

bool ConvertColorFormat(Tensor<HWC, C3, U8>& dst, const Tensor<HWC, C3, U8>& src,
    ColorConversionType type, cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }
    if ((src.getWidth() != dst.getWidth()) || (src.getHeight() != dst.getHeight())) {
        return false;
    }
    if (type == BGR2RGB || type == RGB2BGR) {
        const int order[3] = {2, 1, 0};
        NppStatus status = nppiSwapChannels_8u_C3R_Ctx(
            static_cast<const Npp8u *>(src.getData()),
            src.getStride(TensorDimension::HEIGHT) * sizeof(Npp8u),
            static_cast<Npp8u *>(dst.getData()),
            dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp8u),
            {static_cast<int>(src.getWidth()), static_cast<int>(src.getHeight())},
            order, GetNppStreamContext(stream));
        if (status != NPP_SUCCESS) {
            return false;
        }
    } else {
      return false;
    }
    return true;
}

bool ConvertColorFormat(Tensor<NHWC, C3, U8>& dst, const Tensor<NHWC, C3, U8>& src,
    ColorConversionType type, cudaStream_t stream) {
    return ConvertColorFormatBatch(dst, const_cast<Tensor<NHWC, C3, U8> &>(src), type, stream);
}

bool ConvertColorFormat(Tensor<HWC, C3, U16>& dst, const Tensor<HWC, C3, U16>& src,
    ColorConversionType type, cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }
    if ((src.getWidth() != dst.getWidth()) || (src.getHeight() != dst.getHeight())) {
        return false;
    }
    if (type == BGR2RGB || type == RGB2BGR) {
        const int order[3] = {2, 1, 0};
        NppStatus status   = nppiSwapChannels_16u_C3R_Ctx(
            static_cast<const Npp16u *>(src.getData()),
            src.getStride(TensorDimension::HEIGHT) * sizeof(Npp16u),
            static_cast<Npp16u *>(dst.getData()),
            dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp16u),
            {static_cast<int>(src.getWidth()), static_cast<int>(src.getHeight())},
            order, GetNppStreamContext(stream));
        if (status != NPP_SUCCESS) {
            return false;
        }
    } else {
        return false;
    }
    return true;
}

bool ConvertColorFormat(Tensor<NHWC, C3, U16>& dst, const Tensor<NHWC, C3, U16>& src,
    ColorConversionType type, cudaStream_t stream) {
    return ConvertColorFormatBatch(dst, const_cast<Tensor<NHWC, C3, U16> &>(src), type, stream);
}

bool ConvertColorFormat(Tensor<HWC, C3, F32>& dst, const Tensor<HWC, C3, F32>& src,
    ColorConversionType type, cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }
    if ((src.getWidth() != dst.getWidth()) || (src.getHeight() != dst.getHeight())) {
        return false;
    }

    if (type == BGR2RGB || type == RGB2BGR) {
        const int order[3] = {2, 1, 0};
        NppStatus status   = nppiSwapChannels_32f_C3R_Ctx(
            static_cast<const Npp32f *>(src.getData()),
            src.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f),
            static_cast<Npp32f *>(dst.getData()),
            dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f),
            {static_cast<int>(src.getWidth()), static_cast<int>(src.getHeight())}, order,
            GetNppStreamContext(stream));
        if (status != NPP_SUCCESS) {
            return false;
        }
    } else {
        return false;
    }
    return true;
}

bool ConvertColorFormat(Tensor<NHWC, C3, F32>& dst, const Tensor<NHWC, C3, F32>& src,
    ColorConversionType type, cudaStream_t stream) {
    return ConvertColorFormatBatch(dst, const_cast<Tensor<NHWC, C3, F32> &>(src), type, stream);
}

bool ConvertColorFormat(Tensor<HWC, C1, U8>& dst, const Tensor<HWC, C3, U8>& src,
    ColorConversionType type, cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }
    if ((src.getWidth() != dst.getWidth()) || (src.getHeight() != dst.getHeight())) {
        return false;
    }

    if (type == BGR2GRAY || type == RGB2GRAY) {
        NppStatus status = nppiColorToGray_8u_C3C1R_Ctx(
            static_cast<const Npp8u *>(src.getData()),
            src.getStride(TensorDimension::HEIGHT) * sizeof(Npp8u),
            static_cast<Npp8u *>(dst.getData()),
            dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp8u),
            {static_cast<int>(src.getWidth()), static_cast<int>(src.getHeight())},
            type == BGR2GRAY ? BGR2GRAY_COEFFS : RGB2GRAY_COEFFS,
            GetNppStreamContext(stream));
        if (status != NPP_SUCCESS) {
            return false;
        }
    } else {
        return false;
    }
    return true;
}

bool ConvertColorFormat(Tensor<HWC, C1, U16>& dst, const Tensor<HWC, C3, U16>& src,
    ColorConversionType type, cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }
    if ((src.getWidth() != dst.getWidth()) || (src.getHeight() != dst.getHeight())) {
        return false;
    }

    if (type == BGR2GRAY || type == RGB2GRAY) {
        NppStatus status = nppiColorToGray_16u_C3C1R_Ctx(
            static_cast<const Npp16u *>(src.getData()),
            src.getStride(TensorDimension::HEIGHT) * sizeof(Npp16u),
            static_cast<Npp16u *>(dst.getData()),
            dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp16u),
            {static_cast<int>(src.getWidth()), static_cast<int>(src.getHeight())},
            type == BGR2GRAY ? BGR2GRAY_COEFFS : RGB2GRAY_COEFFS,
            GetNppStreamContext(stream));
        if (status != NPP_SUCCESS) {
            return false;
        }
    } else {
        return false;
    }
    return true;
}

bool ConvertColorFormat(Tensor<HWC, C1, F32>& dst, const Tensor<HWC, C3, F32>& src,
    ColorConversionType type, cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }
    if ((src.getWidth() != dst.getWidth()) || (src.getHeight() != dst.getHeight())) {
        return false;
    }

    if (type == BGR2GRAY || type == RGB2GRAY) {
        NppStatus status = nppiColorToGray_32f_C3C1R_Ctx(
            static_cast<const Npp32f *>(src.getData()),
            src.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f),
            static_cast<Npp32f *>(dst.getData()),
            dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f),
            {static_cast<int>(src.getWidth()), static_cast<int>(src.getHeight())},
            type == BGR2GRAY ? BGR2GRAY_COEFFS : RGB2GRAY_COEFFS,
            GetNppStreamContext(stream));
        if (status != NPP_SUCCESS) {
            return false;
        }
    } else {
        return false;
    }
    return true;
}

bool ConvertColorFormat(Tensor<HWC, C3, U8>& dst, const Tensor<HWC, C1, U8>& src,
    ColorConversionType type, cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }
    if ((src.getWidth() != dst.getWidth()) || (src.getHeight() != dst.getHeight())) {
        return false;
    }

    if (type == GRAY2BGR || type == GRAY2RGB) {
        NppStatus status = nppiDup_8u_C1C3R_Ctx(
            static_cast<const Npp8u *>(src.getData()),
            src.getStride(TensorDimension::HEIGHT) * sizeof(Npp8u),
            static_cast<Npp8u *>(dst.getData()),
            dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp8u),
            {static_cast<int>(src.getWidth()), static_cast<int>(src.getHeight())},
            GetNppStreamContext(stream));
        if (status != NPP_SUCCESS) {
            return false;
        }
    } else {
        return false;
    }
    return true;
}

bool ConvertColorFormat(Tensor<HWC, C3, U16>& dst, const Tensor<HWC, C1, U16>& src,
    ColorConversionType type, cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }
    if ((src.getWidth() != dst.getWidth()) || (src.getHeight() != dst.getHeight())) {
        return false;
    }

    if (type == GRAY2BGR || type == GRAY2RGB) {
        NppStatus status = nppiDup_16u_C1C3R_Ctx(
            static_cast<const Npp16u *>(src.getData()),
            src.getStride(TensorDimension::HEIGHT) * sizeof(Npp16u),
            static_cast<Npp16u *>(dst.getData()),
            dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp16u),
            {static_cast<int>(src.getWidth()), static_cast<int>(src.getHeight())},
            GetNppStreamContext(stream));
        if (status != NPP_SUCCESS) {
            return false;
        }
    } else {
        return false;
    }
    return true;
}

bool ConvertColorFormat(Tensor<HWC, C3, F32>& dst, const Tensor<HWC, C1, F32>& src,
    ColorConversionType type, cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }
    if ((src.getWidth() != dst.getWidth()) || (src.getHeight() != dst.getHeight())) {
        return false;
    }

    if (type == GRAY2BGR || type == GRAY2RGB) {
        NppStatus status = nppiDup_32f_C1C3R_Ctx(
            static_cast<const Npp32f *>(src.getData()),
            src.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f),
            static_cast<Npp32f *>(dst.getData()),
            dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f),
            {static_cast<int>(src.getWidth()), static_cast<int>(src.getHeight())},
            GetNppStreamContext(stream));
        if (status != NPP_SUCCESS) {
            return false;
        }
    } else {
        return false;
    }
    return true;
}

bool ConvertBitDepth(Tensor<HWC, C1, U8>& dst, Tensor<HWC, C1, F32>& src,
    const float scale, cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }
    if ((src.getWidth() != dst.getWidth()) || (src.getHeight() != dst.getHeight())) {
        return false;
    }

    const NppiSize srcSize = {static_cast<int>(src.getWidth()), static_cast<int>(src.getHeight())};

    NppStreamContext streamContext = GetNppStreamContext(stream);

    NppStatus status =
        nppiMulC_32f_C1IR_Ctx(static_cast<Npp32f>(scale), static_cast<Npp32f *>(src.getData()),
            src.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f),
            srcSize, streamContext);
    if (status != NPP_SUCCESS) {
        return false;
    }

    status = nppiConvert_32f8u_C1R_Ctx(
        static_cast<const Npp32f *>(src.getData()),
        src.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f),
        static_cast<Npp8u *>(dst.getData()),
        dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp8u),
        {static_cast<int>(src.getWidth()), static_cast<int>(src.getHeight())},
        NPP_RND_FINANCIAL, streamContext);
    if (status != NPP_SUCCESS) {
        return false;
    }

    return true;
}

bool ConvertBitDepth(Tensor<NHWC, C1, U8>& dst, Tensor<NHWC, C1, F32>& src,
    const float scale, cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU() || (src.getDepth() != dst.getDepth())) {
        return false;
    }

    Tensor<HWC, C1, F32> srcTmp(src.getWidth(), src.getDepth() * src.getHeight(),
        src.getStride(TensorDimension::HEIGHT) * GetChannelSize(F32),
        src.getData(), false);
    Tensor<HWC, C1, U8> dstTmp(dst.getWidth(), dst.getDepth() * dst.getHeight(),
        dst.getStride(TensorDimension::HEIGHT) * GetChannelSize(U8), dst.getData(), false);
    return ConvertBitDepth(dstTmp, srcTmp, scale, stream);
}

bool InterleavedToPlanar(Tensor<CHW, C3, U8>& dst,
    const Tensor<HWC, C3, U8>& src, cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }
    if ((src.getWidth() != dst.getWidth()) || (src.getHeight() != dst.getHeight())) {
        return false;
    }

    NppStatus status;
    NppStreamContext streamContext = GetNppStreamContext(stream);

    const size_t offset       = dst.getStride(TensorDimension::HEIGHT) * dst.getHeight();
    Npp8u *const dstBuffer[3] = {dst.getData(), dst.getData() + offset,
        dst.getData() + 2 * offset};
    status = nppiCopy_8u_C3P3R_Ctx(static_cast<const Npp8u *>(src.getData()),
                src.getStride(TensorDimension::HEIGHT) * sizeof(Npp8u), dstBuffer,
                dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp8u),
                {static_cast<int>(src.getWidth()), static_cast<int>(src.getHeight())},
                streamContext);
    if (status != NPP_SUCCESS) {
        return false;
    }
    return true;
}

bool InterleavedToPlanar(Tensor<NCHW, C3, U8>& dst, const Tensor<NHWC, C3, U8>& src,
    cudaStream_t stream) {
    return InterleavedToPlanarBatch(dst, const_cast<Tensor<NHWC, C3, U8> &>(src), stream);
}

bool InterleavedToPlanar(Tensor<CHW, C3, U16>& dst, const Tensor<HWC, C3, U16>& src,
    cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }
    if ((src.getWidth() != dst.getWidth()) || (src.getHeight() != dst.getHeight())) {
        return false;
    }

    NppStatus status;
    NppStreamContext streamContext = GetNppStreamContext(stream);

    const size_t offset        = dst.getStride(TensorDimension::HEIGHT) * dst.getHeight();
    Npp16u *const dstBuffer[3] = {dst.getData(), dst.getData() + offset,
        dst.getData() + 2 * offset};
    status = nppiCopy_16u_C3P3R_Ctx(static_cast<const Npp16u *>(src.getData()),
                src.getStride(TensorDimension::HEIGHT) * sizeof(Npp16u), dstBuffer,
                dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp16u),
                {static_cast<int>(src.getWidth()), static_cast<int>(src.getHeight())},
                streamContext);
    if (status != NPP_SUCCESS) {
        return false;
    }
    return true;
}

bool InterleavedToPlanar(Tensor<NCHW, C3, U16>& dst, const Tensor<NHWC, C3, U16>& src,
    cudaStream_t stream) {
    return InterleavedToPlanarBatch(dst, const_cast<Tensor<NHWC, C3, U16> &>(src), stream);
}

bool InterleavedToPlanar(Tensor<CHW, C3, F32>& dst, const Tensor<HWC, C3, F32>& src,
    cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }
    if ((src.getWidth() != dst.getWidth()) || (src.getHeight() != dst.getHeight())) {
        return false;
    }

    NppStatus status;
    NppStreamContext streamContext = GetNppStreamContext(stream);

    const size_t offset        = dst.getStride(TensorDimension::HEIGHT) * dst.getHeight();
    Npp32f *const dstBuffer[3] = {dst.getData(), dst.getData() + offset,
        dst.getData() + 2 * offset};
    status = nppiCopy_32f_C3P3R_Ctx(static_cast<const Npp32f *>(src.getData()),
                src.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f), dstBuffer,
                dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f),
                {static_cast<int>(src.getWidth()), static_cast<int>(src.getHeight())},
                streamContext);
    if (status != NPP_SUCCESS) {
        return false;
    }
    return true;
}

bool InterleavedToPlanar(Tensor<NCHW, C3, F32>& dst, const Tensor<NHWC, C3, F32>& src,
    cudaStream_t stream) {
    return InterleavedToPlanarBatch(dst, const_cast<Tensor<NHWC, C3, F32> &>(src), stream);
}

bool InterleavedToPlanar(Tensor<CHW, C1, U8>& dst, const Tensor<HWC, C1, U8>& src,
    cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }
    if ((src.getWidth() != dst.getWidth()) || (src.getHeight() != dst.getHeight())) {
        return false;
    }

    Tensor<HWC, C1, U8> tmp(dst.getWidth(), dst.getHeight(),
                            dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp8u),
                            dst.getData(), false);
    Copy(tmp, src, stream);
    return true;
}

bool InterleavedToPlanar(Tensor<NCHW, C1, U8>& dst, const Tensor<NHWC, C1, U8>& src,
    cudaStream_t stream) {
    return InterleavedToPlanarBatch(dst, const_cast<Tensor<NHWC, C1, U8> &>(src), stream);
}

bool InterleavedToPlanar(Tensor<CHW, C1, U16>& dst, const Tensor<HWC, C1, U16>& src,
    cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }
    if ((src.getWidth() != dst.getWidth()) || (src.getHeight() != dst.getHeight())) {
        return false;
    }

    Tensor<HWC, C1, U16> tmp(dst.getWidth(), dst.getHeight(),
        dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp16u),
        dst.getData(), false);
    Copy(tmp, src, stream);
    return true;
}

bool InterleavedToPlanar(Tensor<NCHW, C1, U16>& dst, const Tensor<NHWC, C1, U16>& src,
    cudaStream_t stream) {
    return InterleavedToPlanarBatch(dst, const_cast<Tensor<NHWC, C1, U16> &>(src), stream);
}

bool InterleavedToPlanar(Tensor<CHW, C1, F32>& dst, const Tensor<HWC, C1, F32>& src,
    cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }
    if ((src.getWidth() != dst.getWidth()) || (src.getHeight() != dst.getHeight())) {
        return false;
    }

    Tensor<HWC, C1, F32> tmp(dst.getWidth(), dst.getHeight(),
        dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f),
        dst.getData(), false);
    Copy(tmp, src, stream);
    return true;
}

bool InterleavedToPlanar(Tensor<NCHW, C1, F32>& dst, const Tensor<NHWC, C1, F32>& src,
    cudaStream_t stream) {
    return InterleavedToPlanarBatch(dst, const_cast<Tensor<NHWC, C1, F32> &>(src), stream);
}

}  // namespace tensor_ops
}  // namespace cvcore
