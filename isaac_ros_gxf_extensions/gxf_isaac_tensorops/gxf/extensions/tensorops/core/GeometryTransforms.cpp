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
#include "nppi_data_exchange_and_initialization.h"
#include "nppi_geometry_transforms.h"

namespace cvcore {
namespace tensor_ops {
namespace {

static NppiInterpolationMode GetNppiInterpolationMode(InterpolationType type) {
    if (type == INTERP_NEAREST) {
        return NPPI_INTER_NN;
    } else if (type == INTERP_LINEAR) {
        return NPPI_INTER_LINEAR;
    } else if (type == INTERP_CUBIC_BSPLINE) {
        return NPPI_INTER_CUBIC2P_BSPLINE;
    } else if (type == INTERP_CUBIC_CATMULLROM) {
        return NPPI_INTER_CUBIC2P_CATMULLROM;
    } else {
        throw std::runtime_error("invalid resizing interpolation mode");
    }
}

static BBox GetScaledROI(int srcW, int srcH, int dstW, int dstH) {
    if (srcW * dstH >= dstW * srcH) {
        int bboxH   = static_cast<int>((static_cast<double>(srcH) / srcW) * dstW);
        int offsetH = (dstH - bboxH) / 2;
        return {0, offsetH, dstW, offsetH + bboxH};
    } else {
        int bboxW   = static_cast<int>((static_cast<double>(srcW) / srcH) * dstH);
        int offsetW = (dstW - bboxW) / 2;
        return {offsetW, 0, offsetW + bboxW, dstH};
    }
}

static bool CheckValidROI(const BBox& roi, int width, int height) {
    if ((roi.xmin >= 0 && roi.xmin < roi.xmax) &&
        (roi.ymin >= 0 && roi.ymin < roi.ymax) &&
        (roi.ymax <= height) &&
        (roi.xmax <= width)) {
        return true;
    } else {
        return false;
    }
}

template<TensorLayout TL>
bool FillBufferC1U8Impl(Tensor<TL, C1, U8>& dst, const Npp8u value, cudaStream_t stream) {
    if (dst.isCPU()) {
        return false;
    }
    NppStatus status = nppiSet_8u_C1R_Ctx(value, static_cast<Npp8u *>(dst.getData()),
                          dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp8u),
                          {static_cast<int>(dst.getWidth()), static_cast<int>(dst.getHeight())},
                          GetNppStreamContext(stream));
    if (status != NPP_SUCCESS) {
        return false;
    }
    return true;
}

template<TensorLayout TL>
bool FillBufferC1U16Impl(Tensor<TL, C1, U16>& dst, const Npp16u value, cudaStream_t stream) {
    if (dst.isCPU()) {
        return false;
    }
    NppStatus status = nppiSet_16u_C1R_Ctx(value, static_cast<Npp16u *>(dst.getData()),
                           dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp16u),
                           {static_cast<int>(dst.getWidth()), static_cast<int>(dst.getHeight())},
                           GetNppStreamContext(stream));
    if (status != NPP_SUCCESS) {
        return false;
    }
    return true;
}

template<TensorLayout TL>
bool FillBufferC1F32Impl(Tensor<TL, C1, F32>& dst, const Npp32f value, cudaStream_t stream) {
    if (dst.isCPU()) {
        return false;
    }
    NppStatus status = nppiSet_32f_C1R_Ctx(value, static_cast<Npp32f *>(dst.getData()),
                           dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f),
                           {static_cast<int>(dst.getWidth()), static_cast<int>(dst.getHeight())},
                           GetNppStreamContext(stream));
    if (status != NPP_SUCCESS) {
        return false;
    }
    return true;
}

static bool FillBuffer(Tensor<HWC, C1, U8>& dst, const Npp8u value, cudaStream_t stream = 0) {
    return FillBufferC1U8Impl(dst, value, stream);
}

static bool FillBuffer(Tensor<HWC, C1, U16>& dst, const Npp16u value, cudaStream_t stream = 0) {
    return FillBufferC1U16Impl(dst, value, stream);
}

static bool FillBuffer(Tensor<HWC, C1, F32>& dst, const Npp32f value, cudaStream_t stream = 0) {
    return FillBufferC1F32Impl(dst, value, stream);
}

static bool FillBuffer(Tensor<CHW, C1, U8>& dst, const Npp8u value, cudaStream_t stream = 0) {
    return FillBufferC1U8Impl(dst, value, stream);
}

static bool FillBuffer(Tensor<CHW, C1, U16>& dst, const Npp16u value, cudaStream_t stream = 0) {
    return FillBufferC1U16Impl(dst, value, stream);
}

static bool FillBuffer(Tensor<CHW, C1, F32>& dst, const Npp32f value, cudaStream_t stream = 0) {
    return FillBufferC1F32Impl(dst, value, stream);
}

static bool FillBuffer(Tensor<HWC, C3, U8>& dst, const Npp8u value, cudaStream_t stream = 0) {
    if (dst.isCPU()) {
        return false;
    }
    const Npp8u padding[3] = {value, value, value};
    NppStatus status = nppiSet_8u_C3R_Ctx(padding, static_cast<Npp8u *>(dst.getData()),
                           dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp8u),
                           {static_cast<int>(dst.getWidth()), static_cast<int>(dst.getHeight())},
                           GetNppStreamContext(stream));
    if (status != NPP_SUCCESS) {
        return false;
    }
    return true;
}

static bool FillBuffer(Tensor<HWC, C3, U16>& dst, const Npp16u value, cudaStream_t stream = 0) {
    if (dst.isCPU()) {
        return false;
    }
    const Npp16u padding[3] = {value, value, value};
    NppStatus status = nppiSet_16u_C3R_Ctx(padding, static_cast<Npp16u *>(dst.getData()),
                           dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp16u),
                           {static_cast<int>(dst.getWidth()), static_cast<int>(dst.getHeight())},
                           GetNppStreamContext(stream));
    if (status != NPP_SUCCESS) {
        return false;
    }
    return true;
}

static bool FillBuffer(Tensor<HWC, C3, F32>& dst, const Npp32f value, cudaStream_t stream = 0) {
    if (dst.isCPU()) {
        return false;
    }
    const Npp32f padding[3] = {value, value, value};
    NppStatus status = nppiSet_32f_C3R_Ctx(padding, static_cast<Npp32f *>(dst.getData()),
                           dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f),
                           {static_cast<int>(dst.getWidth()), static_cast<int>(dst.getHeight())},
                           GetNppStreamContext(stream));
    if (status != NPP_SUCCESS) {
        return false;
    }
    return true;
}

template<TensorLayout TL>
bool CropAndResizeC1U8Impl(Tensor<TL, C1, U8>& dst, const Tensor<TL, C1, U8>& src,
    const BBox& dstROI, const BBox& srcROI, InterpolationType type, cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }
    if (!CheckValidROI(dstROI, dst.getWidth(), dst.getHeight())) {
        return false;
    }
    if (!CheckValidROI(srcROI, src.getWidth(), src.getHeight())) {
        return false;
    }

    NppStatus status = nppiResizeSqrPixel_8u_C1R_Ctx(
        static_cast<const Npp8u *>(src.getData() + srcROI.ymin *
            src.getStride(TensorDimension::HEIGHT) + srcROI.xmin),
        {srcROI.xmax - srcROI.xmin, srcROI.ymax - srcROI.ymin},
        src.getStride(TensorDimension::HEIGHT) * sizeof(Npp8u),
        {0, 0, srcROI.xmax - srcROI.xmin, srcROI.ymax - srcROI.ymin},
        static_cast<Npp8u *>(dst.getData() + dstROI.ymin *
            dst.getStride(TensorDimension::HEIGHT) + dstROI.xmin),
        dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp8u),
        {0, 0, dstROI.xmax - dstROI.xmin, dstROI.ymax - dstROI.ymin},
        static_cast<double>(dstROI.xmax - dstROI.xmin) /
            static_cast<double>(srcROI.xmax - srcROI.xmin),
        static_cast<double>(dstROI.ymax - dstROI.ymin) /
            static_cast<double>(srcROI.ymax - srcROI.ymin),
        0.0, 0.0, GetNppiInterpolationMode(type),
        GetNppStreamContext(stream));
    if (status != NPP_SUCCESS) {
        return false;
    }
    return true;
}

template<TensorLayout TL>
bool CropAndResizeC1U16Impl(Tensor<TL, C1, U16>& dst, const Tensor<TL, C1, U16>& src,
    const BBox& dstROI, const BBox& srcROI, InterpolationType type, cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }
    if (!CheckValidROI(dstROI, dst.getWidth(), dst.getHeight())) {
        return false;
    }
    if (!CheckValidROI(srcROI, src.getWidth(), src.getHeight())) {
        return false;
    }

    NppStatus status = nppiResizeSqrPixel_16u_C1R_Ctx(
        static_cast<const Npp16u *>(src.getData() + srcROI.ymin *
            src.getStride(TensorDimension::HEIGHT) + srcROI.xmin),
        {srcROI.xmax - srcROI.xmin, srcROI.ymax - srcROI.ymin},
        src.getStride(TensorDimension::HEIGHT) * sizeof(Npp16u),
        {0, 0, srcROI.xmax - srcROI.xmin, srcROI.ymax - srcROI.ymin},
        static_cast<Npp16u *>(dst.getData() + dstROI.ymin *
            dst.getStride(TensorDimension::HEIGHT) + dstROI.xmin),
        dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp16u),
        {0, 0, dstROI.xmax - dstROI.xmin, dstROI.ymax - dstROI.ymin},
        static_cast<double>(dstROI.xmax - dstROI.xmin) /
            static_cast<double>(srcROI.xmax - srcROI.xmin),
        static_cast<double>(dstROI.ymax - dstROI.ymin) /
            static_cast<double>(srcROI.ymax - srcROI.ymin),
        0.0, 0.0, GetNppiInterpolationMode(type),
        GetNppStreamContext(stream));
    if (status != NPP_SUCCESS) {
        return false;
    }
    return true;
}

template<TensorLayout TL>
bool CropAndResizeC1F32Impl(Tensor<TL, C1, F32>& dst, const Tensor<TL, C1, F32>& src,
    const BBox& dstROI, const BBox& srcROI, InterpolationType type, cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }
    if (!CheckValidROI(dstROI, dst.getWidth(), dst.getHeight())) {
        return false;
    }
    if (!CheckValidROI(srcROI, src.getWidth(), src.getHeight())) {
        return false;
    }

    NppStatus status = nppiResizeSqrPixel_32f_C1R_Ctx(
        static_cast<const Npp32f *>(src.getData() + srcROI.ymin *
            src.getStride(TensorDimension::HEIGHT) + srcROI.xmin),
        {srcROI.xmax - srcROI.xmin, srcROI.ymax - srcROI.ymin},
        src.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f),
        {0, 0, srcROI.xmax - srcROI.xmin, srcROI.ymax - srcROI.ymin},
        static_cast<Npp32f *>(dst.getData() + dstROI.ymin *
            dst.getStride(TensorDimension::HEIGHT) + dstROI.xmin),
        dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f),
        {0, 0, dstROI.xmax - dstROI.xmin, dstROI.ymax - dstROI.ymin},
        static_cast<double>(dstROI.xmax - dstROI.xmin) /
            static_cast<double>(srcROI.xmax - srcROI.xmin),
        static_cast<double>(dstROI.ymax - dstROI.ymin) /
            static_cast<double>(srcROI.ymax - srcROI.ymin),
        0.0, 0.0, GetNppiInterpolationMode(type),
        GetNppStreamContext(stream));
    if (status != NPP_SUCCESS) {
        return false;
    }
    return true;
}

template<TensorLayout TL, ChannelCount CC, ChannelType CT>
bool ResizeImpl(Tensor<TL, CC, CT>& dst, const Tensor<TL, CC, CT>& src,
    bool keep_aspect_ratio, InterpolationType type,
    cudaStream_t stream) {
    bool ret = true;
    const BBox dstROI = keep_aspect_ratio
                            ? GetScaledROI(src.getWidth(), src.getHeight(),
                              dst.getWidth(), dst.getHeight())
                            : BBox{0, 0, static_cast<int>(dst.getWidth()),
                              static_cast<int>(dst.getHeight())};
    if (keep_aspect_ratio) {
        ret = FillBuffer(dst, 0, stream);
        if (!ret) {
            return false;
        }
    }
    ret = CropAndResize(dst, src, dstROI, {0, 0, static_cast<int>(src.getWidth()),
        static_cast<int>(src.getHeight())}, type, stream);
    return ret;
}

template<TensorLayout TL>
bool CropC1U8Impl(Tensor<TL, C1, U8>& dst, const Tensor<TL, C1, U8>& src,
    const BBox& srcROI, cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }
    if (!CheckValidROI(srcROI, src.getWidth(), src.getHeight())) {
        return false;
    }
    if ((srcROI.xmax - srcROI.xmin) != static_cast<int>(dst.getWidth()) ||
       (srcROI.ymax - srcROI.ymin) != static_cast<int>(dst.getHeight())) {
        return false;
    }

    NppStatus status = nppiCopy_8u_C1R_Ctx(
        static_cast<const Npp8u *>(src.getData() + srcROI.ymin *
            src.getStride(TensorDimension::HEIGHT) + srcROI.xmin),
        src.getStride(TensorDimension::HEIGHT) * sizeof(Npp8u),
        static_cast<Npp8u *>(dst.getData()),
        dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp8u),
        {srcROI.xmax - srcROI.xmin, srcROI.ymax - srcROI.ymin},
        GetNppStreamContext(stream));
    if (status != NPP_SUCCESS) {
        return false;
    }
    return true;
}

template<TensorLayout TL>
bool CropC1U16Impl(Tensor<TL, C1, U16>& dst, const Tensor<TL, C1, U16>& src,
    const BBox& srcROI, cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }
    if (!CheckValidROI(srcROI, src.getWidth(), src.getHeight())) {
        return false;
    }
    if ((srcROI.xmax - srcROI.xmin) != static_cast<int>(dst.getWidth()) ||
       (srcROI.ymax - srcROI.ymin) != static_cast<int>(dst.getHeight())) {
        return false;
    }

    NppStatus status = nppiCopy_16u_C1R_Ctx(
        static_cast<const Npp16u *>(src.getData() + srcROI.ymin *
            src.getStride(TensorDimension::HEIGHT) + srcROI.xmin),
        src.getStride(TensorDimension::HEIGHT) * sizeof(Npp16u),
        static_cast<Npp16u *>(dst.getData()),
        dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp16u),
        {srcROI.xmax - srcROI.xmin, srcROI.ymax - srcROI.ymin},
        GetNppStreamContext(stream));
    if (status != NPP_SUCCESS) {
        return false;
    }
    return true;
}

template<TensorLayout TL>
bool CropC1F32Impl(Tensor<TL, C1, F32>& dst, const Tensor<TL, C1, F32>& src,
    const BBox& srcROI, cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }
    if (!CheckValidROI(srcROI, src.getWidth(), src.getHeight())) {
        return false;
    }
    if ((srcROI.xmax - srcROI.xmin) != static_cast<int>(dst.getWidth()) ||
       (srcROI.ymax - srcROI.ymin) != static_cast<int>(dst.getHeight())) {
        return false;
    }

    NppStatus status = nppiCopy_32f_C1R_Ctx(
        static_cast<const Npp32f *>(src.getData() + srcROI.ymin *
            src.getStride(TensorDimension::HEIGHT) + srcROI.xmin),
        src.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f),
        static_cast<Npp32f *>(dst.getData()),
        dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f),
        {srcROI.xmax - srcROI.xmin, srcROI.ymax - srcROI.ymin},
        GetNppStreamContext(stream));
    if (status != NPP_SUCCESS) {
        return false;
    }
    return true;
}

template<ChannelCount CC, ChannelType CT>
bool ResizeBatch(Tensor<NHWC, CC, CT>& dst, Tensor<NHWC, CC, CT>& src,
    bool keep_aspect_ratio, InterpolationType type, cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }
    if (src.getDepth() != dst.getDepth()) {
        return false;
    }

    for (size_t i = 0; i < src.getDepth(); i++) {
        size_t offsetSrc = i * src.getStride(TensorDimension::DEPTH);
        size_t offsetDst = i * dst.getStride(TensorDimension::DEPTH);
        Tensor<HWC, CC, CT> srcTmp(src.getWidth(), src.getHeight(),
                                   src.getStride(TensorDimension::HEIGHT) * GetChannelSize(CT),
                                   src.getData() + offsetSrc, false);
        Tensor<HWC, CC, CT> dstTmp(dst.getWidth(), dst.getHeight(),
                                   dst.getStride(TensorDimension::HEIGHT) * GetChannelSize(CT),
                                   dst.getData() + offsetDst, false);
        bool ret = Resize(dstTmp, srcTmp, keep_aspect_ratio, type, stream);
        if (!ret) {
            return false;
        }
    }
    return true;
}

}  // anonymous namespace

bool Resize(Tensor<HWC, C1, U8>& dst, const Tensor<HWC, C1, U8>& src,
    bool keep_aspect_ratio, InterpolationType type, cudaStream_t stream) {
    return ResizeImpl(dst, src, keep_aspect_ratio, type, stream);
}

bool Resize(Tensor<NHWC, C1, U8>& dst, const Tensor<NHWC, C1, U8>& src,
    bool keep_aspect_ratio, InterpolationType type, cudaStream_t stream) {
    return ResizeBatch(dst, const_cast<Tensor<NHWC, C1, U8> &>(src),
        keep_aspect_ratio, type, stream);
}

bool CropAndResize(Tensor<HWC, C1, U8>& dst, const Tensor<HWC, C1, U8>& src,
    const BBox& dstROI, const BBox& srcROI, InterpolationType type, cudaStream_t stream) {
    return CropAndResizeC1U8Impl(dst, src, dstROI, srcROI, type, stream);
}

bool CropAndResize(Tensor<HWC, C1, U8>& dst, const Tensor<HWC, C1, U8>& src,
    const BBox& srcROI, InterpolationType type, cudaStream_t stream) {
    return CropAndResize(dst, src, {0, 0, static_cast<int>(dst.getWidth()),
        static_cast<int>(dst.getHeight())}, srcROI, type, stream);
}

bool Resize(Tensor<HWC, C1, U16>& dst, const Tensor<HWC, C1, U16>& src,
    bool keep_aspect_ratio, InterpolationType type, cudaStream_t stream) {
    return ResizeImpl(dst, src, keep_aspect_ratio, type, stream);
}

bool Resize(Tensor<NHWC, C1, U16>& dst, const Tensor<NHWC, C1, U16>& src,
    bool keep_aspect_ratio, InterpolationType type, cudaStream_t stream) {
    return ResizeBatch(dst, const_cast<Tensor<NHWC, C1, U16> &>(src),
        keep_aspect_ratio, type, stream);
}

bool CropAndResize(Tensor<HWC, C1, U16>& dst, const Tensor<HWC, C1, U16>& src,
    const BBox& dstROI, const BBox& srcROI, InterpolationType type, cudaStream_t stream) {
    return CropAndResizeC1U16Impl(dst, src, dstROI, srcROI, type, stream);
}

bool CropAndResize(Tensor<HWC, C1, U16>& dst, const Tensor<HWC, C1, U16>& src,
    const BBox& srcROI, InterpolationType type, cudaStream_t stream) {
    return CropAndResize(dst, src, {0, 0, static_cast<int>(dst.getWidth()),
        static_cast<int>(dst.getHeight())}, srcROI, type, stream);
}

bool Resize(Tensor<HWC, C1, F32>& dst, const Tensor<HWC, C1, F32>& src,
    bool keep_aspect_ratio, InterpolationType type, cudaStream_t stream) {
    return ResizeImpl(dst, src, keep_aspect_ratio, type, stream);
}

bool Resize(Tensor<NHWC, C1, F32>& dst, const Tensor<NHWC, C1, F32>& src,
    bool keep_aspect_ratio, InterpolationType type, cudaStream_t stream) {
    return ResizeBatch(dst, const_cast<Tensor<NHWC, C1, F32> &>(src),
        keep_aspect_ratio, type, stream);
}

bool CropAndResize(Tensor<HWC, C1, F32>& dst, const Tensor<HWC, C1, F32>& src,
    const BBox& dstROI, const BBox& srcROI, InterpolationType type, cudaStream_t stream) {
    return CropAndResizeC1F32Impl(dst, src, dstROI, srcROI, type, stream);
}

bool CropAndResize(Tensor<HWC, C1, F32>& dst, const Tensor<HWC, C1, F32>& src,
    const BBox& srcROI, InterpolationType type, cudaStream_t stream) {
    return CropAndResize(dst, src, {0, 0, static_cast<int>(dst.getWidth()),
        static_cast<int>(dst.getHeight())}, srcROI, type, stream);
}

bool Resize(Tensor<CHW, C1, U8>& dst, const Tensor<CHW, C1, U8>& src,
    bool keep_aspect_ratio, InterpolationType type, cudaStream_t stream) {
    return ResizeImpl(dst, src, keep_aspect_ratio, type, stream);
}

bool CropAndResize(Tensor<CHW, C1, U8>& dst, const Tensor<CHW, C1, U8>& src,
    const BBox& dstROI, const BBox& srcROI, InterpolationType type, cudaStream_t stream) {
    return CropAndResizeC1U8Impl(dst, src, dstROI, srcROI, type, stream);
}

bool CropAndResize(Tensor<CHW, C1, U8>& dst, const Tensor<CHW, C1, U8>& src,
    const BBox& srcROI, InterpolationType type, cudaStream_t stream) {
    return CropAndResize(dst, src, {0, 0, static_cast<int>(dst.getWidth()),
        static_cast<int>(dst.getHeight())}, srcROI, type, stream);
}

bool Resize(Tensor<CHW, C1, U16>& dst, const Tensor<CHW, C1, U16>& src,
    bool keep_aspect_ratio, InterpolationType type, cudaStream_t stream) {
    return ResizeImpl(dst, src, keep_aspect_ratio, type, stream);
}

bool CropAndResize(Tensor<CHW, C1, U16>& dst, const Tensor<CHW, C1, U16>& src,
    const BBox& dstROI, const BBox& srcROI, InterpolationType type, cudaStream_t stream) {
    return CropAndResizeC1U16Impl(dst, src, dstROI, srcROI, type, stream);
}

bool CropAndResize(Tensor<CHW, C1, U16>& dst, const Tensor<CHW, C1, U16>& src,
    const BBox& srcROI, InterpolationType type, cudaStream_t stream) {
    return CropAndResize(dst, src, {0, 0, static_cast<int>(dst.getWidth()),
        static_cast<int>(dst.getHeight())}, srcROI, type, stream);
}

bool Resize(Tensor<CHW, C1, F32>& dst, const Tensor<CHW, C1, F32>& src,
    bool keep_aspect_ratio, InterpolationType type, cudaStream_t stream) {
    return ResizeImpl(dst, src, keep_aspect_ratio, type, stream);
}

bool CropAndResize(Tensor<CHW, C1, F32>& dst, const Tensor<CHW, C1, F32>& src,
    const BBox& dstROI, const BBox& srcROI, InterpolationType type, cudaStream_t stream) {
    return CropAndResizeC1F32Impl(dst, src, dstROI, srcROI, type, stream);
}

bool CropAndResize(Tensor<CHW, C1, F32>& dst, const Tensor<CHW, C1, F32>& src,
    const BBox& srcROI, InterpolationType type, cudaStream_t stream) {
    return CropAndResize(dst, src, {0, 0, static_cast<int>(dst.getWidth()),
        static_cast<int>(dst.getHeight())}, srcROI, type, stream);
}

bool Resize(Tensor<HWC, C3, U8>& dst, const Tensor<HWC, C3, U8>& src,
    bool keep_aspect_ratio, InterpolationType type, cudaStream_t stream) {
    return ResizeImpl(dst, src, keep_aspect_ratio, type, stream);
}

bool Resize(Tensor<NHWC, C3, U8>& dst, const Tensor<NHWC, C3, U8>& src,
    bool keep_aspect_ratio, InterpolationType type, cudaStream_t stream) {
    return ResizeBatch(dst, const_cast<Tensor<NHWC, C3, U8> &>(src),
        keep_aspect_ratio, type, stream);
}

bool CropAndResize(Tensor<HWC, C3, U8>& dst, const Tensor<HWC, C3, U8>& src,
    const BBox& dstROI, const BBox& srcROI, InterpolationType type, cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }
    if (!CheckValidROI(dstROI, dst.getWidth(), dst.getHeight())) {
        return false;
    }
    if (!CheckValidROI(srcROI, src.getWidth(), src.getHeight())) {
        return false;
    }

    NppStatus status = nppiResizeSqrPixel_8u_C3R_Ctx(
        static_cast<const Npp8u *>(src.getData() + srcROI.ymin *
            src.getStride(TensorDimension::HEIGHT) +
            srcROI.xmin * src.getChannelCount()),
        {srcROI.xmax - srcROI.xmin, srcROI.ymax - srcROI.ymin},
        src.getStride(TensorDimension::HEIGHT) * sizeof(Npp8u),
        {0, 0, srcROI.xmax - srcROI.xmin, srcROI.ymax - srcROI.ymin},
        static_cast<Npp8u *>(dst.getData() + dstROI.ymin *
            dst.getStride(TensorDimension::HEIGHT) +
            dstROI.xmin * dst.getChannelCount()),
        dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp8u),
        {0, 0, dstROI.xmax - dstROI.xmin, dstROI.ymax - dstROI.ymin},
        static_cast<double>(dstROI.xmax - dstROI.xmin) /
            static_cast<double>(srcROI.xmax - srcROI.xmin),
        static_cast<double>(dstROI.ymax - dstROI.ymin) /
            static_cast<double>(srcROI.ymax - srcROI.ymin),
        0.0, 0.0, GetNppiInterpolationMode(type),
        GetNppStreamContext(stream));
    if (status != NPP_SUCCESS) {
        return false;
    }
    return true;
}

bool CropAndResize(Tensor<HWC, C3, U8>& dst, const Tensor<HWC, C3, U8>& src,
    const BBox& srcROI, InterpolationType type, cudaStream_t stream) {
    return CropAndResize(dst, src, {0, 0, static_cast<int>(dst.getWidth()),
        static_cast<int>(dst.getHeight())}, srcROI, type, stream);
}

bool Resize(Tensor<HWC, C3, U16>& dst, const Tensor<HWC, C3, U16>& src,
    bool keep_aspect_ratio, InterpolationType type, cudaStream_t stream) {
    return ResizeImpl(dst, src, keep_aspect_ratio, type, stream);
}

bool Resize(Tensor<NHWC, C3, U16>& dst, const Tensor<NHWC, C3, U16>& src,
    bool keep_aspect_ratio, InterpolationType type, cudaStream_t stream) {
    return ResizeBatch(dst, const_cast<Tensor<NHWC, C3, U16> &>(src),
        keep_aspect_ratio, type, stream);
}

bool CropAndResize(Tensor<HWC, C3, U16>& dst, const Tensor<HWC, C3, U16>& src,
    const BBox& dstROI, const BBox& srcROI, InterpolationType type, cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }
    if (!CheckValidROI(dstROI, dst.getWidth(), dst.getHeight())) {
        return false;
    }
    if (!CheckValidROI(srcROI, src.getWidth(), src.getHeight())) {
        return false;
    }

    NppStatus status = nppiResizeSqrPixel_16u_C3R_Ctx(
        static_cast<const Npp16u *>(src.getData() + srcROI.ymin *
            src.getStride(TensorDimension::HEIGHT) +
            srcROI.xmin * src.getChannelCount()),
        {srcROI.xmax - srcROI.xmin, srcROI.ymax - srcROI.ymin},
        src.getStride(TensorDimension::HEIGHT) * sizeof(Npp16u),
        {0, 0, srcROI.xmax - srcROI.xmin, srcROI.ymax - srcROI.ymin},
        static_cast<Npp16u *>(dst.getData() + dstROI.ymin *
            dst.getStride(TensorDimension::HEIGHT) +
            dstROI.xmin * dst.getChannelCount()),
        dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp16u),
        {0, 0, dstROI.xmax - dstROI.xmin, dstROI.ymax - dstROI.ymin},
        static_cast<double>(dstROI.xmax - dstROI.xmin) /
            static_cast<double>(srcROI.xmax - srcROI.xmin),
        static_cast<double>(dstROI.ymax - dstROI.ymin) /
            static_cast<double>(srcROI.ymax - srcROI.ymin),
        0.0, 0.0, GetNppiInterpolationMode(type),
        GetNppStreamContext(stream));
    if (status != NPP_SUCCESS) {
        return false;
    }
    return true;
}

bool CropAndResize(Tensor<HWC, C3, U16>& dst, const Tensor<HWC, C3, U16>& src,
    const BBox& srcROI, InterpolationType type, cudaStream_t stream) {
    return CropAndResize(dst, src, {0, 0, static_cast<int>(dst.getWidth()),
        static_cast<int>(dst.getHeight())}, srcROI, type, stream);
}

bool Resize(Tensor<HWC, C3, F32>& dst, const Tensor<HWC, C3, F32>& src,
    bool keep_aspect_ratio, InterpolationType type, cudaStream_t stream) {
    return ResizeImpl(dst, src, keep_aspect_ratio, type, stream);
}

bool Resize(Tensor<NHWC, C3, F32>& dst, const Tensor<NHWC, C3, F32>& src,
    bool keep_aspect_ratio, InterpolationType type, cudaStream_t stream) {
    return ResizeBatch(dst, const_cast<Tensor<NHWC, C3, F32> &>(src),
        keep_aspect_ratio, type, stream);
}

bool CropAndResize(Tensor<HWC, C3, F32>& dst, const Tensor<HWC, C3, F32>& src,
    const BBox& dstROI, const BBox& srcROI, InterpolationType type, cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }
    if (!CheckValidROI(dstROI, dst.getWidth(), dst.getHeight())) {
        return false;
    }
    if (!CheckValidROI(srcROI, src.getWidth(), src.getHeight())) {
        return false;
    }

    NppStatus status = nppiResizeSqrPixel_32f_C3R_Ctx(
        static_cast<const Npp32f *>(src.getData() + srcROI.ymin *
            src.getStride(TensorDimension::HEIGHT) +
            srcROI.xmin * src.getChannelCount()),
        {srcROI.xmax - srcROI.xmin, srcROI.ymax - srcROI.ymin},
        src.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f),
        {0, 0, srcROI.xmax - srcROI.xmin, srcROI.ymax - srcROI.ymin},
        static_cast<Npp32f *>(dst.getData() + dstROI.ymin *
            dst.getStride(TensorDimension::HEIGHT) +
            dstROI.xmin * dst.getChannelCount()),
        dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f),
        {0, 0, dstROI.xmax - dstROI.xmin, dstROI.ymax - dstROI.ymin},
        static_cast<double>(dstROI.xmax - dstROI.xmin) /
            static_cast<double>(srcROI.xmax - srcROI.xmin),
        static_cast<double>(dstROI.ymax - dstROI.ymin) /
            static_cast<double>(srcROI.ymax - srcROI.ymin),
        0.0, 0.0, GetNppiInterpolationMode(type),
        GetNppStreamContext(stream));
    if (status != NPP_SUCCESS) {
        return false;
    }
    return true;
}

bool CropAndResize(Tensor<HWC, C3, F32>& dst, const Tensor<HWC, C3, F32>& src,
    const BBox& srcROI, InterpolationType type, cudaStream_t stream) {
    return CropAndResize(dst, src, {0, 0, static_cast<int>(dst.getWidth()),
        static_cast<int>(dst.getHeight())}, srcROI, type, stream);
}

bool Crop(Tensor<HWC, C1, U8>& dst, const Tensor<HWC, C1, U8>& src,
    const BBox& srcROI, cudaStream_t stream) {
    return CropC1U8Impl(dst, src, srcROI, stream);
}

bool Crop(Tensor<HWC, C1, U16>& dst, const Tensor<HWC, C1, U16>& src,
    const BBox& srcROI, cudaStream_t stream) {
    return CropC1U16Impl(dst, src, srcROI, stream);
}

bool Crop(Tensor<HWC, C1, F32>& dst, const Tensor<HWC, C1, F32>& src,
    const BBox& srcROI, cudaStream_t stream) {
    return CropC1F32Impl(dst, src, srcROI, stream);
}

bool Crop(Tensor<CHW, C1, U8>& dst, const Tensor<CHW, C1, U8>& src,
    const BBox& srcROI, cudaStream_t stream) {
    return CropC1U8Impl(dst, src, srcROI, stream);
}

bool Crop(Tensor<CHW, C1, U16>& dst, const Tensor<CHW, C1, U16>& src,
    const BBox& srcROI, cudaStream_t stream) {
    return CropC1U16Impl(dst, src, srcROI, stream);
}

bool Crop(Tensor<CHW, C1, F32>& dst, const Tensor<CHW, C1, F32>& src,
    const BBox& srcROI, cudaStream_t stream) {
    return CropC1F32Impl(dst, src, srcROI, stream);
}

bool Crop(Tensor<HWC, C3, U8>& dst, const Tensor<HWC, C3, U8>& src,
    const BBox& srcROI, cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }
    if (!CheckValidROI(srcROI, src.getWidth(), src.getHeight())) {
        return false;
    }
    if ((srcROI.xmax - srcROI.xmin) != static_cast<int>(dst.getWidth()) ||
       (srcROI.ymax - srcROI.ymin) != static_cast<int>(dst.getHeight())) {
        return false;
    }

    NppStatus status = nppiCopy_8u_C3R_Ctx(
        static_cast<const Npp8u *>(src.getData() + srcROI.ymin *
            src.getStride(TensorDimension::HEIGHT) +
            srcROI.xmin * src.getChannelCount()),
        src.getStride(TensorDimension::HEIGHT) * sizeof(Npp8u),
        static_cast<Npp8u *>(dst.getData()),
        dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp8u),
        {srcROI.xmax - srcROI.xmin, srcROI.ymax - srcROI.ymin},
        GetNppStreamContext(stream));
    if (status != NPP_SUCCESS) {
        return false;
    }
    return true;
}

bool Crop(Tensor<HWC, C3, U16>& dst, const Tensor<HWC, C3, U16>& src,
    const BBox& srcROI, cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }
    if (!CheckValidROI(srcROI, src.getWidth(), src.getHeight())) {
        return false;
    }
    if ((srcROI.xmax - srcROI.xmin) != static_cast<int>(dst.getWidth()) ||
       (srcROI.ymax - srcROI.ymin) != static_cast<int>(dst.getHeight())) {
        return false;
    }

    NppStatus status = nppiCopy_16u_C3R_Ctx(
        static_cast<const Npp16u *>(src.getData() + srcROI.ymin *
            src.getStride(TensorDimension::HEIGHT) +
            srcROI.xmin * src.getChannelCount()),
        src.getStride(TensorDimension::HEIGHT) * sizeof(Npp16u),
        static_cast<Npp16u *>(dst.getData()),
        dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp16u),
        {srcROI.xmax - srcROI.xmin, srcROI.ymax - srcROI.ymin},
        GetNppStreamContext(stream));
    if (status != NPP_SUCCESS) {
        return false;
    }
    return true;
}

bool Crop(Tensor<HWC, C3, F32>& dst, const Tensor<HWC, C3, F32>& src,
    const BBox& srcROI, cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }
    if (!CheckValidROI(srcROI, src.getWidth(), src.getHeight())) {
        return false;
    }
    if ((srcROI.xmax - srcROI.xmin) != static_cast<int>(dst.getWidth()) ||
       (srcROI.ymax - srcROI.ymin) != static_cast<int>(dst.getHeight())) {
        return false;
    }

    NppStatus status = nppiCopy_32f_C3R_Ctx(
        static_cast<const Npp32f *>(src.getData() + srcROI.ymin *
            src.getStride(TensorDimension::HEIGHT) +
                                    srcROI.xmin * src.getChannelCount()),
        src.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f),
        static_cast<Npp32f *>(dst.getData()),
        dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f),
        {srcROI.xmax - srcROI.xmin, srcROI.ymax - srcROI.ymin},
        GetNppStreamContext(stream));
    if (status != NPP_SUCCESS) {
        return false;
    }
    return true;
}

bool WarpPerspective(Tensor<HWC, C1, U8>& dst, const Tensor<HWC, C1, U8>& src,
    const double coeffs[3][3], InterpolationType type, cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }

    NppStatus status = nppiWarpPerspective_8u_C1R_Ctx(
        static_cast<const Npp8u *>(src.getData()), {static_cast<int>(src.getWidth()),
        static_cast<int>(src.getHeight())},
        src.getStride(TensorDimension::HEIGHT) * sizeof(Npp8u),
        {0, 0, static_cast<int>(src.getWidth()), static_cast<int>(src.getHeight())},
        static_cast<Npp8u *>(dst.getData()),
        dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp8u),
        {0, 0, static_cast<int>(dst.getWidth()), static_cast<int>(dst.getHeight())}, coeffs,
        GetNppiInterpolationMode(type),
        GetNppStreamContext(stream));
    if (status != NPP_SUCCESS) {
        return false;
    }
    return true;
}

bool WarpPerspective(Tensor<HWC, C1, U16>& dst, const Tensor<HWC, C1, U16>& src,
    const double coeffs[3][3], InterpolationType type, cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }

    NppStatus status = nppiWarpPerspective_16u_C1R_Ctx(
        static_cast<const Npp16u *>(src.getData()),
        {static_cast<int>(src.getWidth()), static_cast<int>(src.getHeight())},
        src.getStride(TensorDimension::HEIGHT) * sizeof(Npp16u),
        {0, 0, static_cast<int>(src.getWidth()), static_cast<int>(src.getHeight())},
        static_cast<Npp16u *>(dst.getData()),
        dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp16u),
        {0, 0, static_cast<int>(dst.getWidth()),
        static_cast<int>(dst.getHeight())}, coeffs, GetNppiInterpolationMode(type),
        GetNppStreamContext(stream));
    if (status != NPP_SUCCESS) {
        return false;
    }
    return true;
}

bool WarpPerspective(Tensor<HWC, C1, F32>& dst, const Tensor<HWC, C1, F32>& src,
    const double coeffs[3][3], InterpolationType type, cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }

    NppStatus status = nppiWarpPerspective_32f_C1R_Ctx(
        static_cast<const Npp32f *>(src.getData()),
        {static_cast<int>(src.getWidth()), static_cast<int>(src.getHeight())},
        src.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f),
        {0, 0, static_cast<int>(src.getWidth()), static_cast<int>(src.getHeight())},
        static_cast<Npp32f *>(dst.getData()),
        dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f),
        {0, 0, static_cast<int>(dst.getWidth()),
        static_cast<int>(dst.getHeight())},
        coeffs, GetNppiInterpolationMode(type),
        GetNppStreamContext(stream));
    if (status != NPP_SUCCESS) {
        return false;
    }
    return true;
}

bool WarpPerspective(Tensor<HWC, C3, U8>& dst, const Tensor<HWC, C3, U8>& src,
    const double coeffs[3][3], InterpolationType type, cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }

    NppStatus status = nppiWarpPerspective_8u_C3R_Ctx(
        static_cast<const Npp8u *>(src.getData()), {static_cast<int>(src.getWidth()),
        static_cast<int>(src.getHeight())},
        src.getStride(TensorDimension::HEIGHT) * sizeof(Npp8u),
        {0, 0, static_cast<int>(src.getWidth()),
        static_cast<int>(src.getHeight())},
        static_cast<Npp8u *>(dst.getData()),
        dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp8u),
        {0, 0, static_cast<int>(dst.getWidth()),
        static_cast<int>(dst.getHeight())}, coeffs,
        GetNppiInterpolationMode(type),
        GetNppStreamContext(stream));
    if (status != NPP_SUCCESS) {
        return false;
    }
    return true;
}

bool WarpPerspective(Tensor<HWC, C3, U16>& dst, const Tensor<HWC, C3, U16>& src,
    const double coeffs[3][3], InterpolationType type, cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }

    NppStatus status = nppiWarpPerspective_16u_C3R_Ctx(
        static_cast<const Npp16u *>(src.getData()),
        {static_cast<int>(src.getWidth()),
        static_cast<int>(src.getHeight())},
        src.getStride(TensorDimension::HEIGHT) * sizeof(Npp16u),
        {0, 0, static_cast<int>(src.getWidth()),
        static_cast<int>(src.getHeight())},
        static_cast<Npp16u *>(dst.getData()),
        dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp16u),
        {0, 0, static_cast<int>(dst.getWidth()),
        static_cast<int>(dst.getHeight())},
        coeffs, GetNppiInterpolationMode(type),
        GetNppStreamContext(stream));
    if (status != NPP_SUCCESS) {
        return false;
    }
    return true;
}

bool WarpPerspective(Tensor<HWC, C3, F32>& dst, const Tensor<HWC, C3, F32>& src,
    const double coeffs[3][3], InterpolationType type, cudaStream_t stream) {
    // src and dst must be GPU tensors
    if (src.isCPU() || dst.isCPU()) {
        return false;
    }

    NppStatus status = nppiWarpPerspective_32f_C3R_Ctx(
        static_cast<const Npp32f *>(src.getData()),
        {static_cast<int>(src.getWidth()),
        static_cast<int>(src.getHeight())},
        src.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f),
        {0, 0, static_cast<int>(src.getWidth()),
        static_cast<int>(src.getHeight())},
        static_cast<Npp32f *>(dst.getData()),
        dst.getStride(TensorDimension::HEIGHT) * sizeof(Npp32f),
        {0, 0, static_cast<int>(dst.getWidth()),
        static_cast<int>(dst.getHeight())}, coeffs,
        GetNppiInterpolationMode(type),
        GetNppStreamContext(stream));
    if (status != NPP_SUCCESS) {
        return false;
    }
    return true;
}

}  // namespace tensor_ops
}  // namespace cvcore
