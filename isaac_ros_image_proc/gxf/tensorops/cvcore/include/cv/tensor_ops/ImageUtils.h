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

#ifndef CVCORE_IMAGE_UTILS_H
#define CVCORE_IMAGE_UTILS_H

#include <memory>

#include <cuda_runtime.h>

#include "cv/core/BBox.h"
#include "cv/core/Tensor.h"

namespace cvcore { namespace tensor_ops {

/**
 * An enum.
 * Enum type for color conversion type.
 */
enum ColorConversionType
{
    BGR2RGB,  /**< convert BGR to RGB. */
    RGB2BGR,  /**< convert RGB to BGR. */
    BGR2GRAY, /**< convert BGR to Grayscale. */
    RGB2GRAY, /**< convert RGB to Grayscale. */
    GRAY2BGR, /**< convert Grayscale to BGR. */
    GRAY2RGB, /**< convert Grayscale to RGB. */
};

/**
 * An enum.
 * Enum type for resize interpolation type.
 */
enum InterpolationType
{
    INTERP_NEAREST,         /**< nearest interpolation. */
    INTERP_LINEAR,          /**< linear interpolation. */
    INTERP_CUBIC_BSPLINE,   /**< cubic bspline interpolation. */
    INTERP_CUBIC_CATMULLROM /**< cubic catmullrom interpolation. */
};

/**
 * An enum.
 * Enum type for resize interpolation type.
 */
enum BorderType
{
    BORDER_ZERO,
    BORDER_REPEAT,
    BORDER_REVERSE,
    BORDER_MIRROR
};

// please note the following functions all require GPU Tensors

/**
 * Image resizing for one channel HWC format uint_8 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param type interpolation type.
 * @param keep_aspect_ratio whether to keep aspect ratio.
 * @param stream specified cuda stream.
 */
void Resize(Tensor<HWC, C1, U8> &dst, const Tensor<HWC, C1, U8> &src, bool keep_aspect_ratio = false,
            InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Image batch resizing for one channel NHWC uint_8 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param type interpolation type.
 * @param keep_aspect_ratio whether to keep aspect ratio.
 * @param stream specified cuda stream.
 */
void Resize(Tensor<NHWC, C1, U8> &dst, const Tensor<NHWC, C1, U8> &src, bool keep_aspect_ratio = false,
            InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Image resizing of a region of interest for one channel HWC format uint_8 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param dstROI destination crop region.
 * @param srcROI source crop region.
 * @param type interpolation type.
 * @param stream specified cuda stream.
 */
void CropAndResize(Tensor<HWC, C1, U8> &dst, const Tensor<HWC, C1, U8> &src, const BBox &dstROI, const BBox &srcROI,
                   InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Image resizing of a region of interest for one channel HWC format uint_8 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param srcROI source crop region.
 * @param type interpolation type.
 * @param stream specified cuda stream.
 */
void CropAndResize(Tensor<HWC, C1, U8> &dst, const Tensor<HWC, C1, U8> &src, const BBox &srcROI,
                   InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Image resizing for one channel HWC format uint_16 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param type interpolation type.
 * @param keep_aspect_ratio whether to keep aspect ratio.
 * @param stream specified cuda stream.
 */
void Resize(Tensor<HWC, C1, U16> &dst, const Tensor<HWC, C1, U16> &src, bool keep_aspect_ratio = false,
            InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Image batch resizing for one channel HWC format uint_16 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param type interpolation type.
 * @param keep_aspect_ratio whether to keep aspect ratio.
 * @param stream specified cuda stream.
 */
void Resize(Tensor<NHWC, C1, U16> &dst, const Tensor<NHWC, C1, U16> &src, bool keep_aspect_ratio = false,
            InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Image resizing of a region of interest for one channel HWC format uint_16 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param dstROI destination crop region.
 * @param srcROI source crop region.
 * @param type interpolation type.
 * @param stream specified cuda stream.
 */
void CropAndResize(Tensor<HWC, C1, U16> &dst, const Tensor<HWC, C1, U16> &src, const BBox &dstROI, const BBox &srcROI,
                   InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Image resizing of a region of interest for one channel HWC format uint_16 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param srcROI source crop region.
 * @param type interpolation type.
 * @param stream specified cuda stream.
 */
void CropAndResize(Tensor<HWC, C1, U16> &dst, const Tensor<HWC, C1, U16> &src, const BBox &srcROI,
                   InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Image resizing for one channel HWC format FP32 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param type interpolation type.
 * @param keep_aspect_ratio whether to keep aspect ratio.
 * @param stream specified cuda stream.
 */
void Resize(Tensor<HWC, C1, F32> &dst, const Tensor<HWC, C1, F32> &src, bool keep_aspect_ratio = false,
            InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Image batch resizing for one channel HWC format FP32 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param type interpolation type.
 * @param keep_aspect_ratio whether to keep aspect ratio.
 * @param stream specified cuda stream.
 */
void Resize(Tensor<NHWC, C1, F32> &dst, const Tensor<NHWC, C1, F32> &src, bool keep_aspect_ratio = false,
            InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Image resizing of a region of interest for one channel HWC format FP32 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param dstROI destination crop region.
 * @param srcROI source crop region.
 * @param type interpolation type.
 * @param stream specified cuda stream.
 */
void CropAndResize(Tensor<HWC, C1, F32> &dst, const Tensor<HWC, C1, F32> &src, const BBox &dstROI, const BBox &srcROI,
                   InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Image resizing of a region of interest for one channel HWC format FP32 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param srcROI source crop region.
 * @param type interpolation type.
 * @param stream specified cuda stream.
 */
void CropAndResize(Tensor<HWC, C1, F32> &dst, const Tensor<HWC, C1, F32> &src, const BBox &srcROI,
                   InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Image resizing for one channel CHW format uint_8 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param type interpolation type.
 * @param keep_aspect_ratio whether to keep aspect ratio.
 * @param stream specified cuda stream.
 */
void Resize(Tensor<CHW, C1, U8> &dst, const Tensor<CHW, C1, U8> &src, bool keep_aspect_ratio = false,
            InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Image resizing of a region of interest for one channel CHW format uint_8 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param dstROI destination crop region.
 * @param srcROI source crop region.
 * @param type interpolation type.
 * @param stream specified cuda stream.
 */
void CropAndResize(Tensor<CHW, C1, U8> &dst, const Tensor<CHW, C1, U8> &src, const BBox &dstROI, const BBox &srcROI,
                   InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Image resizing of a region of interest for one channel CHW format uint_8 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param srcROI source crop region.
 * @param type interpolation type.
 * @param stream specified cuda stream.
 */
void CropAndResize(Tensor<CHW, C1, U8> &dst, const Tensor<CHW, C1, U8> &src, const BBox &srcROI,
                   InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Image resizing for one channel CHW format uint_16 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param type interpolation type.
 * @param keep_aspect_ratio whether to keep aspect ratio.
 * @param stream specified cuda stream.
 */
void Resize(Tensor<CHW, C1, U16> &dst, const Tensor<CHW, C1, U16> &src, bool keep_aspect_ratio = false,
            InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Image resizing of a region of interest for one channel CHW format uint_16 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param dstROI destination crop region.
 * @param srcROI source crop region.
 * @param type interpolation type.
 * @param stream specified cuda stream.
 */
void CropAndResize(Tensor<CHW, C1, U16> &dst, const Tensor<CHW, C1, U16> &src, const BBox &dstROI, const BBox &srcROI,
                   InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Image resizing of a region of interest for one channel CHW format uint_16 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param srcROI source crop region.
 * @param type interpolation type.
 * @param stream specified cuda stream.
 */
void CropAndResize(Tensor<CHW, C1, U16> &dst, const Tensor<CHW, C1, U16> &src, const BBox &srcROI,
                   InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Image resizing for one channel CHW format FP32 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param type interpolation type.
 * @param keep_aspect_ratio whether to keep aspect ratio.
 * @param stream specified cuda stream.
 */
void Resize(Tensor<CHW, C1, F32> &dst, const Tensor<CHW, C1, F32> &src, bool keep_aspect_ratio = false,
            InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Image resizing of a region of interest for one channel CHW format FP32 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param dstROI destination crop region.
 * @param srcROI source crop region.
 * @param type interpolation type.
 * @param stream specified cuda stream.
 */
void CropAndResize(Tensor<CHW, C1, F32> &dst, const Tensor<CHW, C1, F32> &src, const BBox &dstROI, const BBox &srcROI,
                   InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Image resizing of a region of interest for one channel CHW format FP32 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param srcROI source crop region.
 * @param type interpolation type.
 * @param stream specified cuda stream.
 */
void CropAndResize(Tensor<CHW, C1, F32> &dst, const Tensor<CHW, C1, F32> &src, const BBox &srcROI,
                   InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Image resizing for three channels interleaved uint_8 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param type interpolation type.
 * @param keep_aspect_ratio whether to keep aspect ratio.
 * @param stream specified cuda stream.
 */
void Resize(Tensor<HWC, C3, U8> &dst, const Tensor<HWC, C3, U8> &src, bool keep_aspect_ratio = false,
            InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Image batch resizing for three channels interleaved uint_8 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param type interpolation type.
 * @param keep_aspect_ratio whether to keep aspect ratio.
 * @param stream specified cuda stream.
 */
void Resize(Tensor<NHWC, C3, U8> &dst, const Tensor<NHWC, C3, U8> &src, bool keep_aspect_ratio = false,
            InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Image resizing of a region of intesrest for three channel HWC format uint_8 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param dstROI destination crop region.
 * @param srcROI source crop region.
 * @param type interpolation type.
 * @param stream specified cuda stream.
 */
void CropAndResize(Tensor<HWC, C3, U8> &dst, const Tensor<HWC, C3, U8> &src, const BBox &dstROI, const BBox &srcROI,
                   InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Image resizing of a region of intesrest for three channel HWC format uint_8 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param srcROI source crop region.
 * @param type interpolation type.
 * @param stream specified cuda stream.
 */
void CropAndResize(Tensor<HWC, C3, U8> &dst, const Tensor<HWC, C3, U8> &src, const BBox &srcROI,
                   InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Image resizing for three channels interleaved uint_16 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param type interpolation type.
 * @param keep_aspect_ratio whether to keep aspect ratio.
 * @param stream specified cuda stream.
 */
void Resize(Tensor<HWC, C3, U16> &dst, const Tensor<HWC, C3, U16> &src, bool keep_aspect_ratio = false,
            InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Image batch resizing for three channels interleaved uint_16 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param type interpolation type.
 * @param keep_aspect_ratio whether to keep aspect ratio.
 * @param stream specified cuda stream.
 */
void Resize(Tensor<NHWC, C3, U16> &dst, const Tensor<NHWC, C3, U16> &src, bool keep_aspect_ratio = false,
            InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Image resizing of a region of intesrest for three channel HWC format uint_16 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param dstROI destination crop region.
 * @param srcROI source crop region.
 * @param type interpolation type.
 * @param stream specified cuda stream.
 */
void CropAndResize(Tensor<HWC, C3, U16> &dst, const Tensor<HWC, C3, U16> &src, const BBox &dstROI, const BBox &srcROI,
                   InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Image resizing of a region of intesrest for three channel HWC format uint_16 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param srcROI source crop region.
 * @param type interpolation type.
 * @param stream specified cuda stream.
 */
void CropAndResize(Tensor<HWC, C3, U16> &dst, const Tensor<HWC, C3, U16> &src, const BBox &srcROI,
                   InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Image resizing for three channels interleaved float type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param type interpolation type.
 * @param keep_aspect_ratio whether to keep aspect ratio.
 * @param stream specified cuda stream.
 */
void Resize(Tensor<HWC, C3, F32> &dst, const Tensor<HWC, C3, F32> &src, bool keep_aspect_ratio = false,
            InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Image batch resizing for three channels interleaved float type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param type interpolation type.
 * @param keep_aspect_ratio whether to keep aspect ratio.
 * @param stream specified cuda stream.
 */
void Resize(Tensor<NHWC, C3, F32> &dst, const Tensor<NHWC, C3, F32> &src, bool keep_aspect_ratio = false,
            InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Image resizing of a region of intesrest for three channel HWC format float type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param dstROI destination crop region.
 * @param srcROI source crop region.
 * @param type interpolation type.
 * @param stream specified cuda stream.
 */
void CropAndResize(Tensor<HWC, C3, F32> &dst, const Tensor<HWC, C3, F32> &src, const BBox &dstROI, const BBox &srcROI,
                   InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Image resizing of a region of intesrest for three channel HWC format float type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param srcROI source crop region.
 * @param type interpolation type.
 * @param stream specified cuda stream.
 */
void CropAndResize(Tensor<HWC, C3, F32> &dst, const Tensor<HWC, C3, F32> &src, const BBox &srcROI,
                   InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Crop a region of interest for one channel HWC format uint_8 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param srcROI source crop region.
 * @param stream specified cuda stream.
 */
void Crop(Tensor<HWC, C1, U8> &dst, const Tensor<HWC, C1, U8> &src, const BBox &srcROI, cudaStream_t stream = 0);

/**
 * Crop a region of interest for one channel HWC format uint_16 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param srcROI source crop region.
 * @param stream specified cuda stream.
 */
void Crop(Tensor<HWC, C1, U16> &dst, const Tensor<HWC, C1, U16> &src, const BBox &srcROI, cudaStream_t stream = 0);

/**
 * Crop a region of interest for one channel HWC format float type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param srcROI source crop region.
 * @param stream specified cuda stream.
 */
void Crop(Tensor<HWC, C1, F32> &dst, const Tensor<HWC, C1, F32> &src, const BBox &srcROI, cudaStream_t stream = 0);

/**
 * Crop a region of interest for one channel CHW format uint_8 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param srcROI source crop region.
 * @param stream specified cuda stream.
 */
void Crop(Tensor<CHW, C1, U8> &dst, const Tensor<CHW, C1, U8> &src, const BBox &srcROI, cudaStream_t stream = 0);

/**
 * Crop a region of interest for one channel CHW format uint_16 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param srcROI source crop region.
 * @param stream specified cuda stream.
 */
void Crop(Tensor<CHW, C1, U16> &dst, const Tensor<CHW, C1, U16> &src, const BBox &srcROI, cudaStream_t stream = 0);

/**
 * Crop a region of interest for one channel CHW format float type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param srcROI source crop region.
 * @param stream specified cuda stream.
 */
void Crop(Tensor<CHW, C1, F32> &dst, const Tensor<CHW, C1, F32> &src, const BBox &srcROI, cudaStream_t stream = 0);

/**
 * Crop a region of interest for three channels HWC format uint_8 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param srcROI source crop region.
 * @param stream specified cuda stream.
 */
void Crop(Tensor<HWC, C3, U8> &dst, const Tensor<HWC, C3, U8> &src, const BBox &srcROI, cudaStream_t stream = 0);

/**
 * Crop a region of interest for three channels HWC format uint_16 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param srcROI source crop region.
 * @param stream specified cuda stream.
 */
void Crop(Tensor<HWC, C3, U16> &dst, const Tensor<HWC, C3, U16> &src, const BBox &srcROI, cudaStream_t stream = 0);

/**
 * Crop a region of interest for three channels HWC format float type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param srcROI source crop region.
 * @param stream specified cuda stream.
 */
void Crop(Tensor<HWC, C3, F32> &dst, const Tensor<HWC, C3, F32> &src, const BBox &srcROI, cudaStream_t stream = 0);

/**
 * Apply a perspective transformation to one channel HWC format uint_8 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param coeffs 3x3 transformation matrix.
 * @param type interpolation type.
 * @param stream specified cuda stream.
 */
void WarpPerspective(Tensor<HWC, C1, U8> &dst, const Tensor<HWC, C1, U8> &src, const double coeffs[3][3],
                     InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Apply a perspective transformation to one channel HWC format uint_16 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param coeffs 3x3 transformation matrix.
 * @param type interpolation type.
 * @param stream specified cuda stream.
 */
void WarpPerspective(Tensor<HWC, C1, U16> &dst, const Tensor<HWC, C1, U16> &src, const double coeffs[3][3],
                     InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Apply a perspective transformation to one channel HWC format float type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param coeffs 3x3 transformation matrix.
 * @param type interpolation type.
 * @param stream specified cuda stream.
 */
void WarpPerspective(Tensor<HWC, C1, F32> &dst, const Tensor<HWC, C1, F32> &src, const double coeffs[3][3],
                     InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Apply a perspective transformation to three channels HWC format uint_8 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param coeffs 3x3 transformation matrix.
 * @param type interpolation type.
 * @param stream specified cuda stream.
 */
void WarpPerspective(Tensor<HWC, C3, U8> &dst, const Tensor<HWC, C3, U8> &src, const double coeffs[3][3],
                     InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Apply a perspective transformation to three channels HWC format uint_16 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param coeffs 3x3 transformation matrix.
 * @param type interpolation type.
 * @param stream specified cuda stream.
 */
void WarpPerspective(Tensor<HWC, C3, U16> &dst, const Tensor<HWC, C3, U16> &src, const double coeffs[3][3],
                     InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/**
 * Apply a perspective transformation to three channels HWC format float type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param coeffs 3x3 transformation matrix.
 * @param type interpolation type.
 * @param stream specified cuda stream.
 */
void WarpPerspective(Tensor<HWC, C3, F32> &dst, const Tensor<HWC, C3, F32> &src, const double coeffs[3][3],
                     InterpolationType type = INTERP_LINEAR, cudaStream_t stream = 0);

/** Color conversion between two three channels interleaved uint_8 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param type color conversion type.
 * @param stream specified cuda stream.
 */
void ConvertColorFormat(Tensor<HWC, C3, U8> &dst, const Tensor<HWC, C3, U8> &src, ColorConversionType type,
                        cudaStream_t stream = 0);

/** Batch color conversion between three channels interleaved uint_8 type Tensors.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param type color conversion type.
 * @param stream specified cuda stream.
 */
void ConvertColorFormat(Tensor<NHWC, C3, U8> &dst, const Tensor<NHWC, C3, U8> &src, ColorConversionType type,
                        cudaStream_t stream = 0);

/** Color conversion between two three channels interleaved uint_16 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param type color conversion type.
 * @param stream specified cuda stream.
 */
void ConvertColorFormat(Tensor<HWC, C3, U16> &dst, const Tensor<HWC, C3, U16> &src, ColorConversionType type,
                        cudaStream_t stream = 0);

/** Batch color conversion between three channels interleaved uint_16 type Tensors.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param type color conversion type.
 * @param stream specified cuda stream.
 */
void ConvertColorFormat(Tensor<NHWC, C3, U16> &dst, const Tensor<NHWC, C3, U16> &src, ColorConversionType type,
                        cudaStream_t stream = 0);

/** Color conversion between two three channels interleaved float type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param type color conversion type.
 * @param stream specified cuda stream.
 */
void ConvertColorFormat(Tensor<HWC, C3, F32> &dst, const Tensor<HWC, C3, F32> &src, ColorConversionType type,
                        cudaStream_t stream = 0);

/** Batch color conversion between three channels interleaved float type Tensors.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param type color conversion type.
 * @param stream specified cuda stream.
 */
void ConvertColorFormat(Tensor<NHWC, C3, F32> &dst, const Tensor<NHWC, C3, F32> &src, ColorConversionType type,
                        cudaStream_t stream = 0);

/** Color conversion from three channels interleaved uint_8 type Tensor to one channel Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param type color conversion type.
 * @param stream specified cuda stream.
 */
void ConvertColorFormat(Tensor<HWC, C1, U8> &dst, const Tensor<HWC, C3, U8> &src, ColorConversionType type,
                        cudaStream_t stream = 0);

/** Color conversion from three channels interleaved uint_16 type Tensor to one channel Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param type color conversion type.
 * @param stream specified cuda stream.
 */
void ConvertColorFormat(Tensor<HWC, C1, U16> &dst, const Tensor<HWC, C3, U16> &src, ColorConversionType type,
                        cudaStream_t stream = 0);

/** Color conversion from three channels interleaved float type Tensor to one channel Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param type color conversion type.
 * @param stream specified cuda stream.
 */
void ConvertColorFormat(Tensor<HWC, C1, F32> &dst, const Tensor<HWC, C3, F32> &src, ColorConversionType type,
                        cudaStream_t stream = 0);

/** Color conversion from one channel interleaved uint_8 type Tensor to three channels Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param type color conversion type.
 * @param stream specified cuda stream.
 */
void ConvertColorFormat(Tensor<HWC, C3, U8> &dst, const Tensor<HWC, C1, U8> &src, ColorConversionType type,
                        cudaStream_t stream = 0);

/** Color conversion from one channel interleaved uint_16 type Tensor to three channels Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param type color conversion type.
 * @param stream specified cuda stream.
 */
void ConvertColorFormat(Tensor<HWC, C3, U16> &dst, const Tensor<HWC, C1, U16> &src, ColorConversionType type,
                        cudaStream_t stream = 0);

/** Color conversion from one channel interleaved float type Tensor to three channels Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param type color conversion type.
 * @param stream specified cuda stream.
 */
void ConvertColorFormat(Tensor<HWC, C3, F32> &dst, const Tensor<HWC, C1, F32> &src, ColorConversionType type,
                        cudaStream_t stream = 0);

/** Convert bit depth from F32 to U8 of a single channel channel Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param scale multiply the pixel values by a factor.
 * @param stream specified cuda stream.
 */
void ConvertBitDepth(Tensor<HWC, C1, U8> &dst, Tensor<HWC, C1, F32> &src, const float scale, cudaStream_t stream = 0);

/** Convert bit depth from F32 to U8 of a N * single channel Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param scale multiply the pixel values by a factor.
 * @param stream specified cuda stream.
 */
void ConvertBitDepth(Tensor<NHWC, C1, U8> &dst, Tensor<NHWC, C1, F32> &src, const float scale, cudaStream_t stream = 0);

/**
 * Normalization for three channels interleaved uint8_t type Tensor.
 * Each element x will be transformed to (x + offset) * scale
 * @param dst destination tensor.
 * @param src source tensor.
 * @param scale scaling factor for normalization.
 * @param offset offset constant for normalization.
 * @stream specified cuda stream.
 */
void Normalize(Tensor<HWC, C3, F32> &dst, const Tensor<HWC, C3, U8> &src, const float scale[3], const float offset[3],
               cudaStream_t stream = 0);

/**
 * Batch normalization for three channels interleaved uint8_t type Tensor.
 * Each element x will be transformed to (x + offset) * scale
 * @param dst destination tensor.
 * @param src source tensor.
 * @param scale scaling factor for normalization.
 * @param offset offset constant for normalization.
 * @stream specified cuda stream.
 */
void Normalize(Tensor<NHWC, C3, F32> &dst, const Tensor<NHWC, C3, U8> &src, const float scale[3], const float offset[3],
               cudaStream_t stream = 0);

/**
 * Normalization for three channels interleaved uint16_t type Tensor.
 * Each element x will be transformed to (x + offset) * scale
 * @param dst destination tensor.
 * @param src source tensor.
 * @param scale scaling factor for normalization.
 * @param offset offset constant for normalization.
 * @stream specified cuda stream.
 */
void Normalize(Tensor<HWC, C3, F32> &dst, const Tensor<HWC, C3, U16> &src, const float scale[3], const float offset[3],
               cudaStream_t stream = 0);

/**
 * Batch normalization for three channels interleaved uint16_t type Tensor.
 * Each element x will be transformed to (x + offset) * scale
 * @param dst destination tensor.
 * @param src source tensor.
 * @param scale scaling factor for normalization.
 * @param offset offset constant for normalization.
 * @stream specified cuda stream.
 */
void Normalize(Tensor<NHWC, C3, F32> &dst, const Tensor<NHWC, C3, U16> &src, const float scale[3],
               const float offset[3], cudaStream_t stream = 0);

/**
 * Normalization for three channels interleaved float type Tensor.
 * Each element x will be transformed to (x + offset) * scale
 * @param dst destination tensor.
 * @param src source tensor.
 * @param scale scaling factor for normalization.
 * @param offset offset constant for normalization.
 * @stream specified cuda stream.
 */
void Normalize(Tensor<HWC, C3, F32> &dst, const Tensor<HWC, C3, F32> &src, const float scale[3], const float offset[3],
               cudaStream_t stream = 0);

/**
 * Batch normalization for three channels interleaved float type Tensor.
 * Each element x will be transformed to (x + offset) * scale
 * @param dst destination tensor.
 * @param src source tensor.
 * @param scale scaling factor for normalization.
 * @param offset offset constant for normalization.
 * @stream specified cuda stream.
 */
void Normalize(Tensor<NHWC, C3, F32> &dst, const Tensor<NHWC, C3, F32> &src, const float scale[3],
               const float offset[3], cudaStream_t stream = 0);

/**
 * Normalization for one channel interleaved uint8_t type Tensor.
 * Each element x will be transformed to (x + offset) * scale
 * @param dst destination tensor.
 * @param src source tensor.
 * @param scale scaling factor for normalization.
 * @param offset offset constant for normalization.
 * @stream specified cuda stream.
 */
void Normalize(Tensor<HWC, C1, F32> &dst, const Tensor<HWC, C1, U8> &src, const float scale, const float offset,
               cudaStream_t stream = 0);

/**
 * Batch normalization for one channel interleaved uint8_t type Tensor.
 * Each element x will be transformed to (x + offset) * scale
 * @param dst destination tensor.
 * @param src source tensor.
 * @param scale scaling factor for normalization.
 * @param offset offset constant for normalization.
 * @stream specified cuda stream.
 */
void Normalize(Tensor<NHWC, C1, F32> &dst, const Tensor<NHWC, C1, U8> &src, const float scale, const float offset,
               cudaStream_t stream = 0);

/**
 * Normalization for one channel interleaved uint16_t type Tensor.
 * Each element x will be transformed to (x + offset) * scale
 * @param dst destination tensor.
 * @param src source tensor.
 * @param scale scaling factor for normalization.
 * @param offset offset constant for normalization.
 * @stream specified cuda stream.
 */
void Normalize(Tensor<HWC, C1, F32> &dst, const Tensor<HWC, C1, U16> &src, const float scale, const float offset,
               cudaStream_t stream = 0);

/**
 * Batch normalization for one channel interleaved uint16_t type Tensor.
 * Each element x will be transformed to (x + offset) * scale
 * @param dst destination tensor.
 * @param src source tensor.
 * @param scale scaling factor for normalization.
 * @param offset offset constant for normalization.
 * @stream specified cuda stream.
 */
void Normalize(Tensor<NHWC, C1, F32> &dst, const Tensor<NHWC, C1, U16> &src, const float scale, const float offset,
               cudaStream_t stream = 0);

/**
 * Normalization for one channel interleaved float type Tensor.
 * Each element x will be transformed to (x + offset) * scale
 * @param dst destination tensor.
 * @param src source tensor.
 * @param scale scaling factor for normalization.
 * @param offset offset constant for normalization.
 * @stream specified cuda stream.
 */
void Normalize(Tensor<HWC, C1, F32> &dst, const Tensor<HWC, C1, F32> &src, const float scale, const float offset,
               cudaStream_t stream = 0);

/**
 * Batch normalization for one channel interleaved float type Tensor.
 * Each element x will be transformed to (x + offset) * scale
 * @param dst destination tensor.
 * @param src source tensor.
 * @param scale scaling factor for normalization.
 * @param offset offset constant for normalization.
 * @stream specified cuda stream.
 */
void Normalize(Tensor<NHWC, C1, F32> &dst, const Tensor<NHWC, C1, F32> &src, const float scale, const float offset,
               cudaStream_t stream = 0);

/**
 * Normalization for one channel planar uint8_t type Tensor.
 * Each element x will be transformed to (x + offset) * scale
 * @param dst destination tensor.
 * @param src source tensor.
 * @param scale scaling factor for normalization.
 * @param offset offset constant for normalization.
 * @stream specified cuda stream.
 */
void Normalize(Tensor<CHW, C1, F32> &dst, const Tensor<CHW, C1, U8> &src, const float scale, const float offset,
               cudaStream_t stream = 0);

/**
 * Batch normalization for one channel planar uint8_t type Tensor.
 * Each element x will be transformed to (x + offset) * scale
 * @param dst destination tensor.
 * @param src source tensor.
 * @param scale scaling factor for normalization.
 * @param offset offset constant for normalization.
 * @stream specified cuda stream.
 */
void Normalize(Tensor<NCHW, C1, F32> &dst, const Tensor<NCHW, C1, U8> &src, const float scale, const float offset,
               cudaStream_t stream = 0);

/**
 * Normalization for one channel planar uint16_t type Tensor.
 * Each element x will be transformed to (x + offset) * scale
 * @param dst destination tensor.
 * @param src source tensor.
 * @param scale scaling factor for normalization.
 * @param offset offset constant for normalization.
 * @stream specified cuda stream.
 */
void Normalize(Tensor<CHW, C1, F32> &dst, const Tensor<CHW, C1, U16> &src, const float scale, const float offset,
               cudaStream_t stream = 0);

/**
 * Batch normalization for one channel planar uint16_t type Tensor.
 * Each element x will be transformed to (x + offset) * scale
 * @param dst destination tensor.
 * @param src source tensor.
 * @param scale scaling factor for normalization.
 * @param offset offset constant for normalization.
 * @stream specified cuda stream.
 */
void Normalize(Tensor<NCHW, C1, F32> &dst, const Tensor<NCHW, C1, U16> &src, const float scale, const float offset,
               cudaStream_t stream = 0);

/**
 * Normalization for one channel planar float type Tensor.
 * Each element x will be transformed to (x + offset) * scale
 * @param dst destination tensor.
 * @param src source tensor.
 * @param scale scaling factor for normalization.
 * @param offset offset constant for normalization.
 * @stream specified cuda stream.
 */
void Normalize(Tensor<CHW, C1, F32> &dst, const Tensor<CHW, C1, F32> &src, const float scale, const float offset,
               cudaStream_t stream = 0);

/**
 * Batch normalization for one channel planar float type Tensor.
 * Each element x will be transformed to (x + offset) * scale
 * @param dst destination tensor.
 * @param src source tensor.
 * @param scale scaling factor for normalization.
 * @param offset offset constant for normalization.
 * @stream specified cuda stream.
 */
void Normalize(Tensor<NCHW, C1, F32> &dst, const Tensor<NCHW, C1, F32> &src, const float scale, const float offset,
               cudaStream_t stream = 0);

/**
 * Convert interleaved image to planar image for three channels uint8_t type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param stream specified cuda stream.
 */
void InterleavedToPlanar(Tensor<CHW, C3, U8> &dst, const Tensor<HWC, C3, U8> &src, cudaStream_t stream = 0);

/**
 * Batch convert interleaved image to planar image for three channels uint8_t type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param stream specified cuda stream.
 */
void InterleavedToPlanar(Tensor<NCHW, C3, U8> &dst, const Tensor<NHWC, C3, U8> &src, cudaStream_t stream = 0);

/**
 * Convert interleaved image to planar image for three channels uint16_t type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param stream specified cuda stream.
 */
void InterleavedToPlanar(Tensor<CHW, C3, U16> &dst, const Tensor<HWC, C3, U16> &src, cudaStream_t stream = 0);

/**
 * Batch Convert interleaved image to planar image for three channels uint16_t type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param stream specified cuda stream.
 */
void InterleavedToPlanar(Tensor<NCHW, C3, U16> &dst, const Tensor<NHWC, C3, U16> &src, cudaStream_t stream = 0);

/**
 * Convert interleaved image to planar image for three channels float type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param stream specified cuda stream.
 */
void InterleavedToPlanar(Tensor<CHW, C3, F32> &dst, const Tensor<HWC, C3, F32> &src, cudaStream_t stream = 0);

/**
 * Batch convert interleaved image to planar image for three channels float type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param stream specified cuda stream.
 */
void InterleavedToPlanar(Tensor<NCHW, C3, F32> &dst, const Tensor<NHWC, C3, F32> &src, cudaStream_t stream = 0);

/**
 * Convert interleaved image to planar image for single channel uint8_t type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param stream specified cuda stream.
 */
void InterleavedToPlanar(Tensor<CHW, C1, U8> &dst, const Tensor<HWC, C1, U8> &src, cudaStream_t stream = 0);

/**
 * Batch convert interleaved image to planar image for single channel uint8_t type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param stream specified cuda stream.
 */
void InterleavedToPlanar(Tensor<NCHW, C1, U8> &dst, const Tensor<NHWC, C1, U8> &src, cudaStream_t stream = 0);

/**
 * Convert interleaved image to planar image for single channel uint16_t type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param stream specified cuda stream.
 */
void InterleavedToPlanar(Tensor<CHW, C1, U16> &dst, const Tensor<HWC, C1, U16> &src, cudaStream_t stream = 0);

/**
 * Batch Convert interleaved image to planar image for single channel uint16_t type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param stream specified cuda stream.
 */
void InterleavedToPlanar(Tensor<NCHW, C1, U16> &dst, const Tensor<NHWC, C1, U16> &src, cudaStream_t stream = 0);

/**
 * Convert interleaved image to planar image for single channel float type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param stream specified cuda stream.
 */
void InterleavedToPlanar(Tensor<CHW, C1, F32> &dst, const Tensor<HWC, C1, F32> &src, cudaStream_t stream = 0);

/**
 * Batch convert interleaved image to planar image for single channel float type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param stream specified cuda stream.
 */
void InterleavedToPlanar(Tensor<NCHW, C1, F32> &dst, const Tensor<NHWC, C1, F32> &src, cudaStream_t stream = 0);

/**
 * Combines various functions to imitate OpenCV blobFromImage() for various type tensor input.
 * @tparam TL_IN input tensor layout (HWC/NHWC).
 * @tparam TL_OUT output tensor layout(CHW/NCHW).
 * @tparam CC channel count.
 * @tparam CT channel type for input tensor (output is fixed: F32).
 */
template<TensorLayout TL_IN, TensorLayout TL_OUT, ChannelCount CC, ChannelType CT>
class ImageToNormalizedPlanarTensorOperator
{
public:
    /**
     * Implementation for ImageToNormalizedPlanarTensorOperator.
     */
    struct ImageToNormalizedPlanarTensorOperatorImpl;

    /**
     * Constructor for HWC -> CHW tensors.
     */
    template<TensorLayout T = TL_IN, typename std::enable_if<T == HWC>::type * = nullptr>
    ImageToNormalizedPlanarTensorOperator(int width, int height);

    /**
     * Constructor for NHWC -> NCHW tensors.
     */
    template<TensorLayout T = TL_IN, typename std::enable_if<T == NHWC>::type * = nullptr>
    ImageToNormalizedPlanarTensorOperator(int width, int height, int depth);

    /**
     * Destructor for ImageToNormalizedPlanarTensorOperator.
     */
    ~ImageToNormalizedPlanarTensorOperator();

    /**
     * Run the composite operations on three channels tensors.
     * @param dst destination tensor.
     * @param src source tensor.
     * @param scale scale factor for normalization.
     * @param offset offset constant for normalization.
     * @param swapRB whether to swap the first and last channels.
     * @param keep_aspect_ratio whether to keep aspect ratio when resizing.
     * @param stream specified cuda stream.
     */
    template<ChannelCount T = CC, typename std::enable_if<T == C3>::type * = nullptr>
    void operator()(Tensor<TL_OUT, CC, F32> &dst, const Tensor<TL_IN, CC, CT> &src, const float scale[3],
                    const float offset[3], bool swapRB, bool keep_aspect_ratio = false, cudaStream_t stream = 0);

    /**
     * Run the composite operations on single channel tensors.
     * @param dst destination tensor.
     * @param src source tensor.
     * @param scale scale factor for normalization.
     * @param offset offset constant for normalization.
     * @param keep_aspect_ratio whether to keep aspect ratio when resizing.
     * @param stream specified cuda stream.
     */
    template<ChannelCount T = CC, typename std::enable_if<T == C1>::type * = nullptr>
    void operator()(Tensor<TL_OUT, CC, F32> &dst, const Tensor<TL_IN, CC, CT> &src, float scale, float offset,
                    bool keep_aspect_ratio = false, cudaStream_t stream = 0);

private:
    std::unique_ptr<ImageToNormalizedPlanarTensorOperatorImpl> m_pImpl;
};

}} // namespace cvcore::tensor_ops

#endif // CVCORE_IMAGE_UTILS_H
