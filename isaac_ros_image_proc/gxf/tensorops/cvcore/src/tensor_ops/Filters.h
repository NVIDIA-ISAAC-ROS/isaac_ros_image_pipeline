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

#ifndef CVCORE_FILTERS_H
#define CVCORE_FILTERS_H

#include "cv/core/Tensor.h"
#include "cv/core/MathTypes.h"

#include <cuda_runtime.h>

namespace cvcore { namespace tensor_ops {

/**
 * Box type filtering for three channel HWC format uint_8 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param maskSize Size of mask which determines number of pixels to be averaged.
 * @param anchor Offset of mask relative to current pixel index.
 * 	  {0, 0} mask aligns with starting pixel.
 * 	  {mask size/2, mask size/2} mask aligns with center pixel index.
 * @param stream specified cuda stream.
 */

void BoxFilter(Tensor<HWC, C3, U8> &dst, const Tensor<HWC, C3, U8> &src, const Vector2i &maskSize,
               const Vector2i &anchor, cudaStream_t stream = 0);
/**
 * Box type filtering for three channel HWC format uint_16 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param maskSize Size of mask which determines number of pixels to be averaged.
 * @param anchor Offset of mask relative to current pixel index.
 * 	  {0, 0} mask aligns with starting pixel.
 * 	  {mask size/2, mask size/2} mask aligns with center pixel index.
 * @param stream specified cuda stream.
 */
void BoxFilter(Tensor<HWC, C3, U16> &dst, const Tensor<HWC, C3, U16> &src, const Vector2i &maskSize,
               const Vector2i &anchor, cudaStream_t stream = 0);
/**
 * Box type filtering for three channel HWC format float type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param maskSize Size of mask which determines number of pixels to be averaged.
 * @param anchor Offset of mask relative to current pixel index.
 * 	  {0, 0} mask aligns with starting pixel.
 * 	  {mask size/2, mask size/2} mask aligns with center pixel index.
 * @param stream specified cuda stream.
 */
void BoxFilter(Tensor<HWC, C3, F32> &dst, const Tensor<HWC, C3, F32> &src, const Vector2i &maskSize,
               const Vector2i &anchor, cudaStream_t stream = 0);

/**
 * Box type filtering for one channel HWC format uint_8 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param maskSize Size of mask which determines number of pixels to be averaged.
 * @param anchor Offset of mask relative to current pixel index.
 * 	  {0, 0} mask aligns with starting pixel.
 * 	  {mask size/2, mask size/2} mask aligns with center pixel index.
 * @param stream specified cuda stream.
 */
void BoxFilter(Tensor<HWC, C1, U8> &dst, const Tensor<HWC, C1, U8> &src, const Vector2i &maskSize,
               const Vector2i &anchor, cudaStream_t stream = 0);
/**
 * Box type filtering for one channel HWC format uint_16 type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param maskSize Size of mask which determines number of pixels to be averaged.
 * @param anchor Offset of mask relative to current pixel index.
 * 	  {0, 0} mask aligns with starting pixel.
 * 	  {mask size/2, mask size/2} mask aligns with center pixel index.
 * @param stream specified cuda stream.
 */
void BoxFilter(Tensor<HWC, C1, U16> &dst, const Tensor<HWC, C1, U16> &src, const Vector2i &maskSize,
               const Vector2i &anchor, cudaStream_t stream = 0);
/**
 * Box type filtering for one channel HWC format float type Tensor.
 * @param dst destination tensor.
 * @param src source tensor.
 * @param maskSize Size of mask which determines number of pixels to be averaged.
 * @param anchor Offset of mask relative to current pixel index.
 * 	  {0, 0} mask aligns with starting pixel.
 * 	  {mask size/2, mask size/2} mask aligns with center pixel index.
 * @param stream specified cuda stream.
 */
void BoxFilter(Tensor<HWC, C1, F32> &dst, const Tensor<HWC, C1, F32> &src, const Vector2i &maskSize,
               const Vector2i &anchor, cudaStream_t stream = 0);

}} // namespace cvcore::tensor_ops

#endif // CVCORE_FILTERS_H
