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

#ifndef CVCORE_MEMORY_H
#define CVCORE_MEMORY_H

#include <cuda_runtime_api.h>

#include "Tensor.h"

namespace cvcore {

/**
 * Implementation of tensor copy.
 * @param dst destination TensorBase.
 * @param src source TensorBase.
 * @param stream cuda stream.
 */
void TensorBaseCopy(TensorBase &dst, const TensorBase &src, cudaStream_t stream = 0);

/**
 * Implementation of tensor copy for 2D pitch linear tensors.
 * @param dst destination TensorBase.
 * @param src source TensorBase.
 * @param dstPitch pitch of destination Tensor in bytes.
 * @param srcPitch pitch of source Tensor in bytes.
 * @param widthInBytes width in bytes.
 * @param height height of tensor.
 * @param stream cuda stream.
 */
void TensorBaseCopy2D(TensorBase &dst, const TensorBase &src, int dstPitch, int srcPitch, int widthInBytes, int height,
                      cudaStream_t stream = 0);

/**
 * Memory copy function between two non HWC/CHW/NHWC/NCHW Tensors.
 * @tparam TL TensorLayout type.
 * @tparam CC Channel Count.
 * @tparam CT ChannelType.
 * @param dst destination Tensor.
 * @param src source Tensor which copy from.
 * @param stream cuda stream.
 */
template<TensorLayout TL, ChannelCount CC, ChannelType CT,
         typename std::enable_if<TL != HWC && TL != CHW && TL != NHWC && TL != NCHW>::type * = nullptr>
void Copy(Tensor<TL, CC, CT> &dst, const Tensor<TL, CC, CT> &src, cudaStream_t stream = 0)
{
    TensorBaseCopy(dst, src, stream);
}

/**
 * Memory copy function between two HWC Tensors.
 * @tparam TL TensorLayout type.
 * @tparam CC Channel Count.
 * @tparam CT ChannelType.
 * @param dst destination Tensor.
 * @param src source Tensor which copy from.
 * @param stream cuda stream.
 */
template<TensorLayout TL, ChannelCount CC, ChannelType CT, typename std::enable_if<TL == HWC>::type * = nullptr>
void Copy(Tensor<TL, CC, CT> &dst, const Tensor<TL, CC, CT> &src, cudaStream_t stream = 0)
{
    TensorBaseCopy2D(dst, src, dst.getStride(TensorDimension::HEIGHT) * GetChannelSize(CT),
                     src.getStride(TensorDimension::HEIGHT) * GetChannelSize(CT),
                     dst.getWidth() * dst.getChannelCount() * GetChannelSize(CT), src.getHeight(), stream);
}

/**
 * Memory copy function between two NHWC Tensors.
 * @tparam TL TensorLayout type.
 * @tparam CC Channel Count.
 * @tparam CT ChannelType.
 * @param dst destination Tensor.
 * @param src source Tensor which copy from.
 * @param stream cuda stream.
 */
template<TensorLayout TL, ChannelCount CC, ChannelType CT, typename std::enable_if<TL == NHWC>::type * = nullptr>
void Copy(Tensor<TL, CC, CT> &dst, const Tensor<TL, CC, CT> &src, cudaStream_t stream = 0)
{
    TensorBaseCopy2D(dst, src, dst.getStride(TensorDimension::HEIGHT) * GetChannelSize(CT),
                     src.getStride(TensorDimension::HEIGHT) * GetChannelSize(CT),
                     dst.getWidth() * dst.getChannelCount() * GetChannelSize(CT), src.getDepth() * src.getHeight(),
                     stream);
}

/**
 * Memory copy function between two CHW Tensors.
 * @tparam TL TensorLayout type.
 * @tparam CC Channel Count.
 * @tparam CT ChannelType.
 * @param dst destination Tensor.
 * @param src source Tensor which copy from.
 * @param stream cuda stream.
 */
template<TensorLayout TL, ChannelCount CC, ChannelType CT, typename std::enable_if<TL == CHW>::type * = nullptr>
void Copy(Tensor<TL, CC, CT> &dst, const Tensor<TL, CC, CT> &src, cudaStream_t stream = 0)
{
    TensorBaseCopy2D(dst, src, dst.getStride(TensorDimension::HEIGHT) * GetChannelSize(CT),
                     src.getStride(TensorDimension::HEIGHT) * GetChannelSize(CT), dst.getWidth() * GetChannelSize(CT),
                     src.getChannelCount() * src.getHeight(), stream);
}

/**
 * Memory copy function between two NCHW Tensors.
 * @tparam TL TensorLayout type.
 * @tparam CC Channel Count.
 * @tparam CT ChannelType.
 * @param dst destination Tensor.
 * @param src source Tensor which copy from.
 * @param stream cuda stream.
 */
template<TensorLayout TL, ChannelCount CC, ChannelType CT, typename std::enable_if<TL == NCHW>::type * = nullptr>
void Copy(Tensor<TL, CC, CT> &dst, const Tensor<TL, CC, CT> &src, cudaStream_t stream = 0)
{
    TensorBaseCopy2D(dst, src, dst.getStride(TensorDimension::HEIGHT) * GetChannelSize(CT),
                     src.getStride(TensorDimension::HEIGHT) * GetChannelSize(CT), dst.getWidth() * GetChannelSize(CT),
                     src.getDepth() * src.getChannelCount() * src.getHeight(), stream);
}

} // namespace cvcore

#endif // CVCORE_MEMORY_H
