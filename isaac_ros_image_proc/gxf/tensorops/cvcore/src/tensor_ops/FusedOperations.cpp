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

#include <cassert>
#include <cstdint>
#include <stdexcept>

namespace cvcore { namespace tensor_ops {

template<TensorLayout TL_IN, TensorLayout TL_OUT, ChannelCount CC, ChannelType CT>
struct ImageToNormalizedPlanarTensorOperator<TL_IN, TL_OUT, CC, CT>::ImageToNormalizedPlanarTensorOperatorImpl
{
    int m_width;
    int m_height;
    int m_depth;
    std::unique_ptr<Tensor<TL_IN, CC, CT>> m_resizedTensor;
    std::unique_ptr<Tensor<TL_IN, CC, F32>> m_normalizedTensor;

    template<TensorLayout T = TL_IN, typename std::enable_if<T == HWC>::type * = nullptr>
    ImageToNormalizedPlanarTensorOperatorImpl(int width, int height)
        : m_width(width)
        , m_height(height)
        , m_depth(1)
    {
        m_resizedTensor.reset(new Tensor<TL_IN, CC, CT>(width, height, false));
        m_normalizedTensor.reset(new Tensor<TL_IN, CC, F32>(width, height, false));
    }

    template<TensorLayout T = TL_IN, typename std::enable_if<T == NHWC>::type * = nullptr>
    ImageToNormalizedPlanarTensorOperatorImpl(int width, int height, int depth)
        : m_width(width)
        , m_height(height)
        , m_depth(depth)
    {
        m_resizedTensor.reset(new Tensor<TL_IN, CC, CT>(width, height, depth, false));
        m_normalizedTensor.reset(new Tensor<TL_IN, CC, F32>(width, height, depth, false));
    }

    template<TensorLayout T = TL_IN, ChannelCount C = CC,
             typename std::enable_if<T == HWC && C == C3>::type * = nullptr>
    void execute(Tensor<TL_OUT, CC, F32> &dst, const Tensor<TL_IN, CC, CT> &src, const float scale[3],
                 const float offset[3], bool swapRB, bool keep_aspect_ratio, cudaStream_t stream)
    {
        // src and dst must be GPU tensors
        assert(!src.isCPU() && !dst.isCPU());

        // dst image width/height must match width/height of class
        if ((dst.getWidth() != m_width) || (dst.getHeight() != m_height))
        {
            throw std::runtime_error("invalid input width/height");
        }

        // first do the resizing
        Resize(*m_resizedTensor, src, keep_aspect_ratio, INTERP_LINEAR, stream);

        // swap channels if needed
        if (swapRB)
        {
            ConvertColorFormat(*m_resizedTensor, *m_resizedTensor, BGR2RGB, stream);
        }

        // do the normalization
        Normalize(*m_normalizedTensor, *m_resizedTensor, scale, offset, stream);

        // convert interleave to planar tensor
        InterleavedToPlanar(dst, *m_normalizedTensor, stream);
    }

    template<TensorLayout T = TL_IN, ChannelCount C = CC,
             typename std::enable_if<T == NHWC && C == C3>::type * = nullptr>
    void execute(Tensor<TL_OUT, CC, F32> &dst, const Tensor<TL_IN, CC, CT> &src, const float scale[3],
                 const float offset[3], bool swapRB, bool keep_aspect_ratio, cudaStream_t stream)
    {
        // src and dst must be GPU tensors
        assert(!src.isCPU() && !dst.isCPU());

        // dst image width/height must match width/height of class
        if ((dst.getWidth() != m_width) || (dst.getHeight() != m_height))
        {
            throw std::runtime_error("invalid input width/height");
        }

        // dst image depth must be equal to src image depth and no bigger than m_depth
        if ((dst.getDepth() != src.getDepth()) || (dst.getDepth() > m_depth))
        {
            throw std::runtime_error("invalid input depth");
        }

        // wrap the batch tensor with non-owning tensor
        Tensor<TL_IN, CC, CT> resizedTensor(m_width, m_height, dst.getDepth(), m_resizedTensor->getData(), false);
        Tensor<TL_IN, CC, F32> normalizedTensor(m_width, m_height, dst.getDepth(), m_normalizedTensor->getData(),
                                                false);

        // first do the resizing
        Resize(resizedTensor, src, keep_aspect_ratio, INTERP_LINEAR, stream);

        // swap channels if needed
        if (swapRB)
        {
            ConvertColorFormat(resizedTensor, resizedTensor, BGR2RGB, stream);
        }

        // do the normalization
        Normalize(normalizedTensor, resizedTensor, scale, offset, stream);

        // convert interleave to planar tensor
        InterleavedToPlanar(dst, normalizedTensor, stream);
    }

    template<TensorLayout T = TL_IN, ChannelCount C = CC,
             typename std::enable_if<T == HWC && C == C1>::type * = nullptr>
    void execute(Tensor<TL_OUT, CC, F32> &dst, const Tensor<TL_IN, CC, CT> &src, float scale, float offset,
                 bool keep_aspect_ratio, cudaStream_t stream)
    {
        // src and dst must be GPU tensors
        assert(!src.isCPU() && !dst.isCPU());

        // dst image width/height must match width/height of class
        if ((dst.getWidth() != m_width) || (dst.getHeight() != m_height))
        {
            throw std::runtime_error("invalid input width/height");
        }

        // first do the resizing
        Resize(*m_resizedTensor, src, keep_aspect_ratio, INTERP_LINEAR, stream);

        // do the normalization and map to destination tensor directly
        Tensor<TL_IN, CC, F32> output(m_width, m_height, dst.getData(), false);
        Normalize(output, *m_resizedTensor, scale, offset, stream);
    }

    template<TensorLayout T = TL_IN, ChannelCount C = CC,
             typename std::enable_if<T == NHWC && C == C1>::type * = nullptr>
    void execute(Tensor<TL_OUT, CC, F32> &dst, const Tensor<TL_IN, CC, CT> &src, float scale, float offset,
                 bool keep_aspect_ratio, cudaStream_t stream)
    {
        // src and dst must be GPU tensors
        assert(!src.isCPU() && !dst.isCPU());

        // dst image width/height must match width/height of class
        if ((dst.getWidth() != m_width) || (dst.getHeight() != m_height))
        {
            throw std::runtime_error("invalid input width/height");
        }

        // dst image depth must be equal to src image depth and no bigger than m_depth
        if ((dst.getDepth() != src.getDepth()) || (dst.getDepth() > m_depth))
        {
            throw std::runtime_error("invalid input depth");
        }

        // wrap the batch tensor with non-owning tensor
        Tensor<TL_IN, CC, CT> resizedTensor(m_width, m_height, dst.getDepth(), m_resizedTensor->getData(), false);

        // first do the resizing
        Resize(resizedTensor, src, keep_aspect_ratio, INTERP_LINEAR, stream);

        // do the normalization and map to destination tensor directly
        Tensor<TL_IN, CC, F32> output(m_width, m_height, dst.getDepth(), dst.getData(), false);
        Normalize(output, resizedTensor, scale, offset, stream);
    }
};

template<TensorLayout TL_IN, TensorLayout TL_OUT, ChannelCount CC, ChannelType CT>
template<TensorLayout T, typename std::enable_if<T == HWC>::type *>
ImageToNormalizedPlanarTensorOperator<TL_IN, TL_OUT, CC, CT>::ImageToNormalizedPlanarTensorOperator(int width,
                                                                                                    int height)
    : m_pImpl(new ImageToNormalizedPlanarTensorOperatorImpl(width, height))
{
    static_assert(TL_IN == HWC && TL_OUT == CHW, "Tensor Layout is different");
    static_assert(CC == C1 || CC == C3, "Channel count is different");
}

template<TensorLayout TL_IN, TensorLayout TL_OUT, ChannelCount CC, ChannelType CT>
template<TensorLayout T, typename std::enable_if<T == NHWC>::type *>
ImageToNormalizedPlanarTensorOperator<TL_IN, TL_OUT, CC, CT>::ImageToNormalizedPlanarTensorOperator(int width,
                                                                                                    int height,
                                                                                                    int depth)
    : m_pImpl(new ImageToNormalizedPlanarTensorOperatorImpl(width, height, depth))
{
    static_assert(TL_IN == NHWC && TL_OUT == NCHW, "Tensor Layout is different");
    static_assert(CC == C1 || CC == C3, "Channel count is different");
}

template<TensorLayout TL_IN, TensorLayout TL_OUT, ChannelCount CC, ChannelType CT>
ImageToNormalizedPlanarTensorOperator<TL_IN, TL_OUT, CC, CT>::~ImageToNormalizedPlanarTensorOperator()
{
}

template<TensorLayout TL_IN, TensorLayout TL_OUT, ChannelCount CC, ChannelType CT>
template<ChannelCount T, typename std::enable_if<T == C3>::type *>
void ImageToNormalizedPlanarTensorOperator<TL_IN, TL_OUT, CC, CT>::operator()(
    Tensor<TL_OUT, CC, F32> &dst, const Tensor<TL_IN, CC, CT> &src, const float scale[3], const float offset[3],
    bool swapRB, bool keep_aspect_ratio, cudaStream_t stream)
{
    m_pImpl->execute(dst, src, scale, offset, swapRB, keep_aspect_ratio, stream);
}

template<TensorLayout TL_IN, TensorLayout TL_OUT, ChannelCount CC, ChannelType CT>
template<ChannelCount T, typename std::enable_if<T == C1>::type *>
void ImageToNormalizedPlanarTensorOperator<TL_IN, TL_OUT, CC, CT>::operator()(Tensor<TL_OUT, CC, F32> &dst,
                                                                              const Tensor<TL_IN, CC, CT> &src,
                                                                              float scale, float offset,
                                                                              bool keep_aspect_ratio,
                                                                              cudaStream_t stream)
{
    m_pImpl->execute(dst, src, scale, offset, keep_aspect_ratio, stream);
}

// explicit instantiations
template class ImageToNormalizedPlanarTensorOperator<HWC, CHW, C3, U8>;
template void ImageToNormalizedPlanarTensorOperator<HWC, CHW, C3, U8>::operator()<C3>(Tensor<CHW, C3, F32> &,
                                                                                      const Tensor<HWC, C3, U8> &,
                                                                                      const float [], const float [],
                                                                                      bool, bool, cudaStream_t);
template ImageToNormalizedPlanarTensorOperator<HWC, CHW, C3, U8>::ImageToNormalizedPlanarTensorOperator(int, int);

template class ImageToNormalizedPlanarTensorOperator<HWC, CHW, C1, U8>;
template void ImageToNormalizedPlanarTensorOperator<HWC, CHW, C1, U8>::operator()<C1>(Tensor<CHW, C1, F32> &,
                                                                                      const Tensor<HWC, C1, U8> &,
                                                                                      float, float, bool, cudaStream_t);
template ImageToNormalizedPlanarTensorOperator<HWC, CHW, C1, U8>::ImageToNormalizedPlanarTensorOperator(int, int);

template class ImageToNormalizedPlanarTensorOperator<NHWC, NCHW, C3, U8>;
template void ImageToNormalizedPlanarTensorOperator<NHWC, NCHW, C3, U8>::operator()<C3>(Tensor<NCHW, C3, F32> &,
                                                                                        const Tensor<NHWC, C3, U8> &,
                                                                                        const float [], const float [],
                                                                                        bool, bool, cudaStream_t);
template ImageToNormalizedPlanarTensorOperator<NHWC, NCHW, C3, U8>::ImageToNormalizedPlanarTensorOperator(int,
                                                                                                                int,
                                                                                                                int);

template class ImageToNormalizedPlanarTensorOperator<NHWC, NCHW, C1, U8>;
template void ImageToNormalizedPlanarTensorOperator<NHWC, NCHW, C1, U8>::operator()<C1>(Tensor<NCHW, C1, F32> &,
                                                                                        const Tensor<NHWC, C1, U8> &,
                                                                                        float, float, bool,
                                                                                        cudaStream_t);
template ImageToNormalizedPlanarTensorOperator<NHWC, NCHW, C1, U8>::ImageToNormalizedPlanarTensorOperator(int,
                                                                                                                int,
                                                                                                                int);
}} // namespace cvcore::tensor_ops
