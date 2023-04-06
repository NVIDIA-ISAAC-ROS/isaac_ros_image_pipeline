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

#include "cv/core/Memory.h"

#include <cuda_runtime.h>

#include <cassert>
#include <cstring>
#include <stdexcept>

namespace cvcore {

namespace {

// Copy 2D CPU pitch linear tensors
void Memcpy2DCPU(void *dst, size_t dstPitch, const void *src, size_t srcPitch, size_t widthInBytes, size_t height)
{
    uint8_t *dstPt       = reinterpret_cast<uint8_t *>(dst);
    const uint8_t *srcPt = reinterpret_cast<const uint8_t *>(src);
    for (size_t i = 0; i < height; i++)
    {
        memcpy(dstPt, srcPt, widthInBytes);
        dstPt += dstPitch;
        srcPt += srcPitch;
    }
}

} // anonymous namespace

void TensorBaseCopy(TensorBase &dst, const TensorBase &src, cudaStream_t stream)
{
    if (dst.getDataSize() != src.getDataSize())
    {
        throw std::runtime_error("Tensor stride mismatch!");
    }
    assert(dst.getDimCount() == src.getDimCount());
    int dimCount = src.getDimCount();
    for (int i = 0; i < dimCount - 1; i++)
    {
        if (src.getStride(i) != src.getStride(i + 1) * src.getSize(i + 1) ||
            dst.getStride(i) != dst.getStride(i + 1) * dst.getSize(i + 1))
        {
            throw std::runtime_error("Tensor is not contiguous in memory!");
        }
    }
    if (dst.isCPU() && src.isCPU())
    {
        memcpy(dst.getData(), src.getData(), src.getDataSize());
        return;
    }
    cudaError_t error;
    if (!dst.isCPU() && src.isCPU())
    {
        error = cudaMemcpyAsync(dst.getData(), src.getData(), src.getDataSize(), cudaMemcpyHostToDevice, stream);
    }
    else if (dst.isCPU() && !src.isCPU())
    {
        error = cudaMemcpyAsync(dst.getData(), src.getData(), src.getDataSize(), cudaMemcpyDeviceToHost, stream);
    }
    else
    {
        error = cudaMemcpyAsync(dst.getData(), src.getData(), src.getDataSize(), cudaMemcpyDeviceToDevice, stream);
    }
    if (error != cudaSuccess)
    {
        throw std::runtime_error("CUDA memcpy failed!");
    }
}

void TensorBaseCopy2D(TensorBase &dst, const TensorBase &src, int dstPitch, int srcPitch, int widthInBytes, int height,
                      cudaStream_t stream)
{
    assert(dst.getDimCount() == src.getDimCount());
    int dimCount = src.getDimCount();
    for (int i = 0; i < dimCount; i++)
    {
        if (dst.getSize(i) != src.getSize(i))
        {
            throw std::runtime_error("Tensor size mismatch!");
        }
    }
    if (dst.isCPU() && src.isCPU())
    {
        Memcpy2DCPU(dst.getData(), dstPitch, src.getData(), srcPitch, widthInBytes, height);
        return;
    }
    cudaError_t error;
    if (!dst.isCPU() && src.isCPU())
    {
        error = cudaMemcpy2DAsync(dst.getData(), dstPitch, src.getData(), srcPitch, widthInBytes, height,
                                  cudaMemcpyHostToDevice, stream);
    }
    else if (dst.isCPU() && !src.isCPU())
    {
        error = cudaMemcpy2DAsync(dst.getData(), dstPitch, src.getData(), srcPitch, widthInBytes, height,
                                  cudaMemcpyDeviceToHost, stream);
    }
    else
    {
        error = cudaMemcpy2DAsync(dst.getData(), dstPitch, src.getData(), srcPitch, widthInBytes, height,
                                  cudaMemcpyDeviceToDevice, stream);
    }
    if (error != cudaSuccess)
    {
        throw std::runtime_error("CUDA memcpy failed!");
    }
}

} // namespace cvcore
