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

#include <array>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace cvcore { namespace tensor_ops {

constexpr size_t CACHE_SIZE = 20;
static size_t timestamp     = 0;
std::mutex lock;

namespace {

// This function involves GPU query and can be really slow
void SetupNppStreamContext(NppStreamContext &context, cudaStream_t stream)
{
    context.hStream   = stream;
    cudaError_t error = cudaGetDevice(&context.nCudaDeviceId);
    if (error != cudaSuccess)
    {
        throw std::runtime_error("no devices supporting CUDA");
    }
    error = cudaStreamGetFlags(context.hStream, &context.nStreamFlags);
    if (error != cudaSuccess)
    {
        throw std::runtime_error("failed to get cuda stream flags");
    }

    cudaDeviceProp deviceProp;
    error = cudaGetDeviceProperties(&deviceProp, context.nCudaDeviceId);
    if (error != cudaSuccess)
    {
        throw std::runtime_error("no device properties");
    }

    context.nSharedMemPerBlock           = deviceProp.sharedMemPerBlock;
    context.nMaxThreadsPerBlock          = deviceProp.maxThreadsPerBlock;
    context.nMultiProcessorCount         = deviceProp.multiProcessorCount;
    context.nMaxThreadsPerMultiProcessor = deviceProp.maxThreadsPerMultiProcessor;

    // Refer - https://gitlab-master.nvidia.com/cv/core-modules/tensor_ops/-/merge_requests/48#note_6602087
    context.nReserved0 = 0;

    error = cudaDeviceGetAttribute(&(context.nCudaDevAttrComputeCapabilityMajor), cudaDevAttrComputeCapabilityMajor,
                                   context.nCudaDeviceId);
    if (error != cudaSuccess)
    {
        throw std::runtime_error("no device attribute - nCudaDevAttrComputeCapabilityMajor");
    }

    error = cudaDeviceGetAttribute(&(context.nCudaDevAttrComputeCapabilityMinor), cudaDevAttrComputeCapabilityMinor,
                                   context.nCudaDeviceId);
    if (error != cudaSuccess)
    {
        throw std::runtime_error("no device attribute - nCudaDevAttrComputeCapabilityMinor");
    }
}

} // anonymous namespace

struct Context
{
    NppStreamContext nppContext;
    size_t time = 0;
};

NppStreamContext GetNppStreamContext(cudaStream_t stream)
{
    // Create a memory cache, all timestamp would be initialzed to 0 automatically
    static std::array<Context, CACHE_SIZE> contextCache = {};

    // Lock the thread
    std::lock_guard<std::mutex> guard(lock);

    size_t minTimestamp = contextCache[0].time;
    size_t minIdx       = 0;
    for (size_t i = 0; i < CACHE_SIZE; i++)
    {
        auto &it = contextCache[i];
        if (it.time > 0 && it.nppContext.hStream == stream)
        {
            it.time = ++timestamp;
            return it.nppContext;
        }
        if (it.time < minTimestamp)
        {
            minTimestamp = it.time;
            minIdx       = i;
        }
    }
    auto &it = contextCache[minIdx];
    SetupNppStreamContext(it.nppContext, stream);
    it.time = ++timestamp;
    return it.nppContext;
}

}} // namespace cvcore::tensor_ops
