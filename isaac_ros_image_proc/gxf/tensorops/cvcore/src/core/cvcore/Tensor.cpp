// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cv/core/Tensor.h"

#include <cuda_runtime_api.h>

#include <cassert>
#include <cstdlib>
#include <algorithm>
#include <functional>
#include <stdexcept>

namespace cvcore {

TensorBase::TensorBase()
    : m_data(nullptr)
    , m_dimCount(0)
    , m_type(U8)
    , m_isOwning(false)
    , m_isCPU(true)
{
    for (int i = 0; i < kMaxDimCount; ++i)
    {
        m_dimData[i] = {0, 0};
    }
}

TensorBase::TensorBase(ChannelType type, const DimData *dimData, int dimCount, void *dataPtr, bool isCPU)
    : TensorBase()
{
    assert(dimCount >= kMinDimCount && dimCount <= kMaxDimCount);

    m_isOwning = false;
    m_isCPU    = isCPU;

    m_type     = type;
    m_dimCount = dimCount;
    for (int i = 0; i < dimCount; ++i)
    {
        m_dimData[i] = dimData[i];
    }

    m_data = dataPtr;
}

TensorBase::TensorBase(ChannelType type, std::initializer_list<TensorBase::DimData> dimData, void *dataPtr, bool isCPU)
    : TensorBase(type, dimData.begin(), dimData.size(), dataPtr, isCPU)
{
}

TensorBase::TensorBase(ChannelType type, const DimData *dimData, int dimCount, bool isCPU)
    : TensorBase(type, dimData, dimCount, nullptr, isCPU)
{
    m_isOwning = true;

    // compute tensor memory block size
    const std::size_t tensorSize = getDataSize();

    // allocate
    if (isCPU)
    {
        m_data = std::malloc(tensorSize);
    }
    else
    {
        if (cudaMalloc(&m_data, tensorSize) != 0)
        {
            throw std::runtime_error("CUDA alloc() failed!");
        }
    }
}

TensorBase::TensorBase(ChannelType type, std::initializer_list<TensorBase::DimData> dimData, bool isCPU)
    : TensorBase(type, dimData.begin(), dimData.size(), isCPU)
{
}

TensorBase::~TensorBase()
{
    if (m_isOwning)
    {
        if (m_isCPU)
        {
            std::free(m_data);
        }
        else
        {
            cudaFree(m_data);
        }
    }
}

TensorBase::TensorBase(TensorBase &&t)
    : TensorBase()
{
    *this = std::move(t);
}

TensorBase &TensorBase::operator=(TensorBase &&t)
{
    using std::swap;

    swap(m_data, t.m_data);
    swap(m_dimCount, t.m_dimCount);
    swap(m_type, t.m_type);
    swap(m_isOwning, t.m_isOwning);
    swap(m_isCPU, t.m_isCPU);

    for (int i = 0; i < kMaxDimCount; ++i)
    {
        swap(m_dimData[i], t.m_dimData[i]);
    }

    return *this;
}

int TensorBase::getDimCount() const
{
    return m_dimCount;
}

std::size_t TensorBase::getSize(int dimIdx) const
{
    assert(dimIdx >= 0 && dimIdx < m_dimCount);
    return m_dimData[dimIdx].size;
}

std::size_t TensorBase::getStride(int dimIdx) const
{
    assert(dimIdx >= 0 && dimIdx < m_dimCount);
    return m_dimData[dimIdx].stride;
}

ChannelType TensorBase::getType() const
{
    return m_type;
}

void *TensorBase::getData() const
{
    return m_data;
}

std::size_t TensorBase::getDataSize() const
{
    std::size_t tensorSize = m_dimData[0].size * m_dimData[0].stride;
    for (int i = 1; i < m_dimCount; ++i)
    {
        tensorSize = std::max(tensorSize, m_dimData[i].size * m_dimData[i].stride);
    }
    tensorSize *= GetChannelSize(m_type);
    return tensorSize;
}

bool TensorBase::isCPU() const
{
    return m_isCPU;
}

bool TensorBase::isOwning() const
{
    return m_isOwning;
}

std::string GetTensorLayoutAsString(TensorLayout TL)
{
    switch (TL)
    {
    case TensorLayout::CL:
        return "CL";
    case TensorLayout::LC:
        return "LC";
    case TensorLayout::HWC:
        return "HWC";
    case TensorLayout::CHW:
        return "CHW";
    case TensorLayout::DHWC:
        return "DHWC";
    case TensorLayout::DCHW:
        return "DCHW";
    case TensorLayout::CDHW:
        return "CDHW";
    default:
        throw std::runtime_error("Invalid TensorLayout");
    }
}

std::string GetChannelCountAsString(ChannelCount CC)
{
    switch (CC)
    {
    case ChannelCount::C1:
        return "C1";
    case ChannelCount::C2:
        return "C2";
    case ChannelCount::C3:
        return "C3";
    case ChannelCount::C4:
        return "C4";
    case ChannelCount::CX:
        return "CX";
    default:
        throw std::runtime_error("Invalid ChannelCount");
    }
}

std::string GetChannelTypeAsString(ChannelType CT)
{
    switch (CT)
    {
    case ChannelType::U8:
        return "U8";
    case ChannelType::U16:
        return "U16";
    case ChannelType::S8:
        return "S8";
    case ChannelType::S16:
        return "S16";
    case ChannelType::F16:
        return "F16";
    case ChannelType::F32:
        return "F32";
    case ChannelType::F64:
        return "F64";
    default:
        throw std::runtime_error("Invalid ChannelType");
    }
}

std::size_t GetChannelSize(ChannelType CT)
{
    switch (CT)
    {
    case U8:
    case S8:
        return 1;
    case F16:
    case U16:
    case S16:
        return 2;
    case F32:
        return 4;
    case F64:
        return 8;
    default:
        throw std::runtime_error("Invalid ChannelType");
    }
}

std::string GetMemoryTypeAsString(bool isCPU)
{
    return isCPU? "CPU" : "GPU";
}

} // namespace cvcore
