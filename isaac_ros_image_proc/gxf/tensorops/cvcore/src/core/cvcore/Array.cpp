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

#include "cv/core/Array.h"

#include <cuda_runtime_api.h>

#include <cassert>
#include <stdexcept>

namespace cvcore {

ArrayBase::ArrayBase()
    : m_data{nullptr}
    , m_size{0}
    , m_capacity{0}
    , m_elemSize{0}
    , m_isOwning{false}
    , m_isCPU{true}
{
}

ArrayBase::ArrayBase(std::size_t capacity, std::size_t elemSize, void *dataPtr, bool isCPU)
    : ArrayBase()
{
    m_isOwning = false;
    m_isCPU    = isCPU;
    m_elemSize = elemSize;
    m_capacity = capacity;
    m_data     = dataPtr;
}

ArrayBase::ArrayBase(std::size_t capacity, std::size_t elemSize, bool isCPU)
    : ArrayBase(capacity, elemSize, nullptr, isCPU)
{
    m_isOwning = true;

    // allocate
    const size_t arraySize = capacity * elemSize;
    if (arraySize > 0)
    {
        if (isCPU)
        {
            m_data = std::malloc(arraySize);
        }
        else
        {
            if (cudaMalloc(&m_data, arraySize) != 0)
            {
                throw std::runtime_error("CUDA alloc() failed!");
            }
        }
    }
}

ArrayBase::~ArrayBase()
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

ArrayBase::ArrayBase(ArrayBase &&t)
    : ArrayBase()
{
    *this = std::move(t);
}

ArrayBase &ArrayBase::operator=(ArrayBase &&t)
{
    using std::swap;

    swap(m_data, t.m_data);
    swap(m_size, t.m_size);
    swap(m_capacity, t.m_capacity);
    swap(m_elemSize, t.m_elemSize);
    swap(m_isOwning, t.m_isOwning);
    swap(m_isCPU, t.m_isCPU);

    return *this;
}

void *ArrayBase::getElement(int idx) const
{
    assert(idx >= 0 && idx < m_capacity);
    return reinterpret_cast<char *>(m_data) + idx * m_elemSize;
}

std::size_t ArrayBase::getSize() const
{
    return m_size;
}
std::size_t ArrayBase::getCapacity() const
{
    return m_capacity;
}
std::size_t ArrayBase::getElementSize() const
{
    return m_elemSize;
}

void ArrayBase::setSize(std::size_t size)
{
    assert(size <= m_capacity);
    m_size = size;
}

bool ArrayBase::isCPU() const
{
    return m_isCPU;
}

bool ArrayBase::isOwning() const
{
    return m_isOwning;
}

void *ArrayBase::getData() const
{
    return m_data;
}

} // namespace cvcore
