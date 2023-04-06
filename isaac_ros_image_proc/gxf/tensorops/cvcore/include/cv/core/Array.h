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

#ifndef CVCORE_ARRAY_H
#define CVCORE_ARRAY_H

#include <cassert>
#include <new>
#include <utility>

namespace cvcore {

/**
 * Base implementation of Array.
 */
class ArrayBase
{
public:
    /**
     * Constructor of a non-owning arrays.
     * @param capacity capacity of the array.
     * @param elemSize byte size of each element.
     * @param dataPtr data pointer to the raw source array.
     * @param isCPU whether to allocate the array on CPU or GPU.
     */
    ArrayBase(std::size_t capacity, std::size_t elemSize, void *dataPtr, bool isCPU);

    /**
     * Constructor of a memory-owning arrays
     * @param capacity capacity of the array.
     * @param elemSize byte size of each element.
     * @param isCPU whether to allocate the array on CPU or GPU.
     */
    ArrayBase(std::size_t capacity, std::size_t elemSize, bool isCPU);

    /**
     * Destructor of ArrayBase.
     */
    ~ArrayBase();

    /**
     * ArrayBase is non-copyable.
     */
    ArrayBase(const ArrayBase &) = delete;

    /**
     * ArrayBase is non-copyable.
     */
    ArrayBase &operator=(const ArrayBase &) = delete;

    /**
     * Move constructor of ArrayBase. 
     */
    ArrayBase(ArrayBase &&);

    /**
     * Move assignment operator of ArrayBase.
     */
    ArrayBase &operator=(ArrayBase &&);

    /**
     * Get the pointer to specified index.
     * @param idx element index.
     * @return pointer to the specified element.
     */
    void *getElement(int idx) const;

    /**
     * Get the size of the array.
     * @return size of the array.
     */
    std::size_t getSize() const;

    /**
     * Get the capacity of the array.
     * @return size of the array.
     */
    std::size_t getCapacity() const;

    /**
     * Get the size of each element.
     * @return size of each element.
     */
    std::size_t getElementSize() const;

    /**
     * Set the size of the array.
     * @param size size of the array.
     */
    void setSize(std::size_t size);

    /**
     * Get the flag whether the array is CPU or GPU array.
     * @return whether the array is CPU array.
     */
    bool isCPU() const;

    /**
     * Get the flag whether the array is owning memory space.
     * @return whether the array owns memory.
     */
    bool isOwning() const;

    /**
     * Get the raw pointer to the array data.
     * @return void pointer to the first element of the array.
     */
    void *getData() const;

private:
    ArrayBase();

    void *m_data;
    std::size_t m_size;
    std::size_t m_capacity;
    std::size_t m_elemSize;
    bool m_isOwning;
    bool m_isCPU;
};

/**
 * Implementation of Array class.
 * @tparam T type of element in array.
 */
template<typename T>
class Array : public ArrayBase
{
public:
    /**
     * Default constructor of an array.
     */
    Array()
        : ArrayBase{0, sizeof(T), nullptr, true}
    {
    }

    /**
     * Constructor of a non-owning array.
     * @param size size of the array.
     * @param capacity capacity of the array.
     * @param dataPtr data pointer to the raw source array.
     * @param isCPU whether to allocate array on CPU or GPU.
     */
    Array(std::size_t size, std::size_t capacity, void *dataPtr, bool isCPU = true)
        : ArrayBase{capacity, sizeof(T), dataPtr, isCPU}
    {
        ArrayBase::setSize(size);
    }

    /**
     * Constructor of a memory-owning array.
     * @param capacity capacity of the array.
     * @param isCPU whether to allocate array on CPU or GPU.
     */
    Array(std::size_t capacity, bool isCPU = true)
        : ArrayBase{capacity, sizeof(T), isCPU}
    {
    }

    /**
     * Destructor of the Array.
     */
    ~Array()
    {
        // call resize here such that CPU-based destructor
        // will call destructors of the objects stored
        // in the array before deallocating the storage
        setSize(0);
    }

    /**
     * Array is non-copyable.
     */
    Array(const Array &) = delete;

    /**
     * Array is non-copyable.
     */
    Array &operator=(const Array &) = delete;

    /**
     * Move constructor of Array.
     */
    Array(Array &&t)
        : Array()
    {
        *this = std::move(t);
    }

    /**
     * Move assignment operator of Array.
     */
    Array &operator=(Array &&t)
    {
        static_cast<ArrayBase &>(*this) = std::move(t);
        return *this;
    }

    /**
     * Set size of the Array.
     * @param size size of the Array.
     */
    void setSize(std::size_t size)
    {
        const std::size_t oldSize = getSize();
        ArrayBase::setSize(size);
        if (isCPU())
        {
            // shrinking case
            for (std::size_t i = size; i < oldSize; ++i)
            {
                reinterpret_cast<T *>(getElement(i))->~T();
            }
            // expanding case
            for (std::size_t i = oldSize; i < size; ++i)
            {
                new (getElement(i)) T;
            }
        }
    }

    /**
     * Const array index operator.
     * @param idx index of element.
     * @return const reference to the specified element.
     */
    const T &operator[](int idx) const
    {
        assert(idx >= 0 && idx < getSize());
        return *reinterpret_cast<T *>(getElement(idx));
    }

    /**
     * Array index operator.
     * @param idx index of element.
     * @return reference to the specified element.
     */
    T &operator[](int idx)
    {
        assert(idx >= 0 && idx < getSize());
        return *reinterpret_cast<T *>(getElement(idx));
    }
};

/**
 * Implementation of ArrayN class.
 * @tparam T type of element in array.
 * @tparam N capacity of array.
 */
template<typename T, std::size_t N>
class ArrayN : public ArrayBase
{
public:
    /**
     * Default constructor of ArrayN (create an owning Tensor with capacity N).
     */
    ArrayN()
        : ArrayBase{N, sizeof(T), true}
    {
        setSize(N);
    }

    /**
     * Constructor of a non-owning ArrayN.
     * @param size size of the array.
     * @param dataPtr data pointer to the raw source array.
     * @param isCPU whether to allocate array on CPU or GPU.
     */
    ArrayN(std::size_t size, void *dataPtr, bool isCPU = true)
        : ArrayBase{N, sizeof(T), dataPtr, isCPU}
    {
        ArrayBase::setSize(size);
    }

    /**
     * Constructor of a memory-owning ArrayN.
     * @param isCPU whether to allocate array on CPU or GPU.
     */
    ArrayN(bool isCPU)
        : ArrayBase{N, sizeof(T), isCPU}
    {
        setSize(N);
    }

    /**
     * Destructor of the ArrayN.
     */
    ~ArrayN()
    {
        // call resize here such that CPU-based destructor
        // will call destructors of the objects stored
        // in the array before deallocating the storage
        setSize(0);
    }

    /**
     * ArrayN is non-copyable.
     */
    ArrayN(const ArrayN &) = delete;

    /**
     * ArrayN is non-copyable.
     */
    ArrayN &operator=(const ArrayN &) = delete;

    /**
     * Move constructor of ArrayN.
     */
    ArrayN(ArrayN &&t)
        : ArrayN()
    {
        *this = std::move(t);
    }

    /**
     * Move assignment operator of ArrayN.
     */
    ArrayN &operator=(ArrayN &&t)
    {
        static_cast<ArrayBase &>(*this) = std::move(t);
        return *this;
    }

    /**
     * Set size of the ArrayN.
     * @param size size of the ArrayN.
     */
    void setSize(std::size_t size)
    {
        const std::size_t oldSize = getSize();
        ArrayBase::setSize(size);
        if (isCPU())
        {
            // shrinking case
            for (std::size_t i = size; i < oldSize; ++i)
            {
                reinterpret_cast<T *>(getElement(i))->~T();
            }
            // expanding case
            for (std::size_t i = oldSize; i < size; ++i)
            {
                new (getElement(i)) T;
            }
        }
    }

    /**
     * Const ArrayN index operator.
     * @param idx index of element.
     * @return const reference to the specified element.
     */
    const T &operator[](int idx) const
    {
        assert(idx >= 0 && idx < getSize());
        return *reinterpret_cast<T *>(getElement(idx));
    }

    /**
     * ArrayN index operator.
     * @param idx index of element.
     * @return reference to the specified element.
     */
    T &operator[](int idx)
    {
        assert(idx >= 0 && idx < getSize());
        return *reinterpret_cast<T *>(getElement(idx));
    }
};

} // namespace cvcore

#endif // CVCORE_ARRAY_H
