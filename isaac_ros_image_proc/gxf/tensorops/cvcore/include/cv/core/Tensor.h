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

#ifndef CVCORE_TENSOR_H
#define CVCORE_TENSOR_H

#include <initializer_list>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace cvcore {

// there is no CUDA dependency at the data level, so we map half type to uint16_t for now
using half = std::uint16_t;

/**
 * An enum.
 * Enum type for tensor layout type.
 */
enum TensorLayout
{
    LC,          /**< length, channel (channel interleaved). */
    CL,          /**< channel, length (channel planar). */
    HWC,         /**< height, width, channel (channel interleaved). */
    CHW,         /**< channel, height, width (channel planar). */
    DHWC,        /**< depth, height, width, channel (channel interleaved). */
    NHWC = DHWC, /**< alias for DHWC. */
    DCHW,        /**< depth, channel, height, width (channel planar). */
    NCHW = DCHW, /**< alias for DCHW. */
    CDHW,        /**< channel, depth, height, width (channel planar). */
};

/**
 * An enum.
 * Enum type for tensor channel count.
 */
enum ChannelCount
{
    C1, /**< 1 channels. */
    C2, /**< 2 channels. */
    C3, /**< 3 channels. */
    C4, /**< 4 channels. */
    CX, /**< varying number of channels. */
};

/**
 * An enum.
 * Enum type for channel type.
 */
enum ChannelType
{
    U8,  /**< uint8_t. */
    U16, /**< uint16_t. */
    S8,  /**< int8_t. */
    S16, /**< int16_t. */
    F16, /**< cvcore::half. */
    F32, /**< float. */
    F64, /**< double. */
};

/**
 * An enum.
 * Enum type for dimension type.
 */
enum class TensorDimension
{
    LENGTH,  /**< length dimension. */
    HEIGHT,  /**< height dimension. */
    WIDTH,   /**< width dimension. */
    CHANNEL, /**< channel dimension. */
    DEPTH,   /**< depth dimension. */
};

/**
 * Function to get name of a TensorLayout value as string.
 * @param TL the TensorLayout value.
 * @return string name of TL.
 */
std::string GetTensorLayoutAsString(TensorLayout TL);

/**
 * Function to get name of a ChannelCount value as string.
 * @param CC the ChannelCount value.
 * @return string name of CC.
 */
std::string GetChannelCountAsString(ChannelCount CC);

/**
 * Function to get name of a ChannelType value as string.
 * @param CT the ChannelType value.
 * @return string name of CT.
 */
std::string GetChannelTypeAsString(ChannelType CT);

/**
 * Function to get name of a Memory type used.
 * @param bool isCPU of Tensor as input
 * @return string name of Memory type.
 */
std::string GetMemoryTypeAsString(bool isCPU);

/**
 * Function to get element size (in bytes) of a ChannelType.
 * @param CT the ChannelType value.
 * @return size in bytes.
 */
std::size_t GetChannelSize(ChannelType CT);

/**
 * Implementation of TensorBase class.
 */
class TensorBase
{
public:
    /**
     * Struct for storing dimension data.
     */
    struct DimData
    {
        std::size_t size;   /**< size of each dimension. */
        std::size_t stride; /**< stride of each dimension. */
    };

    /**
     * Constructor of a non-owning tensor.
     * @param type ChannelType of the Tensor.
     * @param dimData pointer to the DimData array.
     * @param dimCount number of dimensions.
     * @param dataPtr raw pointer to the source data array.
     * @param isCPU whether to allocate tensor on CPU or GPU.
     */
    TensorBase(ChannelType type, const DimData *dimData, int dimCount, void *dataPtr, bool isCPU);

    /**
     * Constructor of a non-owning tensor.
     * @param type ChannelType of the Tensor.
     * @param dimData initializer_list of DimData.
     * @param dataPtr raw pointer to the source data array.
     * @param isCPU whether to allocate tensor on CPU or GPU.
     */
    TensorBase(ChannelType type, std::initializer_list<DimData> dimData, void *dataPtr, bool isCPU);

    /**
     * Constructor of a memory-owning tensor.
     * @param type ChannelType of the Tensor.
     * @param dimData pointer to the DimData array.
     * @param dimCount number of dimensions.
     * @param isCPU whether to allocate tensor on CPU or GPU.
     */
    TensorBase(ChannelType type, const DimData *dimData, int dimCount, bool isCPU);

    /**
     * Constructor of a memory-owning tensor.
     * @param type ChannelType of the Tensor.
     * @param dimData initializer_list of DimData.
     * @param isCPU whether to allocate tensor on CPU or GPU.
     */
    TensorBase(ChannelType type, std::initializer_list<DimData> dimData, bool isCPU);

    /**
     * Destructor of Tensor.
     */
    ~TensorBase();

    /**
     * TensorBase is non-copyable.
     */
    TensorBase(const TensorBase &) = delete;

    /**
     * TensorBase is non-copyable.
     */
    TensorBase &operator=(const TensorBase &) = delete;

    /**
     * Move Constructor of TensorBase.
     */
    TensorBase(TensorBase &&);

    /**
     * Move operator of TensorBase.
     */
    TensorBase &operator=(TensorBase &&);

    /**
     * Get the dimension count of TensorBase.
     * @return number of dimensions.
     */
    int getDimCount() const;

    /**
     * Get the size of given dimension.
     * @param dimIdx dimension index.
     * @return size of the specified dimension.
     */
    std::size_t getSize(int dimIdx) const;

    /**
     * Get the stride of given dimension.
     * @param dimIdx dimension index.
     * @return stride of the specified dimension.
     */
    std::size_t getStride(int dimIdx) const;

    /**
     * Get the ChannelType of the Tensor.
     * @return ChannelType of the Tensor.
     */
    ChannelType getType() const;

    /**
     * Get the raw data pointer to the Tensor.
     * @return void data pointer to the Tensor.
     */
    void *getData() const;

    /**
     * Get the total size of the Tensor in bytes.
     * @return total bytes of the Tensor.
     */
    std::size_t getDataSize() const;

    /**
     * Get the flag whether the Tensor is allocated in CPU or GPU.
     * @return whether the Tensor is allocated in CPU.
     */
    bool isCPU() const;

    /**
     * Get the flag whether the Tensor owns the data.
     * @return whether the Tensor owns data in memory.
     */
    bool isOwning() const;

protected:
    TensorBase();

private:
    static constexpr int kMinDimCount = 2;
    static constexpr int kMaxDimCount = 4;

    void *m_data;

    int m_dimCount;
    DimData m_dimData[kMaxDimCount];

    ChannelType m_type;
    bool m_isOwning;
    bool m_isCPU;
};

namespace detail {

template<TensorLayout TL>
struct DimToIndex2D
{
    static_assert(TL == LC || TL == CL, "unsupported variant!");
    static constexpr int kLength  = TL == LC ? 0 : 1;
    static constexpr int kChannel = TL == LC ? 1 : 0;
};

template<TensorLayout TL>
struct DimToIndex3D
{
    static_assert(TL == HWC || TL == CHW, "unsupported variant!");
    static constexpr int kWidth   = TL == HWC ? 1 : 2;
    static constexpr int kHeight  = TL == HWC ? 0 : 1;
    static constexpr int kChannel = TL == HWC ? 2 : 0;
};

template<TensorLayout TL>
struct DimToIndex4D
{
    static_assert(TL == DHWC || TL == DCHW || TL == CDHW, "unsupported variant!");
    static constexpr int kWidth   = TL == DHWC ? 2 : (TL == DCHW ? 3 : 3);
    static constexpr int kHeight  = TL == DHWC ? 1 : (TL == DCHW ? 2 : 2);
    static constexpr int kDepth   = TL == DHWC ? 0 : (TL == DCHW ? 0 : 1);
    static constexpr int kChannel = TL == DHWC ? 3 : (TL == DCHW ? 1 : 0);
};

template<TensorLayout TL, typename = void>
struct LayoutToIndex
{
};

template<TensorLayout TL>
struct LayoutToIndex<TL, typename std::enable_if<TL == LC || TL == CL>::type> : public DimToIndex2D<TL>
{
    static constexpr int kDimCount = 2;
};

template<TensorLayout TL>
struct LayoutToIndex<TL, typename std::enable_if<TL == HWC || TL == CHW>::type> : public DimToIndex3D<TL>
{
    static constexpr int kDimCount = 3;
};

template<TensorLayout TL>
struct LayoutToIndex<TL, typename std::enable_if<TL == DHWC || TL == DCHW || TL == CDHW>::type>
    : public DimToIndex4D<TL>
{
    static constexpr int kDimCount = 4;
};

template<ChannelType CT>
struct ChannelTypeToNative
{
};

template<>
struct ChannelTypeToNative<U8>
{
    using Type = std::uint8_t;
};

template<>
struct ChannelTypeToNative<U16>
{
    using Type = std::uint16_t;
};

template<>
struct ChannelTypeToNative<S8>
{
    using Type = std::int8_t;
};

template<>
struct ChannelTypeToNative<S16>
{
    using Type = std::int16_t;
};

template<>
struct ChannelTypeToNative<F32>
{
    using Type = float;
};

template<>
struct ChannelTypeToNative<F16>
{
    using Type = cvcore::half;
};

template<>
struct ChannelTypeToNative<F64>
{
    using Type = double;
};

template<ChannelCount CC>
constexpr std::size_t ChannelToCount()
{
    switch (CC)
    {
    case C1:
        return 1;
    case C2:
        return 2;
    case C3:
        return 3;
    case C4:
        return 4;
    }
    return 0; // this is safe as this function will never be called for dynamic channel counts
}

/**
 * Implementation of 2D tensors.
 * @tparam TL tensor layout type.
 * @tparam CT channel type.
 */
template<TensorLayout TL, ChannelType CT>
class Tensor2D : public TensorBase
{
    using DataType = typename ChannelTypeToNative<CT>::Type;

public:
    /**
     * Default Constructor.
     */
    Tensor2D() = default;

    /**
     * Constructor of a memory-owning 2D tensor.
     * @param dimData initializer_list of DimData.
     * @param isCPU whether to allocate tensor on CPU or GPU.
     */
    Tensor2D(std::initializer_list<DimData> dimData, bool isCPU)
        : TensorBase(CT, dimData, isCPU)
    {
    }

    /**
     * Constructor of a non-owning 2D tensor.
     * @param dimData initializer_list of DimData.
     * @param dataPtr raw pointer to the source data array.
     * @param isCPU whether to allocate tensor on CPU or GPU.
     */
    Tensor2D(std::initializer_list<DimData> dimData, DataType *dataPtr, bool isCPU)
        : TensorBase(CT, dimData, dataPtr, isCPU)
    {
    }

    /**
     * Get the length of the 2D tensor.
     * @return length of the 2D tensor.
     */
    std::size_t getLength() const
    {
        return getSize(DimToIndex2D<TL>::kLength);
    }

    /**
     * Get the channel count of the 2D tensor.
     * @return channel count of the 2D tensor.
     */
    std::size_t getChannelCount() const
    {
        return getSize(DimToIndex2D<TL>::kChannel);
    }

    /**
     * Expose base getStride() function.
     */
    using TensorBase::getStride;

    /**
     * Get the stride of the 2D tensor.
     * @param dim tensor dimension.
     * @return tensor stride of the given dimension.
     */
    std::size_t getStride(TensorDimension dim) const
    {
        switch (dim)
        {
        case TensorDimension::LENGTH:
            return getStride(DimToIndex2D<TL>::kLength);
        case TensorDimension::CHANNEL:
            return getStride(DimToIndex2D<TL>::kChannel);
        default:
            throw std::out_of_range("cvcore::Tensor2D::getStride ==> Requested TensorDimension is out of bounds");
        }
    }

    /**
     * Get the raw data pointer to the 2D tensor.
     * @return data pointer to the 2D tensor.
     */
    DataType *getData()
    {
        return reinterpret_cast<DataType *>(TensorBase::getData());
    }

    /**
     * Get the const raw data pointer to the 2D tensor.
     * @return const data pointer to the 2D tensor.
     */
    const DataType *getData() const
    {
        return reinterpret_cast<DataType *>(TensorBase::getData());
    }
};

/**
 * Implementation of 3D tensors.
 * @tparam TL tensor layout type.
 * @tparam CT channel type.
 */
template<TensorLayout TL, ChannelType CT>
class Tensor3D : public TensorBase
{
    using DataType = typename ChannelTypeToNative<CT>::Type;

public:
    /**
     * Default Constructor.
     */
    Tensor3D() = default;

    /**
     * Constructor of a memory-owning 3D tensor.
     * @param dimData initializer_list of DimData.
     * @param isCPU whether to allocate tensor on CPU or GPU.
     */
    Tensor3D(std::initializer_list<DimData> dimData, bool isCPU)
        : TensorBase(CT, dimData, isCPU)
    {
    }

    /**
     * Constructor of a non-owning 3D tensor.
     * @param dimData initializer_list of DimData.
     * @param dataPtr raw pointer to the source data array.
     * @param isCPU whether to allocate tensor on CPU or GPU.
     */
    Tensor3D(std::initializer_list<DimData> dimData, DataType *dataPtr, bool isCPU)
        : TensorBase(CT, dimData, dataPtr, isCPU)
    {
    }

    /**
     * Get the width of the 3D tensor.
     * @return width of the 3D tensor.
     */
    std::size_t getWidth() const
    {
        return getSize(DimToIndex3D<TL>::kWidth);
    }

    /**
     * Get the height of the 3D tensor.
     * @return height of the 3D tensor.
     */
    std::size_t getHeight() const
    {
        return getSize(DimToIndex3D<TL>::kHeight);
    }

    /**
     * Get the channel count of the 3D tensor.
     * @return channel count of the 3D tensor.
     */
    std::size_t getChannelCount() const
    {
        return getSize(DimToIndex3D<TL>::kChannel);
    }

    /**
     * Expose base getStride() function.
     */
    using TensorBase::getStride;

    /**
     * Get the stride of the 3D tensor.
     * @param dim tensor dimension.
     * @return tensor stride of the given dimension.
     */
    std::size_t getStride(TensorDimension dim) const
    {
        switch (dim)
        {
        case TensorDimension::HEIGHT:
            return getStride(DimToIndex3D<TL>::kHeight);
        case TensorDimension::WIDTH:
            return getStride(DimToIndex3D<TL>::kWidth);
        case TensorDimension::CHANNEL:
            return getStride(DimToIndex3D<TL>::kChannel);
        default:
            throw std::out_of_range("cvcore::Tensor3D::getStride ==> Requested TensorDimension is out of bounds");
        }
    }

    /**
     * Get the raw data pointer to the 3D tensor.
     * @return data pointer to the 3D tensor.
     */
    DataType *getData()
    {
        return reinterpret_cast<DataType *>(TensorBase::getData());
    }

    /**
     * Get the const raw data pointer to the 3D tensor.
     * @return const data pointer to the 3D tensor.
     */
    const DataType *getData() const
    {
        return reinterpret_cast<DataType *>(TensorBase::getData());
    }
};

/**
 * Implementation of 4D tensors.
 * @tparam TL tensor layout type.
 * @tparam CT channel type.
 */
template<TensorLayout TL, ChannelType CT>
class Tensor4D : public TensorBase
{
    using DataType = typename ChannelTypeToNative<CT>::Type;

public:
    /**
     * Default Constructor.
     */
    Tensor4D() = default;

    /**
     * Constructor of a memory-owning 4D tensor.
     * @param dimData initializer_list of DimData.
     * @param isCPU whether to allocate tensor on CPU or GPU.
     */
    Tensor4D(std::initializer_list<DimData> dimData, bool isCPU)
        : TensorBase(CT, dimData, isCPU)
    {
    }

    /**
     * Constructor of a non-owning 4D tensor.
     * @param dimData initializer_list of DimData.
     * @param dataPtr raw pointer to the source data array.
     * @param isCPU whether to allocate tensor on CPU or GPU.
     */
    Tensor4D(std::initializer_list<DimData> dimData, DataType *dataPtr, bool isCPU)
        : TensorBase(CT, dimData, dataPtr, isCPU)
    {
    }

    /**
     * Get the width of the 4D tensor.
     * @return width of the 4D tensor.
     */
    std::size_t getWidth() const
    {
        return getSize(DimToIndex4D<TL>::kWidth);
    }

    /**
     * Get the height of the 4D tensor.
     * @return height of the 4D tensor.
     */
    std::size_t getHeight() const
    {
        return getSize(DimToIndex4D<TL>::kHeight);
    }

    /**
     * Get the depth of the 4D tensor.
     * @return depth of the 4D tensor.
     */
    std::size_t getDepth() const
    {
        return getSize(DimToIndex4D<TL>::kDepth);
    }

    /**
     * Get the channel count of the 4D tensor.
     * @return channel count of the 4D tensor.
     */
    std::size_t getChannelCount() const
    {
        return getSize(DimToIndex4D<TL>::kChannel);
    }

    /**
     * Expose base getStride() function.
     */
    using TensorBase::getStride;

    /**
     * Get the stride of the 4D tensor.
     * @param dim tensor dimension.
     * @return tensor stride of the given dimension.
     */
    std::size_t getStride(TensorDimension dim) const
    {
        switch (dim)
        {
        case TensorDimension::HEIGHT:
            return getStride(DimToIndex4D<TL>::kHeight);
        case TensorDimension::WIDTH:
            return getStride(DimToIndex4D<TL>::kWidth);
        case TensorDimension::CHANNEL:
            return getStride(DimToIndex4D<TL>::kChannel);
        case TensorDimension::DEPTH:
            return getStride(DimToIndex4D<TL>::kDepth);
        default:
            throw std::out_of_range("cvcore::Tensor4D::getStride ==> Requested TensorDimension is out of bounds");
        }
    }

    /**
     * Get the raw data pointer to the 4D tensor.
     * @return data pointer to the 4D tensor.
     */
    DataType *getData()
    {
        return reinterpret_cast<DataType *>(TensorBase::getData());
    }

    /**
     * Get the const raw data pointer to the 4D tensor.
     * @return const data pointer to the 4D tensor.
     */
    const DataType *getData() const
    {
        return reinterpret_cast<DataType *>(TensorBase::getData());
    }
};

} // namespace detail

template<TensorLayout TL, ChannelCount CC, ChannelType CT>
class Tensor;

// 2D Tensors

/**
 * 2D LC tensors.
 * @tparam CC channel count.
 * @tparam CT channel type.
 */
template<ChannelCount CC, ChannelType CT>
class Tensor<LC, CC, CT> : public detail::Tensor2D<LC, CT>
{
public:
    using DataType = typename detail::ChannelTypeToNative<CT>::Type;

    static constexpr ChannelCount kChannelCount = CC;

    Tensor() = default;

    template<ChannelCount T = CC, typename = typename std::enable_if<T != CX>::type>
    Tensor(std::size_t length, bool isCPU = true)
        : detail::Tensor2D<LC, CT>({{length, detail::ChannelToCount<CC>()}, {detail::ChannelToCount<CC>(), 1}}, isCPU)
    {
    }

    template<ChannelCount T = CC, typename = typename std::enable_if<T == CX>::type>
    Tensor(std::size_t length, std::size_t channelCount, bool isCPU = true)
        : detail::Tensor2D<LC, CT>({{length, channelCount}, {channelCount, 1}}, isCPU)
    {
    }

    template<ChannelCount T = CC, typename = typename std::enable_if<T != CX>::type>
    Tensor(std::size_t length, DataType *dataPtr, bool isCPU = true)
        : detail::Tensor2D<LC, CT>({{length, detail::ChannelToCount<CC>()}, {detail::ChannelToCount<CC>(), 1}}, dataPtr,
                                   isCPU)
    {
    }

    template<ChannelCount T = CC, typename = typename std::enable_if<T == CX>::type>
    Tensor(std::size_t length, std::size_t channelCount, DataType *dataPtr, bool isCPU = true)
        : detail::Tensor2D<LC, CT>({{length, channelCount}, {channelCount, 1}}, dataPtr, isCPU)
    {
    }
};

/**
 * 2D CL tensors.
 * @tparam CC channel count.
 * @tparam CT channel type.
 */
template<ChannelCount CC, ChannelType CT>
class Tensor<CL, CC, CT> : public detail::Tensor2D<CL, CT>
{
public:
    using DataType = typename detail::ChannelTypeToNative<CT>::Type;

    static constexpr ChannelCount kChannelCount = CC;

    Tensor() = default;

    template<ChannelCount T = CC, typename = typename std::enable_if<T != CX>::type>
    Tensor(std::size_t length, bool isCPU = true)
        : detail::Tensor2D<CL, CT>({{detail::ChannelToCount<CC>(), length}, {length, 1}}, isCPU)
    {
    }

    template<ChannelCount T = CC, typename = typename std::enable_if<T == CX>::type>
    Tensor(std::size_t length, std::size_t channelCount, bool isCPU = true)
        : detail::Tensor2D<CL, CT>({{channelCount, length}, {length, 1}}, isCPU)
    {
    }

    template<ChannelCount T = CC, typename = typename std::enable_if<T != CX>::type>
    Tensor(std::size_t length, DataType *dataPtr, bool isCPU = true)
        : detail::Tensor2D<CL, CT>({{detail::ChannelToCount<CC>(), length}, {length, 1}}, dataPtr, isCPU)
    {
    }

    template<ChannelCount T = CC, typename = typename std::enable_if<T == CX>::type>
    Tensor(std::size_t length, std::size_t channelCount, DataType *dataPtr, bool isCPU = true)
        : detail::Tensor2D<CL, CT>({{channelCount, length}, {length, 1}}, dataPtr, isCPU)
    {
    }
};

// 3D Tensors

/**
 * 3D HWC tensors.
 * @tparam CC channel count.
 * @tparam CT channel type.
 */
template<ChannelCount CC, ChannelType CT>
class Tensor<HWC, CC, CT> : public detail::Tensor3D<HWC, CT>
{
public:
    using DataType = typename detail::ChannelTypeToNative<CT>::Type;

    static constexpr ChannelCount kChannelCount = CC;

    Tensor() = default;

    template<ChannelCount T = CC, typename B = bool,
             typename std::enable_if<T != CX && std::is_same<B, bool>::value>::type * = nullptr>
    Tensor(std::size_t width, std::size_t height, B isCPU = true)
        : detail::Tensor3D<HWC, CT>({{height, width * detail::ChannelToCount<CC>()},
                                     {width, detail::ChannelToCount<CC>()},
                                     {detail::ChannelToCount<CC>(), 1}},
                                    isCPU)
    {
    }

    template<ChannelCount T = CC, typename B = bool,
             typename std::enable_if<T == CX && std::is_same<B, bool>::value>::type * = nullptr>
    Tensor(std::size_t width, std::size_t height, std::size_t channelCount, B isCPU = true)
        : detail::Tensor3D<HWC, CT>({{height, width * channelCount}, {width, channelCount}, {channelCount, 1}}, isCPU)
    {
    }

    template<ChannelCount T = CC, typename B = bool,
             typename std::enable_if<T != CX && std::is_same<B, bool>::value>::type * = nullptr>
    Tensor(std::size_t width, std::size_t height, DataType *dataPtr, B isCPU = true)
        : detail::Tensor3D<HWC, CT>({{height, width * detail::ChannelToCount<CC>()},
                                     {width, detail::ChannelToCount<CC>()},
                                     {detail::ChannelToCount<CC>(), 1}},
                                    dataPtr, isCPU)
    {
    }

    template<ChannelCount T = CC, typename B = bool,
             typename std::enable_if<T != CX && std::is_same<B, bool>::value>::type * = nullptr>
    Tensor(std::size_t width, std::size_t height, std::size_t rowPitch, DataType *dataPtr, B isCPU = true)
        : detail::Tensor3D<HWC, CT>({{height, rowPitch / GetChannelSize(CT)},
                                     {width, detail::ChannelToCount<CC>()},
                                     {detail::ChannelToCount<CC>(), 1}},
                                    dataPtr, isCPU)
    {
        if (rowPitch % GetChannelSize(CT) != 0)
        {
            throw std::domain_error(
                "cvcore::Tensor<HWC, CC, CT>::Tensor ==> Parameter rowPitch is not evenly divisible by channel size");
        }
    }

    template<ChannelCount T = CC, typename B = bool,
             typename std::enable_if<T == CX && std::is_same<B, bool>::value>::type * = nullptr>
    Tensor(std::size_t width, std::size_t height, std::size_t channelCount, DataType *dataPtr, B isCPU = true)
        : detail::Tensor3D<HWC, CT>({{height, width * channelCount}, {width, channelCount}, {channelCount, 1}}, dataPtr,
                                    isCPU)
    {
    }

    template<ChannelCount T = CC, typename B = bool,
             typename std::enable_if<T == CX && std::is_same<B, bool>::value>::type * = nullptr>
    Tensor(std::size_t width, std::size_t height, std::size_t channelCount, std::size_t rowPitch, DataType *dataPtr,
           B isCPU = true)
        : detail::Tensor3D<HWC, CT>({{height, rowPitch / GetChannelSize(CT)}, {width, channelCount}, {channelCount, 1}},
                                    dataPtr, isCPU)
    {
        if (rowPitch % GetChannelSize(CT) != 0)
        {
            throw std::domain_error(
                "cvcore::Tensor<HWC, CC, CT>::Tensor ==> Parameter rowPitch is not evenly divisible by channel size");
        }
    }
};

/**
 * 3D CHW tensors.
 * @tparam CC channel count.
 * @tparam CT channel type.
 */
template<ChannelCount CC, ChannelType CT>
class Tensor<CHW, CC, CT> : public detail::Tensor3D<CHW, CT>
{
public:
    using DataType = typename detail::ChannelTypeToNative<CT>::Type;

    static constexpr ChannelCount kChannelCount = CC;

    Tensor() = default;

    template<ChannelCount T = CC, typename = typename std::enable_if<T != CX>::type>
    Tensor(std::size_t width, std::size_t height, bool isCPU = true)
        : detail::Tensor3D<CHW, CT>({{detail::ChannelToCount<CC>(), width * height}, {height, width}, {width, 1}},
                                    isCPU)
    {
    }

    template<ChannelCount T = CC, typename = typename std::enable_if<T == CX>::type>
    Tensor(std::size_t width, std::size_t height, std::size_t channelCount, bool isCPU = true)
        : detail::Tensor3D<CHW, CT>({{channelCount, width * height}, {height, width}, {width, 1}}, isCPU)
    {
    }

    template<ChannelCount T = CC, typename = typename std::enable_if<T != CX>::type>
    Tensor(std::size_t width, std::size_t height, DataType *dataPtr, bool isCPU = true)
        : detail::Tensor3D<CHW, CT>({{detail::ChannelToCount<CC>(), width * height}, {height, width}, {width, 1}},
                                    dataPtr, isCPU)
    {
    }

    template<ChannelCount T = CC, typename B = bool,
             typename std::enable_if<T != CX && std::is_same<B, bool>::value>::type * = nullptr>
    Tensor(std::size_t width, std::size_t height, std::size_t rowPitch, DataType *dataPtr, B isCPU = true)
        : detail::Tensor3D<CHW, CT>({{detail::ChannelToCount<CC>(), height * rowPitch / GetChannelSize(CT)},
                                     {height, rowPitch / GetChannelSize(CT)},
                                     {width, 1}},
                                    dataPtr, isCPU)
    {
        if (rowPitch % GetChannelSize(CT) != 0)
        {
            throw std::domain_error(
                "cvcore::Tensor<CHW, CC, CT>::Tensor ==> Parameter rowPitch is not evenly divisible by channel size");
        }
    }

    template<ChannelCount T = CC, typename = typename std::enable_if<T == CX>::type>
    Tensor(std::size_t width, std::size_t height, std::size_t channelCount, DataType *dataPtr, bool isCPU = true)
        : detail::Tensor3D<CHW, CT>({{channelCount, width * height}, {height, width}, {width, 1}}, dataPtr, isCPU)
    {
    }

    template<ChannelCount T = CC, typename B = bool,
             typename std::enable_if<T == CX && std::is_same<B, bool>::value>::type * = nullptr>
    Tensor(std::size_t width, std::size_t height, std::size_t channelCount, std::size_t rowPitch, DataType *dataPtr,
           B isCPU = true)
        : detail::Tensor3D<CHW, CT>({{channelCount, height * rowPitch / GetChannelSize(CT)},
                                     {height, rowPitch / GetChannelSize(CT)},
                                     {width, 1}},
                                    dataPtr, isCPU)
    {
        if (rowPitch % GetChannelSize(CT) != 0)
        {
            throw std::domain_error(
                "cvcore::Tensor<CHW, CC, CT>::Tensor ==> Parameter rowPitch is not evenly divisible by channel size");
        }
    }
};

// 4D Tensors

/**
 * 4D DHWC tensors.
 * @tparam CC channel count.
 * @tparam CT channel type.
 */
template<ChannelCount CC, ChannelType CT>
class Tensor<DHWC, CC, CT> : public detail::Tensor4D<DHWC, CT>
{
public:
    using DataType = typename detail::ChannelTypeToNative<CT>::Type;

    static constexpr ChannelCount kChannelCount = CC;

    Tensor() = default;

    template<ChannelCount T = CC, typename B = bool,
             typename std::enable_if<T != CX && std::is_same<B, bool>::value>::type * = nullptr>
    Tensor(std::size_t width, std::size_t height, std::size_t depth, B isCPU = true)
        : detail::Tensor4D<DHWC, CT>({{depth, height * width * detail::ChannelToCount<CC>()},
                                      {height, width * detail::ChannelToCount<CC>()},
                                      {width, detail::ChannelToCount<CC>()},
                                      {detail::ChannelToCount<CC>(), 1}},
                                     isCPU)
    {
    }

    template<ChannelCount T = CC, typename B = bool,
             typename std::enable_if<T == CX && std::is_same<B, bool>::value>::type * = nullptr>
    Tensor(std::size_t width, std::size_t height, std::size_t depth, std::size_t channelCount, B isCPU = true)
        : detail::Tensor4D<DHWC, CT>({{depth, height * width * channelCount},
                                      {height, width * channelCount},
                                      {width, channelCount},
                                      {channelCount, 1}},
                                     isCPU)
    {
    }

    template<ChannelCount T = CC, typename B = bool,
             typename std::enable_if<T != CX && std::is_same<B, bool>::value>::type * = nullptr>
    Tensor(std::size_t width, std::size_t height, std::size_t depth, DataType *dataPtr, B isCPU = true)
        : detail::Tensor4D<DHWC, CT>({{depth, height * width * detail::ChannelToCount<CC>()},
                                      {height, width * detail::ChannelToCount<CC>()},
                                      {width, detail::ChannelToCount<CC>()},
                                      {detail::ChannelToCount<CC>(), 1}},
                                     dataPtr, isCPU)
    {
    }

    template<ChannelCount T = CC, typename B = bool,
             typename std::enable_if<T != CX && std::is_same<B, bool>::value>::type * = nullptr>
    Tensor(std::size_t width, std::size_t height, std::size_t depth, std::size_t rowPitch, DataType *dataPtr,
           B isCPU = true)
        : detail::Tensor4D<DHWC, CT>({{depth, height * rowPitch / GetChannelSize(CT)},
                                      {height, rowPitch / GetChannelSize(CT)},
                                      {width, detail::ChannelToCount<CC>()},
                                      {detail::ChannelToCount<CC>(), 1}},
                                     dataPtr, isCPU)
    {
        if (rowPitch % GetChannelSize(CT) != 0)
        {
            throw std::domain_error(
                "cvcore::Tensor<DHWC, CC, CT>::Tensor ==> Parameter rowPitch is not evenly divisible by channel size");
        }
    }

    template<ChannelCount T = CC, typename B = bool,
             typename std::enable_if<T == CX && std::is_same<B, bool>::value>::type * = nullptr>
    Tensor(std::size_t width, std::size_t height, std::size_t depth, std::size_t channelCount, DataType *dataPtr,
           B isCPU = true)
        : detail::Tensor4D<DHWC, CT>({{depth, height * width * channelCount},
                                      {height, width * channelCount},
                                      {width, channelCount},
                                      {channelCount, 1}},
                                     dataPtr, isCPU)
    {
    }

    template<ChannelCount T = CC, typename B = bool,
             typename std::enable_if<T == CX && std::is_same<B, bool>::value>::type * = nullptr>
    Tensor(std::size_t width, std::size_t height, std::size_t depth, std::size_t channelCount, std::size_t rowPitch,
           DataType *dataPtr, B isCPU = true)
        : detail::Tensor4D<DHWC, CT>({{depth, height * rowPitch / GetChannelSize(CT)},
                                      {height, rowPitch / GetChannelSize(CT)},
                                      {width, channelCount},
                                      {channelCount, 1}},
                                     dataPtr, isCPU)
    {
        if (rowPitch % GetChannelSize(CT) != 0)
        {
            throw std::domain_error(
                "cvcore::Tensor<DHWC, CC, CT>::Tensor ==> Parameter rowPitch is not evenly divisible by channel size");
        }
    }
};

/**
 * 4D DCHW tensors.
 * @tparam CC channel count.
 * @tparam CT channel type.
 */
template<ChannelCount CC, ChannelType CT>
class Tensor<DCHW, CC, CT> : public detail::Tensor4D<DCHW, CT>
{
public:
    using DataType = typename detail::ChannelTypeToNative<CT>::Type;

    static constexpr ChannelCount kChannelCount = CC;

    Tensor() = default;

    template<ChannelCount T = CC, typename = typename std::enable_if<T != CX>::type>
    Tensor(std::size_t width, std::size_t height, std::size_t depth, bool isCPU = true)
        : detail::Tensor4D<DCHW, CT>({{depth, detail::ChannelToCount<CC>() * width * height},
                                      {detail::ChannelToCount<CC>(), width * height},
                                      {height, width},
                                      {width, 1}},
                                     isCPU)
    {
    }

    template<ChannelCount T = CC, typename = typename std::enable_if<T == CX>::type>
    Tensor(std::size_t width, std::size_t height, std::size_t depth, std::size_t channelCount, bool isCPU = true)
        : detail::Tensor4D<DCHW, CT>(
              {{depth, channelCount * width * height}, {channelCount, width * height}, {height, width}, {width, 1}},
              isCPU)
    {
    }

    template<ChannelCount T = CC, typename = typename std::enable_if<T != CX>::type>
    Tensor(std::size_t width, std::size_t height, std::size_t depth, DataType *dataPtr, bool isCPU = true)
        : detail::Tensor4D<DCHW, CT>({{depth, detail::ChannelToCount<CC>() * width * height},
                                      {detail::ChannelToCount<CC>(), width * height},
                                      {height, width},
                                      {width, 1}},
                                     dataPtr, isCPU)
    {
    }

    template<ChannelCount T = CC, typename B = bool,
             typename std::enable_if<T != CX && std::is_same<B, bool>::value>::type * = nullptr>
    Tensor(std::size_t width, std::size_t height, std::size_t depth, std::size_t rowPitch, DataType *dataPtr,
           B isCPU = true)
        : detail::Tensor4D<DCHW, CT>({{depth, detail::ChannelToCount<CC>() * height * rowPitch / GetChannelSize(CT)},
                                      {detail::ChannelToCount<CC>(), height * rowPitch / GetChannelSize(CT)},
                                      {height, rowPitch / GetChannelSize(CT)},
                                      {width, 1}},
                                     dataPtr, isCPU)
    {
        if (rowPitch % GetChannelSize(CT) != 0)
        {
            throw std::domain_error(
                "cvcore::Tensor<DCHW, CC, CT>::Tensor ==> Parameter rowPitch is not evenly divisible by channel size");
        }
    }

    template<ChannelCount T = CC, typename = typename std::enable_if<T == CX>::type>
    Tensor(std::size_t width, std::size_t height, std::size_t depth, std::size_t channelCount, DataType *dataPtr,
           bool isCPU = true)
        : detail::Tensor4D<DCHW, CT>(
              {{depth, channelCount * width * height}, {channelCount, width * height}, {height, width}, {width, 1}},
              dataPtr, isCPU)
    {
    }

    template<ChannelCount T = CC, typename B = bool,
             typename std::enable_if<T == CX && std::is_same<B, bool>::value>::type * = nullptr>
    Tensor(std::size_t width, std::size_t height, std::size_t depth, std::size_t channelCount, std::size_t rowPitch,
           DataType *dataPtr, B isCPU = true)
        : detail::Tensor4D<DCHW, CT>({{depth, channelCount * height * rowPitch / GetChannelSize(CT)},
                                      {channelCount, height * rowPitch / GetChannelSize(CT)},
                                      {height, rowPitch / GetChannelSize(CT)},
                                      {width, 1}},
                                     dataPtr, isCPU)
    {
        if (rowPitch % GetChannelSize(CT) != 0)
        {
            throw std::domain_error(
                "cvcore::Tensor<DCHW, CC, CT>::Tensor ==> Parameter rowPitch is not evenly divisible by channel size");
        }
    }
};

/**
 * 4D CDHW tensors.
 * @tparam CC channel count.
 * @tparam CT channel type.
 */
template<ChannelCount CC, ChannelType CT>
class Tensor<CDHW, CC, CT> : public detail::Tensor4D<CDHW, CT>
{
public:
    using DataType = typename detail::ChannelTypeToNative<CT>::Type;

    static constexpr ChannelCount kChannelCount = CC;

    Tensor() = default;

    template<ChannelCount T = CC, typename = typename std::enable_if<T != CX>::type>
    Tensor(std::size_t width, std::size_t height, std::size_t depth, bool isCPU = true)
        : detail::Tensor4D<CDHW, CT>({{detail::ChannelToCount<CC>(), depth * width * height},
                                      {depth, width * height},
                                      {height, width},
                                      {width, 1}},
                                     isCPU)
    {
    }

    template<ChannelCount T = CC, typename = typename std::enable_if<T == CX>::type>
    Tensor(std::size_t width, std::size_t height, std::size_t depth, std::size_t channelCount, bool isCPU = true)
        : detail::Tensor4D<CDHW, CT>(
              {{channelCount, depth * width * height}, {depth, width * height}, {height, width}, {width, 1}}, isCPU)
    {
    }

    template<ChannelCount T = CC, typename = typename std::enable_if<T != CX>::type>
    Tensor(std::size_t width, std::size_t height, std::size_t depth, DataType *dataPtr, bool isCPU = true)
        : detail::Tensor4D<CDHW, CT>({{detail::ChannelToCount<CC>(), depth * width * height},
                                      {depth, width * height},
                                      {height, width},
                                      {width, 1}},
                                     dataPtr, isCPU)
    {
    }

    template<ChannelCount T = CC, typename = typename std::enable_if<T == CX>::type>
    Tensor(std::size_t width, std::size_t height, std::size_t depth, std::size_t channelCount, DataType *dataPtr,
           bool isCPU = true)
        : detail::Tensor4D<CDHW, CT>(
              {{channelCount, depth * width * height}, {depth, width * height}, {height, width}, {width, 1}}, dataPtr,
              isCPU)
    {
    }
};

} // namespace cvcore

#endif // CVCORE_TENSOR_H
