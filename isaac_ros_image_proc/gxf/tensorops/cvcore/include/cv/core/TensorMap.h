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

#ifndef CVCORE_TENSORMAP_H
#define CVCORE_TENSORMAP_H

#include <type_traits>
#include <set>
#include <queue>
#include <vector>
#include <unordered_map>

#include "Traits.h"
#include "TensorList.h"

namespace cvcore {

/**
 * @brief Implementation of a map of tensors of the same rank but, potentially, different dimensions, over the batch dimension
 * 
 * @tparam TensorType Any CVCORE tensor type
 * @tparam KeyType Any STL hashable data type
 */
template<typename TensorType, typename KeyType = std::size_t,
         typename = void>
class TensorMap {};

template<TensorLayout TL, ChannelCount CC, ChannelType CT,
         typename KT>
class TensorMap<Tensor<TL, CC, CT>, KT,
                typename std::enable_if<traits::is_batch<Tensor<TL, CC, CT>>::value>::type>
{
    using my_type = TensorMap<Tensor<TL, CC, CT>, KT>;

    public:
        using key_type = KT;
        using unit_type = Tensor<TL, CC, CT>;
        using element_type = traits::remove_batch_t<unit_type>;
        using frame_type = TensorList<element_type>;
        using buffer_type = TensorList<unit_type>;

        template<ChannelCount T = CC, typename = void>
        struct dim_data_type
        {
            std::size_t height;
            std::size_t width;
        };
        
        template<ChannelCount T>
        struct dim_data_type<T, typename std::enable_if<T == CX>::type>
        {
            std::size_t height;
            std::size_t width;
            std::size_t channels;
        };

        TensorMap() = default;
        TensorMap(const my_type &) = delete;
        TensorMap(my_type && other)
        {
            *this = std::move(other);
        }

        /**
         * @brief Construct a new Tensor Map object
         * 
         * @param batchSize The batch dimension of all sub-tensors
         * @param dimData  The dimensional description of all sub-tensors in at least HWC format
         * @param isCPU A boolean flag specifying what device to allocate the sub-tensors
         */
        template<ChannelCount T = CC, typename = typename std::enable_if<T != CX>::type>
        TensorMap(std::size_t batchSize,
                  const std::vector<dim_data_type<T>> & dimData, 
                  bool isCPU = true)
            : m_maxBatchSize{batchSize}, m_size{dimData.size()},
              m_isCPU{isCPU}, m_buffer{dimData.size(), true}
        {
            m_buffer.setSize(m_size);

            int i = 0;
            for(auto dim: dimData)
            {
                m_buffer[i] = std::move(unit_type(dim.width,
                                                  dim.height,
                                                  m_maxBatchSize,
                                                  m_isCPU));
                ++i;
            }

            for(std::size_t i = 0; i < m_maxBatchSize; ++i)
            {
                m_pool.insert(i);
            }
        }

        /**
         * @brief Construct a new Tensor Map object
         * 
         * @param batchSize The batch dimension of all sub-tensors
         * @param dimData  The dimensional description of all sub-tensors in at least HWC format
         * @param isCPU A boolean flag specifying what device to allocate the sub-tensors
         */
        template<ChannelCount T = CC, typename = typename std::enable_if<T == CX>::type>
        TensorMap(std::size_t batchSize,
                  const std::vector<dim_data_type<CX>> & dimData,
                  bool isCPU = true)
            : m_maxBatchSize{batchSize}, m_size{dimData.size()},
              m_isCPU{isCPU}, m_buffer{dimData.size(), true}
        {
            m_buffer.setSize(m_size);

            int i = 0;
            for(auto dim: dimData)
            {
                m_buffer[i] = std::move(unit_type(dim.width,
                                                  dim.height,
                                                  m_maxBatchSize,
                                                  dim.channels,
                                                  m_isCPU));
                ++i;
            }

            for(std::size_t i = 0; i < m_maxBatchSize; ++i)
            {
                m_pool.insert(i);
            }
        }

        ~TensorMap() = default;

        my_type & operator=(const my_type &) = delete;
        my_type & operator=(my_type && other)
        {
            std::swap(m_mapping, other.m_mapping);
            std::swap(m_pool, other.m_pool);
            std::swap(m_maxBatchSize, other.m_maxBatchSize);
            std::swap(m_size, other.m_size);
            std::swap(m_isCPU, other.m_isCPU);
            std::swap(m_buffer, other.m_buffer);

            return *this;
        }

        /**
         * @brief A mapping of the batch dimension index to a given key
         * 
         * @details Given a set of pairs such that the keys AND values are unique
         *   respectively, the key-wise mapping of the batch dimension is reset
         *   to the provided values.
         * 
         * @param pairs An unordered map of the uniqe key value pairs
         * @return true If the length of ``pairs`` is less than the max batch size
         *   and the key value pairs are one-to-one and onto.
         * @return false If the conditions of ``true`` are not met.
         */
        bool remap(const std::unordered_map<key_type, std::size_t> & pairs)
        {
            bool result = false;

            if(pairs.size() <= m_maxBatchSize)
            {
                for(std::size_t i = 0; i < m_maxBatchSize; ++i)
                {
                    m_pool.insert(i);
                }

                m_mapping.clear();
                for(auto mapping: pairs)
                {
                    if(m_pool.erase(mapping.second))
                    {
                        m_mapping[mapping.first] = mapping.second;
                    }
                }

                if((pairs.size() + m_pool.size()) == m_maxBatchSize)
                {
                    result = true;
                }
            }

            return result;
        }

        /**
         * @brief Associates a given key with the first available batch index
         * 
         * @details Assuming the associated keys has not reached `maxBatchSize``
         *   then this function associates a given key with the first available
         *   batch index and returns that index value. If no batch index is
         *   available -1 is returned. NOTE: if ``key`` is already associated
         *   with a batch index, the that index is returned.
         * 
         * @param key The key to be associated with a batch index value
         * @return std::intmax_t The batch index associated with the key or -1
         *   if no index is available. NOTE: because std::intmax_t is not a full
         *   covering of std::size_t, it is possible for wrap around to happen.
         */
        std::intmax_t map(const key_type & key)
        {
            auto it = m_mapping.find(key);

            if(it == m_mapping.end() && !m_pool.empty())
            {
                auto value = m_pool.begin();
                it = m_mapping.insert({key, *value}).first;
                m_pool.erase(value);
            }

            return static_cast<std::intmax_t>(it != m_mapping.end() ? it->second : -1);
        }

        /**
         * @brief Dissociates a given key with a batch index if possible
         * 
         * @details Assuming the given key is associated with a batch index this
         *   function removes the association and returns the batch index is was
         *   associated with. If no batch index is found associated with the given
         *   key, -1 is returned.
         * 
         * @param key The key to be dissociated
         * @return std::intmax_t The batch index associated with the key or -1
         *   if not found. NOTE: because std::intmax_t is not a full covering of
         *   std::size_t, it is possible for wrap around to happen.
         */
        std::intmax_t unmap(const key_type & key)
        {
            std::intmax_t result = -1;

            auto it = m_mapping.find(key);

            if(it != m_mapping.end())
            {
                result = static_cast<std::intmax_t>(it->second);
                m_pool.insert(it->second);
                m_mapping.erase(it);
            }

            return result;
        }

        /**
         * @brief The number of keys associated with a batch index
         * 
         * @return std::size_t 
         */
        std::size_t getKeyCount() const noexcept
        {
            return m_mapping.size();
        }

        /**
         * @brief The maximum number of batch index
         * 
         * @return std::size_t 
         */
        std::size_t getMaxBatchSize() const noexcept
        {
            return m_maxBatchSize;
        }
        
        /**
         * @brief The number of sub-tensors
         * 
         * @return std::size_t 
         */
        std::size_t getUnitCount() const
        {
            return m_size;
        }

        /**
         * Get the size of given dimension.
         * @param dimIdx dimension index.
         * @return size of the specified dimension.
         */
        std::size_t getTensorSize(std::size_t unitIdx, std::size_t dimIdx) const
        {
            return m_buffer[unitIdx].getSize(dimIdx);
        }

        /**
         * Get the stride of given dimension.
         * @param dimIdx dimension index.
         * @return stride of the specified dimension.
         */
        std::size_t getTensorStride(std::size_t unitIdx, std::size_t dimIdx) const
        {
            return m_buffer[unitIdx].getStride(dimIdx);
        }

        template<ChannelCount T = CC, typename = typename std::enable_if<T != CX>::type>
        unit_type getUnit(std::size_t idx)
        {
            unit_type result{m_buffer[idx].getWidth(),
                             m_buffer[idx].getHeight(),
                             m_buffer[idx].getDepth(),
                             m_buffer[idx].getData(),
                             m_buffer[idx].isCPU()};
            return result;
        }

        template<ChannelCount T = CC, typename = typename std::enable_if<T == CX>::type>
        unit_type getUnit(std::size_t idx, ChannelCount UNUSED = T)
        {
            unit_type result{m_buffer[idx].getWidth(),
                             m_buffer[idx].getHeight(),
                             m_buffer[idx].getDepth(),
                             m_buffer[idx].getChannelCount(),
                             m_buffer[idx].getData(),
                             m_buffer[idx].isCPU()};
            return result;
        }

        template<ChannelCount T = CC, typename = typename std::enable_if<T != CX>::type>
        frame_type getFrame(const key_type & idx)
        {
            frame_type result;

            if(m_mapping.find(idx) != m_mapping.end())
            {
                std::size_t at = m_mapping[idx];
                result = std::move(frame_type{m_buffer.getSize(), m_buffer.isCPU()});
                result.setSize(m_size);
                for(std::size_t i = 0; i < m_size; ++i)
                {
                    element_type element{m_buffer[i].getWidth(),
                                         m_buffer[i].getHeight(),
                                         m_buffer[i].getData() +
                                            at * m_buffer[i].getStride(TensorDimension::DEPTH),
                                         m_buffer[i].isCPU()};
                    result[i] = std::move(element);
                }
            }

            return result;
        }
        
        template<ChannelCount T = CC, typename = typename std::enable_if<T == CX>::type>
        frame_type getFrame(const key_type & idx, ChannelCount UNUSED = T)
        {
            frame_type result;

            if(m_mapping.find(idx) != m_mapping.end())
            {
                std::size_t at = m_mapping[idx];
                result = std::move(frame_type{m_buffer.getSize(), m_buffer.isCPU()});
                result.setSize(m_size);
                for(std::size_t i = 0; i < m_size; ++i)
                {
                    element_type element{m_buffer[i].getWidth(),
                                        m_buffer[i].getHeight(),
                                        m_buffer[i].getChannelCount(),
                                        m_buffer[i].getData() +
                                            at * m_buffer[i].getStride(TensorDimension::DEPTH),
                                        m_buffer[i].isCPU()};
                    result[i] = std::move(element);
                }
            }

            return result;
        }

        template<ChannelCount T = CC, typename = typename std::enable_if<T != CX>::type>
        frame_type getFrame(key_type && idx)
        {
            frame_type result;

            if(m_mapping.find(idx) != m_mapping.end())
            {
                std::size_t at = m_mapping[idx];
                result = std::move(frame_type{m_buffer.getSize(), m_buffer.isCPU()});
                result.setSize(m_size);
                for(std::size_t i = 0; i < m_size; ++i)
                {
                    element_type element{m_buffer[i].getWidth(),
                                         m_buffer[i].getHeight(),
                                         m_buffer[i].getData() +
                                            at * m_buffer[i].getStride(TensorDimension::DEPTH),
                                         m_buffer[i].isCPU()};
                    result[i] = std::move(element);
                }
            }

            return result;
        }
        
        template<ChannelCount T = CC, typename = typename std::enable_if<T == CX>::type>
        frame_type getFrame(key_type && idx, ChannelCount UNUSED = T)
        {
            frame_type result;

            if(m_mapping.find(idx) != m_mapping.end())
            {
                std::size_t at = m_mapping[idx];
                result = std::move(frame_type{m_buffer.getSize(), m_buffer.isCPU()});
                result.setSize(m_size);
                for(std::size_t i = 0; i < m_size; ++i)
                {
                    element_type element{m_buffer[i].getWidth(),
                                         m_buffer[i].getHeight(),
                                         m_buffer[i].getChannelCount(),
                                         m_buffer[i].getData() +
                                            at * m_buffer[i].getStride(TensorDimension::DEPTH),
                                         m_buffer[i].isCPU()};
                    result[i] = std::move(element);
                }
            }

            return result;
        }

        template<ChannelCount T = CC, typename = typename std::enable_if<T != CX>::type>
        element_type getElement(const key_type & keyIdx, std::size_t unitIdx)
        {
            element_type element;

            if(m_mapping.find(keyIdx) != m_mapping.end())
            {
                std::size_t at = m_mapping[keyIdx];
                element = std::move(element_type{m_buffer[unitIdx].getWidth(),
                                    m_buffer[unitIdx].getHeight(),
                                    m_buffer[unitIdx].getData() +
                                        at * m_buffer[unitIdx].getStride(TensorDimension::DEPTH),
                                    m_buffer[unitIdx].isCPU()});
            }

            return element;
        }
        
        template<ChannelCount T = CC, typename = typename std::enable_if<T == CX>::type>
        element_type getElement(const key_type & keyIdx, std::size_t unitIdx, ChannelCount UNUSED = T)
        {
            element_type element;

            if(m_mapping.find(keyIdx) != m_mapping.end())
            {
                std::size_t at = m_mapping[keyIdx];
                element = std::move(element_type{m_buffer[unitIdx].getWidth(),
                                    m_buffer[unitIdx].getHeight(),
                                    m_buffer[unitIdx].getChannelCount(),
                                    m_buffer[unitIdx].getData() +
                                        at * m_buffer[unitIdx].getStride(TensorDimension::DEPTH),
                                    m_buffer[unitIdx].isCPU()});
            }

            return element;
        }

        template<ChannelCount T = CC, typename = typename std::enable_if<T != CX>::type>
        element_type getElement(key_type && keyIdx, std::size_t unitIdx)
        {
            element_type element;

            if(m_mapping.find(keyIdx) != m_mapping.end())
            {
                std::size_t at = m_mapping[keyIdx];
                element = std::move(element_type{m_buffer[unitIdx].getWidth(),
                                    m_buffer[unitIdx].getHeight(),
                                    m_buffer[unitIdx].getData() +
                                        at * m_buffer[unitIdx].getStride(TensorDimension::DEPTH),
                                    m_buffer[unitIdx].isCPU()});
            }

            return element;
        }
        
        template<ChannelCount T = CC, typename = typename std::enable_if<T == CX>::type>
        element_type getElement(key_type && keyIdx, std::size_t unitIdx, ChannelCount UNUSED = T)
        {
            element_type element;

            if(m_mapping.find(keyIdx) != m_mapping.end())
            {
                std::size_t at = m_mapping[keyIdx];
                element = std::move(element_type{m_buffer[unitIdx].getWidth(),
                                    m_buffer[unitIdx].getHeight(),
                                    m_buffer[unitIdx].getChannelCount(),
                                    m_buffer[unitIdx].getData() +
                                        at * m_buffer[unitIdx].getStride(TensorDimension::DEPTH),
                                    m_buffer[unitIdx].isCPU()});
            }

            return element;
        }
        
        /**
         * Get the ChannelType of the Tensor.
         * @return ChannelType of the Tensor.
         */
        constexpr ChannelType getType() const noexcept
        {
            return CT;
        }
        
        /**
         * Get the flag whether the Tensor is allocated in CPU or GPU.
         * @return whether the Tensor is allocated in CPU.
         */
        bool isCPU() const noexcept
        {
            return m_isCPU;
        }

    private:
        // Mapping and Pool form a unique-to-unique isometry between
        // the keys and indices of the batch dimension
        mutable std::unordered_map<KT, std::size_t> m_mapping;
        mutable std::set<std::size_t> m_pool;

        std::size_t m_maxBatchSize;
        std::size_t m_size;
        bool m_isCPU;

        buffer_type m_buffer;
};

} // namespace cvcore

#endif // CVCORE_TENSORMAP_H
