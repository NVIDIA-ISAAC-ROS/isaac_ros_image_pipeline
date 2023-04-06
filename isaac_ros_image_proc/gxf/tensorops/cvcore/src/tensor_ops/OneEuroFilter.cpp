// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cv/tensor_ops/OneEuroFilter.h"
#include "cv/core/MathTypes.h"
#include "cv/core/Traits.h"

#include <math.h>
#include <iostream>
#include <system_error>

#ifdef NVBENCH_ENABLE
#include <nvbench/CPU.h>
#endif

namespace cvcore { namespace tensor_ops {

namespace {

// 1/(2*PI)
constexpr float kOneOver2Pi = 0.15915494309189533577f;

// Utilities to get template type from another template type
template<class T, class U = void>
struct deduceDataType;
template<class T>
struct deduceDataType<T, typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, Vector2f>::value ||
                                                 std::is_same<T, Vector3f>::value>::type>
{
    typedef float U;
};
template<class T>
struct deduceDataType<T, typename std::enable_if<std::is_same<T, double>::value || std::is_same<T, Vector2d>::value ||
                                                 std::is_same<T, Vector3d>::value>::type>
{
    typedef double U;
};

} // namespace

/* Low pass filter to apply exponential smoothing*/
template<typename T>
class LowPassfilter
{
public:
    LowPassfilter()
    {
        m_firstIteration = true;
    }

    void resetState()
    {
        m_firstIteration = true;
    }

    bool isInitialized() const
    {
        return !m_firstIteration;
    }

    T getPreviousValue() const
    {
        return m_prevfilteredValue;
    }

    std::error_code filter(T &outValue, T inValue, float alpha)
    {
#ifdef NVBENCH_ENABLE
        std::string funcName       = "LowPassFilter_";
        std::string tag            = funcName + typeid(T).name();
        nv::bench::Timer timerFunc = nv::bench::CPU(tag.c_str(), nv::bench::Flag::DEFAULT);
#endif
        if (m_firstIteration)
        {
            outValue         = inValue;
            m_firstIteration = false;
        }
        else
        {
            outValue = m_prevfilteredValue + (inValue - m_prevfilteredValue) * alpha;
        }
        m_prevRawValue      = inValue;
        m_prevfilteredValue = outValue;
        return ErrorCode::SUCCESS;
    }

private:
    bool m_firstIteration;
    T m_prevRawValue;
    T m_prevfilteredValue;
};

template<typename U>
struct OneEuroFilterState
{
    // Computes alpha value for the filter
    float getAlpha(float dataUpdateRate, float cutOffFreq) const
    {
        float alpha = cutOffFreq / (dataUpdateRate * kOneOver2Pi + cutOffFreq);
        return alpha;
    }

    // Resets the parameters and state of the filter
    std::error_code resetParams(const OneEuroFilterParams &filterParams)
    {
        if (filterParams.dataUpdateRate <= 0.0f || filterParams.minCutoffFreq <= 0 || filterParams.derivCutoffFreq <= 0)
        {
            return ErrorCode::INVALID_ARGUMENT;
        }
        m_freq        = filterParams.dataUpdateRate;
        m_mincutoff   = filterParams.minCutoffFreq;
        m_cutOffSlope = filterParams.cutoffSlope;
        m_derivCutOff = filterParams.derivCutoffFreq;
        m_alphadxFilt = getAlpha(m_freq, filterParams.derivCutoffFreq);

        xFilt->resetState();
        dxFilt->resetState();

        m_currfilteredValue = 0.0f;
        m_prevfilteredValue = m_currfilteredValue;
        return ErrorCode::SUCCESS;
    }

    // Constructor for each filter state
    OneEuroFilterState(const OneEuroFilterParams &filterParams)
    {
        xFilt.reset(new LowPassfilter<U>());
        dxFilt.reset(new LowPassfilter<U>());
        auto err = resetParams(filterParams);
        if (err != make_error_code(ErrorCode::SUCCESS))
        {
            throw err;
        }
    }

    std::error_code filter(U &outValue, U value)
    {
#ifdef NVBENCH_ENABLE
        std::string funcName       = "OneEuroFilterState_";
        std::string tag            = funcName + typeid(U).name();
        nv::bench::Timer timerFunc = nv::bench::CPU(tag.c_str(), nv::bench::Flag::DEFAULT);
#endif
        m_prevfilteredValue = m_currfilteredValue;
        U dxValue           = xFilt->isInitialized() ? (value - xFilt->getPreviousValue()) * m_freq : 0.0f;
        U edxValue;
        auto err = dxFilt->filter(edxValue, dxValue, m_alphadxFilt);
        if (err != make_error_code(ErrorCode::SUCCESS))
        {
            return err;
        }
        // Update the new cutoff frequency
        U newCutoff    = m_mincutoff + m_cutOffSlope * fabsf(edxValue);
        float newAlpha = getAlpha(m_freq, newCutoff);
        err            = xFilt->filter(m_currfilteredValue, value, newAlpha);
        if (err != make_error_code(ErrorCode::SUCCESS))
        {
            return err;
        }

        outValue = m_currfilteredValue;
        return ErrorCode::SUCCESS;
    }
    std::unique_ptr<LowPassfilter<U>> xFilt;
    std::unique_ptr<LowPassfilter<U>> dxFilt;
    float m_alphadxFilt;
    float m_freq;
    float m_mincutoff;
    float m_cutOffSlope;
    float m_derivCutOff;
    U m_prevfilteredValue;
    U m_currfilteredValue;
};

template<typename T>
struct OneEuroFilter<T>::OneEuroFilterImpl
{
    typedef typename deduceDataType<T>::U DT;
    OneEuroFilterImpl(const OneEuroFilterParams &filterParams)
    {
        size_t numStates = traits::get_dim<T>::value;
        m_states.resize(numStates);
        for (size_t i = 0; i < m_states.size(); i++)
        {
            m_states[i].reset(new OneEuroFilterState<DT>(filterParams));
        }
    }

    std::error_code resetParams(const OneEuroFilterParams &filterParams)
    {
        std::error_code err = ErrorCode::SUCCESS;
        for (size_t i = 0; i < m_states.size(); i++)
        {
            err = m_states[i]->resetParams(filterParams);
            if (err != make_error_code(ErrorCode::SUCCESS))
            {
                return err;
            }
        }
        return ErrorCode::SUCCESS;
    }

    ~OneEuroFilterImpl() {}

    template<typename U = T, typename std::enable_if<traits::get_dim<U>::value == 1>::type * = nullptr>
    std::error_code filter(U &outValue, U value)
    {
        if (m_states.size() != 1)
        {
            return ErrorCode::INVALID_OPERATION;
        }
        std::error_code err = m_states[0]->filter(outValue, value);
        return err;
    }

    template<typename U = T, typename std::enable_if<traits::get_dim<U>::value != 1>::type * = nullptr>
    std::error_code filter(U &outValue, U value)
    {
        if (m_states.size() <= 1)
        {
            return ErrorCode::INVALID_OPERATION;
        }
        std::error_code err = ErrorCode::SUCCESS;
        for (size_t i = 0; i < m_states.size(); i++)
        {
            err = m_states[i]->filter(outValue[i], value[i]);
            if (err != make_error_code(ErrorCode::SUCCESS))
            {
                return err;
            }
        }

        return err;
    }

    std::vector<std::unique_ptr<OneEuroFilterState<DT>>> m_states;
};

template<typename T>
OneEuroFilter<T>::OneEuroFilter(const OneEuroFilterParams &filterParams)
    : m_pImpl(new OneEuroFilterImpl(filterParams))
{
}

template<typename T>
OneEuroFilter<T>::~OneEuroFilter()
{
}

template<typename T>
std::error_code OneEuroFilter<T>::resetParams(const OneEuroFilterParams &filterParams)
{
    auto err = m_pImpl->resetParams(filterParams);
    return err;
}

template<typename T>
std::error_code OneEuroFilter<T>::execute(T &filteredValue, T inValue)
{
#ifdef NVBENCH_ENABLE
    std::string funcName       = "OneEuroFilter_";
    std::string tag            = funcName + typeid(T).name();
    nv::bench::Timer timerFunc = nv::bench::CPU(tag.c_str(), nv::bench::Flag::DEFAULT);
#endif
    auto err = m_pImpl->filter(filteredValue, inValue);
    return err;
}

template class OneEuroFilter<float>;
template class OneEuroFilter<Vector2f>;
template class OneEuroFilter<Vector3f>;
template class OneEuroFilter<double>;
template class OneEuroFilter<Vector2d>;
template class OneEuroFilter<Vector3d>;
}} // namespace cvcore::tensor_ops
