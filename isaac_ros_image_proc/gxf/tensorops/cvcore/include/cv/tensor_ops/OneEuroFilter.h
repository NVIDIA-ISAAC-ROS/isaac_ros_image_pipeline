// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CVCORE_ONEEUROFILTER_H
#define CVCORE_ONEEUROFILTER_H

#include <memory>
#include <vector>

#include "cv/core/CVError.h"

namespace cvcore { namespace tensor_ops {

/**
 * Euro Filter Parameters
 */
struct OneEuroFilterParams
{
    float dataUpdateRate;  /**< Data Update rate in Hz. */
    float minCutoffFreq;   /**< Minimum Cut off frequency in Hz. */
    float cutoffSlope;     /**< Beta or Speed coefficient which is a tuning parameter. */
    float derivCutoffFreq; /**< Cutoff frequency for derivative. */
};

/* 
The one euro filter is a low pass filter for filtering noisy signals in real-time. The filtering uses exponential smoothing where the smooting factor is computed dynamically using the input data update rate. The smoothing factor provides a trade off between slow speed jitter vs high speed lag.
There are two main tuning parameters for the filter, the speed coeffcient Beta and the minimum cut off frequency.
If high speed lag is a problem, increase beta; if slow speed jitter is a problem, decrease fcmin.
Reference : http://cristal.univ-lille.fr/~casiez/1euro/
*/
template<typename T>
class OneEuroFilter
{
public:
    struct OneEuroFilterImpl;
    /**
 * Euro Filter Constructor.
 * @param filterParams Filter parameters
 */
    OneEuroFilter(const OneEuroFilterParams &filterParams);

    /**
 * Reset Euro filter Parameters.
 * @param filterParams Filter parameters
 * @return Error code
 */
    std::error_code resetParams(const OneEuroFilterParams &filterParams);

    /**
 * Filter the input. Supports float, double, vector2f, Vector3f, Vector3d, Vector3f input types.
 * @param inValue Noisy input to be filtered.
 * @param filteredValue Filtered output
 * @return Error code
 */
    std::error_code execute(T &filteredValue, T inValue);

    ~OneEuroFilter();

private:
    /**
   * Implementation of EuroFilter.
   */
    std::unique_ptr<OneEuroFilterImpl> m_pImpl;
};

}} // namespace cvcore::tensor_ops

#endif // CVCORE_ONEEUROFILTER_H
