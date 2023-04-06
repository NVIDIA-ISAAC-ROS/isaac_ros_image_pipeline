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

#include "cv/tensor_ops/TensorOperators.h"

#include <iostream>

#ifdef ENABLE_VPI
#include "vpi/VPITensorOperators.h"
#endif

namespace cvcore { namespace tensor_ops {

typename TensorContextFactory::MultitonType TensorContextFactory::instances;
std::mutex TensorContextFactory::instanceMutex;

std::error_code TensorContextFactory::CreateContext(TensorOperatorContext &tensorContext, TensorBackend backend)
{
    using PairType = typename TensorContextFactory::MultitonType::mapped_type;
    using CounterType = typename PairType::first_type;
    using ValuePtrType = typename PairType::second_type;

    std::lock_guard<std::mutex> instanceLock(instanceMutex);

    std::error_code result = ErrorCode::SUCCESS;

    tensorContext = nullptr;
    
    auto contextItr = instances.find(backend);
    if (contextItr == instances.end() && IsBackendSupported(backend))
    {
        switch (backend)
        {
        case TensorBackend::VPI:
#ifdef ENABLE_VPI
            try
            {
                instances[backend] = std::make_pair<CounterType, ValuePtrType>(1, ValuePtrType(new VPITensorContext{}));
            }
            catch (std::error_code &e)
            {
                result = e;
            }
            catch (...)
            {
                result = ErrorCode::INVALID_OPERATION;
            }
#else // _WIN32
            result = ErrorCode::NOT_IMPLEMENTED;
#endif // _WIN32
            break;
        default:
            result = ErrorCode::NOT_IMPLEMENTED;
            break;
        }
        tensorContext = instances[backend].second.get();
    }
    else
    {
        contextItr->second.first++;
        tensorContext = contextItr->second.second.get();
    }

    return result;
}

std::error_code TensorContextFactory::DestroyContext(TensorOperatorContext &context)
{
    std::lock_guard<std::mutex> instanceLock(instanceMutex);

    auto backend = context->Backend();
    context      = nullptr;
    auto contextItr = instances.find(backend);
    if (contextItr != instances.end())
    {
        contextItr->second.first--;
        if (contextItr->second.first == 0)
        {
            instances.erase(backend);
        }
    }
    return ErrorCode::SUCCESS;
}

bool TensorContextFactory::IsBackendSupported(TensorBackend backend)
{
    bool result = false;

    switch (backend)
    {
    case TensorBackend::VPI:
#ifdef ENABLE_VPI
        result = true;
#endif // _WIN32
        break;
    default:
        break;
    }

    return result;
}
}} // namespace cvcore::tensor_ops
