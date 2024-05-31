// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <iostream>
#include <utility>

#include "extensions/tensorops/core/TensorOperators.h"
#include "extensions/tensorops/core/VPITensorOperators.h"

namespace cvcore {
namespace tensor_ops {

typename TensorContextFactory::MultitonType TensorContextFactory::instances;
std::mutex TensorContextFactory::instanceMutex;

std::error_code TensorContextFactory::CreateContext(TensorOperatorContext& tensorContext,
    TensorBackend backend) {
    using PairType = TensorContextFactory::MultitonType::mapped_type;
    using ValuePtrType = PairType::second_type;

    std::lock_guard<std::mutex> instanceLock(instanceMutex);

    std::error_code result = ErrorCode::SUCCESS;

    tensorContext = nullptr;

    auto contextItr = instances.find(backend);
    if (contextItr == instances.end() && IsBackendSupported(backend)) {
        switch (backend) {
        case TensorBackend::VPI:
            try {
                instances[backend] = std::make_pair(1, ValuePtrType(new VPITensorContext{}));
            }
            catch (std::error_code &e) {
                result = e;
            }
            catch (...) {
                result = ErrorCode::INVALID_OPERATION;
            }
            break;
        default:
            result = ErrorCode::NOT_IMPLEMENTED;
            break;
        }
        tensorContext = instances[backend].second.get();
    } else {
        contextItr->second.first++;
        tensorContext = contextItr->second.second.get();
    }

    return result;
}

std::error_code TensorContextFactory::DestroyContext(TensorOperatorContext& context) {
    std::lock_guard<std::mutex> instanceLock(instanceMutex);

    auto backend = context->Backend();
    context      = nullptr;
    auto contextItr = instances.find(backend);
    if (contextItr != instances.end()) {
        contextItr->second.first--;
        if (contextItr->second.first == 0) {
            instances.erase(backend);
        }
    }
    return ErrorCode::SUCCESS;
}

bool TensorContextFactory::IsBackendSupported(TensorBackend backend) {
    bool result = false;

    switch (backend) {
        case TensorBackend::VPI:
            result = true;
            break;
        default:
            break;
    }

    return result;
}

}  // namespace tensor_ops
}  // namespace cvcore
