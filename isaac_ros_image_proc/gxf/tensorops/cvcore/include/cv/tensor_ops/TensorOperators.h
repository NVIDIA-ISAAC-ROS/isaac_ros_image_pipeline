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

#ifndef CVCORE_TENSOROPERATORS_H
#define CVCORE_TENSOROPERATORS_H

#include <memory>
#include <unordered_map>
#include <mutex>

#include "cv/tensor_ops/ITensorOperatorContext.h"
#include "cv/tensor_ops/ITensorOperatorStream.h"

namespace cvcore { namespace tensor_ops {

class TensorContextFactory
{
public:
    static std::error_code CreateContext(TensorOperatorContext &, TensorBackend backend);
    static std::error_code DestroyContext(TensorOperatorContext &context);

    static bool IsBackendSupported(TensorBackend backend);

private:
    using MultitonType = std::unordered_map<TensorBackend,
                                            std::pair<std::size_t, std::unique_ptr<ITensorOperatorContext>>>;

    static MultitonType instances;
    static std::mutex instanceMutex;
};

}} // namespace cvcore::tensor_ops

#endif // CVCORE_TENSOROPERATORS_H
