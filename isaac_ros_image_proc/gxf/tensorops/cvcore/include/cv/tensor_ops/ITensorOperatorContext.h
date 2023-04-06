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

#ifndef CVCORE_ITENSOROPERATORCONTEXT_H
#define CVCORE_ITENSOROPERATORCONTEXT_H

#include <memory>

#include "ITensorOperatorStream.h"
#include "cv/core/CVError.h"
#include "cv/core/ComputeEngine.h"

namespace cvcore { namespace tensor_ops {

enum class TensorBackend : std::uint8_t
{
    NPP,
    VPI,
    DALI
};

class ITensorOperatorContext
{
public:
    // Public Constructor(s)/Destructor
    virtual ~ITensorOperatorContext() noexcept = default;

    // Public Accessor Method(s)
    virtual std::error_code CreateStream(TensorOperatorStream &, const ComputeEngine &) = 0;
    virtual std::error_code DestroyStream(TensorOperatorStream &)                       = 0;

    virtual bool IsComputeEngineCompatible(const ComputeEngine &) const noexcept = 0;

    virtual TensorBackend Backend() const noexcept = 0;

protected:
    // Protected Constructor(s)
    ITensorOperatorContext()                                   = default;
    ITensorOperatorContext(const ITensorOperatorContext &)     = default;
    ITensorOperatorContext(ITensorOperatorContext &&) noexcept = default;

    // Protected Operator(s)
    ITensorOperatorContext &operator=(const ITensorOperatorContext &) = default;
    ITensorOperatorContext &operator=(ITensorOperatorContext &&) noexcept = default;
};

using TensorOperatorContext = ITensorOperatorContext *;

}} // namespace cvcore::tensor_ops

#endif // CVCORE_ITENSOROPERATORCONTEXT_H
