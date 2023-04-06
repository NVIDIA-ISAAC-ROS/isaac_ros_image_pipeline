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

#ifndef CVCORE_ERRORS_H
#define CVCORE_ERRORS_H

#include "cv/core/CVError.h"

namespace cvcore { namespace tensor_ops {

enum class TensorOpsErrorCode : std::int32_t
{
    SUCCESS = 0,
    COMPUTE_ENGINE_UNSUPPORTED_BY_CONTEXT,
    CAMERA_DISTORTION_MODEL_UNSUPPORTED
};

}} // namespace cvcore::tensor_ops

// WARNING: Extending base C++ namespace to cover cvcore error codes
namespace std {

template <>
struct is_error_code_enum<cvcore::tensor_ops::TensorOpsErrorCode> : true_type {};

} // namespace std

namespace cvcore { namespace tensor_ops {

std::error_code make_error_code(TensorOpsErrorCode) noexcept;

}} // namespace cvcore::tensor_ops

#endif // CVCORE_ERRORS_H
