// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CVCORE_CVERROR_H
#define CVCORE_CVERROR_H

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <system_error>

#include <cuda_runtime.h>

namespace cvcore {

// CVCORE ERROR CODES
// -----------------------------------------------------------------------------
// Defining the CVCORE Error Codes on std::error_condition
// std::error_condition creates a set of sub-system independent codes which may
// be used to describe ANY downstream error in a broad sense. An std::error_code
// is defined within the sub-system context (i.e. tensor_ops, trtbackend, ...)
// which is mapped to the cvcore::ErrorCode.
// As an example, cvcore::ErrorCode -1 may not ABSOLUTELY mean the same as
// tensor_ops::FaultCode -1, but does mean the same as tensor_ops:FaultCode 4.
// Thus, tensor_ops::FaultCode 4 needs to be mapped to cvcore::ErrorCode -1.
enum class ErrorCode : std::int32_t
{
    SUCCESS = 0,
    NOT_READY,
    NOT_IMPLEMENTED,
    INVALID_ARGUMENT,
    INVALID_IMAGE_FORMAT,
    INVALID_STORAGE_TYPE,
    INVALID_ENGINE_TYPE,
    INVALID_OPERATION,
    DETECTED_NAN_IN_RESULT,
    OUT_OF_MEMORY,
    DEVICE_ERROR,
    SYSTEM_ERROR,
};

} // namespace cvcore

// WARNING: Extending base C++ namespace to cover cvcore error codes
namespace std {

template<>
struct is_error_condition_enum<cvcore::ErrorCode> : true_type
{
};

template<>
struct is_error_code_enum<cvcore::ErrorCode> : true_type
{
};

} // namespace std

namespace cvcore {

std::error_condition make_error_condition(ErrorCode) noexcept;

std::error_code make_error_code(ErrorCode) noexcept;

// -----------------------------------------------------------------------------

inline void CheckCudaError(cudaError_t code, const char *file, const int line)
{
    if (code != cudaSuccess)
    {
        const char *errorMessage  = cudaGetErrorString(code);
        const std::string message = "CUDA error returned at " + std::string(file) + ":" + std::to_string(line) +
                                    ", Error code: " + std::to_string(code) + " (" + std::string(errorMessage) + ")";
        throw std::runtime_error(message);
    }
}

inline void CheckErrorCode(std::error_code err, const char *file, const int line)
{
    const std::string message = "Error returned at " + std::string(file) + ":" + std::to_string(line) +
                                ", Error code: " + std::string(err.message());

    if (err != cvcore::make_error_code(cvcore::ErrorCode::SUCCESS))
    {
        throw std::runtime_error(message);
    }
}

} // namespace cvcore

#define CHECK_ERROR(val)                                   \
    {                                                      \
        cvcore::CheckCudaError((val), __FILE__, __LINE__); \
    }

#define CHECK_ERROR_CODE(val)                               \
    {                                                    \
        cvcore::CheckErrorCode((val), __FILE__, __LINE__); \
    }

#endif // CVCORE_CVERROR_H
