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

#include "cv/tensor_ops/Errors.h"
#include "vpi/Status.h"

#ifndef __cpp_lib_to_underlying
// Using a C++23 feature by hacking std
namespace std
{
    template<typename Enum>
    constexpr underlying_type_t<Enum> to_underlying(Enum e) noexcept
    {
        return static_cast<underlying_type_t<Enum>>(e);
    }
};
#endif // __cpp_lib_to_underlying

namespace cvcore { namespace tensor_ops {

namespace detail
{
    struct VPIStatusCategory : std::error_category
    {
        virtual const char * name() const noexcept override final
        {
            return "vpi-status";
        }

        virtual std::string message(int value) const override final
        {
            std::string result = "VPI Status";

            return result;
        }

        virtual std::error_condition default_error_condition(int code) const noexcept override final
        {
            std::error_condition result;

            switch(code)
            {
                case VPI_SUCCESS:
                    result = ErrorCode::SUCCESS;
                    break;

                case VPI_ERROR_INVALID_ARGUMENT:
                    result = ErrorCode::INVALID_ARGUMENT;
                    break;

                case VPI_ERROR_INVALID_IMAGE_FORMAT:
                    result = ErrorCode::INVALID_IMAGE_FORMAT;
                    break;

                case VPI_ERROR_INVALID_ARRAY_TYPE:
                    result = ErrorCode::INVALID_STORAGE_TYPE;
                    break;

                case VPI_ERROR_INVALID_PAYLOAD_TYPE:
                    result = ErrorCode::INVALID_STORAGE_TYPE;
                    break;

                case VPI_ERROR_INVALID_OPERATION:
                    result = ErrorCode::INVALID_OPERATION;
                    break;

                case VPI_ERROR_INVALID_CONTEXT:
                    result = ErrorCode::INVALID_ENGINE_TYPE;
                    break;

                case VPI_ERROR_DEVICE:
                    result = ErrorCode::DEVICE_ERROR;
                    break;

                case VPI_ERROR_NOT_READY:
                    result = ErrorCode::NOT_READY;
                    break;

                case VPI_ERROR_BUFFER_LOCKED:
                    result = ErrorCode::SYSTEM_ERROR;
                    break;

                case VPI_ERROR_OUT_OF_MEMORY:
                    result = ErrorCode::OUT_OF_MEMORY;
                    break;

                case VPI_ERROR_INTERNAL:
                    result = ErrorCode::SYSTEM_ERROR;
                    break;

                default:
                    result = ErrorCode::NOT_IMPLEMENTED;
                    break;
            }

            return result;
        }
    };
} // namespace detail

const detail::VPIStatusCategory errorCategory{};

std::error_code make_error_code(VPIStatus ec) noexcept
{
    return {std::to_underlying(ec), errorCategory};
}

}} // namespace cvcore::tensor_ops
