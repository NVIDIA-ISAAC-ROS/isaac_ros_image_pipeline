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

#include "cv/core/CVError.h"

#include <string>
#include <utility>
#include <stdexcept>

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

namespace cvcore {

namespace detail
{
    struct CoreErrorCategory : std::error_category
    {
        virtual const char * name() const noexcept override final
        {
            return "cvcore-error";
        }

        virtual std::string message(int value) const override final
        {
            std::string result;

            switch(value)
            {
                case std::to_underlying(ErrorCode::SUCCESS):
                    result = "(SUCCESS) No errors detected";
                    break;
                case std::to_underlying(ErrorCode::NOT_READY):
                    result = "(NOT_READY) The execution of the requested "
                             "operation is not to return";
                    break;
                case std::to_underlying(ErrorCode::NOT_IMPLEMENTED):
                    result = "(NOT_IMPLEMENTED) The requested operation is not "
                             "implemented";
                    break;
                case std::to_underlying(ErrorCode::INVALID_ARGUMENT):
                    result = "(INVALID_ARGUMENT) The argument provided to the "
                             "operation is not currently supported";
                    break;
                case std::to_underlying(ErrorCode::INVALID_IMAGE_FORMAT):
                    result = "(INVALID_IMAGE_FORMAT) The requested image format "
                             "is not supported by the operation";
                    break;
                case std::to_underlying(ErrorCode::INVALID_STORAGE_TYPE):
                    result = "(INVALID_STORAGE_TYPE) The requested storage type "
                             "is not supported by the operation";
                    break;
                case std::to_underlying(ErrorCode::INVALID_ENGINE_TYPE):
                    result = "(INVALID_ENGINE_TYPE) The requested engine type "
                             "is not supported by the operation";
                    break;
                case std::to_underlying(ErrorCode::INVALID_OPERATION):
                    result = "(INVALID_OPERATION) The requested operation is "
                             "not supported";
                    break;
                case std::to_underlying(ErrorCode::DETECTED_NAN_IN_RESULT):
                    result = "(DETECTED_NAN_IN_RESULT) NaN was detected in the "
                             "return value of the operation";
                    break;
                case std::to_underlying(ErrorCode::OUT_OF_MEMORY):
                    result = "(OUT_OF_MEMORY) The device has run out of memory";
                    break;
                case std::to_underlying(ErrorCode::DEVICE_ERROR):
                    result = "(DEVICE_ERROR) A device level error has been "
                             "encountered";
                    break;
                case std::to_underlying(ErrorCode::SYSTEM_ERROR):
                    result = "(SYSTEM_ERROR) A system level error has been "
                             "encountered";
                    break;
                default:
                    result = "(Unrecognized Condition) Value " + std::to_string(value) +
                             " does not map to known error code literal " +
                             " defined by cvcore::ErrorCode";
                    break;
            }

            return result;
        }
    };
} // namespace detail

const detail::CoreErrorCategory errorCategory{};

std::error_condition make_error_condition(ErrorCode ec) noexcept
{
    return {std::to_underlying(ec), errorCategory};
}

std::error_code make_error_code(ErrorCode ec) noexcept
{
    return {std::to_underlying(ec), errorCategory};
}

} // namespace cvcore
