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
    struct TensorOpsErrorCategory : std::error_category
    {
        virtual const char * name() const noexcept override final
        {
            return "cvcore-tensor-ops-error";
        }

        virtual std::string message(int value) const override final
        {
            std::string result;

            switch(value)
            {
                case std::to_underlying(TensorOpsErrorCode::SUCCESS):
                    result = "(SUCCESS) No errors detected";
                    break;
                case std::to_underlying(TensorOpsErrorCode::COMPUTE_ENGINE_UNSUPPORTED_BY_CONTEXT):
                    result = "(COMPUTE_ENGINE_UNSUPPORTED_BY_CONTEXT) The selected compute "
                             "engine defined by cvcore::ComputeEngine is not avaible in the "
                             "requested context defined by cvcore::tensor_ops::TensorBackend";
                    break;
                case std::to_underlying(TensorOpsErrorCode::CAMERA_DISTORTION_MODEL_UNSUPPORTED):
                    result = "(CAMERA_DISTORTION_MODEL_UNSUPPORTED) The selected camera "
                             "distortion model defined by cvcore::CameraDistortionType is "
                             "currently unsupported";
                    break;
                default:
                    result = "(Unrecognized Condition) Value " + std::to_string(value) +
                             " does not map to known error code literal " +
                             " defined by cvcore::tensor_ops::TensorOpsErrorCode";
                    break;
            }

            return result;
        }

        virtual std::error_condition default_error_condition(int code) const noexcept override final
        {
            std::error_condition result;

            switch(code)
            {
                case std::to_underlying(TensorOpsErrorCode::SUCCESS):
                    result = ErrorCode::SUCCESS;
                    break;
                case std::to_underlying(TensorOpsErrorCode::COMPUTE_ENGINE_UNSUPPORTED_BY_CONTEXT):
                    result = ErrorCode::INVALID_ENGINE_TYPE;
                    break;
                case std::to_underlying(TensorOpsErrorCode::CAMERA_DISTORTION_MODEL_UNSUPPORTED):
                    result = ErrorCode::INVALID_ARGUMENT;
                    break;
                default:
                    result = ErrorCode::NOT_IMPLEMENTED;
                    break;
            }

            return result;
        }
    };
} // namespace detail

const detail::TensorOpsErrorCategory errorCategory{};

std::error_code make_error_code(TensorOpsErrorCode ec) noexcept
{
    return {std::to_underlying(ec), errorCategory};
}

}} // namespace cvcore::tensor_ops
