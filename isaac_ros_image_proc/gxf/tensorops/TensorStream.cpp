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

#include "TensorStream.hpp"

#include "cv/core/ComputeEngine.h"
#include "cv/tensor_ops/TensorOperators.h"

namespace nvidia {
namespace cvcore {
namespace tensor_ops {

namespace detail {

gxf::Expected<::cvcore::tensor_ops::TensorBackend> GetContextType(const std::string& type) {
  if (type == "NPP") {
    return ::cvcore::tensor_ops::TensorBackend::NPP;
  } else if (type == "VPI") {
    return ::cvcore::tensor_ops::TensorBackend::VPI;
  } else if (type == "DALI") {
    return ::cvcore::tensor_ops::TensorBackend::DALI;
  } else {
    return gxf::Unexpected{GXF_FAILURE};
  }
}

gxf::Expected<::cvcore::ComputeEngine> GetComputeEngineType(const std::string& type) {
  if (type == "UNKNOWN") {
    return ::cvcore::ComputeEngine::UNKNOWN;
  } else if (type == "CPU") {
    return ::cvcore::ComputeEngine::CPU;
  } else if (type == "PVA") {
    return ::cvcore::ComputeEngine::PVA;
  } else if (type == "VIC") {
    return ::cvcore::ComputeEngine::VIC;
  } else if (type == "NVENC") {
    return ::cvcore::ComputeEngine::NVENC;
  } else if (type == "GPU") {
    return ::cvcore::ComputeEngine::GPU;
  } else if (type == "DLA") {
    return ::cvcore::ComputeEngine::DLA;
  } else if (type == "COMPUTE_FAULT") {
    return ::cvcore::ComputeEngine::COMPUTE_FAULT;
  } else {
    return gxf::Unexpected{GXF_FAILURE};
  }
}

} // namespace detail

gxf_result_t TensorStream::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;

  result &= registrar->parameter(backend_type_, "backend_type");
  result &= registrar->parameter(engine_type_, "engine_type");

  return gxf::ToResultCode(result);
}

gxf_result_t TensorStream::initialize() {
  // Construct context
  auto backend_type = detail::GetContextType(backend_type_.get());
  if (!backend_type) {
    GXF_LOG_ERROR("unknown backend type.");
    return GXF_FAILURE;
  }
  if (!::cvcore::tensor_ops::TensorContextFactory::IsBackendSupported(backend_type.value())) {
    GXF_LOG_ERROR("unsupported context type.");
    return GXF_FAILURE;
  }
  auto err_code = ::cvcore::tensor_ops::TensorContextFactory::CreateContext(context_, backend_type.value());
  if (err_code != ::cvcore::make_error_code(::cvcore::ErrorCode::SUCCESS)) {
    GXF_LOG_ERROR("tensor context creation failed.");
    return GXF_FAILURE;
  }
  // Construct stream
  auto engine_type = detail::GetComputeEngineType(engine_type_.get());
  if (!engine_type) {
    return GXF_FAILURE;
  }

  if (!context_->IsComputeEngineCompatible(engine_type.value())) {
    GXF_LOG_ERROR("invalid compute engine type.");
    return GXF_FAILURE;
  }
  err_code = context_->CreateStream(stream_, engine_type.value());
  if (err_code != ::cvcore::make_error_code(::cvcore::ErrorCode::SUCCESS)) {
    GXF_LOG_ERROR("tensor stream creation failed.");
    return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}

gxf_result_t TensorStream::deinitialize() {
  auto err_code = context_->DestroyStream(stream_);
  if (err_code != ::cvcore::make_error_code(::cvcore::ErrorCode::SUCCESS)) {
    GXF_LOG_ERROR("tensor stream destroy failed.");
    return GXF_FAILURE;
  }
  err_code = ::cvcore::tensor_ops::TensorContextFactory::DestroyContext(context_);
  if (err_code != ::cvcore::make_error_code(::cvcore::ErrorCode::SUCCESS)) {
    GXF_LOG_ERROR("tensor context destroy failed.");
    return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}

} // namespace tensor_ops
} // namespace cvcore
} // namespace nvidia
