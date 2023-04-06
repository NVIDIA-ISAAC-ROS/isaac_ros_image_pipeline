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
#ifndef NVIDIA_CVCORE_TENSOR_STREAM_HPP
#define NVIDIA_CVCORE_TENSOR_STREAM_HPP

#include "gxf/core/component.hpp"
#include "gxf/std/parameter_parser_std.hpp"

#include "cv/tensor_ops/ITensorOperatorContext.h"
#include "cv/tensor_ops/ITensorOperatorStream.h"

namespace nvidia {
namespace cvcore {
namespace tensor_ops {

// Wrapper of CVCORE ITensorOperatorStream/ITensorOperatorContext
class TensorStream : public gxf::Component {
public:
  virtual ~TensorStream() = default;
  TensorStream()          = default;

  gxf_result_t registerInterface(gxf::Registrar* registrar) override;
  gxf_result_t initialize() override;
  gxf_result_t deinitialize() override;

  ::cvcore::tensor_ops::TensorOperatorContext getContext() const {
    return context_;
  }
  ::cvcore::tensor_ops::TensorOperatorStream getStream() const {
    return stream_;
  }

private:
  gxf::Parameter<std::string> backend_type_;
  gxf::Parameter<std::string> engine_type_;

  ::cvcore::tensor_ops::TensorOperatorContext context_;
  ::cvcore::tensor_ops::TensorOperatorStream stream_;
};

} // namespace tensor_ops
} // namespace cvcore
} // namespace nvidia

#endif
