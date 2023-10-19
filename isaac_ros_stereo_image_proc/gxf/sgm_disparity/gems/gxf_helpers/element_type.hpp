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
#pragma once

#include "engine/core/tensor/element_type.hpp"
#include "gxf/std/tensor.hpp"

namespace nvidia {
namespace isaac {

// Helps determine the Isaac element type given the GXF primitive type
::nvidia::isaac::ElementType ToIsaacElementType(gxf::PrimitiveType gxf_type) {
  switch (gxf_type) {
    case gxf::PrimitiveType::kInt8:
      return ::nvidia::isaac::ElementType::kInt8;
    case gxf::PrimitiveType::kUnsigned8:
      return ::nvidia::isaac::ElementType::kUInt8;
    case gxf::PrimitiveType::kInt16:
      return ::nvidia::isaac::ElementType::kInt16;
    case gxf::PrimitiveType::kUnsigned16:
      return ::nvidia::isaac::ElementType::kUInt16;
    case gxf::PrimitiveType::kInt32:
      return ::nvidia::isaac::ElementType::kInt32;
    case gxf::PrimitiveType::kUnsigned32:
      return ::nvidia::isaac::ElementType::kUInt32;
    case gxf::PrimitiveType::kInt64:
      return ::nvidia::isaac::ElementType::kInt64;
    case gxf::PrimitiveType::kUnsigned64:
      return ::nvidia::isaac::ElementType::kUInt64;
    case gxf::PrimitiveType::kFloat32:
      return ::nvidia::isaac::ElementType::kFloat32;
    case gxf::PrimitiveType::kFloat64:
      return ::nvidia::isaac::ElementType::kFloat64;
    case gxf::PrimitiveType::kCustom:
    default:
      return ::nvidia::isaac::ElementType::kUnknown;
  }
}

}  // namespace isaac
}  // namespace nvidia
