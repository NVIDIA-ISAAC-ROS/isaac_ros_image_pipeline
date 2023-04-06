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

#ifndef CVCORE_COMPUTEENGINE_H
#define CVCORE_COMPUTEENGINE_H

#include <type_traits>

namespace cvcore {

enum class ComputeEngine : unsigned int
{
    UNKNOWN       = 0x00, // 0000_0000

    CPU           = 0x01, // 0000_0001
    PVA           = 0x02, // 0000_0010
    VIC           = 0x04, // 0000_0100
    NVENC         = 0x08, // 0000_1000
    GPU           = 0x10, // 0001_0000
    DLA           = 0x20, // 0010_0000
    DLA_CORE_0    = 0x40, // 0100_0000
    DLA_CORE_1    = 0x80, // 1000_0000

    COMPUTE_FAULT = 0xFF  // 1111_1111
};

} // namespace cvcore

#endif // CVCORE_COMPUTEENGINE_H
