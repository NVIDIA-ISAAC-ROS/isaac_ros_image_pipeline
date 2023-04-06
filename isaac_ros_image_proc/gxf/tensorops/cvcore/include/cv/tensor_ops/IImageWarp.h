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

#ifndef CVCORE_IIMAGEWARP_H
#define CVCORE_IIMAGEWARP_H

#include <system_error>
#include <array>
#include <vector>

namespace cvcore { namespace tensor_ops {

struct ImageGrid
{
    static constexpr std::size_t MAX_HORIZ_REGIONS = 4;
    static constexpr std::size_t MAX_VERT_REGIONS = 4;
    static constexpr std::size_t MIN_REGION_WIDTH = 64;
    static constexpr std::size_t MIN_REGION_HIGHT = 16;

    std::int8_t numHorizRegions{0};
    std::int8_t numVertRegions{0};
    std::array<std::int16_t, MAX_HORIZ_REGIONS> horizInterval;
    std::array<std::int16_t, MAX_VERT_REGIONS> vertInterval;
    std::array<std::int16_t, MAX_VERT_REGIONS> regionHeight;
    std::array<std::int16_t, MAX_HORIZ_REGIONS> regionWidth;
};

class IImageWarp
{
    public:
        // Public Destructor
        virtual ~IImageWarp() = 0;

    protected:
        // Protected Constructor(s)
        IImageWarp()                       = default;
        IImageWarp(const IImageWarp &)     = default;
        IImageWarp(IImageWarp &&) noexcept = default;

        // Protected Operator(s)
        IImageWarp &operator=(const IImageWarp &)     = default;
        IImageWarp &operator=(IImageWarp &&) noexcept = default;
};

using ImageWarp = IImageWarp*;

}} // namespace cvcore::tensor_ops

#endif // CVCORE_IIMAGEWARP_H
