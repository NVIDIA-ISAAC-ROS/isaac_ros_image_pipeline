// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string>

#include "vpi/Types.h"

#include "gxf/core/expected.hpp"
#include "gxf/multimedia/video.hpp"

namespace nvidia {
namespace isaac {
namespace vpi {

constexpr uint32_t VPI_BACKEND_XAVIER = VPI_BACKEND_PVA | VPI_BACKEND_NVENC | VPI_BACKEND_VIC;
constexpr uint32_t VPI_BACKEND_ORIN = VPI_BACKEND_OFA | VPI_BACKEND_PVA | VPI_BACKEND_VIC;

gxf::Expected<VPIBackend> StringToBackend(const std::string& str);

gxf::Expected<VPIImageFormat> VideoFormatToImageFormat(gxf::VideoFormat value);

gxf::Expected<VPIPixelType> VideoFormatToPixelType(gxf::VideoFormat value);

}  // namespace vpi
}  // namespace isaac
}  // namespace nvidia
