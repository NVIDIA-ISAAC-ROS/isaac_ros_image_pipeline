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

#ifndef CVCORE_VPIENUMMAPPING_H
#define CVCORE_VPIENUMMAPPING_H

#include <type_traits>

#include "VPITensorOperators.h"

namespace cvcore { namespace tensor_ops {

constexpr VPIBackend ToVpiBackendType(const ComputeEngine &computeEngine)
{
    switch (computeEngine)
    {
    case ComputeEngine::CPU:
        return VPIBackend::VPI_BACKEND_CPU;
    case ComputeEngine::PVA:
        return VPIBackend::VPI_BACKEND_PVA;
    case ComputeEngine::GPU:
        return VPIBackend::VPI_BACKEND_CUDA;
    case ComputeEngine::VIC:
        return VPIBackend::VPI_BACKEND_VIC;
    case ComputeEngine::NVENC:
        return VPIBackend::VPI_BACKEND_NVENC;
    default:
        return VPIBackend::VPI_BACKEND_INVALID;
    }
}

constexpr VPIInterpolationType ToVpiInterpolationType(InterpolationType value)
{
    VPIInterpolationType result = VPI_INTERP_NEAREST;

    switch (value)
    {
    case INTERP_NEAREST:
        result = VPI_INTERP_NEAREST;
        break;
    case INTERP_LINEAR:
        result = VPI_INTERP_LINEAR;
        break;
    case INTERP_CUBIC_CATMULLROM:
        result = VPI_INTERP_CATMULL_ROM;
        break;
    default:
        break;
    }

    return result;
}

constexpr VPIBorderExtension ToVpiBorderType(BorderType value)
{
    VPIBorderExtension result = VPI_BORDER_ZERO;

    switch (value)
    {
    case BORDER_ZERO:
        result = VPI_BORDER_ZERO;
        break;
    case BORDER_REPEAT:
        result = VPI_BORDER_CLAMP;
        break;
    case BORDER_REVERSE:
        result = VPI_BORDER_REFLECT;
        break;
    case BORDER_MIRROR:
        result = VPI_BORDER_MIRROR;
        break;
    default:
        break;
    }

    return result;
}

constexpr VPIImageFormat ToVpiImageFormat(ImageType value)
{
    VPIImageFormat result = VPI_IMAGE_FORMAT_Y8_ER;

    switch (value)
    {
    case Y_U8:
        result = VPI_IMAGE_FORMAT_Y8_ER;
        break;
    case Y_U16:
        result = VPI_IMAGE_FORMAT_Y16_ER;
        break;
    case Y_S8:
        result = VPI_IMAGE_FORMAT_S8;
        break;
    case Y_S16:
        result = VPI_IMAGE_FORMAT_S16;
        break;
    case Y_F32:
        result = VPI_IMAGE_FORMAT_F32;
        break;
    case RGB_U8:
        result = VPI_IMAGE_FORMAT_RGB8;
        break;
    case BGR_U8:
        result = VPI_IMAGE_FORMAT_BGR8;
        break;
    case RGBA_U8:
        result = VPI_IMAGE_FORMAT_RGBA8;
        break;
    case NV12:
        result = VPI_IMAGE_FORMAT_NV12_ER;
        break;
    case NV24:
        result = VPI_IMAGE_FORMAT_NV24_ER;
        break;
    default:
        break;
    }

    return result;
}

constexpr VPIPixelType ToVpiPixelType(ImageType value)
{
    VPIPixelType result = VPI_PIXEL_TYPE_U8;

    switch (value)
    {
    case Y_U8:
        result = VPI_PIXEL_TYPE_U8;
        break;
    case Y_U16:
        result = VPI_PIXEL_TYPE_U16;
        break;
    case Y_S8:
        result = VPI_PIXEL_TYPE_S8;
        break;
    case Y_S16:
        result = VPI_PIXEL_TYPE_S16;
        break;
    case Y_F32:
        result = VPI_PIXEL_TYPE_F32;
        break;
    case RGB_U8:
        result = VPI_PIXEL_TYPE_3U8;
        break;
    case BGR_U8:
        result = VPI_PIXEL_TYPE_3U8;
        break;
    case RGBA_U8:
        result = VPI_PIXEL_TYPE_4U8;
        break;
    default:
        break;
    }

    return result;
}

static inline std::string getVPIBackendString(VPIBackend vpiBackend)
{
    switch (vpiBackend)
    {
    case VPIBackend::VPI_BACKEND_CPU:
        return "CPU";
    case VPIBackend::VPI_BACKEND_CUDA:
        return "GPU";
    case VPIBackend::VPI_BACKEND_VIC:
        return "VIC";
    case VPIBackend::VPI_BACKEND_PVA:
        return "PVA";
    case VPIBackend::VPI_BACKEND_NVENC:
        return "NVENC";
    case VPIBackend::VPI_BACKEND_INVALID:
        return "INVALID";
    default:
        return "INVALID";
    }
}

}} // namespace cvcore::tensor_ops

#endif // CVCORE_VPIENUMMAPPING_H
