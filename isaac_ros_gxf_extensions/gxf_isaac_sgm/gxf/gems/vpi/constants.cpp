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

#include "gems/vpi/constants.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>

namespace nvidia {
namespace isaac {
namespace vpi {

namespace {

const std::unordered_map<std::string, uint32_t> kStrToVpiBackend({
      {"CPU", VPI_BACKEND_CPU},
      {"CUDA", VPI_BACKEND_CUDA},
      {"XAVIER", VPI_BACKEND_XAVIER},
      {"ORIN", VPI_BACKEND_ORIN},
      {"PVA", VPI_BACKEND_PVA},
      {"ALL", VPI_BACKEND_ALL},
    });

}  // namespace

gxf::Expected<VPIBackend> StringToBackend(const std::string& str) {
  auto it = kStrToVpiBackend.find(str);

  if (it == kStrToVpiBackend.end()) {
    GXF_LOG_ERROR("Invalid VPI backend string '%s'", str.c_str());
    return gxf::Unexpected{GXF_ARGUMENT_INVALID};
  }

  return static_cast<VPIBackend>(it->second);
}

gxf::Expected<VPIImageFormat> VideoFormatToImageFormat(gxf::VideoFormat value) {
  switch (value) {
    case gxf::VideoFormat::GXF_VIDEO_FORMAT_D32F:
      return VPI_IMAGE_FORMAT_F32;
    case gxf::VideoFormat::GXF_VIDEO_FORMAT_D64F:
      return VPI_IMAGE_FORMAT_F64;
    case gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB:
      return VPI_IMAGE_FORMAT_RGB8;
    case gxf::VideoFormat::GXF_VIDEO_FORMAT_BGR:
      return VPI_IMAGE_FORMAT_BGR8;
    case gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12_ER:
      return VPI_IMAGE_FORMAT_Y8_ER;
    default:
      // TODO(kpatzwaldt): add cases for default return value and then throw error
      return gxf::Unexpected{GXF_INVALID_DATA_FORMAT};
  }
}

gxf::Expected<VPIPixelType> VideoFormatToPixelType(gxf::VideoFormat value) {
  switch (value) {
    case gxf::VideoFormat::GXF_VIDEO_FORMAT_D32F:
      return VPI_PIXEL_TYPE_F32;
    case gxf::VideoFormat::GXF_VIDEO_FORMAT_D64F:
      return VPI_PIXEL_TYPE_F64;
    case gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB:
    case gxf::VideoFormat::GXF_VIDEO_FORMAT_BGR:
      return VPI_PIXEL_TYPE_3U8;
    case gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12_ER:
      return VPI_PIXEL_TYPE_U8;
    default:
      // TODO(kpatzwaldt): add cases for default return value and then throw error
      return gxf::Unexpected{GXF_INVALID_DATA_FORMAT};
  }
}

}  // namespace vpi
}  // namespace isaac
}  // namespace nvidia
