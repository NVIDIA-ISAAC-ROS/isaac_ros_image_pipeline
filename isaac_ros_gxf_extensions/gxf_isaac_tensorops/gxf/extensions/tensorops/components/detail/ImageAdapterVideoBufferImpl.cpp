// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "ImageAdapterVideoBufferImpl.hpp"

namespace nvidia {
namespace isaac {
namespace tensor_ops {
namespace detail {
using ImageType = cvcore::tensor_ops::ImageType;
gxf::Expected<cvcore::tensor_ops::ImageType> GetImageTypeFromVideoFormat(
    const gxf::VideoFormat format) {
  switch (format) {
  case gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12:
  case gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12_ER: {
    return ImageType::NV12;
  }
  case gxf::VideoFormat::GXF_VIDEO_FORMAT_NV24:
  case gxf::VideoFormat::GXF_VIDEO_FORMAT_NV24_ER: {
    return ImageType::NV24;
  }
  case gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA: {
    return ImageType::RGBA_U8;
  }
  case gxf::VideoFormat::GXF_VIDEO_FORMAT_BGRA: {
    return ImageType::BGRA_U8;
  }
  case gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB: {
    return ImageType::RGB_U8;
  }
  case gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB32: {
    return ImageType::RGB_F32;
  }
  case gxf::VideoFormat::GXF_VIDEO_FORMAT_BGR: {
    return ImageType::BGR_U8;
  }
  case gxf::VideoFormat::GXF_VIDEO_FORMAT_BGR32: {
    return ImageType::BGR_F32;
  }
  case gxf::VideoFormat::GXF_VIDEO_FORMAT_R8_G8_B8: {
    return ImageType::PLANAR_RGB_U8;
  }
  case gxf::VideoFormat::GXF_VIDEO_FORMAT_R32_G32_B32: {
    return ImageType::PLANAR_RGB_F32;
  }
  case gxf::VideoFormat::GXF_VIDEO_FORMAT_B8_G8_R8: {
    return ImageType::PLANAR_BGR_U8;
  }
  case gxf::VideoFormat::GXF_VIDEO_FORMAT_B32_G32_R32: {
    return ImageType::PLANAR_BGR_F32;
  }
  case gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY: {
    return ImageType::Y_U8;
  }
  case gxf::VideoFormat::GXF_VIDEO_FORMAT_D32F: {
    return ImageType::Y_F32;
  }
  default: {
    GXF_LOG_ERROR("invalid video format.");
    return gxf::Unexpected{GXF_FAILURE};
  }
  }
}

gxf::Expected<ImageInfo> GetVideoBufferInfo(gxf::Handle<gxf::VideoBuffer> video_buffer) {
  const auto buffer_info  = video_buffer->video_frame_info();
  const auto storage_type = video_buffer->storage_type();
  auto image_type         = GetImageTypeFromVideoFormat(buffer_info.color_format);
  if (!image_type) {
    return gxf::Unexpected{GXF_FAILURE};
  }
  return ImageInfo{image_type.value(), buffer_info.width, buffer_info.height,
                   storage_type != gxf::MemoryStorageType::kDevice};
}

}  // namespace detail
}  // namespace tensor_ops
}  // namespace isaac
}  // namespace nvidia
