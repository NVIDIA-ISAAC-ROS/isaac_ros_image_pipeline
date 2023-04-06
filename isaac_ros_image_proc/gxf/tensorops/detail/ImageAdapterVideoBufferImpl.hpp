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
#ifndef NVIDIA_CVCORE_VIDEO_BUFFER_ADAPTER_HPP
#define NVIDIA_CVCORE_VIDEO_BUFFER_ADAPTER_HPP

#include "../ImageUtils.hpp"

#include "gxf/multimedia/video.hpp"
#include "gxf/std/allocator.hpp"

#include "cv/core/Image.h"

namespace nvidia {
namespace cvcore {
namespace tensor_ops {
namespace detail {

gxf::Expected<::cvcore::ImageType> GetImageTypeFromVideoFormat(const gxf::VideoFormat format);

gxf::Expected<ImageInfo> GetVideoBufferInfo(gxf::Handle<gxf::VideoBuffer> video_buffer);

template<::cvcore::ImageType T,
         typename std::enable_if<T != ::cvcore::ImageType::NV12 && T != ::cvcore::ImageType::NV24>::type* = nullptr>
gxf::Expected<::cvcore::Image<T>> WrapImageFromVideoBuffer(gxf::Handle<gxf::VideoBuffer> video_buffer) {
  const auto info = GetVideoBufferInfo(video_buffer);
  if (!info) {
    return gxf::Unexpected{GXF_FAILURE};
  }
  using D      = typename ::cvcore::detail::ChannelTypeToNative<::cvcore::ImageTraits<T, 3>::CT>::Type;
  auto pointer = reinterpret_cast<D*>(video_buffer->pointer());
  if (!pointer) {
    return gxf::Unexpected{GXF_FAILURE};
  }
  const auto& color_planes = video_buffer->video_frame_info().color_planes;
  return ::cvcore::Image<T>(info.value().width, info.value().height, color_planes[0].stride, pointer,
                            info.value().is_cpu);
}

template<::cvcore::ImageType T,
         typename std::enable_if<T == ::cvcore::ImageType::NV12 || T == ::cvcore::ImageType::NV24>::type* = nullptr>
gxf::Expected<::cvcore::Image<T>> WrapImageFromVideoBuffer(gxf::Handle<gxf::VideoBuffer> video_buffer) {
  const auto info = GetVideoBufferInfo(video_buffer);
  if (!info) {
    return gxf::Unexpected{GXF_FAILURE};
  }
  // Note only U8 is supported in NV12/NV24
  auto pointer = reinterpret_cast<uint8_t*>(video_buffer->pointer());
  if (!pointer) {
    return gxf::Unexpected{GXF_FAILURE};
  }
  const auto& color_planes = video_buffer->video_frame_info().color_planes;
  return ::cvcore::Image<T>(info.value().width, info.value().height, color_planes[0].stride, color_planes[1].stride,
                            pointer, pointer + color_planes[1].offset, info.value().is_cpu);
}

template<::cvcore::ImageType T>
struct ImageTypeToVideoFormat {
  static constexpr gxf::VideoFormat format = gxf::VideoFormat::GXF_VIDEO_FORMAT_CUSTOM;
};

template<>
struct ImageTypeToVideoFormat<::cvcore::ImageType::NV12> {
  static constexpr gxf::VideoFormat format = gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12_ER;
};

template<>
struct ImageTypeToVideoFormat<::cvcore::ImageType::NV24> {
  static constexpr gxf::VideoFormat format = gxf::VideoFormat::GXF_VIDEO_FORMAT_NV24_ER;
};

template<>
struct ImageTypeToVideoFormat<::cvcore::ImageType::RGBA_U8> {
  static constexpr gxf::VideoFormat format = gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA;
};

template<>
struct ImageTypeToVideoFormat<::cvcore::ImageType::RGB_U8> {
  static constexpr gxf::VideoFormat format = gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB;
};

template<>
struct ImageTypeToVideoFormat<::cvcore::ImageType::RGB_F32> {
  static constexpr gxf::VideoFormat format = gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB32;
};

template<>
struct ImageTypeToVideoFormat<::cvcore::ImageType::BGR_U8> {
  static constexpr gxf::VideoFormat format = gxf::VideoFormat::GXF_VIDEO_FORMAT_BGR;
};

template<>
struct ImageTypeToVideoFormat<::cvcore::ImageType::BGR_F32> {
  static constexpr gxf::VideoFormat format = gxf::VideoFormat::GXF_VIDEO_FORMAT_BGR32;
};

template<>
struct ImageTypeToVideoFormat<::cvcore::ImageType::PLANAR_RGB_U8> {
  static constexpr gxf::VideoFormat format = gxf::VideoFormat::GXF_VIDEO_FORMAT_R8_G8_B8;
};

template<>
struct ImageTypeToVideoFormat<::cvcore::ImageType::PLANAR_RGB_F32> {
  static constexpr gxf::VideoFormat format = gxf::VideoFormat::GXF_VIDEO_FORMAT_R32_G32_B32;
};

template<>
struct ImageTypeToVideoFormat<::cvcore::ImageType::PLANAR_BGR_U8> {
  static constexpr gxf::VideoFormat format = gxf::VideoFormat::GXF_VIDEO_FORMAT_B8_G8_R8;
};

template<>
struct ImageTypeToVideoFormat<::cvcore::ImageType::PLANAR_BGR_F32> {
  static constexpr gxf::VideoFormat format = gxf::VideoFormat::GXF_VIDEO_FORMAT_B32_G32_R32;
};

template<>
struct ImageTypeToVideoFormat<::cvcore::ImageType::Y_U8> {
  static constexpr gxf::VideoFormat format = gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY;
};

template<>
struct ImageTypeToVideoFormat<::cvcore::ImageType::Y_F32> {
  static constexpr gxf::VideoFormat format = gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY32;
};

template<::cvcore::ImageType T>
struct DefaultNoPaddingColorPlanes {};

template<>
struct DefaultNoPaddingColorPlanes<::cvcore::ImageType::NV12> {
  DefaultNoPaddingColorPlanes(size_t width)
    : planes({gxf::ColorPlane("Y", 1, width), gxf::ColorPlane("UV", 2, width)}) {}
  std::array<gxf::ColorPlane, 2> planes;
};

template<>
struct DefaultNoPaddingColorPlanes<::cvcore::ImageType::NV24> {
  DefaultNoPaddingColorPlanes(size_t width)
    : planes({gxf::ColorPlane("Y", 1, width), gxf::ColorPlane("UV", 2, width * 2)}) {}
  std::array<gxf::ColorPlane, 2> planes;
};

template<>
struct DefaultNoPaddingColorPlanes<::cvcore::ImageType::RGBA_U8> {
  DefaultNoPaddingColorPlanes(size_t width)
    : planes({gxf::ColorPlane("RGBA", 4, width * 4)}) {}
  std::array<gxf::ColorPlane, 1> planes;
};

template<>
struct DefaultNoPaddingColorPlanes<::cvcore::ImageType::RGB_U8> {
  DefaultNoPaddingColorPlanes(size_t width)
    : planes({gxf::ColorPlane("RGB", 3, width * 3)}) {}
  std::array<gxf::ColorPlane, 1> planes;
};

template<>
struct DefaultNoPaddingColorPlanes<::cvcore::ImageType::RGB_F32> {
  DefaultNoPaddingColorPlanes(size_t width)
    : planes({gxf::ColorPlane("RGB", 12, width * 12)}) {}
  std::array<gxf::ColorPlane, 1> planes;
};

template<>
struct DefaultNoPaddingColorPlanes<::cvcore::ImageType::BGR_U8> {
  DefaultNoPaddingColorPlanes(size_t width)
    : planes({gxf::ColorPlane("BGR", 3, width * 3)}) {}
  std::array<gxf::ColorPlane, 1> planes;
};

template<>
struct DefaultNoPaddingColorPlanes<::cvcore::ImageType::BGR_F32> {
  DefaultNoPaddingColorPlanes(size_t width)
    : planes({gxf::ColorPlane("BGR", 12, width * 12)}) {}
  std::array<gxf::ColorPlane, 1> planes;
};

template<>
struct DefaultNoPaddingColorPlanes<::cvcore::ImageType::PLANAR_RGB_U8> {
  DefaultNoPaddingColorPlanes(size_t width)
    : planes({gxf::ColorPlane("R", 1, width), gxf::ColorPlane("G", 1, width), gxf::ColorPlane("B", 1, width)}) {}
  std::array<gxf::ColorPlane, 3> planes;
};

template<>
struct DefaultNoPaddingColorPlanes<::cvcore::ImageType::PLANAR_RGB_F32> {
  DefaultNoPaddingColorPlanes(size_t width)
    : planes(
        {gxf::ColorPlane("R", 4, width * 4), gxf::ColorPlane("G", 4, width * 4), gxf::ColorPlane("B", 4, width * 4)}) {}
  std::array<gxf::ColorPlane, 3> planes;
};

template<>
struct DefaultNoPaddingColorPlanes<::cvcore::ImageType::PLANAR_BGR_U8> {
  DefaultNoPaddingColorPlanes(size_t width)
    : planes({gxf::ColorPlane("B", 1, width), gxf::ColorPlane("G", 1, width), gxf::ColorPlane("R", 1, width)}) {}
  std::array<gxf::ColorPlane, 3> planes;
};

template<>
struct DefaultNoPaddingColorPlanes<::cvcore::ImageType::PLANAR_BGR_F32> {
  DefaultNoPaddingColorPlanes(size_t width)
    : planes(
        {gxf::ColorPlane("B", 4, width * 4), gxf::ColorPlane("G", 4, width * 4), gxf::ColorPlane("R", 4, width * 4)}) {}
  std::array<gxf::ColorPlane, 3> planes;
};

template<>
struct DefaultNoPaddingColorPlanes<::cvcore::ImageType::Y_U8> {
  DefaultNoPaddingColorPlanes(size_t width)
    : planes({gxf::ColorPlane("gray", 1, width)}) {}
  std::array<gxf::ColorPlane, 1> planes;
};

template<>
struct DefaultNoPaddingColorPlanes<::cvcore::ImageType::Y_F32> {
  DefaultNoPaddingColorPlanes(size_t width)
    : planes({gxf::ColorPlane("gray", 4, width * 4)}) {}
  std::array<gxf::ColorPlane, 1> planes;
};

// This include the list of image types that GXF supported so far
constexpr bool IsValidGXFImageType(const ::cvcore::ImageType type) {
  return type == ::cvcore::ImageType::NV12 || type == ::cvcore::ImageType::NV24 ||
         type == ::cvcore::ImageType::RGBA_U8 || type == ::cvcore::ImageType::RGB_U8 ||
         type == ::cvcore::ImageType::BGR_U8 || type == ::cvcore::ImageType::RGB_F32 ||
         type == ::cvcore::ImageType::BGR_F32 || type == ::cvcore::ImageType::PLANAR_RGB_U8 ||
         type == ::cvcore::ImageType::PLANAR_BGR_U8 || type == ::cvcore::ImageType::PLANAR_RGB_F32 ||
         type == ::cvcore::ImageType::PLANAR_BGR_F32 || type == ::cvcore::ImageType::Y_U8 ||
         type == ::cvcore::ImageType::Y_F32;
}

template<::cvcore::ImageType T, typename std::enable_if<IsValidGXFImageType(T)>::type* = nullptr>
gxf_result_t AllocateVideoBuffer(gxf::Handle<gxf::VideoBuffer> video_buffer, size_t width, size_t height,
                                 gxf::Handle<gxf::Allocator> allocator, bool is_cpu, bool allocate_pitch_linear) {
  if (width % 2 != 0 || height % 2 != 0) {
    GXF_LOG_ERROR("image width/height must be even for creation of gxf::VideoBuffer");
    return GXF_FAILURE;
  }
  if (allocate_pitch_linear) {
    auto result = video_buffer->resize<ImageTypeToVideoFormat<T>::format>(
      static_cast<uint32_t>(width), static_cast<uint32_t>(height), gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR,
      is_cpu ? gxf::MemoryStorageType::kHost : gxf::MemoryStorageType::kDevice, allocator);

    if (!result) {
      GXF_LOG_ERROR("resize VideoBuffer failed.");
      return GXF_FAILURE;
    }
  } else {
    DefaultNoPaddingColorPlanes<T> planes_trait(width);
    gxf::VideoFormatSize<ImageTypeToVideoFormat<T>::format> buffer_type_trait;
    uint64_t size = buffer_type_trait.size(width, height, planes_trait.planes);
    std::vector<gxf::ColorPlane> planes_filled{planes_trait.planes.begin(), planes_trait.planes.end()};
    gxf::VideoBufferInfo buffer_info{static_cast<uint32_t>(width), static_cast<uint32_t>(height),
                                     ImageTypeToVideoFormat<T>::format, planes_filled,
                                     gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR};
    auto result = video_buffer->resizeCustom(
      buffer_info, size, is_cpu ? gxf::MemoryStorageType::kHost : gxf::MemoryStorageType::kDevice, allocator);

    if (!result) {
      GXF_LOG_ERROR("custom resize VideoBuffer failed.");
      return GXF_FAILURE;
    }
  }
  return GXF_SUCCESS;
}

template<::cvcore::ImageType T, typename std::enable_if<!IsValidGXFImageType(T)>::type* = nullptr>
gxf_result_t AllocateVideoBuffer(gxf::Handle<gxf::VideoBuffer> video_buffer, size_t width, size_t height,
                                 gxf::Handle<gxf::Allocator> allocator, bool is_cpu, bool allocate_pitch_linear) {
  GXF_LOG_ERROR("image type not supported in gxf::VideoBuffer");
  return GXF_FAILURE;
}

} // namespace detail
} // namespace tensor_ops
} // namespace cvcore
} // namespace nvidia

#endif
