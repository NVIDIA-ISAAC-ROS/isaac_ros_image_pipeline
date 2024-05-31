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
#pragma once

#include <string>

#include "extensions/tensorops/components/detail/ImageAdapterTensorImpl.hpp"
#include "extensions/tensorops/components/detail/ImageAdapterVideoBufferImpl.hpp"
#include "extensions/tensorops/components/ImageUtils.hpp"
#include "extensions/tensorops/core/Image.h"
#include "gxf/core/component.hpp"
#include "gxf/core/parameter_parser_std.hpp"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/allocator.hpp"
#include "gxf/std/tensor.hpp"

namespace nvidia {
namespace isaac {
namespace tensor_ops {

// Enum class: gxf::Tensor and gxf::VideoBuffer
enum class BufferType {
  TENSOR,
  VIDEO_BUFFER,
};

// Utility component for conversion between message and cvcore image type
class ImageAdapter : public gxf::Component {
 public:
  virtual ~ImageAdapter() = default;
  ImageAdapter()          = default;

  gxf_result_t registerInterface(gxf::Registrar* registrar) override;
  gxf_result_t initialize() override;

  gxf::Expected<ImageInfo> GetImageInfo(const gxf::Entity& message, const char* name = nullptr);

  template<cvcore::tensor_ops::ImageType T>
  gxf::Expected<cvcore::tensor_ops::Image<T>> WrapImageFromMessage(const gxf::Entity& message,
      const char* name = nullptr) {
    if (message_type_ == BufferType::TENSOR) {
      auto tensor = message.get<gxf::Tensor>(name);
      if (!tensor) {
        return gxf::Unexpected{GXF_FAILURE};
      }
      return detail::WrapImageFromTensor<T>(tensor.value());
    } else {
      auto video_buffer = message.get<gxf::VideoBuffer>(name);
      if (!video_buffer) {
        return gxf::Unexpected{GXF_FAILURE};
      }
      return detail::WrapImageFromVideoBuffer<T>(video_buffer.value());
    }
  }

  template<cvcore::tensor_ops::ImageType T>
  gxf_result_t AddImageToMessage(gxf::Entity& message, size_t width, size_t height,
      gxf::Handle<gxf::Allocator> allocator,
      bool is_cpu, const char* name = nullptr) {
    if (message_type_ == BufferType::TENSOR) {
      auto tensor = message.add<gxf::Tensor>(name);
      if (!tensor) {
        return GXF_FAILURE;
      }
      return detail::AllocateTensor<T>(tensor.value(), width, height,
          allocator, is_cpu, allocate_pitch_linear_.get());
    } else {
      auto video_buffer = message.add<gxf::VideoBuffer>(name);
      if (!video_buffer) {
        return GXF_FAILURE;
      }
      return detail::AllocateVideoBuffer<T>(video_buffer.value(), width,
          height, allocator, is_cpu, allocate_pitch_linear_.get());
    }
  }

 private:
  gxf::Parameter<std::string> message_type_param_;
  gxf::Parameter<std::string> image_type_param_;
  gxf::Parameter<bool> allocate_pitch_linear_;

  cvcore::tensor_ops::ImageType image_type_;
  BufferType message_type_;
};

}  // namespace tensor_ops
}  // namespace isaac
}  // namespace nvidia
