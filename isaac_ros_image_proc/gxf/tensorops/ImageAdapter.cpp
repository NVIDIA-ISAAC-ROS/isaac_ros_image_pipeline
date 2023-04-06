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
#include "ImageAdapter.hpp"

namespace nvidia {
namespace cvcore {
namespace tensor_ops {

gxf_result_t ImageAdapter::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;

  result &= registrar->parameter(message_type_param_, "message_type");
  result &= registrar->parameter(image_type_param_, "image_type", "image type", "optional image type",
                                 gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(allocate_pitch_linear_, "allocate_pitch_linear",
                                 "if true, allocate output buffers as padded pitch linear surfaces", "", false);

  return gxf::ToResultCode(result);
}

gxf_result_t ImageAdapter::initialize() {
  if (message_type_param_.get() == "Tensor") {
    message_type_ = BufferType::TENSOR;
  } else if (message_type_param_.get() == "VideoBuffer") {
    message_type_ = BufferType::VIDEO_BUFFER;
  } else {
    GXF_LOG_ERROR("unknown buffer type.");
    return GXF_FAILURE;
  }

  const auto& image_type_param = image_type_param_.try_get();
  if (message_type_ == BufferType::TENSOR && !image_type_param) {
    GXF_LOG_INFO("image type must be specified for gxf::Tensor.");
    return GXF_FAILURE;
  }
  if (image_type_param) {
    const auto image_type = GetImageTypeFromString(image_type_param.value());
    if (!image_type) {
      return GXF_FAILURE;
    }
    image_type_ = image_type.value();
  }
  return GXF_SUCCESS;
}

gxf::Expected<ImageInfo> ImageAdapter::GetImageInfo(const gxf::Entity& message, const char* name) {
  if (message_type_ == BufferType::TENSOR) {
    auto tensor = message.get<gxf::Tensor>(name);
    if (!tensor) {
      return gxf::Unexpected{GXF_FAILURE};
    }
    return detail::GetTensorInfo(tensor.value(), image_type_);
  } else {
    auto video_buffer = message.get<gxf::VideoBuffer>(name);
    if (!video_buffer) {
      return gxf::Unexpected{GXF_FAILURE};
    }
    return detail::GetVideoBufferInfo(video_buffer.value());
  }
}

} // namespace tensor_ops
} // namespace cvcore
} // namespace nvidia
