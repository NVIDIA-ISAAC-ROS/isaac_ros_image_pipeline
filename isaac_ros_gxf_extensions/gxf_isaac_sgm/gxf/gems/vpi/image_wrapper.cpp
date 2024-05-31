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

#include "gems/vpi/image_wrapper.hpp"

#include "gems/gxf_helpers/expected_macro_gxf.hpp"
#include "gems/vpi/constants.hpp"

namespace nvidia {
namespace isaac {
namespace vpi {

gxf::Expected<void> ImageWrapper::createFromVideoBuffer(
    const gxf::VideoBuffer& video_buffer, uint64_t flags) {
  nvidia::gxf::VideoBufferInfo image_info = video_buffer.video_frame_info();

  auto image_format = UNWRAP_OR_RETURN(vpi::VideoFormatToImageFormat(image_info.color_format));
  auto pixel_type = UNWRAP_OR_RETURN(vpi::VideoFormatToPixelType(image_info.color_format));

  image_data_.bufferType = VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;
  image_data_.buffer.pitch.format = image_format;
  image_data_.buffer.pitch.numPlanes = 1;
  image_data_.buffer.pitch.planes[0].data = video_buffer.pointer();
  image_data_.buffer.pitch.planes[0].height = image_info.height;
  image_data_.buffer.pitch.planes[0].width = image_info.width;
  image_data_.buffer.pitch.planes[0].pixelType = pixel_type;
  image_data_.buffer.pitch.planes[0].pitchBytes = image_info.color_planes[0].stride;

  auto ret = vpiImageCreateWrapper(&image_data_, nullptr, flags, &image_);

  if (ret != VPI_SUCCESS) {
    GXF_LOG_ERROR("Failed to create image wrapper");
    return gxf::Unexpected{GXF_FAILURE};
  }

  return gxf::Expected<void>{};
}

gxf::Expected<void> ImageWrapper::update(const gxf::VideoBuffer& video_buffer) {
  image_data_.buffer.pitch.planes[0].data = video_buffer.pointer();
  VPIStatus ret = vpiImageSetWrapper(image_, &image_data_);

  if (ret != VPI_SUCCESS) {
    GXF_LOG_ERROR("Failed to update image wrapper");
    return gxf::Unexpected{GXF_FAILURE};
  }

  return gxf::Expected<void>{};
}

void ImageWrapper::release() {
  vpiImageDestroy(image_);
}

}  // namespace vpi
}  // namespace isaac
}  // namespace nvidia
