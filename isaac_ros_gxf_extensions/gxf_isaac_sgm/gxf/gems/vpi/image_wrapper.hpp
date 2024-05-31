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

#include "gxf/core/expected.hpp"
#include "gxf/multimedia/video.hpp"

#include "vpi/Image.h"
#include "vpi/Types.h"

namespace nvidia {
namespace isaac {
namespace vpi {

class ImageWrapper {
 public:
  ~ImageWrapper() { release(); }

  gxf::Expected<void> createFromVideoBuffer(
      const gxf::VideoBuffer& video_buffer, uint64_t flags);

  gxf::Expected<void> update(
      const gxf::VideoBuffer& video_buffer);

  void release();

  VPIImage& getImage() { return image_; }
  VPIImageData& getImageData() { return image_data_; }

 private:
  VPIImage image_{};
  VPIImageData image_data_{};
};

}  // namespace vpi
}  // namespace isaac
}  // namespace nvidia
