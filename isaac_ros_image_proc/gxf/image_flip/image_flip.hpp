// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NVIDIA_ISAAC_ROS_EXTENSIONS_IMAGE_FLIP_HPP_
#define NVIDIA_ISAAC_ROS_EXTENSIONS_IMAGE_FLIP_HPP_

#include <memory>
#include <queue>
#include <string>

#include "gxf/core/entity.hpp"
#include "gxf/core/gxf.h"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/allocator.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"

#include "vpi/VPI.h"
#include "vpi/algo/ConvertImageFormat.h"
#include "vpi/algo/ImageFlip.h"

namespace nvidia
{
namespace isaac_ros
{
// GXF codelet that subscribes image using videobuffer,
// then publishes a flipped image
class ImageFlip : public gxf::Codelet
{
public:
  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;
  gxf_result_t registerInterface(gxf::Registrar * registrar) override;

private:
  // The name of the input image
  gxf::Parameter<std::string> image_name_;
  // The name of the output video buffer
  gxf::Parameter<std::string> output_name_;
  // Data allocator to create a video buffer
  gxf::Parameter<gxf::Handle<gxf::Allocator>> pool_;
  // Data receiver to get image data
  gxf::Parameter<gxf::Handle<gxf::Receiver>> image_receiver_;
  // Data transmitter to send the data
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> output_transmitter_;
  // The backend for computing flip
  gxf::Parameter<std::string> vpi_backends_param_;
  // Mode for computing flip
  gxf::Parameter<std::string> vpi_flip_mode_param_;

  // Shared VPI stream for submitting all operations
  VPIStream vpi_stream_{};

  // VPI image in original input dimensions
  VPIImage input_{};
  VPIImageData input_data_{};

  // Final flip output in display-friendly format
  VPIImage flipped_{};
  VPIImageData flipped_data_{};

  // VPI backends
  uint64_t vpi_backends_{};
  uint64_t vpi_flags_{};

  // VPI flip mode
  VPIFlipMode vpi_flip_mode_{};

  // Cached values from previous iteration to compare against
  uint32_t prev_height_{};
  uint32_t prev_width_{};
  nvidia::gxf::VideoFormat prev_color_format_{};
};

}  // namespace isaac_ros
}  // namespace nvidia

#endif  // NVIDIA_ISAAC_ROS_EXTENSIONS_IMAGE_FLIP_HPP_
