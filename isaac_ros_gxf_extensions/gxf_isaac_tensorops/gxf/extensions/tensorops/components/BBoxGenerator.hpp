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

#include "gxf/std/allocator.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace isaac {
namespace tensor_ops {

// BBox Generator Codelet.
class BBoxGenerator : public gxf::Codelet {
 public:
  gxf_result_t registerInterface(gxf::Registrar* registrar) override;
  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override {
    return GXF_SUCCESS;
  }

 private:
  // The width of the image
  gxf::Parameter<size_t> image_width_;
  // The height of the image
  gxf::Parameter<size_t> image_height_;
  // The number of the output bboxes
  gxf::Parameter<size_t> bbox_count_;
  // The width of the output bbox
  gxf::Parameter<size_t> bbox_width_;
  // The height of the output bbox
  gxf::Parameter<size_t> bbox_height_;
  // The x coordinate of the output bbox left top corner
  gxf::Parameter<size_t> bbox_loc_x_;
  // The y coordinate of the output bbox left top corner
  gxf::Parameter<size_t> bbox_loc_y_;
  // Data allocator to create a tensor
  gxf::Parameter<gxf::Handle<gxf::Allocator>> pool_;
  // Data transmitter to send the data
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> transmitter_;
};

}  // namespace tensor_ops
}  // namespace isaac
}  // namespace nvidia
