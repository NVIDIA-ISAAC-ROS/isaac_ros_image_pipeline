// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <string>

#include "gxf/std/allocator.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace isaac {

// GXF codelet that subscribe left and right image using videobuffer,
// and publish the disparity using videobuffer format.
class SGMDisparity : public gxf::Codelet {
 public:
  SGMDisparity();
  ~SGMDisparity();

  gxf_result_t registerInterface(gxf::Registrar * registrar) override;
  gxf_result_t initialize() override;
  gxf_result_t deinitialize() override;

  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;

 private:
  // Data allocator to create a video buffer
  gxf::Parameter<gxf::Handle<gxf::Allocator>> pool_;
  // Data receiver to get left image data
  gxf::Parameter<gxf::Handle<gxf::Receiver>> left_image_receiver_;
  // Data receiver to get right image data
  gxf::Parameter<gxf::Handle<gxf::Receiver>> right_image_receiver_;
  // Data transmitter to send the data
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> output_transmitter_;
  // The backend for compute disparity
  gxf::Parameter<std::string> backend_;
  // Max disparity search range
  gxf::Parameter<float> max_disparity_;

  // Parameters to tune SGM performance
  gxf::Parameter<int32_t> confidence_threshold_;
  gxf::Parameter<int32_t> confidence_type_;
  gxf::Parameter<int32_t> window_size_;
  gxf::Parameter<int32_t> num_passes_;
  gxf::Parameter<int32_t> p1_;
  gxf::Parameter<int32_t> p2_;
  gxf::Parameter<int32_t> p2_alpha_;
  gxf::Parameter<int32_t> quality_;

  // Hide implementation details
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace isaac
}  // namespace nvidia
