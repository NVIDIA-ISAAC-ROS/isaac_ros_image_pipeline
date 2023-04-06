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

#ifndef NVIDIA_ISAAC_ROS_EXTENSIONS_DISPARITY_COMPOSITOR_HPP_
#define NVIDIA_ISAAC_ROS_EXTENSIONS_DISPARITY_COMPOSITOR_HPP_

#include "gxf/core/entity.hpp"
#include "gxf/core/gxf.h"
#include "gxf/core/parameter.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/parameter_parser_std.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"


namespace nvidia
{
namespace isaac_ros
{

// GXF codelet that copies CameraModel and Timestamp components (if existed) from the reciever
class DisparityCompositor : public gxf::Codelet
{
public:
  gxf_result_t registerInterface(gxf::Registrar * registrar) noexcept override;
  gxf_result_t start() noexcept override;
  gxf_result_t tick() noexcept override;
  gxf_result_t stop() noexcept override {return GXF_SUCCESS;}

private:
  gxf::Parameter<gxf::Handle<gxf::Receiver>> disparity_receiver_;
  gxf::Parameter<gxf::Handle<gxf::Receiver>> left_camera_model_receiver_;
  gxf::Parameter<gxf::Handle<gxf::Receiver>> right_camera_model_receiver_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> output_transmitter_;

  // Min and Max disparity search range
  gxf::Parameter<float> min_disparity_;
  gxf::Parameter<float> max_disparity_;

  gxf::Parameter<bool> debug_;
};

}  // namespace isaac_ros
}  // namespace nvidia

#endif  // NVIDIA_ISAAC_ROS_EXTENSIONS_DISPARITY_COMPOSITOR_HPP_
