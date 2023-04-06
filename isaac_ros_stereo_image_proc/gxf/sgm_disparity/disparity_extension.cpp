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
#include <string>

#include "gxf/core/gxf.h"
#include "gxf/std/extension_factory_helper.hpp"
#include "disparity_extension/sgm_disparity.hpp"
#include "disparity_extension/disparity_compositor.hpp"

extern "C" {

GXF_EXT_FACTORY_BEGIN()

GXF_EXT_FACTORY_SET_INFO(
  0x0a0bbee8d30411ec, 0x9d640242ac120002, "DisparityExtension",
  "Disparity GXF extension",
  "NVIDIA", "1.0.0", "LICENSE");

GXF_EXT_FACTORY_ADD(
  0x7aaf40aa9bb340dd, 0x84e81a21e0603442,
  nvidia::isaac_ros::SGMDisparity,
  nvidia::gxf::Codelet,
  "Codelet calculate disparity using SGM.");

GXF_EXT_FACTORY_ADD(
  0x96791a52be1d4b27, 0x891d4a5ceaa6dff3,
  nvidia::isaac_ros::DisparityCompositor, nvidia::gxf::Codelet,
  "Copies CameraModel and Timestamp components into the main message forwarded to transmitter");

GXF_EXT_FACTORY_END()

}  // extern "C"
