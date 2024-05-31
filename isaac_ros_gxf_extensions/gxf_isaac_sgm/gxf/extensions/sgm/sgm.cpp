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
#include <string>

#include "extensions/sgm/camera_message_compositor.hpp"
#include "extensions/sgm/disparity_compositor.hpp"
#include "extensions/sgm/sgm_disparity.hpp"

#include "gxf/core/gxf.h"
#include "gxf/std/extension_factory_helper.hpp"

extern "C" {

GXF_EXT_FACTORY_BEGIN()

GXF_EXT_FACTORY_SET_INFO(
  0x41c079d2b60c11ed, 0xa31f9b6a01b12786, "DisparityExtension",
  "Disparity GXF extension",
  "NVIDIA", "2.0.0", "LICENSE");

GXF_EXT_FACTORY_ADD(
  0x50b31328b60c11ed, 0xbae02b37db5190f1,
  nvidia::isaac::SGMDisparity,
  nvidia::gxf::Codelet,
  "Codelet calculate disparity using SGM.");

GXF_EXT_FACTORY_ADD(
  0x8ea5176b0db388ea, 0x816acf080055e756,
  nvidia::isaac::CameraMessageCompositor, nvidia::gxf::Codelet,
  "Combines video buffer with intrinsics and extrinsics to create CameraMessage");

GXF_EXT_FACTORY_ADD(
  0x620edc9cb60c11ed, 0xaa55a38997417df8,
  nvidia::isaac::DisparityCompositor, nvidia::gxf::Codelet,
  "Copies CameraModel and Timestamp components into the main message forwarded to transmitter");

GXF_EXT_FACTORY_END()

}  // extern "C"
