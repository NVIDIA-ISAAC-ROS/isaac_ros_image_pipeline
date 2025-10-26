// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "extensions/depth_image_proc/depth_to_point_cloud.hpp"
#include "gxf/core/gxf.h"
#include "gxf/std/extension_factory_helper.hpp"

extern "C" {

GXF_EXT_FACTORY_BEGIN()

GXF_EXT_FACTORY_SET_INFO(
  0x8aed076e3bcb8170, 0xbb544f6115700daa, "DepthProcExtension",
  "Depth Proc GXF extension",
  "NVIDIA", "1.0.0", "LICENSE");

GXF_EXT_FACTORY_ADD(
  0x8b9613bf39953335, 0x8e35df7d160ed5c9,
  nvidia::isaac_ros::depth_image_proc::DepthToPointCloud,
  nvidia::gxf::Codelet,
  "Codelet to convert a depth image to pointcloud.");

GXF_EXT_FACTORY_END()

}  // extern "C"
