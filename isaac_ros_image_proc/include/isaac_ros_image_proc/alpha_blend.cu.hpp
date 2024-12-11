// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef ISAAC_ROS_IMAGE_PROC__ALPHA_BLEND_CU_HPP_
#define ISAAC_ROS_IMAGE_PROC__ALPHA_BLEND_CU_HPP_

#include <cstdint>

#include "cuda.h"  // NOLINT - include .h without directory
#include "cuda_runtime.h"  // NOLINT - include .h without directory

namespace nvidia
{
namespace isaac_ros
{

void AlphaBlend(
  uint8_t * output_image, const uint8_t * segmentation_mask, const uint8_t * original_image,
  const uint32_t width, const uint32_t height, const double alpha, const bool mono_channel,
  const cudaStream_t stream);

}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_IMAGE_PROC__ALPHA_BLEND_CU_HPP_
