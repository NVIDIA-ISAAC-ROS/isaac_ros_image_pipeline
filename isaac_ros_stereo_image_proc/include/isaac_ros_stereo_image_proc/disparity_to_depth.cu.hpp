// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime.h>

namespace nvidia
{
namespace isaac_ros
{
namespace stereo_image_proc
{

cudaError_t disparity_to_depth_cuda(
  const float * input, float * output, float baseline, float focal_length,
  int imageHeight, int imageWidth, cudaStream_t stream = nullptr) noexcept;

}  // namespace stereo_image_proc
}  // namespace isaac_ros
}  // namespace nvidia
