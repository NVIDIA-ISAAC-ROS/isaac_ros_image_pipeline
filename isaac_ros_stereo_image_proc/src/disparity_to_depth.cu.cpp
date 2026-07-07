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

#include "isaac_ros_stereo_image_proc/disparity_to_depth.cu.hpp"

#include <cstdint>
#include <limits>

#include "cuda.h"  // NOLINT
#include "cuda_runtime.h"  // NOLINT

namespace nvidia
{
namespace isaac_ros
{
namespace stereo_image_proc
{

__global__ void disparity_to_depth_kernel(
  const float * input, float * output, float baseline, float focal_length,
  int image_height, int image_width)
{
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  const uint32_t index = y * image_width + x;
  if (x < image_width && y < image_height) {
    if (input[index] > 0) {
      output[index] = (baseline * focal_length) / input[index];
    } else {
      output[index] = 0;
    }
  }
}

uint32_t ceil_div(uint32_t numerator, uint32_t denominator)
{
  uint32_t accumulator = numerator + denominator - 1;
  return accumulator / denominator;
}

cudaError_t disparity_to_depth_cuda(
  const float * input, float * output, float baseline, float focal_length,
  int image_height, int image_width, cudaStream_t stream) noexcept
{
  dim3 block(16, 16);
  dim3 grid(ceil_div(image_width, 16), ceil_div(image_height, 16), 1);
  disparity_to_depth_kernel << < grid, block, 0, stream >> > (input, output, baseline, focal_length,
  image_height, image_width);
  cudaError_t err = cudaGetLastError();
  return err;
}

}  // namespace stereo_image_proc
}  // namespace isaac_ros
}  // namespace nvidia
