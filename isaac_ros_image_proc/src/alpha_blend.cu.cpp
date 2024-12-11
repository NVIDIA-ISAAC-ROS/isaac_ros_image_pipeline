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

#include "isaac_ros_image_proc/alpha_blend.cu.hpp"

#include <string>
#include <stdexcept>

namespace nvidia
{
namespace isaac_ros
{
namespace
{

constexpr const int kNumColorChannels = 3;

/**
 * v: row number of the pixel
 * u: column number of the pixel
 * c: color channel number of the pixel
 */
__device__ inline uint32_t GetFlattenedIndex(
  const uint32_t v, const uint32_t u, const uint32_t c,
  const uint32_t width, const uint32_t channels)
{
  return v * width * channels + u * channels + c;
}

__device__ inline uint32_t GetMonoIndex(const uint32_t v, const uint32_t u, const uint32_t width)
{
  return GetFlattenedIndex(v, u, 0, width, 1);
}

__global__ void AlphaBlendImpl(
  uint8_t * output_image, const uint8_t * segmentation_mask, const uint8_t * original_image,
  const uint32_t width, const uint32_t height, const double alpha, const bool mono_channel)
{
  uint32_t u_idx{blockIdx.x * blockDim.x + threadIdx.x};
  uint32_t u_stride{gridDim.x * blockDim.x};

  uint32_t v_idx{blockIdx.y * blockDim.y + threadIdx.y};
  uint32_t v_stride{gridDim.y * blockDim.y};

  for (uint32_t v = v_idx; v < height; v += v_stride) {
    for (uint32_t u = u_idx; u < width; u += u_stride) {
      for (int c = 0; c < kNumColorChannels; c++) {
        uint8_t original_value =
          original_image[GetFlattenedIndex(v, u, c, width, kNumColorChannels)];
        uint8_t mask_value;
        mask_value = mono_channel ? segmentation_mask[GetMonoIndex(v, u, width)] :
          segmentation_mask[GetFlattenedIndex(v, u, c, width, kNumColorChannels)];

        output_image[GetFlattenedIndex(
            v, u, c, width, kNumColorChannels)] = original_value * alpha + mask_value * (1 - alpha);
      }
    }
  }
}

}  // namespace

void AlphaBlend(
  uint8_t * output_image, const uint8_t * segmentation_mask, const uint8_t * original_image,
  const uint32_t width, const uint32_t height, const double alpha, const bool mono_channel,
  const cudaStream_t stream)
{
  dim3 threads_per_block{32, 32, 1};
  dim3 blocks{(width + threads_per_block.x - 1) / threads_per_block.x,
    (height + threads_per_block.y - 1) / threads_per_block.y, 1};
  AlphaBlendImpl << < blocks, threads_per_block, 0, stream >> > (
    output_image, segmentation_mask, original_image, width, height, alpha, mono_channel);
}
}  // namespace isaac_ros
}  // namespace nvidia
