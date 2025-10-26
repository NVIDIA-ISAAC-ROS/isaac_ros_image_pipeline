// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cstdint>

#include <sensor_msgs/msg/camera_info.hpp>

namespace nvidia
{
namespace isaac_ros
{
namespace depth_image_proc
{

#define CUDA_RETURN_IF_ERROR(cuda_call) \
  do { \
    cudaError_t error = (cuda_call); \
    if (error != cudaSuccess) { \
      fprintf( \
        stderr, "CUDA error at %s:%d - %s: %s\n", \
        __FILE__, __LINE__, #cuda_call, cudaGetErrorString(error)); \
      return error; \
    } \
  } while(0)

// Align depth image to color image
cudaError_t AlignDepthToColor(
  const float * depth_data,
  float * aligned_depth,
  const sensor_msgs::msg::CameraInfo & depth_camera_info,
  const sensor_msgs::msg::CameraInfo & color_camera_info,
  const double * depth_pose_color,
  cudaStream_t stream,
  float * gpu_time_ms = nullptr);

}  // namespace depth_image_proc
}  // namespace isaac_ros
}  // namespace nvidia
