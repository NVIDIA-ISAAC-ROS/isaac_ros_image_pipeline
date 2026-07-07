// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_stereo_image_proc/point_cloud_cuda.cu.hpp"

#include <iostream>
#include <string>
#include <unordered_map>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wgcc-compat"   // or "-Wline-directive" if that's the one
#include "cuda.h"  // NOLINT
#include "cuda_runtime.h"  // NOLINT
#pragma GCC diagnostic pop

#include "isaac_ros_common/cuda_stream.hpp"

namespace
{
__device__ inline uint32_t GetBGRPixel_CUDA(uint8_t r, uint8_t g, uint8_t b)
{
  // Data format: B, G, R, don't care
  return static_cast<uint32_t>(b) << 24 | static_cast<uint32_t>(g) << 16 |
         static_cast<uint32_t>(r) << 8;
}

__device__ inline uint32_t GetRGBPixel_CUDA(uint8_t r, uint8_t g, uint8_t b)
{
  // Data format: don't care, R, G, B
  return static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 |
         static_cast<uint32_t>(b);
}

__device__ inline uint32_t GetPixel_CUDA(uint8_t r, uint8_t g, uint8_t b, bool is_bigendian)
{
  return is_bigendian ? GetBGRPixel_CUDA(r, g, b) : GetRGBPixel_CUDA(r, g, b);
}

__device__ inline void ExtractR_G_B_Pixel_CUDA(
  uint8_t & r_pixel, uint8_t & g_pixel, uint8_t & b_pixel, const uint8_t * rgb_buffer,
  unsigned int u, unsigned int v,
  const nvidia::isaac_ros::stereo_image_proc::RGBProperties & rgb_properties)
{
  int rgb_idx = v * rgb_properties.row_step + u * rgb_properties.color_step;
  r_pixel = rgb_buffer[rgb_idx + rgb_properties.red_offset];
  g_pixel = rgb_buffer[rgb_idx + rgb_properties.green_offset];
  b_pixel = rgb_buffer[rgb_idx + rgb_properties.blue_offset];
}

__device__ inline void WriteColorPointToBuffer_CUDA(
  float * point_cloud_buffer,
  uint32_t pixel, unsigned int u, unsigned int v,
  const nvidia::isaac_ros::stereo_image_proc::PointCloudProperties & cloud_properties)
{
  int pointcloud_idx = v * cloud_properties.point_row_step + u * cloud_properties.point_step +
    cloud_properties.rgb_offset;
  point_cloud_buffer[pointcloud_idx] = *reinterpret_cast<float *>(&pixel);
}

__global__ void AddColorToPointCloud_CUDA(
  float * point_cloud_buffer, const uint8_t * rgb_buffer,
  const nvidia::isaac_ros::stereo_image_proc::PointCloudProperties cloud_properties,
  const nvidia::isaac_ros::stereo_image_proc::RGBProperties rgb_properties)
{
  unsigned int u_idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int u_stride = gridDim.x * blockDim.x;

  unsigned int v_idx = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int v_stride = gridDim.y * blockDim.y;

  for (unsigned int v = v_idx; v < rgb_properties.height; v += v_stride) {
    for (unsigned int u = u_idx; u < rgb_properties.width; u += u_stride) {
      uint8_t r_pixel, g_pixel, b_pixel;
      ExtractR_G_B_Pixel_CUDA(r_pixel, g_pixel, b_pixel, rgb_buffer, u, v, rgb_properties);
      uint32_t pixel = GetPixel_CUDA(r_pixel, g_pixel, b_pixel, cloud_properties.is_bigendian);
      WriteColorPointToBuffer_CUDA(point_cloud_buffer, pixel, u, v, cloud_properties);
    }
  }
}

__device__ inline void ComputeXYZWPoint_CUDA(
  float & X, float & Y, float & Z, float & W,
  float disparity, const unsigned int u, const unsigned int v,
  const nvidia::isaac_ros::stereo_image_proc::CameraIntrinsics & intrinsics)
{
  // Compute X, Y, Z and W using the reprojection matrix
  W =
    static_cast<float>(intrinsics.reprojection_matrix[3][2] * disparity +
    intrinsics.reprojection_matrix[3][3]);
  X =
    static_cast<float>((intrinsics.reprojection_matrix[0][0] * static_cast<float>(u) +
    intrinsics.reprojection_matrix[0][3]) / W);
  Y =
    static_cast<float>((intrinsics.reprojection_matrix[1][1] * static_cast<float>(v) +
    intrinsics.reprojection_matrix[1][3]) / W);
  Z = static_cast<float>(intrinsics.reprojection_matrix[2][3] / W);
}

__device__ inline bool IsBadPoint_CUDA(float X, float Y, float Z, float W)
{
  return !isfinite(Z) || !isfinite(W) || !isfinite(X) || !isfinite(Y);
}

__device__ inline void WriteBadPointToBuffer_CUDA(float & X, float & Y, float & Z, float bad_point)
{
  X = bad_point;
  Y = bad_point;
  Z = bad_point;
}

__device__ inline void WriteXYZPointToBuffer_CUDA(
  float * point_cloud_buffer, float X, float Y,
  float Z, unsigned int u, unsigned int v,
  const nvidia::isaac_ros::stereo_image_proc::PointCloudProperties & cloud_properties,
  float unit_scaling)
{
  int cloud_idx = v * cloud_properties.point_row_step + u * cloud_properties.point_step;
  point_cloud_buffer[cloud_idx + cloud_properties.x_offset] = X * unit_scaling;
  point_cloud_buffer[cloud_idx + cloud_properties.y_offset] = Y * unit_scaling;
  point_cloud_buffer[cloud_idx + cloud_properties.z_offset] = Z * unit_scaling;
}

template<typename T>
__global__ void ConvertDisparityToPointCloud_CUDA(
  float * point_cloud_buffer, const T * disparity_buffer,
  const nvidia::isaac_ros::stereo_image_proc::PointCloudProperties cloud_properties,
  const nvidia::isaac_ros::stereo_image_proc::DisparityProperties disparity_properties,
  const nvidia::isaac_ros::stereo_image_proc::CameraIntrinsics intrinsics,
  float unit_scaling)
{
  unsigned int u_idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int u_stride = gridDim.x * blockDim.x;

  unsigned int v_idx = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int v_stride = gridDim.y * blockDim.y;

  for (unsigned int v = v_idx; v < disparity_properties.height; v += v_stride) {
    for (unsigned int u = u_idx; u < disparity_properties.width; u += u_stride) {
      float disparity = static_cast<float>(disparity_buffer[v * disparity_properties.row_step + u]);
      float X, Y, Z, W;
      ComputeXYZWPoint_CUDA(X, Y, Z, W, disparity, u, v, intrinsics);
      if (IsBadPoint_CUDA(X, Y, Z, W)) {
        WriteBadPointToBuffer_CUDA(X, Y, Z, cloud_properties.bad_point);
      }
      WriteXYZPointToBuffer_CUDA(point_cloud_buffer, X, Y, Z, u, v, cloud_properties, unit_scaling);
    }
  }
}
}  // namespace

namespace nvidia
{
namespace isaac_ros
{
namespace stereo_image_proc
{
PointCloudNodeCUDA::PointCloudNodeCUDA()
: use_color_{false}, unit_scaling_{1.0f}
{
}

PointCloudNodeCUDA::~PointCloudNodeCUDA() {}

template<typename T>
void PointCloudNodeCUDA::ComputePointCloudData(
  float * output,
  const PointCloudProperties & cloud_properties,
  const T * disparity_buffer,
  const DisparityProperties & disparity_properties,
  const uint8_t * rgb_buffer,
  const RGBProperties & rgb_properties,
  const CameraIntrinsics & intrinsics,
  const cudaStream_t & stream)
{
  ConvertDisparityToPointCloud<T>(
    output, cloud_properties, disparity_buffer, disparity_properties,
    intrinsics, stream);

  if (use_color_) {
    AddColorToPointCloud(output, rgb_buffer, cloud_properties, rgb_properties, stream);
  }
}

template<typename T>
void PointCloudNodeCUDA::ConvertDisparityToPointCloud(
  float * point_cloud_buffer,
  const PointCloudProperties & cloud_properties,
  const T * disparity_buffer,
  const DisparityProperties & disparity_properties,
  const CameraIntrinsics & intrinsics, const cudaStream_t & stream)
{
  // Get the number of CUDA blocks & threads
  constexpr int num_threads_per_block_x = 16;
  constexpr int num_threads_per_block_y = 16;

  // Validate inputs
  if (!point_cloud_buffer || !disparity_buffer) {
    return;
  }
  if (disparity_properties.width == 0 || disparity_properties.height == 0) {
    return;
  }

  int num_blocks_x = (disparity_properties.width + num_threads_per_block_x - 1) /
    num_threads_per_block_x;
  int num_blocks_y = (disparity_properties.height + num_threads_per_block_y - 1) /
    num_threads_per_block_y;

  dim3 block_num(num_blocks_x, num_blocks_y, 1);
  dim3 threads_per_block(num_threads_per_block_x, num_threads_per_block_y, 1);

  ConvertDisparityToPointCloud_CUDA<T><< < block_num, threads_per_block, 0, stream >> > (
    point_cloud_buffer, disparity_buffer, cloud_properties, disparity_properties, intrinsics,
    unit_scaling_);

  // Wait for CUDA to finish
  CHECK_CUDA_ERROR(cudaGetLastError(), __FILE__, __LINE__);
}

void PointCloudNodeCUDA::AddColorToPointCloud(
  float * point_cloud_buffer,
  const uint8_t * rgb_buffer,
  const PointCloudProperties & cloud_properties,
  const RGBProperties & rgb_properties,
  const cudaStream_t & stream)
{
  // Get the number of CUDA blocks and threads
  constexpr int num_threads_per_block_x = 16;
  constexpr int num_threads_per_block_y = 16;

  // Validate inputs
  if (!point_cloud_buffer || !rgb_buffer) {
    return;
  }
  if (rgb_properties.width == 0 || rgb_properties.height == 0) {
    return;
  }

  int num_blocks_x = (rgb_properties.width + num_threads_per_block_x - 1) /
    num_threads_per_block_x;
  int num_blocks_y = (rgb_properties.height + num_threads_per_block_y - 1) /
    num_threads_per_block_y;

  dim3 block_num(num_blocks_x, num_blocks_y, 1);
  dim3 threads_per_block(num_threads_per_block_x, num_threads_per_block_y, 1);

  AddColorToPointCloud_CUDA << < block_num, threads_per_block, 0, stream >> > (
    point_cloud_buffer, rgb_buffer, cloud_properties, rgb_properties);

  // Wait for CUDA to finish
  CHECK_CUDA_ERROR(cudaGetLastError(), __FILE__, __LINE__);
}

// template instantiation
template void PointCloudNodeCUDA::ComputePointCloudData<uint8_t>(
  float * output,
  const PointCloudProperties & cloud_properties,
  const uint8_t * disparity_buffer,
  const DisparityProperties & disparity_properties,
  const uint8_t * rgb_buffer,
  const RGBProperties & rgb_properties,
  const CameraIntrinsics & intrinsics,
  const cudaStream_t & stream);

template void PointCloudNodeCUDA::ComputePointCloudData<uint16_t>(
  float * output,
  const PointCloudProperties & cloud_properties,
  const uint16_t * disparity_buffer,
  const DisparityProperties & disparity_properties,
  const uint8_t * rgb_buffer,
  const RGBProperties & rgb_properties,
  const CameraIntrinsics & intrinsics,
  const cudaStream_t & stream);

template void PointCloudNodeCUDA::ComputePointCloudData<float>(
  float * output,
  const PointCloudProperties & cloud_properties,
  const float * disparity_buffer,
  const DisparityProperties & disparity_properties,
  const uint8_t * rgb_buffer,
  const RGBProperties & rgb_properties,
  const CameraIntrinsics & intrinsics,
  const cudaStream_t & stream);

}  // namespace stereo_image_proc
}  // namespace isaac_ros
}  // namespace nvidia
