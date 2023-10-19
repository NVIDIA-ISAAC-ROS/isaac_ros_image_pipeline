// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "depth_to_point_cloud_cuda.cu.hpp"

#include <unordered_map>

#include "cuda.h"  // NOLINT - include .h without directory
#include "cuda_runtime.h"  // NOLINT - include .h without directory

namespace
{
#define CHECK_CUDA_ERRORS(result){checkCudaErrors(result, __FILE__, __LINE__);}
inline void checkCudaErrors(cudaError_t result, const char * filename, int line_number)
{
  if (result != cudaSuccess) {
    GXF_LOG_ERROR(
      ("CUDA Error: " + std::string(cudaGetErrorString(result)) +
      " (error code: " + std::to_string(result) + ") at " +
      std::string(filename) + " in line " + std::to_string(line_number)).c_str());
  }
}

__device__ inline uint32_t GetRGBPixel_CUDA(uint8_t r, uint8_t g, uint8_t b)
{
  // Data format: don't care, R, G, B
  return static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 |
         static_cast<uint32_t>(b);
}

__device__ inline void ExtractR_G_B_Pixel_CUDA(
  uint8_t & r_pixel, uint8_t & g_pixel, uint8_t & b_pixel, const uint8_t * rgb_buffer,
  unsigned int rgb_index,
  const nvidia::isaac_ros::depth_image_proc::DepthProperties & depth_properties)
{
  r_pixel = rgb_buffer[rgb_index + depth_properties.red_offset];
  g_pixel = rgb_buffer[rgb_index + depth_properties.green_offset];
  b_pixel = rgb_buffer[rgb_index + depth_properties.blue_offset];
}

__device__ inline void WriteColorPointToBuffer_CUDA(
  float * point_cloud_buffer,
  uint32_t pixel, unsigned int point_cloud_index,
  const nvidia::isaac_ros::depth_image_proc::PointCloudProperties & point_cloud_properties)
{
  point_cloud_buffer[point_cloud_properties.point_step * point_cloud_index + point_cloud_properties.rgb_offset] = *reinterpret_cast<float *>(&pixel);
}

__device__ inline void ComputeXYZWPoint_CUDA(
  float & X, float & Y, float & Z,
  float depth, const float depth_x_index, const float depth_y_index,
  const nvidia::isaac_ros::depth_image_proc::DepthProperties depth_properties)
{
  // Compute X, Y, Z point coordinates
  X =
    (depth_x_index - depth_properties.c_x) * depth / depth_properties.f_x;
  Y =
    (depth_y_index - depth_properties.c_y) * depth / depth_properties.f_y;
  Z = depth;
}

__device__ inline bool IsBadPoint_CUDA(float X, float Y, float Z)
{
  return !isfinite(Z) || !isfinite(X) || !isfinite(Y);
}

__device__ inline void WriteBadPointToBuffer_CUDA(float & X, float & Y, float & Z, float bad_point)
{
  X = bad_point;
  Y = bad_point;
  Z = bad_point;
}

__device__ inline void WriteXYZPointToBuffer_CUDA(
  float * point_cloud_buffer, float X, float Y,
  float Z, unsigned int point_cloud_index,
  const nvidia::isaac_ros::depth_image_proc::PointCloudProperties & point_cloud_properties)
{
  point_cloud_buffer[point_cloud_properties.point_step * point_cloud_index + point_cloud_properties.x_offset] = X;
  point_cloud_buffer[point_cloud_properties.point_step * point_cloud_index + point_cloud_properties.y_offset] = Y;
  point_cloud_buffer[point_cloud_properties.point_step * point_cloud_index + point_cloud_properties.z_offset] = Z;
}

__global__ void ConvertDepthToPointCloud_CUDA(
  const float * depth_buffer,
  float * point_cloud_buffer,
  nvidia::isaac_ros::depth_image_proc::PointCloudProperties point_cloud_properties,
  nvidia::isaac_ros::depth_image_proc::DepthProperties depth_properties,
  int skip)
{
  unsigned int point_cloud_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_cloud_index < point_cloud_properties.n_points) {
    unsigned int depth_index = point_cloud_index*skip;
    if(depth_index < depth_properties.height * depth_properties.width){
      float depth = static_cast<float>(depth_buffer[depth_index]);
      float X, Y, Z;
      unsigned int depth_x_index = depth_index % depth_properties.width;
      unsigned int depth_y_index = depth_index / depth_properties.width;
      ComputeXYZWPoint_CUDA(X, Y, Z, depth, static_cast<float>(depth_x_index), static_cast<float>(depth_y_index), depth_properties);
      if (IsBadPoint_CUDA(X, Y, Z)) {
        WriteBadPointToBuffer_CUDA(X, Y, Z, point_cloud_properties.bad_point);
      }
      WriteXYZPointToBuffer_CUDA(point_cloud_buffer, X, Y, Z, point_cloud_index, point_cloud_properties); 
    }
  }
}

__global__ void ColorizePointCloud_CUDA(
  const uint8_t * rgb_buffer,
  float * point_cloud_buffer,
  nvidia::isaac_ros::depth_image_proc::PointCloudProperties point_cloud_properties,
  nvidia::isaac_ros::depth_image_proc::DepthProperties depth_properties,
  int skip)
{
  unsigned int point_cloud_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_cloud_index < point_cloud_properties.n_points) {
    int depth_index = point_cloud_index*skip;
    if(depth_index < depth_properties.height * depth_properties.width){
      unsigned int rgb_index = depth_index*point_cloud_properties.rgb_offset;
      uint8_t r_pixel, g_pixel, b_pixel;
      ExtractR_G_B_Pixel_CUDA(r_pixel, g_pixel, b_pixel, rgb_buffer, rgb_index, depth_properties);
      uint32_t pixel = GetRGBPixel_CUDA(r_pixel, g_pixel, b_pixel);
      WriteColorPointToBuffer_CUDA(point_cloud_buffer, pixel, point_cloud_index, point_cloud_properties);
    }
  }
}

}  // namespace

namespace nvidia
{
namespace isaac_ros
{
namespace depth_image_proc
{
DepthToPointCloudNodeCUDA::DepthToPointCloudNodeCUDA()
{
  // TODO(yuankunz):consider using gxf stream
  cudaStreamCreate(&stream_);
}

DepthToPointCloudNodeCUDA::~DepthToPointCloudNodeCUDA()
{
  cudaStreamDestroy(stream_);
}

gxf_result_t DepthToPointCloudNodeCUDA::DepthToPointCloudCuda(
  nvidia::isaac::CameraMessageParts depth_message,
  nvidia::isaac::CameraMessageParts rgb_image_message,
  nvidia::isaac_ros::messages::PointCloudMessageParts point_cloud_message,
  PointCloudProperties point_cloud_properties,
  DepthProperties depth_properties,
  bool colorize_point_cloud,
  int skip)
{

  // Get the number of CUDA blocks & threads
  constexpr int num_threads_per_block = 16;
  int num_blocks_per_grid = ((point_cloud_properties.n_points) / num_threads_per_block);
  dim3 threads_per_block(num_threads_per_block, 1, 1);
  dim3 block_num(num_blocks_per_grid, 1, 1);
  ConvertDepthToPointCloud_CUDA << < block_num, threads_per_block, 0, stream_ >> > (
    reinterpret_cast<float *>(depth_message.frame->pointer()),
    point_cloud_message.points->data<float>().value(), 
    point_cloud_properties, depth_properties, skip);
  CHECK_CUDA_ERRORS(cudaGetLastError());
  CHECK_CUDA_ERRORS(cudaStreamSynchronize(stream_));

  if(colorize_point_cloud){
    ColorizePointCloud_CUDA << < block_num, threads_per_block, 0, stream_ >> > (
    rgb_image_message.frame->pointer(),
    point_cloud_message.points->data<float>().value(), 
    point_cloud_properties, depth_properties, skip);
    CHECK_CUDA_ERRORS(cudaGetLastError());
    CHECK_CUDA_ERRORS(cudaStreamSynchronize(stream_));
  }

  return GXF_SUCCESS;
}

}  // namespace depth_image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#undef CHECK_CUDA_ERRORS
