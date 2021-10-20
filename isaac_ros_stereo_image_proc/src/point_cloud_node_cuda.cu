/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "isaac_ros_stereo_image_proc/point_cloud_node_cuda.hpp"

#include <unordered_map>

#include "cuda.h"  // NOLINT - include .h without directory
#include "cuda_runtime.h"  // NOLINT - include .h without directory

namespace
{
#define CHECK_CUDA_ERRORS(result){checkCudaErrors(result, __FILE__, __LINE__);}
inline void checkCudaErrors(cudaError_t result, const char * filename, int line_number)
{
  if (result != cudaSuccess) {
    throw std::runtime_error(
            "CUDA Error: " + std::string(cudaGetErrorString(result)) +
            " (error code: " + std::to_string(result) + ") at " +
            std::string(filename) + " in line " + std::to_string(line_number));
  }
}

void AllocateStreamedCudaMem(
  void ** buffer_handle, const size_t buffer_size,
  const cudaStream_t & stream)
{
  CHECK_CUDA_ERRORS(cudaFree(*buffer_handle));
  CHECK_CUDA_ERRORS(cudaMallocManaged(buffer_handle, buffer_size, cudaMemAttachHost));
  CHECK_CUDA_ERRORS(cudaStreamAttachMemAsync(stream, *buffer_handle));
  CHECK_CUDA_ERRORS(cudaStreamSynchronize(stream));
}

void TransferCpuMemToCuda(
  void * cuda_mem, const void * cpu_mem, const size_t byte_size,
  const cudaStream_t & stream)
{
  CHECK_CUDA_ERRORS(
    cudaMemcpyAsync(
      cuda_mem, cpu_mem, byte_size,
      cudaMemcpyHostToDevice, stream));

  CHECK_CUDA_ERRORS(cudaStreamSynchronize(stream));
}

void TransferCudaMemToCpu(
  void * cpu_mem, const void * cuda_mem, const size_t byte_size,
  const cudaStream_t & stream)
{
  CHECK_CUDA_ERRORS(cudaMemcpyAsync(cpu_mem, cuda_mem, byte_size, cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA_ERRORS(cudaStreamSynchronize(stream));
}

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
  if (is_bigendian) {
    return GetBGRPixel_CUDA(r, g, b);
  } else {
    return GetRGBPixel_CUDA(r, g, b);
  }
}

__device__ inline void ExtractR_G_B_Pixel_CUDA(
  uint8_t & r_pixel, uint8_t & g_pixel, uint8_t & b_pixel, const uint8_t * rgb_buffer,
  unsigned int u, unsigned int v,
  const isaac_ros::stereo_image_proc::RGBProperties & rgb_properties)
{
  int rgb_idx = v * rgb_properties.row_step + u * rgb_properties.color_step;
  r_pixel = rgb_buffer[rgb_idx + rgb_properties.red_offset];
  g_pixel = rgb_buffer[rgb_idx + rgb_properties.green_offset];
  b_pixel = rgb_buffer[rgb_idx + rgb_properties.blue_offset];
}

__device__ inline void WriteColorPointToBuffer_CUDA(
  float * point_cloud_buffer,
  uint32_t pixel, unsigned int u, unsigned int v,
  const isaac_ros::stereo_image_proc::PointCloudProperties & cloud_properties)
{
  int pointcloud_idx = v * cloud_properties.point_row_step + u * cloud_properties.point_step +
    cloud_properties.rgb_offset;
  point_cloud_buffer[pointcloud_idx] = *reinterpret_cast<float *>(&pixel);
}

__global__ void AddColorToPointCloud_CUDA(
  float * point_cloud_buffer, const uint8_t * rgb_buffer,
  const isaac_ros::stereo_image_proc::PointCloudProperties cloud_properties,
  const isaac_ros::stereo_image_proc::RGBProperties rgb_properties)
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
  const isaac_ros::stereo_image_proc::CameraIntrinsics & intrinsics)
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
  const isaac_ros::stereo_image_proc::PointCloudProperties & cloud_properties, float unit_scaling)
{
  int cloud_idx = v * cloud_properties.point_row_step + u * cloud_properties.point_step;
  point_cloud_buffer[cloud_idx + cloud_properties.x_offset] = X * unit_scaling;
  point_cloud_buffer[cloud_idx + cloud_properties.y_offset] = Y * unit_scaling;
  point_cloud_buffer[cloud_idx + cloud_properties.z_offset] = Z * unit_scaling;
}

template<typename T>
__global__ void ConvertDisparityToPointCloud_CUDA(
  float * point_cloud_buffer, const T * disparity_buffer,
  const isaac_ros::stereo_image_proc::PointCloudProperties cloud_properties,
  const isaac_ros::stereo_image_proc::DisparityProperties disparity_properties,
  const isaac_ros::stereo_image_proc::CameraIntrinsics intrinsics,
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

namespace isaac_ros
{
namespace stereo_image_proc
{
PointCloudNodeCUDA::PointCloudNodeCUDA()
: use_color_{false}, unit_scaling_{1.0f}
{
  cudaStreamCreate(&stream_);
}

PointCloudNodeCUDA::~PointCloudNodeCUDA()
{
  cudaFree(rgb_buffer_);
  cudaFree(disparity_buffer_);
  cudaFree(point_cloud_buffer_);
  cudaStreamDestroy(stream_);
}

template<typename T>
void PointCloudNodeCUDA::ComputePointCloudData(
  std::vector<uint8_t> & output,
  const PointCloudProperties & cloud_properties,
  const std::vector<uint8_t> & disparity_buffer,
  const DisparityProperties & disparity_properties,
  const std::vector<uint8_t> & rgb_buffer,
  const RGBProperties & rgb_properties,
  const CameraIntrinsics & intrinsics)
{
  AllocatePointCloudBuffer(cloud_properties.buffer_size, stream_);
  UpdateDisparityBuffer(disparity_buffer, disparity_properties.buffer_size, stream_);
  ConvertDisparityToPointCloud<T>(cloud_properties, disparity_properties, intrinsics, stream_);

  if (use_color_) {
    UpdateRGBBuffer(rgb_buffer, rgb_properties.buffer_size, stream_);
    AddColorToPointCloud(cloud_properties, rgb_properties, stream_);
  }
  WritePointsTo(output, cloud_properties.buffer_size, stream_);
}

void PointCloudNodeCUDA::AllocatePointCloudBuffer(size_t size, const cudaStream_t & stream)
{
  if (point_cloud_buffer_capacity_ < size) {
    AllocateStreamedCudaMem((void **)&point_cloud_buffer_, size, stream);
    point_cloud_buffer_capacity_ = size;
  }
}

void PointCloudNodeCUDA::UpdateDisparityBuffer(
  const std::vector<uint8_t> & disparity_buffer,
  size_t size, const cudaStream_t & stream)
{
  if (disparity_buffer_capacity_ < size) {
    AllocateStreamedCudaMem((void **)&disparity_buffer_, size, stream);
    disparity_buffer_capacity_ = size;
  }
  TransferCpuMemToCuda(disparity_buffer_, disparity_buffer.data(), size, stream);
}

void PointCloudNodeCUDA::UpdateRGBBuffer(
  const std::vector<uint8_t> & rgb_buffer, size_t size,
  const cudaStream_t & stream)
{
  if (rgb_buffer_capacity_ < size) {
    AllocateStreamedCudaMem((void **)&rgb_buffer_, size, stream);
    rgb_buffer_capacity_ = size;
  }
  TransferCpuMemToCuda(rgb_buffer_, rgb_buffer.data(), size, stream);
}

void PointCloudNodeCUDA::WritePointsTo(
  std::vector<uint8_t> & output, size_t size,
  const cudaStream_t & stream)
{
  output.resize(size);
  TransferCudaMemToCpu(output.data(), point_cloud_buffer_, size, stream);
}

template<typename T>
void PointCloudNodeCUDA::ConvertDisparityToPointCloud(
  const PointCloudProperties & cloud_properties,
  const DisparityProperties & disparity_properties,
  const CameraIntrinsics & intrinsics, const cudaStream_t & stream)
{
  // Reinterpret so that disparity is accessed correctly
  const T * disparity_buffer = reinterpret_cast<const T *>(disparity_buffer_);

  // Get the number of CUDA blocks & threads
  int num_blocks_x = (disparity_properties.width + num_threads_per_block_x_ - 1) /
    num_threads_per_block_x_;
  int num_blocks_y = (disparity_properties.height + num_threads_per_block_y_ - 1) /
    num_threads_per_block_y_;

  dim3 block_num(num_blocks_x, num_blocks_y, 1);
  dim3 threads_per_block(num_threads_per_block_x_, num_threads_per_block_y_, 1);

  ConvertDisparityToPointCloud_CUDA<T><<<block_num, threads_per_block, 0, stream>>>(
    point_cloud_buffer_, disparity_buffer, cloud_properties, disparity_properties, intrinsics,
    unit_scaling_);

  // Wait for CUDA to finish
  CHECK_CUDA_ERRORS(cudaGetLastError());
  CHECK_CUDA_ERRORS(cudaStreamSynchronize(stream));
}

void PointCloudNodeCUDA::AddColorToPointCloud(
  const PointCloudProperties & cloud_properties,
  const RGBProperties & rgb_properties,
  const cudaStream_t & stream)
{
  // Get the number of CUDA blocks and threads
  int num_blocks_x = (rgb_properties.width + num_threads_per_block_x_ - 1) /
    num_threads_per_block_x_;
  int num_blocks_y = (rgb_properties.height + num_threads_per_block_y_ - 1) /
    num_threads_per_block_y_;

  dim3 block_num(num_blocks_x, num_blocks_y, 1);
  dim3 threads_per_block(num_threads_per_block_x_, num_threads_per_block_y_, 1);

  AddColorToPointCloud_CUDA<<<block_num, threads_per_block, 0, stream>>>(
    point_cloud_buffer_, rgb_buffer_, cloud_properties, rgb_properties);

  // Wait for CUDA to finish
  CHECK_CUDA_ERRORS(cudaGetLastError());
  CHECK_CUDA_ERRORS(cudaStreamSynchronize(stream));
}

// template instantation
template void PointCloudNodeCUDA::ComputePointCloudData<uint8_t>(
  std::vector<uint8_t> & output,
  const PointCloudProperties & cloud_properties,
  const std::vector<uint8_t> & disparity_buffer,
  const DisparityProperties & disparity_properties,
  const std::vector<uint8_t> & rgb_buffer,
  const RGBProperties & rgb_properties,
  const CameraIntrinsics & intrinsics);

template void PointCloudNodeCUDA::ComputePointCloudData<uint16_t>(
  std::vector<uint8_t> & output,
  const PointCloudProperties & cloud_properties,
  const std::vector<uint8_t> & disparity_buffer,
  const DisparityProperties & disparity_properties,
  const std::vector<uint8_t> & rgb_buffer,
  const RGBProperties & rgb_properties,
  const CameraIntrinsics & intrinsics);


template void PointCloudNodeCUDA::ComputePointCloudData<float>(
  std::vector<uint8_t> & output,
  const PointCloudProperties & cloud_properties,
  const std::vector<uint8_t> & disparity_buffer,
  const DisparityProperties & disparity_properties,
  const std::vector<uint8_t> & rgb_buffer,
  const RGBProperties & rgb_properties,
  const CameraIntrinsics & intrinsics);
}  // namespace stereo_image_proc
}  // namespace isaac_ros

#undef CHECK_CUDA_ERRORS
