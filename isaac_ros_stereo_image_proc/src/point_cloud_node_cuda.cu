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
#include <cuda.h>
#include <cuda_runtime.h>

namespace isaac_ros
{
namespace stereo_image_proc
{
namespace PointCloudNode_CUDA
{
__global__ void addColorToPointCloud_CUDA(
  float * pointcloud_data, const uint8_t * rgb, const int rgb_row_step,
  const int red_offset, const int green_offset, const int blue_offset, const int color_step,
  const nvPointCloudIntrinsics * intrinsics, const nvPointCloudProperties * cloud_properties)
{
  /**
   * @brief CUDA function that writes RGB values from a 8-bit RGB byte stream into a pointcloud data buffer
   *
   * @param pointcloud_data The input point cloud data buffer that will be modified
   * @param rgb The RGB image represented as a 1D raw byte stream
   * @param rgb_row_step The step size for a row of the RGB image
   * @param red_offset The offset of the red pixel for one pixel
   * @param green_offset The offset of the green pixel for one pixel
   * @param blue_offset The offset of the blue pixel for one pixel
   * @param color_step The number of points in one pixel
   * @param intrinsics A struct that holds relevant camera intrinsic information
   * @param cloud_properties A struct that holds relevant point cloud properties
   */

  // Calculate the index and stride for the loop
  int u_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int u_stride = gridDim.x * blockDim.x;

  int v_idx = blockIdx.y * blockDim.y + threadIdx.y;
  int v_stride = gridDim.y * blockDim.y;

  for (int v = v_idx; v < intrinsics->height; v += v_stride) {
    for (int u = u_idx; u < intrinsics->width; u += u_stride) {
      // Compute the location that we should access
      // And then access the R,G and B pixels
      int base_idx = v * rgb_row_step + u * color_step;
      uint8_t r_pixel = rgb[base_idx + red_offset];
      uint8_t g_pixel = rgb[base_idx + green_offset];
      uint8_t b_pixel = rgb[base_idx + blue_offset];

      uint32_t pixel;

      // Little endian: DC, R, G, B (DC = don't care)
      if (!cloud_properties->is_bigendian) {
        // Assumes 32-bit
        pixel = (static_cast<uint32_t>(r_pixel) << 16 |
          static_cast<uint32_t>(g_pixel) << 8 |
          static_cast<uint32_t>(b_pixel));
      }
      // Big endian: B, G, R, DC (DC = don't care)
      else {
        pixel = (static_cast<uint32_t>(b_pixel) << 24 |
          static_cast<uint32_t>(g_pixel) << 16 |
          static_cast<uint32_t>(r_pixel) << 8);
      }

      // Compute the point cloud data index to write to and write to it
      // Assumes 32-bit floating point data buffer
      int pointcloud_idx = v * cloud_properties->point_row_step + u * cloud_properties->point_step +
        cloud_properties->rgb_offset;
      pointcloud_data[pointcloud_idx] = *reinterpret_cast<float *>(&pixel);
    }
  }
}

void addColorToPointCloud(
  float * pointcloud_data, const uint8_t * rgb, int rgb_row_step,
  int red_offset, int green_offset, int blue_offset, int color_step,
  const nvPointCloudIntrinsics * intrinsics, const nvPointCloudProperties * cloud_properties,
  cudaStream_t stream)
{

  // The number of blocks per thread and the minimum number of blocks to use
  const int num_threads_per_block_dim_x = 16;
  const int num_threads_per_block_dim_y = 16;
  int num_blocks_x = (intrinsics->width + num_threads_per_block_dim_y - 1) / num_threads_per_block_dim_x;
  int num_blocks_y = (intrinsics->height + num_threads_per_block_dim_y - 1) / num_threads_per_block_dim_x;

  dim3 block_num(num_blocks_x, num_blocks_y, 1);
  dim3 threads_per_block(num_threads_per_block_dim_x, num_threads_per_block_dim_y, 1);
  
  addColorToPointCloud_CUDA <<<block_num, threads_per_block, 0, stream>>> (pointcloud_data, rgb, rgb_row_step,
  red_offset, green_offset, blue_offset, color_step, intrinsics, cloud_properties);

  cudaError_t cuda_result = cudaGetLastError();
  throwIfCudaError(cuda_result);

  // Wait for stream to finish before continuing
  cuda_result = cudaStreamSynchronize(stream);
  throwIfCudaError(cuda_result);
}

template<typename T>
__global__ void convertDisparityToPointCloud_CUDA(
  float * pointcloud_data, const T * disparity_img, const int disparity_row_step,
  const nvPointCloudIntrinsics * intrinsics,
  const nvPointCloudProperties * cloud_properties)
{
  /**
   * @brief CUDA function that writes computes 3D points from a disparity image and writes
   *         them to point cloud data buffer
   *
   * @param T The data type of the disparity image
   * @param pointcloud_data The input point cloud data buffer that will be modified
   * @param disparity_img The disparity image array, represented as a 1D array
   * @param disparity_row_step The step size for a row of the disparity image
   * @param intrinsics A struct that holds relevant camera intrinsic information
   * @param cloud_properties A struct that holds relevant point cloud properties
   */

  // Calculate the index and stride for the loop
  int u_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int u_stride = gridDim.x * blockDim.x;

  int v_idx = blockIdx.y * blockDim.y + threadIdx.y;
  int v_stride = gridDim.y * blockDim.y;

  for (int v = v_idx; v < intrinsics->height; v += v_stride) {
    for (int u = u_idx; u < intrinsics->width; u += u_stride) {
      // Type cast the data type to a float (ensure it's 32 bit) then assign it to a double
      // In order to check the error
      float disparity = static_cast<float>(disparity_img[v * disparity_row_step + u]);
      float X, Y, Z, W;

      // Calculate the X, Y, Z and W (scaling factor) values using the reprojection matrix
      W =
        static_cast<float>(intrinsics->reproject_matrix[3][2] * disparity +
        intrinsics->reproject_matrix[3][3]);
      X =
        static_cast<float>((intrinsics->reproject_matrix[0][0] * static_cast<float>(u) +
        intrinsics->reproject_matrix[0][3]) / W);
      Y =
        static_cast<float>((intrinsics->reproject_matrix[1][1] * static_cast<float>(v) +
        intrinsics->reproject_matrix[1][3]) / W);
      Z = static_cast<float>(intrinsics->reproject_matrix[2][3] / W);
      
      // The computed value is not finite; it's a bad point
      if (!isfinite(Z) || !isfinite(W) || !isfinite(X) || !isfinite(Y)) {
        X = cloud_properties->bad_point;
        Z = cloud_properties->bad_point;
        Y = cloud_properties->bad_point;
      }

      // Compute the point cloud data index to write to and write to it
      // Also scales it by the unit scaling (if necessary)
      int base_idx = v * cloud_properties->point_row_step + u * cloud_properties->point_step;
      pointcloud_data[base_idx +
        cloud_properties->x_offset] = static_cast<float>(X * intrinsics->unit_scaling);
      pointcloud_data[base_idx +
        cloud_properties->y_offset] = static_cast<float>(Y * intrinsics->unit_scaling);
      pointcloud_data[base_idx +
        cloud_properties->z_offset] = static_cast<float>(Z * intrinsics->unit_scaling);
    }
  }
}

template<typename T>
void convertDisparityToPointCloud(
  float * pointcloud_data, const T * disparity_img, const int disparity_row_step,
  const nvPointCloudIntrinsics * intrinsics, const nvPointCloudProperties * cloud_properties,
  cudaStream_t stream)
{
  // The number of blocks per thread and the minimum number of blocks to use
  const int num_threads_per_block_dim_x = 16;
  const int num_threads_per_block_dim_y = 16;
  int num_blocks_x = (intrinsics->width + num_threads_per_block_dim_y - 1) / num_threads_per_block_dim_x;
  int num_blocks_y = (intrinsics->height + num_threads_per_block_dim_y - 1) / num_threads_per_block_dim_x;

  dim3 block_num(num_blocks_x, num_blocks_y, 1);
  dim3 threads_per_block(num_threads_per_block_dim_x, num_threads_per_block_dim_y, 1);
  convertDisparityToPointCloud_CUDA<T><<< block_num, threads_per_block, 0, stream>>>
  (pointcloud_data, disparity_img, disparity_row_step, intrinsics, cloud_properties);

  cudaError_t cuda_result = cudaGetLastError();
  throwIfCudaError(cuda_result);

  // Wait for stream to finish before continuing
  cuda_result = cudaStreamSynchronize(stream);
  throwIfCudaError(cuda_result);
}

// Template initialization
template void convertDisparityToPointCloud<uint8_t>(
  float * pointcloud_data, const uint8_t * disparity_img,
  const int disparity_row_step, const nvPointCloudIntrinsics * intrinsics,
  const nvPointCloudProperties * cloud_properties, cudaStream_t stream);

template void convertDisparityToPointCloud<float>(
  float * pointcloud_data, const float * disparity_img,
  const int disparity_row_step, const nvPointCloudIntrinsics * intrinsics,
  const nvPointCloudProperties * cloud_properties, cudaStream_t stream);

template void convertDisparityToPointCloud<uint16_t>(
  float * pointcloud_data, const uint16_t * disparity_img,
  const int disparity_row_step, const nvPointCloudIntrinsics * intrinsics,
  const nvPointCloudProperties * cloud_properties, cudaStream_t stream);

}  // namespace PointCloudNode_CUDA
}  // namespace stereo_image_proc
}  // namespace isaac_ros
