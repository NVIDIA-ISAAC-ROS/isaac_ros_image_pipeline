/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef ISAAC_ROS_STEREO_IMAGE_PROC__POINT_CLOUD_NODE_CUDA_HPP_
#define ISAAC_ROS_STEREO_IMAGE_PROC__POINT_CLOUD_NODE_CUDA_HPP_

#include "nvPointCloud.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <float.h>
#include <cstdint>
#include <stdexcept>
#include <string>

/**
 * @namespace PointCloudNode_CUDA
 *
 * @brief Functions for calling CUDA code for the PointCloudNode
 *
 */
namespace isaac_ros
{
namespace stereo_image_proc
{
namespace PointCloudNode_CUDA
{

/**
 * @brief Macro that calls the _checkCudaErrors function
 *
 * @param result The result from the CUDA function call
 */
#define throwIfCudaError(result) {_throwIfCudaError((result), __FILE__, __LINE__); \
}

/**
 * @brief Checks if a CUDA error has occurred. If so, throws a runtime error.
 *
 * @note This does not perform any ROS logging and thus should be used
 *        for CUDA-only files
 *
 * @param result The result of a CUDA function call
 * @param filename The name of the file calling
 * @param line_number The line number that the error occurred
 */
inline void _throwIfCudaError(cudaError_t result, const char * filename, int line_number)
{
  if (result != cudaSuccess) {
    throw std::runtime_error(
            "CUDA Error: " + std::string(cudaGetErrorString(result)) + " (error code: " +
            std::to_string(result) + ") at " + std::string(filename) + " in line " +
            std::to_string(line_number));
  }
}

/**
 * @brief Host function that calls the addColorToPointCloud kernel
 *
 * @warning This function only synchronizes with respect to the input stream
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
 * @param stream The CUDA stream to execute the kernel
 */

void addColorToPointCloud(
  float * pointcloud_data, const uint8_t * rgb, const int rgb_row_step,
  const int red_offset, const int green_offset, const int blue_offset, const int color_step,
  const nvPointCloudIntrinsics * intrinsics,
  const nvPointCloudProperties * cloud_properties,
  cudaStream_t stream);

/**
 * @brief Host function that calls the convertDisparityToPointCloud kernel
 *
 * @warning This function only synchronizes with respect to the input stream
 *
 * @param T The data type of the disparity image
 * @param pointcloud_data The input point cloud data buffer that will be modified
 * @param disparity_img The disparity image array, represented as a 1D array
 * @param disparity_row_step The step size for a row of the disparity image
 * @param intrinsics A struct that holds relevant camera intrinsic information
 * @param cloud_properties A struct that holds relevant point cloud properties
 * @param stream The CUDA stream to execute the kernel
 */
template<typename T>
void convertDisparityToPointCloud(
  float * pointcloud_data, const T * disparity_img,
  const int disparity_row_step, const nvPointCloudIntrinsics * intrinsics,
  const nvPointCloudProperties * cloud_properties,
  cudaStream_t stream);

}  // namespace PointCloudNode_CUDA
}  // namespace stereo_image_proc
}  // namespace isaac_ros

#endif  // ISAAC_ROS_STEREO_IMAGE_PROC__POINT_CLOUD_NODE_CUDA_HPP_
