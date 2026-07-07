// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdint>
#include <stdexcept>
#include <vector>
#include <string>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wgcc-compat"   // or "-Wline-directive" if that's the one
#include "cuda.h"  // NOLINT
#include "cuda_runtime.h"  // NOLINT
#pragma GCC diagnostic pop

namespace nvidia
{
namespace isaac_ros
{
namespace stereo_image_proc
{
/**
 * @brief Struct that holds relevant camera intrinsics
 * using a left & right camera info pair
 */
struct CameraIntrinsics
{
  double reprojection_matrix[4][4] = {0};
  static const int reprojection_matrix_rows = 4;  // Number of rows of reprojection matrix
  static const int reprojection_matrix_cols = 4;   // Number of cols of reprojection matrix
};

/**
 * @brief Struct that holds relevant cloud properties
 *  from a PointCloud2 message
 */
struct PointCloudProperties
{
  unsigned int point_row_step{0};   // Length of a row
  unsigned int point_step{0};   // Length of a point
  unsigned int n_points{0};   // Number of points in pointcloud
  unsigned int x_offset{0};   // Position of x for one point
  unsigned int y_offset{0};   // Position of y for one point
  unsigned int z_offset{0};   // Position of z for one point
  unsigned int rgb_offset{0};   // Position of rgb for one point
  unsigned int buffer_size{0};   // Size of pointcloud buffer (in bytes)
  bool is_bigendian{false};   // Is byte stream big or little endian
  float bad_point{0.0f};   // Representation of a bad point in the cloud
};

/**
 * @brief Struct that holds relevant disparity properties
 * from a DisparityImage message
 */
struct DisparityProperties
{
  unsigned int row_step{0};  // Length of a row
  unsigned int height{0};  // Height of the disparity image
  unsigned int width{0};  // Width of the disparity image
  unsigned int buffer_size{0};  // Size of disparity buffer (in bytes)
  std::string encoding{""};  // Data format of the disparity image
};

/**
 * @brief Struct that holds relevant RGB properties
 *  from an Image message
 */
struct RGBProperties
{
  unsigned int row_step{0};  // Length of a row
  unsigned int height{0};  // Height of the RGB image
  unsigned int width{0};  // Width of the RGB image
  unsigned int buffer_size{0};  // Size of RGB buffer (in bytes)
  std::string encoding{""};  // Data format of the RGB image
  unsigned int red_offset{0};  // Position of red point for one pixel
  unsigned int green_offset{0};  // Position of green point for one pixel
  unsigned int blue_offset{0};  // Position of blue point for one pixel
  unsigned int color_step{0};  // Number of points per pixel
};

/**
 * @brief Class that computes a PointCloud2 formatted point cloud given
 *  a 1D disparity byte stream and a 1D RGB byte stream
 */
class PointCloudNodeCUDA
{
public:
  PointCloudNodeCUDA();
  ~PointCloudNodeCUDA();
  PointCloudNodeCUDA(const PointCloudNodeCUDA &) = delete;
  PointCloudNodeCUDA(const PointCloudNodeCUDA &&) = delete;
  PointCloudNodeCUDA & operator=(const PointCloudNodeCUDA &) = delete;
  PointCloudNodeCUDA & operator=(const PointCloudNodeCUDA &&) = delete;

  /**
   * @brief Sets the amount to scale the xyz points
   *
   * @param unit_scaling The desired unit scaling value
   */
  void SetUnitScaling(const float & unit_scaling) {unit_scaling_ = unit_scaling;}

  /**
   * @brief Sets whether to use color or not
   *
   * @param use_color The boolean value for whether to use color or not
   */
  void SetUseColor(const bool & use_color) {use_color_ = use_color;}

  /**
   * @brief Performs all the necessary steps to generate a PointCloud2 formatted
   *        byte stream and writes it to the output vector
   *
   * @tparam T The data type of the disparity buffer
   * @param[out] output The vector to write the generated PointCloud2 to
   * @param[in] cloud_properties A struct that contains relevant cloud properties
   * @param[in] disparity_buffer The vector that represents the disparity image as a byte stream
   * @param[in] disparity_properties A struct that contains relevant disparity properties
   * @param[in] rgb_buffer The vector that represents the RGB image as a byte stream
   * @param[in] rgb_properties A struct that contains relevant RGB properties
   * @param[in] intrinsics A struct that contains a reprojection matrix
   */
  template<typename T>
  void ComputePointCloudData(
    float * output,
    const PointCloudProperties & cloud_properties,
    const T * disparity_buffer,
    const DisparityProperties & disparity_properties,
    const uint8_t * rgb_buffer,
    const RGBProperties & rgb_properties,
    const CameraIntrinsics & intrinsics,
    const cudaStream_t & stream);

private:
  // The amount to scale the xyz points by
  float unit_scaling_;

  // Whether to use color or not
  bool use_color_;

  // The CUDA stream that all the processing will happen on
  cudaStream_t stream_;

  /**
   * @brief Host function that calls the CUDA kernel to compute the XYZ points
   *        using a disparity image. This function will modify the point cloud CUDA buffer
   *        with the points at the correct location using the disparity CUDA buffer.
   *        Warning: this function only synchronizes with respect to the input stream
   *
   * @tparam T The data type of the disparity buffer
   * @param cloud_properties A struct that contains relevant cloud properties
   * @param disparity_properties A struct that contains relevant disparity properties
   * @param intrinsics A struct that contains the reprojection matrix
   * @param stream The CUDA stream to perform computation and synchronize on
   */
  template<typename T>
  void ConvertDisparityToPointCloud(
    float * point_cloud_buffer,
    const PointCloudProperties & cloud_properties,
    const T * disparity_buffer,
    const DisparityProperties & disparity_properties,
    const CameraIntrinsics & intrinsics,
    const cudaStream_t & stream);

/**
 * @brief Host function that calls the CUDA kernel to add color using a RGB image.
 *        This function will modify the point cloud CUDA buffer with the RGB values
 *        at the correct location using the RGB CUDA buffer.
 *        Warning: this function only synchronizes with respect to the input stream
 *
 * @param cloud_properties A struct that contains relevant cloud properties
 * @param rgb_properties A struct that contains relevant RGB properties
 * @param stream The CUDA stream perform computation and synchronize on
 */
  void AddColorToPointCloud(
    float * point_cloud_buffer,
    const uint8_t * rgb_buffer,
    const PointCloudProperties & cloud_properties,
    const RGBProperties & rgb_properties,
    const cudaStream_t & stream);
};

}  // namespace stereo_image_proc
}  // namespace isaac_ros
}  // namespace nvidia
