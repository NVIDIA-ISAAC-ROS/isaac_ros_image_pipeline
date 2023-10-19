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
#ifndef NVIDIA_ISAAC_ROS_EXTENSIONS_DEPTH_TO_POINT_CLOUD_CUDA_HPP_
#define NVIDIA_ISAAC_ROS_EXTENSIONS_DEPTH_TO_POINT_CLOUD_CUDA_HPP_

#include <cstdint>
#include <stdexcept>
#include <vector>
#include <string>

#include "gxf/core/gxf.h"
#include "gxf/core/parameter.hpp"

#include "cuda.h"  // NOLINT - include .h without directory
#include "cuda_runtime.h"  // NOLINT - include .h without directory
#include "extensions/messages/camera_message.hpp"
#include "point_cloud_message.hpp"


namespace nvidia
{
namespace isaac_ros
{
namespace depth_image_proc
{
/**
 * @brief Struct that holds relevant cloud properties
 */
struct PointCloudProperties
{
  unsigned int point_step{0};   // Length of a point
  unsigned int n_points{0};   // Number of points in pointcloud
  unsigned int x_offset{0};   // Position of x for one point
  unsigned int y_offset{0};   // Position of y for one point
  unsigned int z_offset{0};   // Position of z for one point
  unsigned int rgb_offset{0};   // Position of rgb for one point
  float bad_point{0.0f};   // Representation of a bad point in the cloud
};

/**
 * @brief Struct that holds relevant depth properties
 */
struct DepthProperties
{
  float f_x{0};  // Focal length X
  float f_y{0};  // Focal length Y
  float c_x{0};  // Principal point X
  float c_y{0};  // Principal point Y
  unsigned int height{0};  // Height of the Depth image
  unsigned int width{0};  // Width of the Depth image
  unsigned int red_offset{0};  // Height of the Depth image
  unsigned int green_offset{0};  // Width of the Depth image
  unsigned int blue_offset{0};  // Width of the Depth image
};

/**
 * @brief Class that computes a PointCloud2 formatted point cloud given
 *  a depth image
 */
class DepthToPointCloudNodeCUDA
{
public:
  DepthToPointCloudNodeCUDA();
  ~DepthToPointCloudNodeCUDA();
  DepthToPointCloudNodeCUDA(const DepthToPointCloudNodeCUDA &) = delete;
  DepthToPointCloudNodeCUDA(const DepthToPointCloudNodeCUDA &&) = delete;
  DepthToPointCloudNodeCUDA & operator=(const DepthToPointCloudNodeCUDA &) = delete;
  DepthToPointCloudNodeCUDA & operator=(const DepthToPointCloudNodeCUDA &&) = delete;

  /**
   * @brief Host function that calls the CUDA kernel to compute the XYZ points
   *        using a depth image. This function will modify the point cloud CUDA buffer
   *        with the points at the correct location using the depth CUDA buffer.
   *        Warning: this function only synchronizes with respect to the input stream
   *
   * @param depth_message The input depth image message
   * @param rgb_image_message The optional input rgb image message
   * @param point_cloud_message The ouput PointCloud message
   * @param point_cloud_properties A struct that contains relevant cloud properties
   * @param depth_properties A struct that contains relevant depth properties
   * @param stream The CUDA stream to perform computation and synchronize on
   */
  gxf_result_t DepthToPointCloudCuda(
    nvidia::isaac::CameraMessageParts depth_message,
    nvidia::isaac::CameraMessageParts rgb_image_message,
    nvidia::isaac_ros::messages::PointCloudMessageParts point_cloud_message,
    PointCloudProperties point_cloud_properties,
    DepthProperties depth_properties,
    bool colorize_point_cloud,
    int skip);

private:
  // The CUDA stream that all the processing will happen on
  cudaStream_t stream_;
};
} // namespace depth_image_proc
} // namespace isaac_ros
} // namespace nvidia
#endif  // NVIDIA_ISAAC_ROS_EXTENSIONS_DEPTH_TO_POINT_CLOUD_CUDA_HPP_
