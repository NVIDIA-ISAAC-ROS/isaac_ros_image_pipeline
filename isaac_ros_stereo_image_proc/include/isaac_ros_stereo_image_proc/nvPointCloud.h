/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef ISAAC_ROS_STEREO_IMAGE_PROC__NVPOINTCLOUD_H_
#define ISAAC_ROS_STEREO_IMAGE_PROC__NVPOINTCLOUD_H_

/**
 * @brief Struct that holds relevant camera intrinsics
 *
 */
typedef struct nvPointCloudIntrinsics
{
  double reproject_matrix[4][4];  // 4x4 reprojection matrix
  int reproject_matrix_rows;  // Number of rows of reprojection matrix
  int reproject_matrix_cols;  // Number of cols of reprojection matrix
  int height;  // Height of the camera image
  int width;  // Width of the camera image
  float unit_scaling;  // Unit to scale the reprojected 3D points
} nvPointCloudIntrinsics;

/**
 * @brief struct that holds relevant cloud properties
 *
 */
typedef struct nvPointCloudProperties
{
  int point_row_step;  // Length of a row
  int point_step;  // Length of a point
  int x_offset;  // Position of x for one point
  int y_offset;  // Position of y for one point
  int z_offset;  // Position of z for one point
  int rgb_offset;  // Position of rgb for one point
  int buffer_size;  // Size of pointcloud buffer (in bytes)
  bool is_bigendian;  // Is byte stream big or little endian
  float bad_point;  // Representation of a bad point in the cloud
} nvPointCloudProperties;

#endif  // ISAAC_ROS_STEREO_IMAGE_PROC__NVPOINTCLOUD_H_
