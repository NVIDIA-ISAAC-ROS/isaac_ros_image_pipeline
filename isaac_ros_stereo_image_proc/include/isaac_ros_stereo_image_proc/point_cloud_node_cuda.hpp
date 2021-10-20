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

#include <cstdint>
#include <stdexcept>
#include <vector>
#include <string>

#include "cuda.h"  // NOLINT - include .h without directory
#include "cuda_runtime.h"  // NOLINT - include .h without directory

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
  double reprojection_matrix[4][4];  // 4x4 reprojection matrix
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
    std::vector<uint8_t> & output,
    const PointCloudProperties & cloud_properties,
    const std::vector<uint8_t> & disparity_buffer,
    const DisparityProperties & disparity_properties,
    const std::vector<uint8_t> & rgb_buffer,
    const RGBProperties & rgb_properties,
    const CameraIntrinsics & intrinsics);

private:
  // The amount to scale the xyz points by
  float unit_scaling_;

  // Whether to use color or not
  bool use_color_;

  // CUDA buffer that will hold the PointCloud2 data
  float * point_cloud_buffer_{};

  // Max capacity of point_cloud_buffer_ in bytes
  size_t point_cloud_buffer_capacity_{0};

  // CUDA buffer that will hold the RGB image
  uint8_t * rgb_buffer_{};

  // Max capacity of rgb_buffer_ in bytes
  size_t rgb_buffer_capacity_{0};

  // CUDA buffer that will hold the disparity image
  uint8_t * disparity_buffer_{};

  // Max capacity of disparity_buffer_ in bytes
  size_t disparity_buffer_capacity_{0};

  // The CUDA stream that all the processing will happen on
  cudaStream_t stream_;

  // The number of CUDA threads per block in the x direction
  const int num_threads_per_block_x_{16};

  // The number of CUDA threads per block in the y direction
  const int num_threads_per_block_y_{16};

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
    const PointCloudProperties & cloud_properties,
    const DisparityProperties & disparity_properties,
    const CameraIntrinsics & intrinsics, const cudaStream_t & stream);

/**
 * @brief Updates the RGB CUDA buffer using the RGB buffer from
 *        the ROS message and the size of the ROS RGB buffer
 *
 * @param rgb_buffer The RGB buffer that will be copied into CUDA
 * @param size The number of bytes from the RGB buffer that should be copied
 * @param stream The stream to perform the operations on
 */
  void UpdateRGBBuffer(
    const std::vector<uint8_t> & rgb_buffer, size_t size,
    const cudaStream_t & stream);

  /**
   * @brief Updates the disparity CUDA buffer using the disparity
   *        buffer from the ROS message and the size of the ROS
   *        disparity buffer
   *
   * @param disparity_buffer The disparity buffer that will be copied into CUDA
   * @param size The number of bytes from the disparity buffer that should be copied
   * @param stream The stream to perform the operations on
   */
  void UpdateDisparityBuffer(
    const std::vector<uint8_t> & disparity_buffer,
    size_t size, const cudaStream_t & stream);

/**
 * @brief Writes points from the point cloud CUDA buffer to the output vector
 *
 * @param[out] output The output vector that will be modified
 * @param[in] size The number of bytes from the point cloud buffer that should be copied
 * @param[in] stream The stream to perform the operations on
 */
  void WritePointsTo(
    std::vector<uint8_t> & output, size_t size,
    const cudaStream_t & stream);

/**
 * @brief Updates the size of the point cloud CUDA buffer
 *
 * @param size The number of bytes that the point cloud CUDA buffer should be
 * @param stream The stream to perform the operations on
 */
  void AllocatePointCloudBuffer(size_t size, const cudaStream_t & stream);

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
    const PointCloudProperties & cloud_properties,
    const RGBProperties & rgb_properties,
    const cudaStream_t & stream);
};
}  // namespace stereo_image_proc
}  // namespace isaac_ros

#endif  // ISAAC_ROS_STEREO_IMAGE_PROC__POINT_CLOUD_NODE_CUDA_HPP_
