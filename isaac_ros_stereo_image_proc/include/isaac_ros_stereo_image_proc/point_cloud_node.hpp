/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#ifndef ISAAC_ROS_STEREO_IMAGE_PROC__POINT_CLOUD_NODE_HPP_
#define ISAAC_ROS_STEREO_IMAGE_PROC__POINT_CLOUD_NODE_HPP_

#include "nvPointCloud.h"
#include "point_cloud_node_cuda.hpp"

#include <image_transport/subscriber_filter.hpp>
#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rcutils/logging_macros.h>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <stereo_msgs/msg/disparity_image.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <image_geometry/pinhole_camera_model.h>
#include <image_geometry/stereo_camera_model.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <limits>
#include <memory>
#include <string>

namespace isaac_ros
{
namespace stereo_image_proc
{
/**
 * @class PointCloudNode
 * @brief This node combines a disparity image message and a rectified color image and outputs a point cloud
 */

class PointCloudNode : public rclcpp::Node
{
public:
  /**
  * @brief Construct a new Point Cloud Node object
  */
  explicit PointCloudNode(const rclcpp::NodeOptions & options);

  /**
   * @brief Destroy the Point Cloud Node object
   */
  ~PointCloudNode();

private:
  /**
   * @brief Function that assigns ROS parameters to member variables
   *
   * The following parameters are available:
   * @param queue_size Determines the queue size of the subscriber
   * @param unit_scaling Determines the amount to scale the point cloud xyz points by
   * @param use_color Determines whether the output point cloud should have color or not
   */
  void initializeParameters();

  /**
   * @brief Function that sets up the message sync policy and subscribes the node to the relevant subscribers
   *
   */
  void setupSubscribers();

  /**
   * @brief Function that sets up the topic that the node will publish to
   *
   */
  void setupPublishers();

  message_filters::Subscriber<sensor_msgs::msg::Image> sub_left_image_;  // Left rectified image
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> sub_left_info_;  // Left camera info
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> sub_right_info_;  // Right camera info
  message_filters::Subscriber<stereo_msgs::msg::DisparityImage> sub_disparity_;  // Disparity image

  using ExactPolicy = message_filters::sync_policies::ExactTime<
    sensor_msgs::msg::Image,
    sensor_msgs::msg::CameraInfo,
    sensor_msgs::msg::CameraInfo,
    stereo_msgs::msg::DisparityImage>;
  using ExactSync = message_filters::Synchronizer<ExactPolicy>;

  std::shared_ptr<ExactSync> exact_sync_;  // Exact message sync policy

  std::shared_ptr<
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> pub_points2_;  // PointCloud2 publisher

  /**
   * @brief Callback function for the left image, left camera info,
   *        right camera info and disparity image subscriber.
   *        This method will also publish the point cloud to the relevant topic
   *
   * @param left_image_msg The left rectified image received
   * @param left_info_msg  The left image info message received
   * @param right_info_msg  The right image info message received
   * @param disp_msg The disparity image message received
   */
  void image_cb(
    const sensor_msgs::msg::Image::ConstSharedPtr & left_image_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & left_info_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & right_info_msg,
    const stereo_msgs::msg::DisparityImage::ConstSharedPtr & disp_msg);

  /**
   * @brief Initializes an empty PointCloud2 message using information from
   *        the disparity image and use_color parameters
   *
   * @warning Method does not fill the PointCloud2 message with points
   *
   * @param cloud_msg The input PointCloud2 message that will be modified
   * @param disp_msg The disparity image message that determines the PointCloud2 size
   */
  void formatPointCloudMessage(
    sensor_msgs::msg::PointCloud2::SharedPtr & cloud_msg,
    const stereo_msgs::msg::DisparityImage::ConstSharedPtr & disp_msg);

  /**
   * @brief Updates RGB, Disparity and PointCloud2 data buffer sizes for use in GPU
   *
   * @note If the buffers are not initialized, this method will call initializeCuda
   *
   * @param cloud_msg The PointCloud2 message that determines the PointCloud2 data buffer size
   * @param left_image_msg The left rectified image that determines the RGB image buffer size
   * @param disp_msg The disparity image that determines the disparity image buffer size
   */
  void updateCudaBuffers(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr & left_image_msg,
    const stereo_msgs::msg::DisparityImage::ConstSharedPtr & disp_msg);

  /**
   * @brief Initializes the PointCloud2, RGB and disparity data buffers,
   *        and the intrinsics and cloud properties structs for use in GPU.
   *        This function also initializes the CUDA streams
   *
   * @param cloud_size The PointCloud2 data buffer size that will be allocated
   * @param rgb_image_size The RGB image data buffer size that will be allocated
   * @param disparity_image_size The disparity image data buffer size that will allocated
   */
  void initializeCuda(int cloud_size, int rgb_image_size, int disparity_image_size);

  /**
   * @brief Macro that calls the _checkCudaErrors function
   *
   * @param result The result from the CUDA function call
   */
  #define checkCudaErrors(result) {_checkCudaErrors((result), __FILE__, __LINE__); \
}
  /**
   * @brief Checks if a CUDA error occurred. If so, reports the error and terminates the program
   *
   * @param result The result from a CUDA function call
   * @param filename The file that called the CUDA function
   * @param line_number The line number that the error occurred
   */
  void _checkCudaErrors(cudaError_t result, const char * filename, int line_number);

  /**
   * @brief Updates the camera intrinsics struct variables
   *
   * @param disp_msg The disparity image that updates the height,
   *                 width and minimum disparity
   */
  void updateIntrinsics(
    const stereo_msgs::msg::DisparityImage::ConstSharedPtr & disp_msg);

  /**
   * @brief Updates the cloud properties struct variables
   *
   * @param cloud_msg The PointCloud2 message to extract the cloud properties from
   */
  void updateCloudProperties(
    const sensor_msgs::msg::PointCloud2::SharedPtr & cloud_msg);

  /**
   * @brief Converts a disparity image into 3D points
   *        and writes them into a PointCloud2 data buffer
   *
   * @note This method internally copies the disparity image into GPU memory
   * @note Only MONO8, 8UC1, MONO16, 16UC1 and 32FC1 disparity image encodings are supported
   * @note The parameter T is used to decode the disparity image's buffer
   *
   * @param T Template type that represents the type of the disparity image's pixels
   * @param cloud_buffer Empty float array that represents the PointCloud2's data buffer
   * @param disp_msg The disparity image message that will be used to generate 3D points
   */
  template<typename T>
  void convertDisparityToPointCloud(
    float * cloud_buffer,
    const stereo_msgs::msg::DisparityImage::ConstSharedPtr & disp_msg);

  /**
   * @brief Writes RGB from the left rectified image into the PointCloud2 data buffer
   *
   * @note This method internally copies the RGB image into GPU memory
   * @note Only RGB8, BGR8 and MONO8 are supported
   *
   * @param cloud_buffer Float array that represents the Pointcloud2's data buffer
   * @param rgb_image_msg The image that will be used to extract RGB values
   */
  void addColorToPointCloud(
    float * cloud_buffer,
    const sensor_msgs::msg::Image::ConstSharedPtr & rgb_image_msg);

  /**
   * @brief Copies the GPU PointCloud data buffer into a PointCloud2 message
   *
   * @param cloud_msg The input PointCloud2 message with a data buffer
   *                  that will be overwritten
   */
  void copyPointCloudBufferToMessage(
    sensor_msgs::msg::PointCloud2::SharedPtr & cloud_msg);

  nvPointCloudIntrinsics * intrinsics_ = nullptr;  // Relevant camera info struct
  nvPointCloudProperties * cloud_properties_ = nullptr;  // Relevant cloud properties struct

  uint8_t * disparity_image_buffer_ = nullptr;  // GPU disparity image data buffer
  int disparity_image_buffer_size_ = 0;  // GPU disparity image data buffer size
  cudaStream_t stream_;  // CUDA stream

  uint8_t * rgb_image_buffer_ = nullptr;  // GPU RGB image data buffer
  int rgb_image_buffer_size_ = 0;  // GPU RGB image data buffer size

  uint8_t * pointcloud_data_buffer_ = nullptr;  // GPU point cloud data buffer
  int pointcloud_data_buffer_size_ = 0;  // GPU point cloud data buffer size

  image_geometry::StereoCameraModel model_;  // Stereo image geometry
  bool cuda_initialized_ = false;  // Boolean to check if CUDA memory has been initialized
  float unit_scaling_;  // Parameter to scale the x, y and z output of the node
  int queue_size_;  // Queue size of the subscriber
  bool use_color_;  // Boolean to decide whether to use color or not
};

}  // namespace stereo_image_proc
}  // namespace isaac_ros
#endif  // ISAAC_ROS_STEREO_IMAGE_PROC__POINT_CLOUD_NODE_HPP_
