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

#include "point_cloud_node_cuda.hpp"

#include <limits>
#include <memory>
#include <string>

#include "cuda.h"  // NOLINT - include .h without directory
#include "cuda_runtime.h"  // NOLINT - include .h without directory
#include "image_geometry/stereo_camera_model.h"
#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/exact_time.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "stereo_msgs/msg/disparity_image.hpp"

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
  explicit PointCloudNode(const rclcpp::NodeOptions & options);

private:
  // Queue size of the subscriber
  int queue_size_;

  // Boolean to decide whether to use color or not
  bool use_color_;

  // Left rectified image subscriber
  message_filters::Subscriber<sensor_msgs::msg::Image> sub_left_image_;

  // Left camera info subscriber
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> sub_left_info_;

  // Right camera info subscriber
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> sub_right_info_;

  // Disparity image subscriber
  message_filters::Subscriber<stereo_msgs::msg::DisparityImage> sub_disparity_;

  // PointCloud2 publisher
  std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> pub_;

  using ExactPolicy = message_filters::sync_policies::ExactTime<
    sensor_msgs::msg::Image,
    sensor_msgs::msg::CameraInfo,
    sensor_msgs::msg::CameraInfo,
    stereo_msgs::msg::DisparityImage>;
  using ExactSync = message_filters::Synchronizer<ExactPolicy>;

  // Exact message sync policy
  std::shared_ptr<ExactSync> exact_sync_;

  // Stereo camera model for getting reprojection matrix
  image_geometry::StereoCameraModel stereo_camera_model_;

  // Performs the computation of the point cloud
  PointCloudNodeCUDA cloud_compute_;

  /**
   * @brief Callback to calculate and publish point cloud to
   *        the relevant topic using left image, left camera info,
   *        right camera info and disparity image.
   *
   * @param left_image_msg The left rectified image received
   * @param left_info_msg  The left image info message received
   * @param right_info_msg  The right image info message received
   * @param disp_msg The disparity image message received
   */
  void PointCloudCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr & left_image_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & left_info_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & right_info_msg,
    const stereo_msgs::msg::DisparityImage::ConstSharedPtr & disp_msg);

  /**
   * @brief Initializes an empty PointCloud2 message using information from
   *        the disparity image and use_color ROS parameter. This method
   *        does not fill the PointCloud2 message with points
   *
   * @param cloud_msg The input PointCloud2 message that will be modified
   * @param disp_msg The disparity image message whose data will be read
   */
  void FormatPointCloudMessage(
    sensor_msgs::msg::PointCloud2::SharedPtr & cloud_msg,
    const stereo_msgs::msg::DisparityImage::ConstSharedPtr & disp_msg);

  /**
   * @brief Creates a point cloud properties struct using information from the cloud message
   *
   * @param cloud_msg The PointCloud2 message whose data will be read
   */
  PointCloudProperties CreatePointCloudProperties(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud_msg);

/**
 * @brief Creates a disparity properties struct using information from the disparity message
 *
 * @param disp_msg The disparity image message whose data will be read
 */
  DisparityProperties CreateDisparityProperties(
    const stereo_msgs::msg::DisparityImage::ConstSharedPtr & disp_msg);

  /**
   * @brief Creates a RGB properties struct using information from the RGB message
   *
   * @param rgb_msg The RGB image message whose data will be read
   */
  RGBProperties CreateRGBProperties(
    const sensor_msgs::msg::Image::ConstSharedPtr & rgb_msg);

  /**
   * @brief Creates a camera properties struct using information from camera info messages
   *
   * @param stereo_camera_model The stereo camera model that will generate the reprojection matrix
   * @param left_info_msg The left camera info message
   * @param right_info_msg The right camera info message
   */
  CameraIntrinsics CreateCameraIntrinsics(
    image_geometry::StereoCameraModel & stereo_camera_model,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & left_info_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & right_info_msg);

  /**
   * @brief Selects the correct disparity format to interpret the disparity data as
   *        and then calls PointCloudNodeCUDA object's compute function
   *        to generate the point cloud
   *
   * @param[out] cloud_msg The cloud message where the point cloud will be written to
   * @param[in] cloud_properties A struct that contains information about the point cloud message
   * @param[in] disp_msg The disparity message whose data buffer will be read
   * @param[in] disparity_properties A struct that contains information about the disparity message
   * @param[in] rgb_msg The RGB message whose data buffer will be read
   * @param[in] rgb_properties A struct that contains information about the RGB message
   * @param[in] intrinsics A struct that contains the reprojection matrix
   */
  void SelectDisparityFormatAndCompute(
    sensor_msgs::msg::PointCloud2::SharedPtr & cloud_msg,
    const PointCloudProperties & cloud_properties,
    const stereo_msgs::msg::DisparityImage::ConstSharedPtr & disp_msg,
    const DisparityProperties & disparity_properties,
    const sensor_msgs::msg::Image::ConstSharedPtr & rgb_msg,
    const RGBProperties & rgb_properties,
    const CameraIntrinsics & intrinsics);
};
}  // namespace stereo_image_proc
}  // namespace isaac_ros
#endif  // ISAAC_ROS_STEREO_IMAGE_PROC__POINT_CLOUD_NODE_HPP_
