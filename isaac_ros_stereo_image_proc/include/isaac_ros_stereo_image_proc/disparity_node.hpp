/**
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef ISAAC_ROS_STEREO_IMAGE_PROC__DISPARITY_NODE_HPP_
#define ISAAC_ROS_STEREO_IMAGE_PROC__DISPARITY_NODE_HPP_

#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <stereo_msgs/msg/disparity_image.hpp>

#include <image_geometry/stereo_camera_model.h>
#include <image_transport/subscriber_filter.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>

#include <cv_bridge/cv_bridge.h>
#include <vpi/OpenCVInterop.hpp>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/Rescale.h>
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/StereoDisparity.h>

#include <memory>
#include <string>

namespace isaac_ros
{
namespace stereo_image_proc
{

class DisparityNode : public rclcpp::Node
{
public:
  /**
   * @brief Construct a new Disparity Node object
   */
  explicit DisparityNode(const rclcpp::NodeOptions &);

  ~DisparityNode();

private:
  message_filters::Subscriber<sensor_msgs::msg::Image> sub_left_image_;  // Left rectified image
  message_filters::Subscriber<sensor_msgs::msg::Image> sub_right_image_;  // Right rectified image
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> sub_left_info_;  // Left image info
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> sub_right_info_;  // Right image info

  using ExactPolicy = message_filters::sync_policies::ExactTime<
    sensor_msgs::msg::Image,
    sensor_msgs::msg::Image,
    sensor_msgs::msg::CameraInfo,
    sensor_msgs::msg::CameraInfo>;

  using ExactSync = message_filters::Synchronizer<ExactPolicy>;

  std::shared_ptr<ExactSync> exact_sync_;  // The message sync policy

  std::shared_ptr<rclcpp::Publisher<stereo_msgs::msg::DisparityImage>> pub_disparity_;

  /**
   * @brief Callback for the left and right rectified image subscriber
   *
   * @param left_rectified The left rectified image message received
   * @param right_rectified The right rectified image message received
   * @param sub_left_info_ The left image info message received
   * @param sub_right_info_ The right image info message received
   */
  void cam_cb(
    const sensor_msgs::msg::Image::ConstSharedPtr & left_rectified_,
    const sensor_msgs::msg::Image::ConstSharedPtr & right_rectified_,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & sub_left_info_,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & sub_right_info_);

  VPIStream vpi_stream_ = nullptr;  // The VPI stream for VPI tasks
  VPIImage vpi_left_ = nullptr;  // The VPI object for left image
  VPIImage vpi_right_ = nullptr;  // The VPI object for left image
  VPIImage vpi_disparity_ = nullptr;  // The VPI object for disparity image
  VPIImage tmp_left_ = nullptr;  // The VPI object for temporary image conversions
  VPIImage tmp_right_ = nullptr;  // The VPI object for temporary image conversions
  VPIImage confidence_map_ = nullptr;  // The VPI object for confidence map
  VPIImage temp_scale_ = nullptr;  // The VPI object for final scaling
  VPIImage stereo_left_ = nullptr;  // The VPI object for stereo output
  VPIImage stereo_right_ = nullptr;  // The VPI object for stereo output
  uint32_t backends_;  // VPI backends flags
  VPIStereoDisparityEstimatorCreationParams params_;  // Disparity sgm parameters
  VPIConvertImageFormatParams conv_params_;  // Image conversion params
  VPIConvertImageFormatParams scale_params_;  // Image conversion params
  VPIPayload stereo_payload_ = nullptr;  // The VPI object for disparity job payload
  VPIImageFormat stereo_format_;  // The VPI object for format specification

  int max_disparity_ = 0;  // Maximum value for pixel in disparity image
  int prev_height_ = 0;  // Cached height
  int prev_width_ = 0;  // Cached width
  int stereo_width_ = 0;  // Stereo image height
  int stereo_height_ = 0;  // Stereo image width
};

}  // namespace stereo_image_proc
}  // namespace isaac_ros

#endif  // ISAAC_ROS_STEREO_IMAGE_PROC__DISPARITY_NODE_HPP_
