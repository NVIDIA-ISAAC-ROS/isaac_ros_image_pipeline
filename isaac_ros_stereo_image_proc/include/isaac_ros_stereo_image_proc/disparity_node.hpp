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

#include <memory>

#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/exact_time.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "stereo_msgs/msg/disparity_image.hpp"

#include "vpi/algo/ConvertImageFormat.h"
#include "vpi/algo/StereoDisparity.h"
#include "vpi/OpenCVInterop.hpp"
#include "vpi/VPI.h"

namespace isaac_ros
{
namespace stereo_image_proc
{

class DisparityNode : public rclcpp::Node
{
public:
  explicit DisparityNode(const rclcpp::NodeOptions &);

  ~DisparityNode();

  DisparityNode(const DisparityNode &) = delete;

  DisparityNode & operator=(const DisparityNode &) = delete;

private:
  /**
   * @brief Callback to calculate disparity between left and right images
   *
   * @param left_image_msg The left rectified image message received
   * @param right_image_msg The right rectified image message received
   * @param left_info_msg The left camera info message received
   * @param right_info_msg The right camera info message received
   */
  void DisparityCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr & left_image_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr & right_image_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & left_info_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & right_info_msg);

  using ExactPolicy = message_filters::sync_policies::ExactTime<
    sensor_msgs::msg::Image,
    sensor_msgs::msg::Image,
    sensor_msgs::msg::CameraInfo,
    sensor_msgs::msg::CameraInfo>;

  using ExactSync = message_filters::Synchronizer<ExactPolicy>;

  // Set of four ROS2 topic subscribers for input (left/right camera streams)
  message_filters::Subscriber<sensor_msgs::msg::Image> sub_left_image_;
  message_filters::Subscriber<sensor_msgs::msg::Image> sub_right_image_;
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> sub_left_info_;
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> sub_right_info_;

  // ROS2 parameter for maximum per-pixel value in output disparity image
  int max_disparity_{};

  // ROS2 parameter for per-pixel scaling factor to apply to disparity image
  double scale_{};

  // ROS2 parameter for maximum size of the subscriber queue
  int queue_size_{};

  // ROS2 parameter for VPI backend flags
  uint32_t vpi_backends_{};

  // Synchronizer to fire callback only once all four subscribers receive messages
  std::shared_ptr<ExactSync> exact_sync_;

  // ROS2 DisparityImage publisher for output
  std::shared_ptr<rclcpp::Publisher<stereo_msgs::msg::DisparityImage>> pub_;


  // Shared VPI stream for submitting all operations
  VPIStream vpi_stream_{};

  // Left and right VPI images in original input dimensions
  VPIImage left_input_{}, right_input_{};

  // Left and right VPI images in stereo algorithm-specific format
  VPIImage left_formatted_{}, right_formatted_{};

  // Left and right VPI images in stereo algorithm-specific format and size
  VPIImage left_stereo_{}, right_stereo_{};

  // Raw disparity, resized disparity, and confidence map in VPI-specific format
  VPIImage disparity_raw_{}, disparity_resized_{}, confidence_map_{};

  // Final disparity output in display-friendly format
  VPIImage disparity_{};

  // VPI algorithm parameters
  VPIConvertImageFormatParams stereo_input_scale_params_{};
  VPIStereoDisparityEstimatorCreationParams disparity_params_{};
  VPIConvertImageFormatParams disparity_scale_params_{};

  // VPI stereo calculation parameters
  VPIPayload stereo_payload_{};
  VPIImageFormat stereo_format_{};

  // Output stereo image dimensions
  int stereo_height_{}, stereo_width_{};

  // Cached values from previous iteration to compare against
  size_t prev_height_{}, prev_width_{};

  // Special configuration values used for Tegra backend
  const int kTegraSupportedStereoWidth{1920};
  const int kTegraSupportedStereoHeight{1080};
  const int kTegraSupportedScale{256};
};

}  // namespace stereo_image_proc
}  // namespace isaac_ros

#endif  // ISAAC_ROS_STEREO_IMAGE_PROC__DISPARITY_NODE_HPP_
