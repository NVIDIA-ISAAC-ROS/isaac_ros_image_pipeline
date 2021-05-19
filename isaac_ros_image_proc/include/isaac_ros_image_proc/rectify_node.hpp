/**
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef ISAAC_ROS_IMAGE_PROC__RECTIFY_NODE_HPP_
#define ISAAC_ROS_IMAGE_PROC__RECTIFY_NODE_HPP_

#include <string>
#include <utility>

#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.hpp"
#include "opencv2/opencv.hpp"
#include "rclcpp/rclcpp.hpp"
#include "vpi/VPI.h"

namespace isaac_ros
{
namespace image_proc
{

class RectifyNode : public rclcpp::Node
{
public:
  explicit RectifyNode(const rclcpp::NodeOptions &);

  ~RectifyNode();

  RectifyNode(const RectifyNode &) = delete;

  RectifyNode & operator=(const RectifyNode &) = delete;

private:
  /**
   * @brief Callback to rectify the image received by subscription
   *
   * @param image_msg The image message received
   * @param info_msg The information of the camera that produced the image
   */
  void RectifyCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr & image_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & info_msg);

  /**
   * @brief Rectify the input image in-place according to the distortion parameters
   *
   * @param image_ptr [in/out] Pointer to the image to be rectified in-place
   * @param K_mat The 3x3 K matrix of camera intrinsics
   * @param P_mat The 3x4 P matrix mapping from world points to image points
   * @param D_mat The distortion parameters
   * @param distortion_model The distortion model according to which to interpret the distortion parameters
   * @param roi The region of interest to crop the final rectified image to
   */
  void RectifyImage(
    cv_bridge::CvImagePtr & image_ptr, const cv::Matx33d & K_mat, const cv::Matx34d & P_mat,
    const cv::Mat_<double> & D_mat, const std::string & distortion_model, cv::Rect roi);

  // ROS2 Camera subscriber for input and Image publisher for output
  image_transport::CameraSubscriber sub_;
  image_transport::Publisher pub_;

  // ROS2 parameter for specifying interpolation type
  VPIInterpolationType interpolation_;

  // ROS2 parameter for VPI backend flags
  uint32_t vpi_backends_{};

  // Shared VPI resources for all callbacks
  VPIStream stream_{};
  VPIImage image_{};
  VPIPayload remap_{};

  // Temporary input and output images in NV12 format
  VPIImage tmp_in_{}, tmp_out_{};

  // Cached VPI camera intrinsics and extrinsics
  VPICameraIntrinsic K_{};
  VPICameraExtrinsic X_{};

  // Cached values from CameraInfo message to avoid unnecessary computation
  std::pair<int, int> image_dims_{-1, -1};
  cv::Matx33d curr_K_mat_{};
  cv::Matx34d curr_P_mat_{};
  cv::Mat_<double> curr_D_mat_{};
  cv::Rect current_roi_{};
};

}  // namespace image_proc
}  // namespace isaac_ros

#endif  // ISAAC_ROS_IMAGE_PROC__RECTIFY_NODE_HPP_
