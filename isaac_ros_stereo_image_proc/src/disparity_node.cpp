/**
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "isaac_ros_stereo_image_proc/disparity_node.hpp"

#include <memory>
#include <string>
#include <unordered_map>

#include "cv_bridge/cv_bridge.h"

#include "vpi/OpenCVInterop.hpp"
#include "vpi/algo/ConvertImageFormat.h"
#include "vpi/algo/Rescale.h"
#include "vpi/algo/StereoDisparity.h"
#include "vpi/VPI.h"

#include "isaac_ros_common/vpi_utilities.hpp"

namespace isaac_ros
{
namespace stereo_image_proc
{

DisparityNode::DisparityNode(const rclcpp::NodeOptions & options)
: Node{"disparity_node", options},
  sub_left_image_{this, "left/image_rect"},
  sub_right_image_{this, "right/image_rect"},
  sub_left_info_{this, "left/camera_info"},
  sub_right_info_{this, "right/camera_info"},
  max_disparity_{static_cast<int>(declare_parameter("max_disparity", 64))},
  scale_{static_cast<double>(declare_parameter("scale", 1 / 32.0))},
  queue_size_{static_cast<int>(declare_parameter<int>(
      "queue_size", rmw_qos_profile_default.depth))},
  vpi_backends_{isaac_ros::common::DeclareVPIBackendParameter(this, VPI_BACKEND_CUDA)},
  exact_sync_{std::make_shared<ExactSync>(
      ExactPolicy(queue_size_), sub_left_image_, sub_right_image_, sub_left_info_,
      sub_right_info_)},
  pub_{create_publisher<stereo_msgs::msg::DisparityImage>("disparity", queue_size_)}
{
  if (max_disparity_ <= 0) {
    throw std::runtime_error("Max disparity must be strictly positive");
  }

  exact_sync_->registerCallback(&DisparityNode::DisparityCallback, this);

  CHECK_STATUS(vpiInitStereoDisparityEstimatorCreationParams(&disparity_params_));
  disparity_params_.maxDisparity = max_disparity_;

  CHECK_STATUS(vpiInitConvertImageFormatParams(&stereo_input_scale_params_));
  if (vpi_backends_ == VPI_BACKEND_TEGRA) {
    stereo_input_scale_params_.scale = kTegraSupportedScale;
  }

  CHECK_STATUS(vpiInitConvertImageFormatParams(&disparity_scale_params_));
  // Scale the per-pixel disparity output
  disparity_scale_params_.scale = scale_;

  // Initialize VPI stream
  CHECK_STATUS(vpiStreamCreate(0, &vpi_stream_));

  // TODO(jaiveers): Use a universal utility to log the backend being used
  // Print out backend used
  std::string backend_string = "CUDA";
  if (vpi_backends_ == VPI_BACKEND_TEGRA) {
    backend_string = "PVA-NVENC-VIC";
  }
  RCLCPP_INFO(this->get_logger(), "Using backend %s", backend_string.c_str());
}

DisparityNode::~DisparityNode()
{
  // Wait for stream to complete before destroying
  if (!vpi_stream_) {
    vpiStreamSync(vpi_stream_);
  }
  vpiStreamDestroy(vpi_stream_);

  vpiImageDestroy(left_input_);
  vpiImageDestroy(right_input_);
  vpiImageDestroy(left_formatted_);
  vpiImageDestroy(right_formatted_);
  vpiImageDestroy(left_stereo_);
  vpiImageDestroy(right_stereo_);
  vpiPayloadDestroy(stereo_payload_);
  vpiImageDestroy(disparity_raw_);
  vpiImageDestroy(confidence_map_);
  vpiImageDestroy(disparity_resized_);
  vpiImageDestroy(disparity_);
}

void DisparityNode::DisparityCallback(
  const sensor_msgs::msg::Image::ConstSharedPtr & left_image_msg,
  const sensor_msgs::msg::Image::ConstSharedPtr & right_image_msg,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & left_info_msg,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & right_info_msg)
{
  // Extract images from ROS2 messages as grayscale
  const cv::Mat left_input_cv = cv_bridge::toCvShare(left_image_msg, "mono8")->image;
  const cv::Mat right_input_cv = cv_bridge::toCvShare(right_image_msg, "mono8")->image;

  // Check current dimensions against previous dimensions to determine if cache is valid
  const size_t height{left_info_msg->height}, width{left_info_msg->width};
  const bool cache_valid = (height == prev_height_) && (width == prev_width_);

  if (cache_valid) {
    // The image dimensions are the same, so rewrap using existing VPI images
    vpiImageSetWrappedOpenCVMat(left_input_, left_input_cv);
    vpiImageSetWrappedOpenCVMat(right_input_, right_input_cv);
  } else {
    // The image dimensions have changed, so we need to recreate VPI images with the new size

    // Recreate left and right input VPI images
    vpiImageDestroy(left_input_);
    vpiImageDestroy(right_input_);
    CHECK_STATUS(vpiImageCreateOpenCVMatWrapper(left_input_cv, 0, &left_input_));
    CHECK_STATUS(vpiImageCreateOpenCVMatWrapper(right_input_cv, 0, &right_input_));

    if (vpi_backends_ == VPI_BACKEND_TEGRA) {
      // PVA-NVENC-VIC backend only accepts 1920x1080 images and Y16 Block linear format.
      stereo_format_ = VPI_IMAGE_FORMAT_Y16_ER_BL;
      stereo_width_ = kTegraSupportedStereoWidth;
      stereo_height_ = kTegraSupportedStereoHeight;
    } else {
      stereo_format_ = VPI_IMAGE_FORMAT_Y16_ER;
      stereo_width_ = width;
      stereo_height_ = height;
    }

    // Recreate temporaries for changing format from input to stereo format
    vpiImageDestroy(left_formatted_);
    vpiImageDestroy(right_formatted_);
    CHECK_STATUS(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_Y16_ER, 0, &left_formatted_));
    CHECK_STATUS(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_Y16_ER, 0, &right_formatted_));

    if (vpi_backends_ == VPI_BACKEND_TEGRA) {
      // Recreate left and right Tegra-specific resized VPI images
      vpiImageDestroy(left_stereo_);
      vpiImageDestroy(right_stereo_);
      CHECK_STATUS(
        vpiImageCreate(stereo_width_, stereo_height_, stereo_format_, 0, &left_stereo_));
      CHECK_STATUS(
        vpiImageCreate(stereo_width_, stereo_height_, stereo_format_, 0, &right_stereo_));

      // Recreate disparity VPI image with original input dimensions
      vpiImageDestroy(disparity_resized_);
      CHECK_STATUS(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_U16, 0, &disparity_resized_));

      // Recreate confidence map, which is used only on Tegra backend
      vpiImageDestroy(confidence_map_);
      CHECK_STATUS(
        vpiImageCreate(
          stereo_width_, stereo_height_, VPI_IMAGE_FORMAT_U16, 0,
          &confidence_map_));
    } else {
      left_stereo_ = nullptr;
      right_stereo_ = nullptr;
      confidence_map_ = nullptr;
    }

    // Recreate stereo payload with parameters for stereo disparity algorithm
    vpiPayloadDestroy(stereo_payload_);
    CHECK_STATUS(
      vpiCreateStereoDisparityEstimator(
        vpi_backends_, stereo_width_, stereo_height_,
        stereo_format_, &disparity_params_, &stereo_payload_));

    // Recreate raw disparity VPI image
    vpiImageDestroy(disparity_raw_);
    CHECK_STATUS(
      vpiImageCreate(
        stereo_width_, stereo_height_,
        VPI_IMAGE_FORMAT_U16, 0, &disparity_raw_));

    // Recreate final output disparity VPI image
    vpiImageDestroy(disparity_);
    CHECK_STATUS(
      vpiImageCreate(
        width, height,
        VPI_IMAGE_FORMAT_F32, 0, &disparity_));

    // Update cached dimensions
    prev_height_ = height;
    prev_width_ = width;
  }

  // Convert input-format images to stereo-format images
  CHECK_STATUS(
    vpiSubmitConvertImageFormat(
      vpi_stream_, VPI_BACKEND_CUDA,
      left_input_, left_formatted_, &stereo_input_scale_params_));
  CHECK_STATUS(
    vpiSubmitConvertImageFormat(
      vpi_stream_, VPI_BACKEND_CUDA,
      right_input_, right_formatted_, &stereo_input_scale_params_));

  if (vpi_backends_ == VPI_BACKEND_TEGRA) {
    // Submit resize operation from input dimensions to Tegra-specific stereo dimensions
    CHECK_STATUS(
      vpiSubmitRescale(
        vpi_stream_, VPI_BACKEND_VIC, left_formatted_, left_stereo_,
        VPI_INTERP_LINEAR, VPI_BORDER_CLAMP, 0));
    CHECK_STATUS(
      vpiSubmitRescale(
        vpi_stream_, VPI_BACKEND_VIC, right_formatted_, right_stereo_,
        VPI_INTERP_LINEAR, VPI_BORDER_CLAMP, 0));
  } else {
    // Since stereo dimensions match input dimensions on non-Tegra backend, no resize necessary
    left_stereo_ = left_formatted_;
    right_stereo_ = right_formatted_;
  }

  // Calculate raw disparity and confidence map
  CHECK_STATUS(
    vpiSubmitStereoDisparityEstimator(
      vpi_stream_, vpi_backends_,
      stereo_payload_, left_stereo_, right_stereo_,
      disparity_raw_, confidence_map_, nullptr));

  if (vpi_backends_ == VPI_BACKEND_TEGRA) {
    // Submit resize operation from Tegra-specific stereo dimensions back to input dimensions
    CHECK_STATUS(
      vpiSubmitRescale(
        vpi_stream_, VPI_BACKEND_CUDA, disparity_raw_, disparity_resized_,
        VPI_INTERP_LINEAR, VPI_BORDER_CLAMP, 0));
  } else {
    // Since stereo dimensions match input dimensions on non-Tegra backend, no resize necessary
    disparity_resized_ = disparity_raw_;
  }

  // Convert to ROS2 standard 32-bit float format
  CHECK_STATUS(
    vpiSubmitConvertImageFormat(
      vpi_stream_, VPI_BACKEND_CUDA,
      disparity_resized_, disparity_, &disparity_scale_params_));

  // Wait for operations to complete
  CHECK_STATUS(vpiStreamSync(vpi_stream_));

  // Lock output disparity VPI image to extract data
  VPIImageData data;
  CHECK_STATUS(vpiImageLock(disparity_, VPI_LOCK_READ, &data));
  try {
    cv::Mat cvOut;
    CHECK_STATUS(vpiImageDataExportOpenCVMat(data, &cvOut));

    stereo_msgs::msg::DisparityImage disparity_image;
    disparity_image.header = left_info_msg->header;
    disparity_image.image =
      *cv_bridge::CvImage(left_info_msg->header, "32FC1", cvOut).toImageMsg();
    disparity_image.f = right_info_msg->p[0];   // Focal length in pixels
    disparity_image.t = -right_info_msg->p[3] / right_info_msg->p[0];   // Baseline in world units
    disparity_image.min_disparity = 0;
    disparity_image.max_disparity = disparity_params_.maxDisparity;
    pub_->publish(disparity_image);
  } catch (...) {
    // If any exception occurs, we must release the VPI image lock
    vpiImageUnlock(disparity_);
    RCLCPP_ERROR(this->get_logger(), "Exception occurred with locked image");
  }
  CHECK_STATUS(vpiImageUnlock(disparity_));
}

}  // namespace stereo_image_proc
}  // namespace isaac_ros

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(isaac_ros::stereo_image_proc::DisparityNode)
