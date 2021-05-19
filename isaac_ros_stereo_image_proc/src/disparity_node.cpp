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

#include <vpi/OpenCVInterop.hpp>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/Rescale.h>
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/StereoDisparity.h>

#include <string>
#include <unordered_map>

#include "isaac_ros_common/vpi_utilities.hpp"

// VPI status check macro
#define CHECK_STATUS(STMT) \
  do { \
    VPIStatus status = (STMT); \
    if (status != VPI_SUCCESS) { \
      char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH]; \
      vpiGetLastStatusMessage(buffer, sizeof(buffer)); \
      std::ostringstream ss; \
      ss << vpiStatusGetName(status) << ": " << buffer; \
      throw std::runtime_error(ss.str()); \
    } \
  } while (0);

namespace isaac_ros
{
namespace stereo_image_proc
{

DisparityNode::DisparityNode(const rclcpp::NodeOptions & options)
: Node("disp_node", options),
  backends_{isaac_ros::common::DeclareVPIBackendParameter(this, VPI_BACKEND_CUDA)}
{
  using namespace std::placeholders;

  // Initialize message time sync filter
  exact_sync_.reset(
    new ExactSync(
      ExactPolicy(5),
      sub_left_image_, sub_right_image_, sub_left_info_, sub_right_info_));
  exact_sync_->registerCallback(&DisparityNode::cam_cb, this);

  // Publisher
  pub_disparity_ = create_publisher<stereo_msgs::msg::DisparityImage>("disparity", 1);

  // Subscriber
  sub_left_image_.subscribe(this, "left/image_rect");
  sub_right_image_.subscribe(this, "right/image_rect");
  sub_left_info_.subscribe(this, "left/camera_info");
  sub_right_info_.subscribe(this, "right/camera_info");

  // Get parameters
  max_disparity_ = declare_parameter<int>("max_disparity", 64);
  if (max_disparity_ < 1) {
    RCLCPP_ERROR(this->get_logger(), "Max disparity cannot be 0");
    return;
  }

  // Initialize parameters
  CHECK_STATUS(vpiInitConvertImageFormatParams(&conv_params_));
  CHECK_STATUS(vpiInitConvertImageFormatParams(&scale_params_));
  CHECK_STATUS(vpiInitStereoDisparityEstimatorCreationParams(&params_));
  params_.maxDisparity = max_disparity_;
  scale_params_.scale = 255.0 / (32 * params_.maxDisparity);

  // Print out backend used
  std::string backend_string = "CUDA";
  if (backends_ == VPI_BACKEND_TEGRA) {
    backend_string = "PVA-NVENC-VIC";
  }
  RCLCPP_INFO(this->get_logger(), "Using backend %s", backend_string.c_str());

  // Initialize VPI Stream
  vpiStreamCreate(0, &vpi_stream_);
}

DisparityNode::~DisparityNode()
{
  // Close VPI Stream when if still open
  if (!vpi_stream_) {
    vpiStreamSync(vpi_stream_);
  }
  if (backends_ == VPI_BACKEND_TEGRA) {
    vpiImageDestroy(confidence_map_);
    vpiImageDestroy(tmp_left_);
    vpiImageDestroy(tmp_right_);
  }
  vpiPayloadDestroy(stereo_payload_);
  vpiImageDestroy(temp_scale_);
  vpiImageDestroy(stereo_left_);
  vpiImageDestroy(stereo_right_);
  vpiImageDestroy(vpi_disparity_);
  vpiStreamDestroy(vpi_stream_);
  vpiImageDestroy(vpi_left_);
  vpiImageDestroy(vpi_right_);
}

void DisparityNode::cam_cb(
  const sensor_msgs::msg::Image::ConstSharedPtr & left_rectified_,
  const sensor_msgs::msg::Image::ConstSharedPtr & right_rectified_,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & sub_left_info_,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & sub_right_info_)
{
  // Convert images to mono8
  const cv::Mat left_mono8 = cv_bridge::toCvShare(left_rectified_, "mono8")->image;
  const cv::Mat right_mono8 = cv_bridge::toCvShare(right_rectified_, "mono8")->image;
  int inputWidth, inputHeight;
  inputWidth = left_mono8.cols;
  inputHeight = left_mono8.rows;

  const bool resolution_change =
    prev_height_ != inputHeight || prev_width_ != inputWidth;
  const bool resolution_cache_check = prev_height_ != 0 || prev_width_ != 0;
  if ((!vpi_left_ || !vpi_right_ ) || resolution_change) {
    if (resolution_change && resolution_cache_check) {
      vpiImageDestroy(vpi_left_);
      vpiImageDestroy(vpi_right_);
    }
    vpiImageCreateOpenCVMatWrapper(left_mono8, 0, &vpi_left_);
    vpiImageCreateOpenCVMatWrapper(right_mono8, 0, &vpi_right_);
  } else {
    vpiImageSetWrappedOpenCVMat(vpi_left_, left_mono8);
    vpiImageSetWrappedOpenCVMat(vpi_right_, right_mono8);
  }

  if (!stereo_payload_ || resolution_change) {
    if (resolution_change && resolution_cache_check) {
      vpiPayloadDestroy(stereo_payload_);
      vpiImageDestroy(stereo_left_);
      vpiImageDestroy(stereo_right_);
      vpiImageDestroy(vpi_disparity_);
    }

    if (backends_ == VPI_BACKEND_TEGRA) {
      if (resolution_change && resolution_cache_check) {
        vpiImageDestroy(confidence_map_);
        vpiImageDestroy(tmp_left_);
        vpiImageDestroy(tmp_right_);
      }

      // PVA-NVENC-VIC backend only accepts 1920x1080 images and Y16 Block linear format.
      stereo_format_ = VPI_IMAGE_FORMAT_Y16_ER_BL;
      stereo_width_ = 1920;
      stereo_height_ = 1080;
      conv_params_.scale = 256;
      CHECK_STATUS(
        vpiImageCreate(
          stereo_width_, stereo_height_, VPI_IMAGE_FORMAT_U16, 0,
          &confidence_map_));
      CHECK_STATUS(vpiImageCreate(inputWidth, inputHeight, VPI_IMAGE_FORMAT_Y16_ER, 0, &tmp_left_));
      CHECK_STATUS(
        vpiImageCreate(
          inputWidth, inputHeight, VPI_IMAGE_FORMAT_Y16_ER, 0,
          &tmp_right_));
    } else {
      stereo_format_ = VPI_IMAGE_FORMAT_Y16_ER;
      stereo_width_ = inputWidth;
      stereo_height_ = inputHeight;
      confidence_map_ = nullptr;
    }
    prev_height_ = inputHeight;
    prev_width_ = inputWidth;
    CHECK_STATUS(
      vpiCreateStereoDisparityEstimator(
        backends_, stereo_width_, stereo_height_,
        stereo_format_, &params_, &stereo_payload_));
    CHECK_STATUS(
      vpiImageCreate(
        stereo_width_, stereo_height_, VPI_IMAGE_FORMAT_U16,
        0, &vpi_disparity_));
    CHECK_STATUS(vpiImageCreate(stereo_width_, stereo_height_, stereo_format_, 0, &stereo_left_));
    CHECK_STATUS(vpiImageCreate(stereo_width_, stereo_height_, stereo_format_, 0, &stereo_right_));
  }

  if (backends_ == VPI_BACKEND_TEGRA) {
    CHECK_STATUS(
      vpiSubmitConvertImageFormat(
        vpi_stream_, VPI_BACKEND_CUDA,
        vpi_left_, tmp_left_, &conv_params_));
    CHECK_STATUS(
      vpiSubmitConvertImageFormat(
        vpi_stream_, VPI_BACKEND_CUDA,
        vpi_right_, tmp_right_, &conv_params_));
    CHECK_STATUS(
      vpiSubmitRescale(
        vpi_stream_, VPI_BACKEND_VIC, tmp_left_, stereo_left_,
        VPI_INTERP_LINEAR, VPI_BORDER_CLAMP, 0));
    CHECK_STATUS(
      vpiSubmitRescale(
        vpi_stream_, VPI_BACKEND_VIC, tmp_right_, stereo_right_,
        VPI_INTERP_LINEAR, VPI_BORDER_CLAMP, 0));
  } else {
    CHECK_STATUS(
      vpiSubmitConvertImageFormat(
        vpi_stream_, VPI_BACKEND_CUDA,
        vpi_left_, stereo_left_, &conv_params_));
    CHECK_STATUS(
      vpiSubmitConvertImageFormat(
        vpi_stream_, VPI_BACKEND_CUDA,
        vpi_right_, stereo_right_, &conv_params_));
  }
  CHECK_STATUS(
    vpiSubmitStereoDisparityEstimator(
      vpi_stream_, backends_,
      stereo_payload_, stereo_left_, stereo_right_,
      vpi_disparity_, confidence_map_, nullptr));

  cv::Mat cvOut;
  if (!temp_scale_) {
    CHECK_STATUS(
      vpiImageCreate(
        stereo_width_, stereo_height_, VPI_IMAGE_FORMAT_U16, 0,
        &temp_scale_));
  }
  CHECK_STATUS(
    vpiSubmitConvertImageFormat(
      vpi_stream_, VPI_BACKEND_CUDA,
      vpi_disparity_, temp_scale_, &scale_params_));
  VPIImageData data;

  CHECK_STATUS(vpiStreamSync(vpi_stream_));
  CHECK_STATUS(vpiImageLock(temp_scale_, VPI_LOCK_READ, &data));

  try {
    CHECK_STATUS(vpiImageDataExportOpenCVMat(data, &cvOut));

    stereo_msgs::msg::DisparityImage disparity_image;
    disparity_image.header = sub_left_info_->header;
    disparity_image.image =
      *cv_bridge::CvImage(sub_left_info_->header, "mono16", cvOut).toImageMsg();
    disparity_image.f = sub_right_info_->p[0];
    disparity_image.t = sub_right_info_->p[3];
    disparity_image.min_disparity = 0;
    disparity_image.max_disparity = params_.maxDisparity;
    pub_disparity_->publish(disparity_image);
  } catch (...) {
    vpiImageUnlock(temp_scale_);
    RCLCPP_ERROR(this->get_logger(), "Exception occurred with locked image");
  }
  vpiImageUnlock(temp_scale_);
}

}  // namespace stereo_image_proc
}  // namespace isaac_ros

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(isaac_ros::stereo_image_proc::DisparityNode)
