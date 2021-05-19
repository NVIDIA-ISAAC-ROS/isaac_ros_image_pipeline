/**
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "isaac_ros_image_proc/resize_node.hpp"

#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.hpp"
#include "opencv2/opencv.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "vpi/algo/ConvertImageFormat.h"
#include "vpi/algo/Rescale.h"
#include "vpi/OpenCVInterop.hpp"
#include "vpi/VPI.h"

#include "isaac_ros_common/vpi_utilities.hpp"

namespace isaac_ros
{
namespace image_proc
{

ResizeNode::ResizeNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("ResizeNode", options),
  sub_{image_transport::create_camera_subscription(
      this, "image", std::bind(
        &ResizeNode::ResizeCallback,
        this, std::placeholders::_1, std::placeholders::_2), "raw")},
  pub_{image_transport::create_camera_publisher(this, "resized/image")},
  use_relative_scale_{declare_parameter("use_relative_scale", true)},
  scale_height_{declare_parameter("scale_height", 1.0)},
  scale_width_{declare_parameter("scale_width", 1.0)},
  height_{static_cast<int>(declare_parameter("height", -1))},
  width_{static_cast<int>(declare_parameter("width", -1))},
  vpi_backends_{isaac_ros::common::DeclareVPIBackendParameter(this, VPI_BACKEND_CUDA)} {}

void ResizeNode::ResizeCallback(
  const sensor_msgs::msg::Image::ConstSharedPtr & image_msg,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & info_msg)
{
  cv_bridge::CvImagePtr image_ptr;
  try {
    image_ptr = cv_bridge::toCvCopy(image_msg);
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  // Set the output image's header and encoding
  cv_bridge::CvImage output_image;
  output_image.header = image_ptr->header;
  output_image.encoding = image_ptr->encoding;

  VPIImage input{};
  CHECK_STATUS(vpiImageCreateOpenCVMatWrapper(image_ptr->image, 0, &input));

  VPIImageFormat type{VPI_IMAGE_FORMAT_U8};
  CHECK_STATUS(vpiImageGetFormat(input, &type));

  // Initialize VPI stream for all VPI operations
  VPIStream stream{};
  CHECK_STATUS(vpiStreamCreate(vpi_backends_, &stream));

  // Prepare CameraInfo output with desired size
  // The original dimensions will either be scaled or replaced entirely
  double scale_y, scale_x;
  sensor_msgs::msg::CameraInfo output_info_msg{*info_msg};
  if (use_relative_scale_) {
    if (scale_height_ <= 0 || scale_width_ <= 0) {
      RCLCPP_ERROR(get_logger(), "scale_height and scale_width must be greater than 0");
      return;
    }

    scale_y = scale_height_;
    scale_x = scale_width_;
    output_info_msg.height = static_cast<int>(info_msg->height * scale_height_);
    output_info_msg.width = static_cast<int>(info_msg->width * scale_width_);
  } else {
    if (height_ <= 0 || width_ <= 0) {
      RCLCPP_ERROR(get_logger(), "height and width must be greater than 0");
      return;
    }

    scale_y = static_cast<double>(height_) / info_msg->height;
    scale_x = static_cast<double>(width_) / info_msg->width;
    output_info_msg.height = height_;
    output_info_msg.width = width_;
  }

  // Rescale the relevant entries of the intrinsic and extrinsic matrices
  output_info_msg.k[0] = output_info_msg.k[0] * scale_x;  // fx
  output_info_msg.k[2] = output_info_msg.k[2] * scale_x;  // cx
  output_info_msg.k[4] = output_info_msg.k[4] * scale_y;  // fy
  output_info_msg.k[5] = output_info_msg.k[5] * scale_y;  // cy

  output_info_msg.p[0] = output_info_msg.p[0] * scale_x;  // fx
  output_info_msg.p[2] = output_info_msg.p[2] * scale_x;  // cx
  output_info_msg.p[3] = output_info_msg.p[3] * scale_x;  // T
  output_info_msg.p[5] = output_info_msg.p[5] * scale_y;  // fy
  output_info_msg.p[6] = output_info_msg.p[6] * scale_y;  // cy

  // Prepare intermediate image with input dimensions
  VPIImage tmp_in{};
  CHECK_STATUS(
    vpiImageCreate(
      info_msg->width, info_msg->height, VPI_IMAGE_FORMAT_NV12_ER, 0,
      &tmp_in));

  // Prepare intermediate and output images with output dimensions
  VPIImage tmp_out{}, output{};
  CHECK_STATUS(
    vpiImageCreate(
      output_info_msg.width, output_info_msg.height, VPI_IMAGE_FORMAT_NV12_ER, 0,
      &tmp_out));
  CHECK_STATUS(
    vpiImageCreate(
      output_info_msg.width, output_info_msg.height, type, 0,
      &output));

  // Convert input to NV12 format
  CHECK_STATUS(vpiSubmitConvertImageFormat(stream, vpi_backends_, input, tmp_in, nullptr));

  // Rescale while still in NV12 format
  CHECK_STATUS(
    vpiSubmitRescale(
      stream, vpi_backends_, tmp_in, tmp_out, VPI_INTERP_LINEAR, VPI_BORDER_ZERO,
      0));

  // Convert output to original format
  CHECK_STATUS(vpiSubmitConvertImageFormat(stream, vpi_backends_, tmp_out, output, nullptr));

  // Wait until all operations are complete
  CHECK_STATUS(vpiStreamSync(stream));

  // Transfer VPI output to image output
  VPIImageData outData{};
  CHECK_STATUS(vpiImageLock(output, VPI_LOCK_READ, &outData));
  output_image.image =
    cv::Mat{outData.planes[0].height, outData.planes[0].width, CV_8UC3, outData.planes[0].data,
    static_cast<size_t>(outData.planes[0].pitchBytes)};
  CHECK_STATUS(vpiImageUnlock(output));

  pub_.publish(*output_image.toImageMsg(), output_info_msg);

  vpiStreamDestroy(stream);
  vpiImageDestroy(input);
  vpiImageDestroy(output);
  vpiImageDestroy(tmp_in);
  vpiImageDestroy(tmp_out);
}

}  // namespace image_proc
}  // namespace isaac_ros

// Register as a component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(isaac_ros::image_proc::ResizeNode)
