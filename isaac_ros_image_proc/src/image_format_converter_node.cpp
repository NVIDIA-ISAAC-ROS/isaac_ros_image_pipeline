/**
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "isaac_ros_image_proc/image_format_converter_node.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/qos.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "vpi/algo/ConvertImageFormat.h"
#include "vpi/Image.h"
#include "vpi/ImageFormat.h"
#include "vpi/OpenCVInterop.hpp"
#include "vpi/Stream.h"

#include "isaac_ros_common/vpi_utilities.hpp"

namespace
{

// Create OpenCV Supported Conversions
// (https://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython)
const std::unordered_set<std::string> g_cv_supported_types({
    {"mono8", "mono16", "bgr8", "rgb8", "bgra8", "rgba8"}});

// Map the encoding desired string to the VPI Image Format needed
const std::unordered_map<std::string, VPIImageFormat> g_str_to_vpi_format({
    {"mono8", VPI_IMAGE_FORMAT_U8},
    {"mono16", VPI_IMAGE_FORMAT_U16},
    {"bgr8", VPI_IMAGE_FORMAT_BGR8},
    {"rgb8", VPI_IMAGE_FORMAT_RGB8},
    {"bgra8", VPI_IMAGE_FORMAT_BGRA8},
    {"rgba8", VPI_IMAGE_FORMAT_RGBA8}});

// Map the encoding desired string to the number of channels needed
const std::unordered_map<std::string, int> g_str_to_channels({
    {"mono8", CV_8UC1},
    {"mono16", CV_16UC1},
    {"bgr8", CV_8UC3},
    {"rgb8", CV_8UC3},
    {"bgra8", CV_8UC4},
    {"rgba8", CV_8UC4}});

// Perform image format conversion using VPI
cv::Mat GetConvertedMat(
  VPIImage & input, VPIImage & output, VPIStream & stream, const cv::Mat & cv_image,
  const uint32_t backends, const std::string encoding_current, const std::string encoding_desired)
{
  CHECK_STATUS(vpiStreamCreate(backends, &stream));
  CHECK_STATUS(
    vpiImageCreateOpenCVMatWrapper(
      cv_image, g_str_to_vpi_format.at(encoding_current), 0, &input));
  CHECK_STATUS(
    vpiImageCreate(
      cv_image.cols, cv_image.rows, g_str_to_vpi_format.at(encoding_desired), 0, &output));

  // Convert input from current encoding to encoding desired
  CHECK_STATUS(vpiSubmitConvertImageFormat(stream, backends, input, output, nullptr));

  CHECK_STATUS(vpiStreamSync(stream));

  // Retrieve the output image contents into OpenCV matrix and return result
  VPIImageData out_data;
  CHECK_STATUS(vpiImageLock(output, VPI_LOCK_READ, &out_data));
  cv::Mat output_mat{out_data.planes[0].height, out_data.planes[0].width,
    g_str_to_channels.at(encoding_desired), out_data.planes[0].data,
    static_cast<size_t>(out_data.planes[0].pitchBytes)};
  CHECK_STATUS(vpiImageUnlock(output));

  return output_mat;
}
}  // namespace

namespace isaac_ros
{
namespace image_proc
{

ImageFormatConverterNode::ImageFormatConverterNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("ImageFormatConverterNode", options),
  sub_{image_transport::create_subscription(
      this, "image_raw", std::bind(
        &ImageFormatConverterNode::FormatCallback,
        this, std::placeholders::_1), "raw")},
  pub_{image_transport::create_publisher(this, "image")},
  encoding_desired_{static_cast<std::string>(
      this->declare_parameter("encoding_desired", "bgr8"))},
  vpi_backends_{isaac_ros::common::DeclareVPIBackendParameter(this, VPI_BACKEND_CUDA)} {}

void ImageFormatConverterNode::FormatCallback(
  const sensor_msgs::msg::Image::ConstSharedPtr & image_msg)
{
  cv_bridge::CvImagePtr image_ptr;
  try {
    image_ptr = cv_bridge::toCvCopy(image_msg);
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  // Identify source encoding from the image message
  std::string encoding_current{image_ptr->encoding};

  // Skip processing if image is already in desired encoding
  if (encoding_current == encoding_desired_) {
    pub_.publish(*image_msg);
    return;
  }

  const bool cv_supported = g_cv_supported_types.find(encoding_current) !=
    g_cv_supported_types.end() &&
    g_cv_supported_types.find(encoding_desired_) != g_cv_supported_types.end();

  VPIImage input = nullptr;
  VPIImage output = nullptr;
  VPIStream stream = nullptr;

  cv_bridge::CvImage output_image;
  output_image.header = image_ptr->header;
  output_image.encoding = encoding_desired_;

  bool publish = true;
  try {
    // Convert image from source to target encoding using VPI
    output_image.image = GetConvertedMat(
      input, output, stream, image_ptr->image, vpi_backends_, encoding_current,
      encoding_desired_);
  } catch (std::runtime_error & e) {
    // If there is an error converting the image using VPI, use OpenCV instead
    auto & clk = *get_clock();
    RCLCPP_WARN_THROTTLE(get_logger(), clk, 5000, "Exception: %s", e.what());
    if (cv_supported) {
      RCLCPP_INFO_THROTTLE(get_logger(), clk, 5000, "Attempting conversion using OpenCV");
      output_image = *cv_bridge::cvtColor(image_ptr, encoding_desired_);
    } else {
      publish = false;
    }
  }

  if (publish) {
    pub_.publish(output_image.toImageMsg());
  } else {
    auto & clk = *get_clock();
    RCLCPP_ERROR_THROTTLE(
      get_logger(), clk, 5000,
      "Image format conversion from '%s' to '%s' is not supported for both VPI and OpenCV",
      encoding_current, encoding_desired_);
  }
  vpiStreamDestroy(stream);
  vpiImageDestroy(input);
  vpiImageDestroy(output);
}

}  // namespace image_proc
}  // namespace isaac_ros

// Register as a component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(isaac_ros::image_proc::ImageFormatConverterNode)
