/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "isaac_ros_stereo_image_proc/point_cloud_node.hpp"

#include <memory>
#include <limits>
#include <string>
#include <unordered_map>

#include "cuda.h"  // NOLINT - include .h without directory
#include "cuda_runtime.h"  // NOLINT - include .h without directory

namespace
{
const std::unordered_map<std::string, unsigned int> disparity_byte_conversion(
  {
    {sensor_msgs::image_encodings::MONO8, sizeof(uint8_t)},
    {sensor_msgs::image_encodings::TYPE_8UC1, sizeof(uint8_t)},
    {sensor_msgs::image_encodings::MONO16, sizeof(uint16_t)},
    {sensor_msgs::image_encodings::TYPE_16UC1, sizeof(uint16_t)},
    {sensor_msgs::image_encodings::TYPE_32FC1, sizeof(float)}
  });

struct InvalidImageFormatError
{
  std::string error_msg_;
  InvalidImageFormatError(std::string error_msg)
  : error_msg_{error_msg} {}
  const char * what() {return error_msg_.c_str();}
};

}  // namespace

namespace isaac_ros
{
namespace stereo_image_proc
{
PointCloudNode::PointCloudNode(const rclcpp::NodeOptions & options)
: Node("point_cloud_node", options),
  queue_size_(declare_parameter<int>("queue_size", rmw_qos_profile_default.depth)),
  use_color_(declare_parameter<bool>("use_color", false)),
  sub_left_image_(message_filters::Subscriber<sensor_msgs::msg::Image>(
      this, "left/image_rect_color")),
  sub_left_info_(message_filters::Subscriber<sensor_msgs::msg::CameraInfo>(
      this, "left/camera_info")),
  sub_right_info_(message_filters::Subscriber<sensor_msgs::msg::CameraInfo>(
      this, "right/camera_info")),
  sub_disparity_(message_filters::Subscriber<stereo_msgs::msg::DisparityImage>(
      this, "disparity")),
  pub_(create_publisher<sensor_msgs::msg::PointCloud2>("points2", 1))
{
  // Ensure hardware has 32 bit floating point
  if (!std::numeric_limits<float>::is_iec559) {
    throw std::runtime_error(
            "Hardware does not support 32-bit IEEE754 floating point standard");
  }

  // Set the unit scaling
  cloud_compute_.SetUnitScaling(declare_parameter<float>("unit_scaling", 1.0f));

  // Set the sync policy
  exact_sync_.reset(
    new ExactSync(
      ExactPolicy(queue_size_),
      sub_left_image_, sub_left_info_,
      sub_right_info_, sub_disparity_));

  using namespace std::placeholders;
  exact_sync_->registerCallback(
    std::bind(&PointCloudNode::PointCloudCallback, this, _1, _2, _3, _4));
}

void PointCloudNode::PointCloudCallback(
  const sensor_msgs::msg::Image::ConstSharedPtr & left_image_msg,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & left_info_msg,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & right_info_msg,
  const stereo_msgs::msg::DisparityImage::ConstSharedPtr & disp_msg)
{
  if (left_image_msg->height != disp_msg->image.height ||
    left_image_msg->width != disp_msg->image.width)
  {
    RCLCPP_ERROR(get_logger(), "Error: RGB & disparity image dimensions do not match!");
    return;
  }

  auto cloud_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
  FormatPointCloudMessage(cloud_msg, disp_msg);

  PointCloudProperties cloud_properties;
  cloud_properties = CreatePointCloudProperties(cloud_msg);

  CameraIntrinsics intrinsics;
  intrinsics = CreateCameraIntrinsics(
    stereo_camera_model_,
    left_info_msg,
    right_info_msg);

  DisparityProperties disparity_properties;
  try {
    disparity_properties = CreateDisparityProperties(disp_msg);
  } catch (InvalidImageFormatError & e) {
    RCLCPP_ERROR(get_logger(), e.what());
    return;
  }

  RGBProperties rgb_properties;
  if (use_color_) {
    try {
      rgb_properties = CreateRGBProperties(left_image_msg);
      cloud_compute_.SetUseColor(true);
    } catch (InvalidImageFormatError & e) {
      RCLCPP_ERROR(get_logger(), e.what());
      cloud_compute_.SetUseColor(false);
    }
  }

  SelectDisparityFormatAndCompute(
    cloud_msg, cloud_properties, disp_msg, disparity_properties,
    left_image_msg, rgb_properties, intrinsics);

  pub_->publish(*cloud_msg);
}

void PointCloudNode::FormatPointCloudMessage(
  sensor_msgs::msg::PointCloud2::SharedPtr & cloud_msg,
  const stereo_msgs::msg::DisparityImage::ConstSharedPtr & disp_msg)
{
  cloud_msg->header = disp_msg->image.header;
  cloud_msg->height = disp_msg->image.height;
  cloud_msg->width = disp_msg->image.width;
  cloud_msg->is_bigendian = false;
  cloud_msg->is_dense = false;

  sensor_msgs::PointCloud2Modifier pc2_modifier(*cloud_msg);

  if (use_color_) {
    // Data format: x,y,z,rgb; 16 bytes per point
    pc2_modifier.setPointCloud2Fields(
      4,
      "x", 1, sensor_msgs::msg::PointField::FLOAT32,
      "y", 1, sensor_msgs::msg::PointField::FLOAT32,
      "z", 1, sensor_msgs::msg::PointField::FLOAT32,
      "rgb", 1, sensor_msgs::msg::PointField::FLOAT32);
  } else {
    // Data format: x,y,z; 12 bytes per point
    pc2_modifier.setPointCloud2Fields(
      3,
      "x", 1, sensor_msgs::msg::PointField::FLOAT32,
      "y", 1, sensor_msgs::msg::PointField::FLOAT32,
      "z", 1, sensor_msgs::msg::PointField::FLOAT32);
  }
}

PointCloudProperties PointCloudNode::CreatePointCloudProperties(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud_msg)
{
  const unsigned int byte_unit_conversion_factor = sizeof(float);

  PointCloudProperties cloud_properties;
  cloud_properties.point_row_step = cloud_msg->row_step / byte_unit_conversion_factor;
  cloud_properties.point_step = cloud_msg->point_step / byte_unit_conversion_factor;
  cloud_properties.x_offset = cloud_msg->fields[0].offset / byte_unit_conversion_factor;
  cloud_properties.y_offset = cloud_msg->fields[1].offset / byte_unit_conversion_factor;
  cloud_properties.z_offset = cloud_msg->fields[2].offset / byte_unit_conversion_factor;
  cloud_properties.is_bigendian = cloud_msg->is_bigendian;
  cloud_properties.bad_point = std::numeric_limits<float>::quiet_NaN();
  if (use_color_) {
    cloud_properties.rgb_offset = cloud_msg->fields[3].offset / byte_unit_conversion_factor;
  }
  cloud_properties.buffer_size = cloud_msg->row_step * cloud_msg->height;
  return cloud_properties;
}

DisparityProperties PointCloudNode::CreateDisparityProperties(
  const stereo_msgs::msg::DisparityImage::ConstSharedPtr & disp_msg)
{
  if (disparity_byte_conversion.find(disp_msg->image.encoding) ==
    disparity_byte_conversion.end())
  {
    throw InvalidImageFormatError(
            "Unsupported encoding " + disp_msg->image.encoding + "! Not publishing");
  }

  DisparityProperties disparity_properties;
  disparity_properties.row_step = disp_msg->image.step /
    disparity_byte_conversion.at(disp_msg->image.encoding);

  disparity_properties.height = disp_msg->image.height;
  disparity_properties.width = disp_msg->image.width;
  disparity_properties.buffer_size = disp_msg->image.step * disp_msg->image.height;
  disparity_properties.encoding = disp_msg->image.encoding;
  return disparity_properties;
}

RGBProperties PointCloudNode::CreateRGBProperties(
  const sensor_msgs::msg::Image::ConstSharedPtr & rgb_msg)
{
  RGBProperties rgb_properties;
  rgb_properties.row_step = rgb_msg->step;
  rgb_properties.height = rgb_msg->height;
  rgb_properties.width = rgb_msg->width;
  rgb_properties.buffer_size = rgb_msg->step * rgb_msg->height;
  rgb_properties.encoding = rgb_msg->encoding;

  // Only support 8 bit encodings (since point cloud can only support each color point with 8 bits)
  if (rgb_msg->encoding == sensor_msgs::image_encodings::RGB8) {
    rgb_properties.red_offset = 0;
    rgb_properties.green_offset = 1;
    rgb_properties.blue_offset = 2;
    rgb_properties.color_step = 3;
  } else if (rgb_msg->encoding == sensor_msgs::image_encodings::BGR8) {
    rgb_properties.blue_offset = 0;
    rgb_properties.green_offset = 1;
    rgb_properties.red_offset = 2;
    rgb_properties.color_step = 3;
  } else if (rgb_msg->encoding == sensor_msgs::image_encodings::MONO8) {
    rgb_properties.red_offset = 0;
    rgb_properties.green_offset = 0;
    rgb_properties.blue_offset = 0;
    rgb_properties.color_step = 1;
  } else {
    throw InvalidImageFormatError(
            "Unsupported encoding " + rgb_msg->encoding + "! Not publishing color");
  }
  return rgb_properties;
}

CameraIntrinsics PointCloudNode::CreateCameraIntrinsics(
  image_geometry::StereoCameraModel & stereo_camera_model,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & left_info_msg,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & right_info_msg)
{
  CameraIntrinsics intrinsics;
  stereo_camera_model.fromCameraInfo(left_info_msg, right_info_msg);
  const cv::Matx44d reprojection_matrix = stereo_camera_model.reprojectionMatrix();
  for (auto i = 0; i < intrinsics.reprojection_matrix_rows; ++i) {
    for (auto j = 0; j < intrinsics.reprojection_matrix_cols; ++j) {
      intrinsics.reprojection_matrix[i][j] = reprojection_matrix(i, j);
    }
  }
  return intrinsics;
}

void PointCloudNode::SelectDisparityFormatAndCompute(
  sensor_msgs::msg::PointCloud2::SharedPtr & cloud_msg,
  const PointCloudProperties & cloud_properties,
  const stereo_msgs::msg::DisparityImage::ConstSharedPtr & disp_msg,
  const DisparityProperties & disparity_properties,
  const sensor_msgs::msg::Image::ConstSharedPtr & rgb_msg,
  const RGBProperties & rgb_properties,
  const CameraIntrinsics & intrinsics)
{
  auto disparity_encoding = disp_msg->image.encoding;

  // Select the appropiate disparity type so that the disparity image is interpreted correctly
  if (disparity_encoding == sensor_msgs::image_encodings::MONO8) {
    cloud_compute_.ComputePointCloudData<uint8_t>(
      cloud_msg->data, cloud_properties, disp_msg->image.data, disparity_properties,
      rgb_msg->data, rgb_properties, intrinsics);
  } else if (disparity_encoding == sensor_msgs::image_encodings::TYPE_8UC1) {
    cloud_compute_.ComputePointCloudData<uint8_t>(
      cloud_msg->data, cloud_properties, disp_msg->image.data, disparity_properties,
      rgb_msg->data, rgb_properties, intrinsics);
  } else if (disparity_encoding == sensor_msgs::image_encodings::MONO16) {
    cloud_compute_.ComputePointCloudData<uint16_t>(
      cloud_msg->data, cloud_properties, disp_msg->image.data, disparity_properties,
      rgb_msg->data, rgb_properties, intrinsics);
  } else if (disparity_encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
    cloud_compute_.ComputePointCloudData<uint16_t>(
      cloud_msg->data, cloud_properties, disp_msg->image.data, disparity_properties,
      rgb_msg->data, rgb_properties, intrinsics);
  } else if (disparity_encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
    cloud_compute_.ComputePointCloudData<float>(
      cloud_msg->data, cloud_properties, disp_msg->image.data, disparity_properties,
      rgb_msg->data, rgb_properties, intrinsics);
  }
}

}  // namespace stereo_image_proc
}  // namespace isaac_ros

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(isaac_ros::stereo_image_proc::PointCloudNode)
