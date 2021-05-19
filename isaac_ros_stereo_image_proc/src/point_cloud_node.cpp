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

#include <cuda.h>
#include <cuda_runtime.h>

#include <memory>
#include <limits>
#include <string>

namespace isaac_ros
{
namespace stereo_image_proc
{
PointCloudNode::PointCloudNode(const rclcpp::NodeOptions & options)
: Node("point_cloud_node", options)
{
  // Make sure assumption of 32 bit floating point value is valid
  if (!std::numeric_limits<float>::is_iec559) {
    throw std::runtime_error(
            "Hardware does not support 32-bit IEEE754 floating point standard");
  }

  initializeParameters();
  setupSubscribers();
  setupPublishers();
}

PointCloudNode::~PointCloudNode()
{
  cudaStreamDestroy(stream_);
  cudaFree(intrinsics_);
  cudaFree(cloud_properties_);
  cudaFree(disparity_image_buffer_);
  cudaFree(rgb_image_buffer_);
  cudaFree(pointcloud_data_buffer_);
}

void PointCloudNode::initializeParameters()
{
  // Declare ROS parameters
  queue_size_ = declare_parameter<int>("queue_size", rmw_qos_profile_default.depth);
  use_color_ = declare_parameter<bool>("use_color", false);
  unit_scaling_ = declare_parameter<float>("unit_scaling", 1.0f);
}

void PointCloudNode::setupSubscribers()
{
  using namespace std::placeholders;

  // Initialize message sync policy
  exact_sync_.reset(
    new ExactSync(
      ExactPolicy(queue_size_),
      sub_left_image_, sub_left_info_,
      sub_right_info_, sub_disparity_));

  exact_sync_->registerCallback(
    std::bind(&PointCloudNode::image_cb, this, _1, _2, _3, _4));

  // Subscribe to the relevant topics
  sub_left_image_.subscribe(this, "left/image_rect_color");
  sub_left_info_.subscribe(this, "left/camera_info");
  sub_right_info_.subscribe(this, "right/camera_info");
  sub_disparity_.subscribe(this, "disparity");
}

void PointCloudNode::setupPublishers()
{
  // Create a publisher to the relevant topic
  pub_points2_ = create_publisher<sensor_msgs::msg::PointCloud2>("points2", 1);
}

void PointCloudNode::image_cb(
  const sensor_msgs::msg::Image::ConstSharedPtr & left_image_msg,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & left_info_msg,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & right_info_msg,
  const stereo_msgs::msg::DisparityImage::ConstSharedPtr & disp_msg)
{
  auto cloud_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
  formatPointCloudMessage(cloud_msg, disp_msg);

  updateCudaBuffers(cloud_msg, left_image_msg, disp_msg);

  // Update camera model to get reprojection matrix
  model_.fromCameraInfo(left_info_msg, right_info_msg);
  updateIntrinsics(disp_msg);

  // Reinterpret the point cloud data buffer as a float
  float * pointcloud_data_buffer = reinterpret_cast<float *>(pointcloud_data_buffer_);
  updateCloudProperties(cloud_msg);

  // Determine the image format of the disparity image and convert
  // point cloud according to the datatype
  if (disp_msg->image.encoding == sensor_msgs::image_encodings::TYPE_8UC1 ||
    disp_msg->image.encoding == sensor_msgs::image_encodings::MONO8)
  {
    convertDisparityToPointCloud<uint8_t>(pointcloud_data_buffer, disp_msg);
  } else if (disp_msg->image.encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
    convertDisparityToPointCloud<uint16_t>(pointcloud_data_buffer, disp_msg);
  } else if (disp_msg->image.encoding == sensor_msgs::image_encodings::MONO16) {
    convertDisparityToPointCloud<uint16_t>(pointcloud_data_buffer, disp_msg);
  } else if (disp_msg->image.encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
    convertDisparityToPointCloud<float>(pointcloud_data_buffer, disp_msg);
  } else {
    RCLCPP_ERROR(
      this->get_logger(),
      "Unsupported image encoding [%s]. Not publishing",
      disp_msg->image.encoding.c_str());
    return;
  }

  if (use_color_) {
    addColorToPointCloud(pointcloud_data_buffer, left_image_msg);
  }

  // Wait for CUDA to finish before continuing
  cudaError_t cuda_result = cudaStreamSynchronize(stream_);
  checkCudaErrors(cuda_result);

  copyPointCloudBufferToMessage(cloud_msg);

  pub_points2_->publish(*cloud_msg);
}

void PointCloudNode::formatPointCloudMessage(
  sensor_msgs::msg::PointCloud2::SharedPtr & cloud_msg,
  const stereo_msgs::msg::DisparityImage::ConstSharedPtr & disp_msg)
{
  cloud_msg->header = disp_msg->header;
  cloud_msg->height = disp_msg->image.height;
  cloud_msg->width = disp_msg->image.width;
  cloud_msg->is_bigendian = false;
  cloud_msg->is_dense = false;

  sensor_msgs::PointCloud2Modifier pc2_modifier(*cloud_msg);

  // DC = don't care, all items are 32-bit floats
  if (use_color_) {
    // Data format: x, y, z, rgb
    // 16 bytes per point
    pc2_modifier.setPointCloud2Fields(
      4,
      "x", 1, sensor_msgs::msg::PointField::FLOAT32,
      "y", 1, sensor_msgs::msg::PointField::FLOAT32,
      "z", 1, sensor_msgs::msg::PointField::FLOAT32,
      "rgb", 1, sensor_msgs::msg::PointField::FLOAT32);
  } else {
    // Data format: x, y, z
    // 12 bytes per point
    pc2_modifier.setPointCloud2Fields(
      3,
      "x", 1, sensor_msgs::msg::PointField::FLOAT32,
      "y", 1, sensor_msgs::msg::PointField::FLOAT32,
      "z", 1, sensor_msgs::msg::PointField::FLOAT32);
  }
}

void PointCloudNode::updateCudaBuffers(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud_msg,
  const sensor_msgs::msg::Image::ConstSharedPtr & left_image_msg,
  const stereo_msgs::msg::DisparityImage::ConstSharedPtr & disp_msg)
{
  // Calculate the size of each buffer (in bytes)
  int cloud_size = cloud_msg->row_step * cloud_msg->height;
  int rgb_image_size = left_image_msg->step * left_image_msg->height;
  int disparity_image_size = disp_msg->image.step * disp_msg->image.height;
  cudaError_t cuda_result;

  if (!cuda_initialized_) {
    initializeCuda(cloud_size, rgb_image_size, disparity_image_size);
    cuda_initialized_ = true;
    return;
  }

  // Checks if every buffer is the correct size. If not, update it.
  if (disparity_image_buffer_size_ < disparity_image_size) {
    cuda_result = cudaFree(disparity_image_buffer_);
    checkCudaErrors(cuda_result);

    cuda_result =
      cudaMallocManaged(&disparity_image_buffer_, disparity_image_size, cudaMemAttachHost);
    checkCudaErrors(cuda_result);

    cuda_result = cudaStreamAttachMemAsync(stream_, disparity_image_buffer_);
    checkCudaErrors(cuda_result);

    disparity_image_buffer_size_ = disparity_image_size;
  }

  // If color isn't necessary, avoid allocating the buffer
  if (rgb_image_buffer_size_ < rgb_image_size && use_color_) {
    cuda_result = cudaFree(rgb_image_buffer_);
    checkCudaErrors(cuda_result);

    cuda_result = cudaMallocManaged(&rgb_image_buffer_, rgb_image_size, cudaMemAttachHost);
    checkCudaErrors(cuda_result);

    cuda_result = cudaStreamAttachMemAsync(stream_, rgb_image_buffer_);
    checkCudaErrors(cuda_result);

    rgb_image_buffer_size_ = rgb_image_size;
  }

  if (pointcloud_data_buffer_size_ < cloud_size) {
    cuda_result = cudaFree(pointcloud_data_buffer_);
    checkCudaErrors(cuda_result);

    cuda_result = cudaMallocManaged(&pointcloud_data_buffer_, cloud_size, cudaMemAttachHost);
    checkCudaErrors(cuda_result);

    cuda_result = cudaStreamAttachMemAsync(stream_, pointcloud_data_buffer_);
    checkCudaErrors(cuda_result);

    pointcloud_data_buffer_size_ = cloud_size;
  }

  cuda_result = cudaStreamSynchronize(stream_);
  checkCudaErrors(cuda_result);
}

void PointCloudNode::initializeCuda(int cloud_size, int rgb_image_size, int disparity_image_size)
{
  // Allocates memory in GPU for every struct and buffer needed in the GPU
  cudaError_t cuda_result = cudaMallocManaged(&intrinsics_, sizeof(nvPointCloudIntrinsics));
  checkCudaErrors(cuda_result);

  cuda_result = cudaMallocManaged(&cloud_properties_, sizeof(nvPointCloudProperties));
  checkCudaErrors(cuda_result);

  cuda_result =
    cudaMallocManaged(&disparity_image_buffer_, disparity_image_size, cudaMemAttachHost);
  checkCudaErrors(cuda_result);
  disparity_image_buffer_size_ = disparity_image_size;

  // If color isn't necessary, don't allocate the buffer
  if (use_color_) {
    cuda_result = cudaMallocManaged(&rgb_image_buffer_, rgb_image_size, cudaMemAttachHost);
    checkCudaErrors(cuda_result);
    rgb_image_buffer_size_ = rgb_image_size;
  }

  cuda_result = cudaMallocManaged(&pointcloud_data_buffer_, cloud_size, cudaMemAttachHost);
  checkCudaErrors(cuda_result);
  pointcloud_data_buffer_size_ = cloud_size;

  // Create the CUDA streams; do both just in case color parameter changes
  cuda_result = cudaStreamCreate(&stream_);
  checkCudaErrors(cuda_result);

  if (use_color_) {
    cuda_result = cudaStreamAttachMemAsync(stream_, rgb_image_buffer_);
    checkCudaErrors(cuda_result);
  }

  cuda_result = cudaStreamAttachMemAsync(stream_, disparity_image_buffer_);
  checkCudaErrors(cuda_result);

  cuda_result = cudaStreamAttachMemAsync(stream_, pointcloud_data_buffer_);
  checkCudaErrors(cuda_result);

  cuda_result = cudaStreamSynchronize(stream_);
  checkCudaErrors(cuda_result);
}

inline void PointCloudNode::_checkCudaErrors(
  cudaError_t result, const char * filename, int line_number)
{
  if (result != cudaSuccess) {
    RCLCPP_ERROR(
      this->get_logger(),
      "CUDA Error: [%s] (error code: [%d]) at [%s] in line [%d]",
      cudaGetErrorString(result), result, filename, line_number);

    throw std::runtime_error(
            "CUDA Error: " + std::string(cudaGetErrorString(result)) + " (error code: " +
            std::to_string(result) + ") at " + std::string(filename) + " in line " +
            std::to_string(line_number));
  }
}

void PointCloudNode::updateIntrinsics(
  const stereo_msgs::msg::DisparityImage::ConstSharedPtr & disp_msg)
{
  const cv::Matx44d reproject_matrix = model_.reprojectionMatrix();

  // Only getting the relevant entries
  intrinsics_->reproject_matrix_rows = 4;
  intrinsics_->reproject_matrix_cols = 4;
  intrinsics_->reproject_matrix[0][0] = reproject_matrix(0, 0);
  intrinsics_->reproject_matrix[0][3] = reproject_matrix(0, 3);
  intrinsics_->reproject_matrix[1][1] = reproject_matrix(1, 1);
  intrinsics_->reproject_matrix[1][3] = reproject_matrix(1, 3);
  intrinsics_->reproject_matrix[2][3] = reproject_matrix(2, 3);
  intrinsics_->reproject_matrix[3][2] = reproject_matrix(3, 2);
  intrinsics_->reproject_matrix[3][3] = reproject_matrix(3, 3);

  intrinsics_->height = disp_msg->image.height;
  intrinsics_->width = disp_msg->image.width;
  intrinsics_->unit_scaling = unit_scaling_;
}

void PointCloudNode::updateCloudProperties(
  const sensor_msgs::msg::PointCloud2::SharedPtr & cloud_msg)
{
  // The offsets are given in bytes, but the cloud data will be in floats
  // The byte unit conversion is to ensure that the parameters work with a float array
  const int byte_unit_conversion_factor = sizeof(float);
  cloud_properties_->point_row_step = cloud_msg->row_step / byte_unit_conversion_factor;
  cloud_properties_->point_step = cloud_msg->point_step / byte_unit_conversion_factor;
  cloud_properties_->x_offset = cloud_msg->fields[0].offset / byte_unit_conversion_factor;
  cloud_properties_->y_offset = cloud_msg->fields[1].offset / byte_unit_conversion_factor;
  cloud_properties_->z_offset = cloud_msg->fields[2].offset / byte_unit_conversion_factor;
  cloud_properties_->is_bigendian = cloud_msg->is_bigendian;
  cloud_properties_->bad_point = std::numeric_limits<float>::quiet_NaN();

  if (use_color_) {
    cloud_properties_->rgb_offset = cloud_msg->fields[3].offset / byte_unit_conversion_factor;
  }

  cloud_properties_->buffer_size = cloud_msg->row_step * cloud_msg->height;
}

void PointCloudNode::copyPointCloudBufferToMessage(
  sensor_msgs::msg::PointCloud2::SharedPtr & cloud_msg)
{
  // Resize the cloud message buffer and then copy from GPU to CPU
  cloud_msg->data.resize(cloud_properties_->buffer_size);
  cudaError_t cuda_result = cudaMemcpyAsync(
    cloud_msg->data.data(),
    pointcloud_data_buffer_,
    cloud_properties_->buffer_size, cudaMemcpyDeviceToHost, stream_);
  checkCudaErrors(cuda_result);

  cuda_result = cudaStreamSynchronize(stream_);
  checkCudaErrors(cuda_result);
}

template<typename T>
void PointCloudNode::convertDisparityToPointCloud(
  float * pointcloud_data,
  const stereo_msgs::msg::DisparityImage::ConstSharedPtr & disp_msg)
{
  int disparity_image_size = disp_msg->image.step * disp_msg->image.height;

  // Redunant sanity check to ensure that memory accesses are valid
  if (disparity_image_buffer_size_ < disparity_image_size ||
    pointcloud_data_buffer_size_ < cloud_properties_->buffer_size)
  {
    RCLCPP_ERROR(
      this->get_logger(),
      "Size allocated to CUDA buffer cannot fit message's data");
    throw std::runtime_error(
            "Size allocated to CUDA buffer cannot fit message's data");
  }

  // Copy the disparity image into GPU
  cudaError_t cuda_result;
  cuda_result = cudaMemcpyAsync(
    disparity_image_buffer_,
    disp_msg->image.data.data(),
    disparity_image_size, cudaMemcpyHostToDevice,
    stream_);
  checkCudaErrors(cuda_result);

  cuda_result = cudaStreamSynchronize(stream_);
  checkCudaErrors(cuda_result);

  // Reinterpret the disparity buffer as template type T
  // And adjust the step size accordingly
  const T * disparity_buffer = reinterpret_cast<const T *>(disparity_image_buffer_);
  const int disparity_row_step = disp_msg->image.step / sizeof(T);

  PointCloudNode_CUDA::convertDisparityToPointCloud<T>(
    pointcloud_data,
    disparity_buffer, disparity_row_step, intrinsics_, cloud_properties_, stream_);
}

void PointCloudNode::addColorToPointCloud(
  float * pointcloud_data,
  const sensor_msgs::msg::Image::ConstSharedPtr & rgb_image_msg)
{
  const int rgb_image_size = rgb_image_msg->step * rgb_image_msg->height;

  // Redunant sanity check to ensure that memory accesses are valid
  if (rgb_image_buffer_size_ < rgb_image_size ||
    pointcloud_data_buffer_size_ < cloud_properties_->buffer_size)
  {
    RCLCPP_ERROR(
      this->get_logger(),
      "Size allocated to CUDA buffer cannot fit message's data");
    throw std::runtime_error("Size allocated to CUDA buffer cannot fit message's data");
  }

  // Copy the RGB image into GPU
  cudaError_t cuda_result;
  cuda_result = cudaMemcpyAsync(
    rgb_image_buffer_,
    rgb_image_msg->data.data(),
    rgb_image_size, cudaMemcpyHostToDevice,
    stream_);
  checkCudaErrors(cuda_result);

  cuda_result = cudaStreamSynchronize(stream_);
  checkCudaErrors(cuda_result);

  int red_offset, green_offset, blue_offset, color_step;
  const int rgb_row_step = rgb_image_msg->step;

  // Determines the R,G,B and size of each pixel based on image encoding
  if (rgb_image_msg->encoding == sensor_msgs::image_encodings::RGB8) {
    red_offset = 0;
    green_offset = 1;
    blue_offset = 2;
    color_step = 3;
  } else if (rgb_image_msg->encoding == sensor_msgs::image_encodings::BGR8) {
    red_offset = 2;
    green_offset = 1;
    blue_offset = 0;
    color_step = 3;
  } else if (rgb_image_msg->encoding == sensor_msgs::image_encodings::MONO8) {
    red_offset = 0;
    green_offset = 0;
    blue_offset = 0;
    color_step = 1;
  } else {
    RCLCPP_ERROR(
      this->get_logger(),
      "Unsupported data encoding [%s]. Publishing without color",
      rgb_image_msg->encoding.c_str());
    return;
  }

  PointCloudNode_CUDA::addColorToPointCloud(
    pointcloud_data, rgb_image_buffer_,
    rgb_row_step, red_offset, green_offset, blue_offset,
    color_step, intrinsics_, cloud_properties_, stream_);
}
}  // namespace stereo_image_proc
}  // namespace isaac_ros

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(isaac_ros::stereo_image_proc::PointCloudNode)
