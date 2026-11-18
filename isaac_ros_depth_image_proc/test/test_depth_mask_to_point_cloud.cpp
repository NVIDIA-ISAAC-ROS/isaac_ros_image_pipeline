// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "cuda_buffer/cuda_buffer_impl.hpp"
#include "isaac_ros_depth_image_proc/depth_mask_to_point_cloud.hpp"
#include "pointcloud_conversions/pointcloud_conversions.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace depth_image_proc
{
namespace
{

//! Convert a CUDA test-fixture failure into an exception.
//!
//! `status` is the CUDA result to check.
//!
//! Returns normally for `cudaSuccess`; otherwise throws `std::runtime_error` with CUDA's error
//! text.
void CheckCuda(cudaError_t status)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

// Provides a deliberately unsupported backend name without exposing or accessing its storage.
class UnsupportedBufferImpl final : public cuda_buffer_backend::CudaBufferImpl<uint8_t>
{
public:
  std::string get_backend_type() const override {return "unsupported";}
};

//! Construct a tightly packed image with either CPU or CUDA payload storage.
//!
//! `T` is the test pixel representation and must agree with `encoding`.
//! `width` and `height` become the image dimensions in pixels.
//! `encoding` is copied into the ROS image metadata without validation.
//! `values` supplies payload elements copied verbatim into storage owned by the returned message.
//! It must provide enough bytes for every declared row. Extra elements are retained and may become
//! row padding when a test subsequently increases `step`.
//! `cuda_backed` promotes the CPU payload to CUDA storage when true.
//!
//! Returns the populated image with `step` equal to `width * sizeof(T)`. CUDA allocation, upload,
//! and handle exceptions propagate directly.
template<typename T>
sensor_msgs::msg::Image MakeImage(
  uint32_t width,
  uint32_t height,
  const std::string & encoding,
  const std::vector<T> & values,
  bool cuda_backed = false)
{
  sensor_msgs::msg::Image image;
  image.width = width;
  image.height = height;
  image.encoding = encoding;
  image.step = width * sizeof(T);
  const std::size_t bytes = values.size() * sizeof(T);
  image.data.resize(bytes);
  std::memcpy(image.data.data(), values.data(), bytes);
  if (cuda_backed) {
    auto handle = cuda_buffer_backend::from_input_buffer(image.data, nullptr);
    image.data = std::move(*handle.get_promoted_buffer());
  }
  return image;
}

//! Construct rectified pinhole calibration for a test image.
//!
//! `width` and `height` are copied as optional expected image dimensions; zero leaves a dimension
//! unspecified by the builder contract.
//! `fx` and `fy` are focal lengths in pixels.
//! `cx` and `cy` are principal-point coordinates in pixels.
//!
//! Returns CameraInfo with the supplied values in the corresponding K-matrix entries.
sensor_msgs::msg::CameraInfo MakeCameraInfo(
  uint32_t width, uint32_t height, double fx, double fy, double cx, double cy)
{
  sensor_msgs::msg::CameraInfo camera_info;
  camera_info.width = width;
  camera_info.height = height;
  camera_info.k[0] = fx;
  camera_info.k[2] = cx;
  camera_info.k[4] = fy;
  camera_info.k[5] = cy;
  return camera_info;
}

//! Copy a packed CUDA XYZ cloud into host floats for mathematical assertions.
//!
//! `cloud` is a borrowed, CUDA-backed PointCloud2 with packed 12-byte XYZ points, packed rows, and
//! enough payload for `width * height * 3` floats. It remains unmodified.
//! `stream` is a reusable nonblocking CUDA stream owned by the test suite.
//!
//! Returns x, y, and z values in message order after waiting for the device-to-host copy. CUDA
//! failures throw. The stream remains valid for later assertions.
std::vector<float> DownloadPoints(
  const sensor_msgs::msg::PointCloud2 & cloud, const cudaStream_t & stream)
{
  // Widen width before multiplying metadata so the host vector size cannot overflow uint32.
  const std::size_t float_count = static_cast<std::size_t>(cloud.width) * cloud.height * 3U;
  std::vector<float> points(float_count);
  {
    auto handle = pointcloud_conversions::from_input_point_cloud_msg(cloud, stream);
    CheckCuda(cudaMemcpyAsync(
      points.data(), handle.get_ptr(), points.size() * sizeof(float),
      cudaMemcpyDeviceToHost, stream));
    CheckCuda(cudaStreamSynchronize(stream));
  }
  return points;
}

//! Verify the compact x/y/z layout required by CUDA consumers.
//!
//! `cloud` is the PointCloud2 metadata and payload to inspect.
//! `point_count` is the expected number of points in its single row.
//!
//! Emits GoogleTest failures for an incorrect backend, dimensions, byte sizes, or field layout.
void ExpectPackedCudaCloud(const sensor_msgs::msg::PointCloud2 & cloud, uint32_t point_count)
{
  EXPECT_EQ(cloud.data.get_backend_type(), "cuda");
  EXPECT_EQ(cloud.width, point_count);
  EXPECT_EQ(cloud.height, 1U);
  EXPECT_EQ(cloud.point_step, 12U);
  EXPECT_EQ(cloud.row_step, point_count * 12U);

  // Widen point_count so the expected byte count uses the payload size type without uint32
  // overflow.
  EXPECT_EQ(cloud.data.size(), static_cast<std::size_t>(point_count) * 12U);
  ASSERT_EQ(cloud.fields.size(), 3U);
  EXPECT_EQ(cloud.fields[0].name, "x");
  EXPECT_EQ(cloud.fields[0].offset, 0U);
  EXPECT_EQ(cloud.fields[0].datatype, sensor_msgs::msg::PointField::FLOAT32);
  EXPECT_EQ(cloud.fields[0].count, 1U);
  EXPECT_EQ(cloud.fields[1].name, "y");
  EXPECT_EQ(cloud.fields[1].offset, 4U);
  EXPECT_EQ(cloud.fields[1].datatype, sensor_msgs::msg::PointField::FLOAT32);
  EXPECT_EQ(cloud.fields[1].count, 1U);
  EXPECT_EQ(cloud.fields[2].name, "z");
  EXPECT_EQ(cloud.fields[2].offset, 8U);
  EXPECT_EQ(cloud.fields[2].datatype, sensor_msgs::msg::PointField::FLOAT32);
  EXPECT_EQ(cloud.fields[2].count, 1U);
}

// Reuses one CUDA download stream across every builder test in this executable.
class DepthMaskToPointCloudBuilderTest : public ::testing::Test
{
protected:
  static void SetUpTestSuite()
  {
    CheckCuda(cudaStreamCreateWithFlags(&download_stream_, cudaStreamNonBlocking));
  }

  static void TearDownTestSuite()
  {
    CheckCuda(cudaStreamSynchronize(download_stream_));
    CheckCuda(cudaStreamDestroy(download_stream_));
    download_stream_ = nullptr;
  }

  static cudaStream_t download_stream_;
};

cudaStream_t DepthMaskToPointCloudBuilderTest::download_stream_{nullptr};

TEST_F(DepthMaskToPointCloudBuilderTest, UploadsCpuFloatDepthAndEightBitMask)
{
  auto depth =
    MakeImage<float>(2U, 2U, "32FC1", {1.0F, 2.0F, std::numeric_limits<float>::quiet_NaN(), 4.0F});
  depth.header.frame_id = "depth_optical";
  depth.header.stamp.sec = 7;
  const auto mask = MakeImage<uint8_t>(2U, 2U, "mono8", {1U, 0U, 1U, 1U});

  DepthMaskToPointCloudBuilder builder(0.5F, 5.0F);
  const auto cloud = builder.Build(depth, MakeCameraInfo(2U, 2U, 2.0, 4.0, 0.5, 0.5), mask);

  ASSERT_NE(cloud, nullptr);
  EXPECT_EQ(cloud->header, depth.header);
  EXPECT_TRUE(cloud->is_dense);
  ExpectPackedCudaCloud(*cloud, 2U);

  // Pixels zero and three survive mask, finite-depth, and range filtering and retain raster order.
  const std::vector<float> points = DownloadPoints(*cloud, download_stream_);
  const std::vector<float> expected{-0.25F, -0.125F, 1.0F, 1.0F, 0.5F, 4.0F};
  EXPECT_EQ(points, expected);
}

TEST_F(DepthMaskToPointCloudBuilderTest, ReturnedCloudOutlivesBuilder)
{
  const auto depth = MakeImage<float>(1U, 1U, "32FC1", {1.0F});
  const auto mask = MakeImage<uint8_t>(1U, 1U, "mono8", {1U});
  std::unique_ptr<sensor_msgs::msg::PointCloud2> cloud;

  {
    DepthMaskToPointCloudBuilder builder;
    cloud = builder.Build(depth, MakeCameraInfo(1U, 1U, 1.0, 1.0, 0.0, 0.0), mask);
    ASSERT_NE(cloud, nullptr);

    ExpectPackedCudaCloud(*cloud, 1U);
  }

  // Read through the buffer's completion event on a consumer-owned stream after builder teardown.
  ASSERT_NE(cloud, nullptr);
  EXPECT_EQ(
    DownloadPoints(*cloud, download_stream_), (std::vector<float>{0.0F, 0.0F, 1.0F}));
}

TEST_F(DepthMaskToPointCloudBuilderTest, KeepsUint16DepthAndMaskOnCuda)
{
  const auto depth = MakeImage<uint16_t>(2U, 1U, "16UC1", {500U, 1500U}, true);
  const auto mask = MakeImage<uint16_t>(2U, 1U, "mono16", {0U, 7U}, true);
  ASSERT_EQ(depth.data.get_backend_type(), "cuda");
  ASSERT_EQ(mask.data.get_backend_type(), "cuda");

  DepthMaskToPointCloudBuilder builder;
  const auto cloud = builder.Build(depth, MakeCameraInfo(2U, 1U, 100.0, 100.0, 0.0, 0.0), mask);

  ASSERT_NE(cloud, nullptr);
  ExpectPackedCudaCloud(*cloud, 1U);

  // Only column one is selected: 1500 mm becomes 1.5 m and x = column * z / fx = 0.015 m.
  const std::vector<float> points = DownloadPoints(*cloud, download_stream_);
  ASSERT_EQ(points.size(), 3U);
  EXPECT_FLOAT_EQ(points[0], 0.015F);
  EXPECT_FLOAT_EQ(points[1], 0.0F);
  EXPECT_FLOAT_EQ(points[2], 1.5F);
}

TEST_F(DepthMaskToPointCloudBuilderTest, HonorsPaddedImageRows)
{
  // Every second element is row padding. Correct stride handling must skip those zeros rather than
  // interpret them as image pixels.
  auto depth = MakeImage<float>(1U, 2U, "32FC1", {1.0F, 0.0F, 2.0F, 0.0F});
  depth.step = 2U * sizeof(float);
  auto mask = MakeImage<uint8_t>(1U, 2U, "8UC1", {1U, 0U, 1U, 0U});
  mask.step = 2U;

  DepthMaskToPointCloudBuilder builder;
  const auto cloud = builder.Build(depth, MakeCameraInfo(1U, 2U, 1.0, 1.0, 0.0, 0.0), mask);

  ASSERT_NE(cloud, nullptr);
  const std::vector<float> points = DownloadPoints(*cloud, download_stream_);
  const std::vector<float> expected{0.0F, 0.0F, 1.0F, 0.0F, 2.0F, 2.0F};
  EXPECT_EQ(points, expected);
}

TEST_F(DepthMaskToPointCloudBuilderTest, ReturnsNoCloudForEmptyMask)
{
  const auto depth = MakeImage<float>(2U, 1U, "32FC1", {1.0F, 2.0F});
  const auto mask = MakeImage<uint8_t>(2U, 1U, "mono8", {0U, 0U});

  DepthMaskToPointCloudBuilder builder;

  // Zero CameraInfo dimensions mean unspecified, so valid intrinsics still apply to this image.
  const auto cloud = builder.Build(depth, MakeCameraInfo(0U, 0U, 1.0, 1.0, 0.0, 0.0), mask);

  EXPECT_EQ(cloud, nullptr);
}

TEST_F(DepthMaskToPointCloudBuilderTest, RejectsInvalidMetadataAndStorage)
{
  const auto depth = MakeImage<float>(2U, 1U, "32FC1", {1.0F, 2.0F});
  const auto mask = MakeImage<uint8_t>(2U, 1U, "mono8", {1U, 1U});
  const auto camera_info = MakeCameraInfo(2U, 1U, 1.0, 1.0, 0.0, 0.0);
  DepthMaskToPointCloudBuilder builder;

  // Reject unknown backends before attempting to interpret either payload as host or device memory.
  auto invalid_depth = depth;
  invalid_depth.data = rosidl::Buffer<uint8_t>(std::make_unique<UnsupportedBufferImpl>());
  EXPECT_THROW(builder.Build(invalid_depth, camera_info, mask), std::invalid_argument);

  auto invalid_mask = mask;
  invalid_mask.data = rosidl::Buffer<uint8_t>(std::make_unique<UnsupportedBufferImpl>());
  EXPECT_THROW(builder.Build(depth, camera_info, invalid_mask), std::invalid_argument);

  // Reject encodings that do not identify one of the implemented pixel representations.
  invalid_depth = depth;
  invalid_depth.encoding = "8UC1";
  EXPECT_THROW(builder.Build(invalid_depth, camera_info, mask), std::invalid_argument);

  invalid_mask = mask;
  invalid_mask.encoding = "rgb8";
  EXPECT_THROW(builder.Build(depth, camera_info, invalid_mask), std::invalid_argument);

  // Depth and mask geometry and byte order must describe the same little-endian raster.
  invalid_mask = mask;
  invalid_mask.width = 1U;
  EXPECT_THROW(builder.Build(depth, camera_info, invalid_mask), std::invalid_argument);

  invalid_depth = depth;
  invalid_depth.is_bigendian = true;
  EXPECT_THROW(builder.Build(invalid_depth, camera_info, mask), std::invalid_argument);

  // Row metadata must address a complete payload and align to the encoded element width.
  invalid_depth = depth;
  invalid_depth.data.resize(sizeof(float));
  EXPECT_THROW(builder.Build(invalid_depth, camera_info, mask), std::invalid_argument);

  invalid_depth = depth;
  invalid_depth.step = 2U * sizeof(float) + 1U;
  invalid_depth.data.resize(invalid_depth.step);
  EXPECT_THROW(builder.Build(invalid_depth, camera_info, mask), std::invalid_argument);

  // Calibration dimensions must match when specified, and all four projection values must be
  // finite floats with nonzero focal lengths.
  auto invalid_camera = camera_info;
  invalid_camera.width = 1U;
  EXPECT_THROW(builder.Build(depth, invalid_camera, mask), std::invalid_argument);

  invalid_camera = camera_info;
  invalid_camera.k[0] = 0.0;
  EXPECT_THROW(builder.Build(depth, invalid_camera, mask), std::invalid_argument);

  invalid_camera = camera_info;
  invalid_camera.k[4] = std::numeric_limits<double>::infinity();
  EXPECT_THROW(builder.Build(depth, invalid_camera, mask), std::invalid_argument);

  invalid_camera = camera_info;
  invalid_camera.k[2] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(builder.Build(depth, invalid_camera, mask), std::invalid_argument);

  invalid_camera = camera_info;
  invalid_camera.k[5] = std::numeric_limits<double>::max();
  EXPECT_THROW(builder.Build(depth, invalid_camera, mask), std::invalid_argument);
}

TEST_F(DepthMaskToPointCloudBuilderTest, UsesUnboundedDefaultDepthRange)
{
  const auto depth = MakeImage<float>(2U, 1U, "32FC1", {0.0F, 1000.0F});
  const auto mask = MakeImage<uint8_t>(2U, 1U, "mono8", {1U, 1U});

  DepthMaskToPointCloudBuilder builder;
  const auto cloud = builder.Build(
    depth, MakeCameraInfo(2U, 1U, 1.0, 1.0, 0.0, 0.0), mask);

  ASSERT_NE(cloud, nullptr);
  EXPECT_EQ(cloud->width, 2U);
  EXPECT_EQ(
    DownloadPoints(*cloud, download_stream_),
    (std::vector<float>{0.0F, 0.0F, 0.0F, 1000.0F, 0.0F, 1000.0F}));
}

TEST_F(DepthMaskToPointCloudBuilderTest, RejectsInvalidLimits)
{
  EXPECT_THROW(
    DepthMaskToPointCloudBuilder(2.0F, 1.0F),
    std::invalid_argument);
  EXPECT_THROW(
    DepthMaskToPointCloudBuilder(0.0F, std::numeric_limits<float>::quiet_NaN()),
    std::invalid_argument);
}

TEST_F(DepthMaskToPointCloudBuilderTest, ReportsSupportedEncodings)
{
  EXPECT_TRUE(DepthMaskToPointCloudBuilder::IsSupportedDepthEncoding("32FC1"));
  EXPECT_TRUE(DepthMaskToPointCloudBuilder::IsSupportedDepthEncoding("16UC1"));
  EXPECT_FALSE(DepthMaskToPointCloudBuilder::IsSupportedDepthEncoding("mono8"));
  EXPECT_TRUE(DepthMaskToPointCloudBuilder::IsSupportedMaskEncoding("mono8"));
  EXPECT_TRUE(DepthMaskToPointCloudBuilder::IsSupportedMaskEncoding("8UC1"));
  EXPECT_TRUE(DepthMaskToPointCloudBuilder::IsSupportedMaskEncoding("mono16"));
  EXPECT_TRUE(DepthMaskToPointCloudBuilder::IsSupportedMaskEncoding("16UC1"));
  EXPECT_FALSE(DepthMaskToPointCloudBuilder::IsSupportedMaskEncoding("32FC1"));
}

}  // namespace
}  // namespace depth_image_proc
}  // namespace isaac_ros
}  // namespace nvidia
