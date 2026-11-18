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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <future>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "isaac_ros_depth_image_proc/point_cloud_xyz_masked_node.hpp"
#include "pointcloud_conversions/pointcloud_conversions.hpp"
#include "rclcpp/create_timer.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "gtest/gtest.h"

namespace nvidia
{
namespace isaac_ros
{
namespace depth_image_proc
{
namespace
{

using namespace std::chrono_literals;

//! Produce a process-unique suffix for ROS graph resources and temporary files.
//!
//! Returns a decimal sequence value. The atomic counter permits concurrent fixture construction
//! without duplicate names.
std::string NextTestSuffix()
{
  static std::atomic<uint64_t> next_id{1U};
  return std::to_string(next_id.fetch_add(1U));
}

//! Construct a deterministic ROS timestamp without consulting a clock.
//!
//! `seconds` is copied into the signed seconds field.
//! `nanoseconds` is copied into the fractional field and must be less than one billion; this helper
//! does not validate normalization.
//!
//! Returns the populated timestamp.
builtin_interfaces::msg::Time MakeStamp(int32_t seconds, uint32_t nanoseconds = 0U)
{
  builtin_interfaces::msg::Time stamp;
  stamp.sec = seconds;
  stamp.nanosec = nanoseconds;
  return stamp;
}

//! Create a fixed-size CPU-backed float depth image for node ingress tests.
//!
//! `stamp` becomes the image timestamp in the `depth_camera` frame.
//! `depths` supplies exactly four row-major float32 meter values copied into the payload. By
//! default, it includes in-range, out-of-range, and non-finite samples.
//!
//! Returns a 2x2 `32FC1` image with a tightly packed row step.
sensor_msgs::msg::Image MakeDepthImage(
  const builtin_interfaces::msg::Time & stamp,
  const std::vector<float> & depths = {1.0F, 3.0F, std::numeric_limits<float>::quiet_NaN(), 0.5F})
{
  sensor_msgs::msg::Image image;
  image.header.stamp = stamp;
  image.header.frame_id = "depth_camera";
  image.width = 2U;
  image.height = 2U;
  image.encoding = "32FC1";
  image.step = image.width * sizeof(float);
  image.data.resize(depths.size() * sizeof(float));
  std::memcpy(image.data.data(), depths.data(), image.data.size());
  return image;
}

//! Create a fixed-size CPU-backed mask whose nonzero values select depth pixels.
//!
//! `stamp` becomes the image timestamp in the deliberately different `mask_camera` frame, verifying
//! that production output retains the depth frame.
//! `values` supplies exactly four row-major uint8 mask elements copied into the payload.
//!
//! Returns a 2x2 tightly packed `mono8` image.
sensor_msgs::msg::Image MakeMaskImage(
  const builtin_interfaces::msg::Time & stamp, std::vector<uint8_t> values = {1U, 1U, 1U, 1U})
{
  sensor_msgs::msg::Image image;
  image.header.stamp = stamp;
  image.header.frame_id = "mask_camera";
  image.width = 2U;
  image.height = 2U;
  image.encoding = "mono8";
  image.step = image.width;
  image.data.assign(values.begin(), values.end());
  return image;
}

//! Create identity-like pinhole calibration matching the fixed test depth image.
//!
//! `stamp` becomes the calibration timestamp in the `depth_camera` frame.
//!
//! Returns 2x2 CameraInfo with unit focal lengths and a zero principal point.
sensor_msgs::msg::CameraInfo MakeCameraInfo(const builtin_interfaces::msg::Time & stamp)
{
  sensor_msgs::msg::CameraInfo info;
  info.header.stamp = stamp;
  info.header.frame_id = "depth_camera";
  info.width = 2U;
  info.height = 2U;
  info.k = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  return info;
}

//! Copy a packed CUDA XYZ cloud into host floats for assertions.
//!
//! `cloud` is a borrowed CPU- or CUDA-backed PointCloud2 with packed 12-byte XYZ points, packed
//! rows, and enough payload for `width * height * 3` floats. It remains unmodified.
//!
//! Returns x, y, and z values in message order. The helper creates a temporary nonblocking
//! stream, waits for the device-to-host copy, and destroys the stream before returning. CUDA
//! failures throw.
std::vector<float> DownloadPoints(const sensor_msgs::msg::PointCloud2 & cloud)
{
  cudaStream_t stream = nullptr;
  const cudaError_t create_status =
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (create_status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(create_status));
  }

  try {
    // Widen width before multiplying metadata so the host vector size cannot overflow uint32.
    std::vector<float> points(static_cast<std::size_t>(cloud.width) * cloud.height * 3U);
    {
      auto handle = pointcloud_conversions::from_input_point_cloud_msg(cloud, stream);
      const cudaError_t copy_status =
        cudaMemcpyAsync(
        points.data(), handle.get_ptr(), points.size() * sizeof(float),
        cudaMemcpyDeviceToHost, stream);
      if (copy_status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(copy_status));
      }
      const cudaError_t sync_status = cudaStreamSynchronize(stream);
      if (sync_status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(sync_status));
      }
    }

    const cudaError_t destroy_status = cudaStreamDestroy(stream);
    stream = nullptr;
    if (destroy_status != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(destroy_status));
    }
    return points;
  } catch (...) {
    if (stream != nullptr) {
      // Preserve the original exception; cleanup failures cannot replace the error under test.
      (void)cudaStreamSynchronize(stream);

      // The same best-effort rule applies to destruction of the temporary stream.
      (void)cudaStreamDestroy(stream);
    }
    throw;
  }
}

//! Read a complete benchmark file after the single-threaded callback finishes.
//!
//! `path` names the text file to open.
//!
//! Returns all readable bytes, or an empty string when the file is absent, unreadable, or empty.
std::string ReadBenchmark(const std::string & path)
{
  std::ifstream benchmark(path);
  std::stringstream contents;
  contents << benchmark.rdbuf();
  return contents.str();
}

// Owns an isolated in-process ROS graph for synchronized masked-cloud publication tests.
class PointCloudXYZMaskedNodeTest : public ::testing::Test
{
protected:
  //! Initialize one ROS context shared by all tests in this suite.
  static void SetUpTestSuite()
  {
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
    }
  }

  //! Shut down the suite's ROS context when it is still active.
  static void TearDownTestSuite()
  {
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
  }

  //! Create uniquely named topics, I/O endpoints, the node under test, and its executor graph.
  //!
  //! Waits until every expected publisher/subscriber match is visible before allowing the test body
  //! to publish. The fixture configures a 100 ms synchronization slop and a process-unique
  //! benchmark path.
  void SetUp() override
  {
    suffix_ = NextTestSuffix();
    const std::string topic_prefix = "/point_cloud_xyz_masked_test_" + suffix_;
    depth_topic_ = topic_prefix + "/depth";
    camera_info_topic_ = topic_prefix + "/camera_info";
    mask_topic_ = topic_prefix + "/mask";
    cloud_topic_ = topic_prefix + "/cloud";
    benchmark_path_ = "/tmp/point_cloud_xyz_masked_test_" + suffix_ + ".jsonl";
    std::remove(benchmark_path_.c_str());

    io_node_ = std::make_shared<rclcpp::Node>("point_cloud_xyz_masked_io_" + suffix_);
    const auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
    depth_pub_ = io_node_->create_publisher<sensor_msgs::msg::Image>(depth_topic_, qos);
    camera_info_pub_ =
      io_node_->create_publisher<sensor_msgs::msg::CameraInfo>(camera_info_topic_, qos);
    mask_pub_ = io_node_->create_publisher<sensor_msgs::msg::Image>(mask_topic_, qos);

    // Exercise ordinary ROS delivery here. The builder tests verify CUDA output storage directly;
    // the RMW may deliver a CPU fallback when a platform lacks a compatible CUDA backend plugin.
    cloud_sub_ = io_node_->create_subscription<sensor_msgs::msg::PointCloud2>(
      cloud_topic_,
      qos,
      [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr message) {
        cloud_ = std::move(message);
        if (cloud_ready_.wait_for(0ms) != std::future_status::ready) {
          cloud_received_.set_value();
        }
      });

    masked_node_ = CreateMaskedNode();

    executor_.add_node(io_node_);
    executor_.add_node(masked_node_);
    std::promise<void> graph_matched;
    auto graph_ready = graph_matched.get_future();
    // Check discovery on the ROS node's clock while the executor waits for the handoff.
    auto discovery_timer = rclcpp::create_timer(
      io_node_, io_node_->get_clock(), rclcpp::Duration(10ms), [this, &graph_matched]() {
        if (depth_pub_->get_subscription_count() == 1U &&
        camera_info_pub_->get_subscription_count() == 1U &&
        mask_pub_->get_subscription_count() == 1U && cloud_sub_->get_publisher_count() == 1U)
        {
          graph_matched.set_value();
        }
      });
    const auto discovery_result = executor_.spin_until_future_complete(graph_ready, 2s);
    discovery_timer->cancel();
    ASSERT_EQ(discovery_result, rclcpp::FutureReturnCode::SUCCESS);
  }

  //! Remove both nodes from the executor and attempt to delete the fixture's benchmark file.
  void TearDown() override
  {
    executor_.remove_node(masked_node_);
    executor_.remove_node(io_node_);
    masked_node_.reset();
    io_node_.reset();
    std::remove(benchmark_path_.c_str());
  }

  //! Publish depth and cached calibration before the matching mask arrives.
  //!
  //! `depth_stamp` becomes the depth timestamp queued by the message filter.
  //! `camera_info_stamp` becomes the cached calibration timestamp.
  //!
  //! Runs a bounded executor spin to allow both subscription callbacks to cache their messages.
  void PublishDepthAndCameraInfo(
    const builtin_interfaces::msg::Time & depth_stamp,
    const builtin_interfaces::msg::Time & camera_info_stamp)
  {
    depth_pub_->publish(MakeDepthImage(depth_stamp));
    camera_info_pub_->publish(MakeCameraInfo(camera_info_stamp));
    EXPECT_EQ(
      executor_.spin_until_future_complete(cloud_ready_, 100ms), rclcpp::FutureReturnCode::TIMEOUT);
  }

  //! Construct the component with fixture topics and approximate synchronization.
  std::shared_ptr<PointCloudXYZMaskedNode> CreateMaskedNode()
  {
    rclcpp::NodeOptions options;
    options.append_parameter_override("depth_topic", depth_topic_);
    options.append_parameter_override("camera_info_topic", camera_info_topic_);
    options.append_parameter_override("segmentation_mask_topic", mask_topic_);
    options.append_parameter_override("output_topic", cloud_topic_);
    options.append_parameter_override("sync_slop_sec", 0.1);
    options.append_parameter_override("benchmark_output_path", benchmark_path_);
    return std::make_shared<PointCloudXYZMaskedNode>(options);
  }

  // Process-local unique names prevent collisions between tests in this executable.
  std::string suffix_;
  std::string depth_topic_;
  std::string camera_info_topic_;
  std::string mask_topic_;
  std::string cloud_topic_;
  std::string benchmark_path_;

  // Single-threaded callback execution keeps fixture state deterministic.
  rclcpp::executors::SingleThreadedExecutor executor_;
  std::shared_ptr<PointCloudXYZMaskedNode> masked_node_;
  rclcpp::Node::SharedPtr io_node_;

  // Test-side endpoints feed each input and capture the published output.
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr mask_pub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud_;
  std::promise<void> cloud_received_;
  std::shared_future<void> cloud_ready_{cloud_received_.get_future().share()};
};

TEST_F(PointCloudXYZMaskedNodeTest, PublishesPackedCloudAndBenchmark)
{
  const auto stamp = MakeStamp(20, 123U);
  PublishDepthAndCameraInfo(stamp, stamp);
  mask_pub_->publish(MakeMaskImage(stamp));

  ASSERT_EQ(
    executor_.spin_until_future_complete(cloud_ready_, 2s), rclcpp::FutureReturnCode::SUCCESS);
  EXPECT_EQ(cloud_->header.frame_id, "depth_camera");
  EXPECT_EQ(cloud_->header.stamp, stamp);
  EXPECT_EQ(cloud_->width, 3U);
  EXPECT_EQ(cloud_->height, 1U);
  EXPECT_EQ(cloud_->point_step, 12U);

  // The unbounded default range rejects only NaN, leaving raster pixels (0, 0), (1, 0), and (1, 1).
  // The output keeps the depth frame even though the selecting mask uses a different frame.
  const std::vector<float> points = DownloadPoints(*cloud_);
  const std::vector<float> expected{
    0.0F, 0.0F, 1.0F, 3.0F, 0.0F, 3.0F, 0.5F, 0.5F, 0.5F};
  EXPECT_EQ(points, expected);

  // The executor sets cloud_ only after the synchronized callback returns and flushes the record.
  const std::string benchmark = ReadBenchmark(benchmark_path_);
  EXPECT_NE(benchmark.find("\"point_count\":3"), std::string::npos);
  EXPECT_NE(benchmark.find("\"conversion_ms\":"), std::string::npos);
  EXPECT_NE(benchmark.find("\"mask_to_cloud_total_ms\":"), std::string::npos);
}

TEST_F(PointCloudXYZMaskedNodeTest, RequiresCalibrationAndRejectsDepthMaskOutsideSlop)
{
  const auto missing_calibration_stamp = MakeStamp(10);

  // A synchronized pair cannot trigger conversion before calibration is available.
  depth_pub_->publish(MakeDepthImage(missing_calibration_stamp));
  mask_pub_->publish(MakeMaskImage(missing_calibration_stamp));
  EXPECT_EQ(
    executor_.spin_until_future_complete(cloud_ready_, 100ms), rclcpp::FutureReturnCode::TIMEOUT);
  EXPECT_EQ(cloud_, nullptr);

  // CameraInfo is cached independently; its timestamp does not participate in image sync.
  camera_info_pub_->publish(MakeCameraInfo(MakeStamp(100)));
  EXPECT_EQ(
    executor_.spin_until_future_complete(cloud_ready_, 100ms), rclcpp::FutureReturnCode::TIMEOUT);

  // Approximate synchronization does not emit a callback outside the configured slop.
  depth_pub_->publish(MakeDepthImage(MakeStamp(20)));
  mask_pub_->publish(MakeMaskImage(MakeStamp(21)));
  EXPECT_EQ(
    executor_.spin_until_future_complete(cloud_ready_, 100ms), rclcpp::FutureReturnCode::TIMEOUT);
  EXPECT_EQ(cloud_, nullptr);

  // An all-zero mask is valid input but produces no publication or benchmark record.
  const auto empty_stamp = MakeStamp(30);
  depth_pub_->publish(MakeDepthImage(empty_stamp));
  mask_pub_->publish(MakeMaskImage(empty_stamp, {0U, 0U, 0U, 0U}));
  EXPECT_EQ(
    executor_.spin_until_future_complete(cloud_ready_, 100ms), rclcpp::FutureReturnCode::TIMEOUT);
  EXPECT_EQ(cloud_, nullptr);
  EXPECT_TRUE(ReadBenchmark(benchmark_path_).empty());

  // Only the successful publication produces a benchmark record.
  const auto valid_stamp = MakeStamp(40);
  depth_pub_->publish(MakeDepthImage(valid_stamp));
  mask_pub_->publish(MakeMaskImage(valid_stamp));
  ASSERT_EQ(
    executor_.spin_until_future_complete(cloud_ready_, 2s), rclcpp::FutureReturnCode::SUCCESS);
  const std::string benchmark = ReadBenchmark(benchmark_path_);
  EXPECT_NE(benchmark.find("\"point_count\":3"), std::string::npos);
  EXPECT_EQ(std::count(benchmark.begin(), benchmark.end(), '\n'), 1);
}

TEST_F(PointCloudXYZMaskedNodeTest, ApproximatelySynchronizesDepthAndMaskWithinSlop)
{
  camera_info_pub_->publish(MakeCameraInfo(MakeStamp(100)));
  EXPECT_EQ(
    executor_.spin_until_future_complete(cloud_ready_, 100ms), rclcpp::FutureReturnCode::TIMEOUT);

  depth_pub_->publish(MakeDepthImage(MakeStamp(20)));
  mask_pub_->publish(MakeMaskImage(MakeStamp(20, 50000000U)));
  // Advance the depth stream so ApproximateTime can finalize the offset pair.
  depth_pub_->publish(MakeDepthImage(MakeStamp(21)));
  ASSERT_EQ(
    executor_.spin_until_future_complete(cloud_ready_, 2s), rclcpp::FutureReturnCode::SUCCESS);
  EXPECT_EQ(cloud_->header.stamp, MakeStamp(20));

  cloud_.reset();
  cloud_received_ = std::promise<void>{};
  cloud_ready_ = cloud_received_.get_future().share();
  depth_pub_->publish(MakeDepthImage(MakeStamp(30)));
  mask_pub_->publish(MakeMaskImage(MakeStamp(31)));
  EXPECT_EQ(
    executor_.spin_until_future_complete(cloud_ready_, 100ms), rclcpp::FutureReturnCode::TIMEOUT);
  EXPECT_EQ(cloud_, nullptr);
}

TEST(PointCloudXYZMaskedNodeConstructor, RejectsInvalidConfiguration)
{
  if (!rclcpp::ok()) {
    rclcpp::init(0, nullptr);
  }

  rclcpp::NodeOptions depth_options;
  depth_options.append_parameter_override("min_depth_m", 2.0);
  depth_options.append_parameter_override("max_depth_m", 1.0);
  EXPECT_THROW(std::make_shared<PointCloudXYZMaskedNode>(depth_options), std::invalid_argument);
  rclcpp::shutdown();
}

}  // namespace
}  // namespace depth_image_proc
}  // namespace isaac_ros
}  // namespace nvidia
