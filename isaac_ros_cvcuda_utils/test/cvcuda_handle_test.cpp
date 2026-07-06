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

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <memory>

#include "isaac_ros_cvcuda_utils/cvcuda_handle.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "nvcv/ImageFormat.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace cvcuda_utils
{
namespace test
{

class CVCUDATensorHandleTest : public ::testing::Test
{
protected:
  static constexpr uint32_t kTestWidth = 640;
  static constexpr uint32_t kTestHeight = 480;
  static constexpr int kNumChannels = 3;
  static constexpr int kBytesPerChannel = 1;
  static constexpr uint32_t kStepBytes = kTestWidth * kNumChannels * kBytesPerChannel;
  static constexpr size_t kBufferSize = kStepBytes * kTestHeight;

  void SetUp() override
  {
    // Allocate CUDA memory for test
    cudaError_t err = cudaMalloc(&device_ptr_, kBufferSize);
    ASSERT_EQ(err, cudaSuccess) << "Failed to allocate CUDA memory";

    // Create a stream for testing
    err = cudaStreamCreate(&stream_);
    ASSERT_EQ(err, cudaSuccess) << "Failed to create CUDA stream";
  }

  void TearDown() override
  {
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
    // Note: device_ptr_ ownership is transferred to NitrosImage, so we don't free it here
  }

  // Helper to create a NitrosImage with test data
  nitros::NitrosImage CreateTestNitrosImage()
  {
    nitros::NitrosImage img;
    // Allocate new memory for each image (ownership transfers to NitrosImage)
    void * ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, kBufferSize);
    if (err != cudaSuccess) {
      throw std::runtime_error("Failed to allocate CUDA memory for test image");
    }
    // from_external takes ownership of the pointer
    // WriteHandle goes out of scope here, which records the write event
    {
      auto write_handle = img.from_external(
        ptr,
        kBufferSize,
        kTestWidth,
        kTestHeight,
        kStepBytes,
        "rgb8",
        stream_);
    }  // WriteHandle destructor records event, making data available for reading
    return img;
  }

  void * device_ptr_{nullptr};
  cudaStream_t stream_{nullptr};
};

// Test CVCUDATensorHandle default construction
TEST_F(CVCUDATensorHandleTest, DefaultConstruction)
{
  CVCUDATensorHandle<ReadHandle> handle;
  // Default constructed handle should have default tensor
  // Just verify it doesn't crash
  SUCCEED();
}

// Test CVCUDATensorHandle move construction
TEST_F(CVCUDATensorHandleTest, MoveConstruction)
{
  auto img = CreateTestNitrosImage();
  auto read_handle = img.get_read_handle(stream_);

  auto tensor_handle = CVCUDATensorHandle<ReadHandle>::from_nitros_image(
    img,
    std::move(read_handle),
    nvcv::FMT_RGB8,
    kNumChannels,
    kBytesPerChannel);

  // Move construct
  CVCUDATensorHandle<ReadHandle> moved_handle(std::move(tensor_handle));

  // Verify moved handle has valid tensor
  const auto & tensor = moved_handle.get_tensor();
  auto shape = tensor.shape();
  EXPECT_EQ(shape[0], 1);  // Batch size
  EXPECT_EQ(shape[1], static_cast<int64_t>(kTestHeight));
  EXPECT_EQ(shape[2], static_cast<int64_t>(kTestWidth));
  EXPECT_EQ(shape[3], kNumChannels);
}

// Test CVCUDATensorHandle move assignment
TEST_F(CVCUDATensorHandleTest, MoveAssignment)
{
  auto img = CreateTestNitrosImage();
  auto read_handle = img.get_read_handle(stream_);

  auto tensor_handle = CVCUDATensorHandle<ReadHandle>::from_nitros_image(
    img,
    std::move(read_handle),
    nvcv::FMT_RGB8,
    kNumChannels,
    kBytesPerChannel);

  // Move assign
  CVCUDATensorHandle<ReadHandle> assigned_handle;
  assigned_handle = std::move(tensor_handle);

  // Verify assigned handle has valid tensor
  const auto & tensor = assigned_handle.get_tensor();
  auto shape = tensor.shape();
  EXPECT_EQ(shape[0], 1);
  EXPECT_EQ(shape[1], static_cast<int64_t>(kTestHeight));
  EXPECT_EQ(shape[2], static_cast<int64_t>(kTestWidth));
  EXPECT_EQ(shape[3], kNumChannels);
}

// Test from_nitros_image static factory method with RGB8 format
TEST_F(CVCUDATensorHandleTest, FromNitrosImageRGB8)
{
  auto img = CreateTestNitrosImage();
  auto read_handle = img.get_read_handle(stream_);

  auto tensor_handle = CVCUDATensorHandle<ReadHandle>::from_nitros_image(
    img,
    std::move(read_handle),
    nvcv::FMT_RGB8,
    kNumChannels,
    kBytesPerChannel);

  // Verify tensor properties
  const auto & tensor = tensor_handle.get_tensor();
  auto shape = tensor.shape();

  // NHWC format: [batch, height, width, channels]
  EXPECT_EQ(shape.rank(), 4);
  EXPECT_EQ(shape[0], 1);  // Batch size
  EXPECT_EQ(shape[1], static_cast<int64_t>(kTestHeight));
  EXPECT_EQ(shape[2], static_cast<int64_t>(kTestWidth));
  EXPECT_EQ(shape[3], kNumChannels);

  // Verify layout is HWC
  EXPECT_EQ(tensor.layout(), nvcv::TENSOR_NHWC);
}

// Test WrapCVCUDATensor free function (backward compatibility)
TEST_F(CVCUDATensorHandleTest, WrapCVCUDATensorFreeFunction)
{
  auto img = CreateTestNitrosImage();
  auto read_handle = img.get_read_handle(stream_);

  // Use the free function
  auto tensor_handle = WrapCVCUDATensor(
    img,
    std::move(read_handle),
    nvcv::FMT_RGB8,
    kNumChannels,
    kBytesPerChannel);

  // Verify tensor properties
  const auto & tensor = tensor_handle.get_tensor();
  auto shape = tensor.shape();

  EXPECT_EQ(shape.rank(), 4);
  EXPECT_EQ(shape[0], 1);
  EXPECT_EQ(shape[1], static_cast<int64_t>(kTestHeight));
  EXPECT_EQ(shape[2], static_cast<int64_t>(kTestWidth));
  EXPECT_EQ(shape[3], kNumChannels);
}

// Test with WriteHandle
TEST_F(CVCUDATensorHandleTest, FromNitrosImageWithWriteHandle)
{
  nitros::NitrosImage img;
  void * ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, kBufferSize);
  ASSERT_EQ(err, cudaSuccess);

  auto write_handle = img.from_external(
    ptr,
    kBufferSize,
    kTestWidth,
    kTestHeight,
    kStepBytes,
    "rgb8",
    stream_);

  auto tensor_handle = CVCUDATensorHandle<WriteHandle>::from_nitros_image(
    img,
    std::move(write_handle),
    nvcv::FMT_RGB8,
    kNumChannels,
    kBytesPerChannel);

  // Verify tensor properties
  auto & tensor = tensor_handle.get_tensor();
  auto shape = tensor.shape();

  EXPECT_EQ(shape.rank(), 4);
  EXPECT_EQ(shape[0], 1);
  EXPECT_EQ(shape[1], static_cast<int64_t>(kTestHeight));
  EXPECT_EQ(shape[2], static_cast<int64_t>(kTestWidth));
  EXPECT_EQ(shape[3], kNumChannels);
}

// Test with RGBA8 format (4 channels)
TEST_F(CVCUDATensorHandleTest, FromNitrosImageRGBA8)
{
  constexpr int kRGBA8Channels = 4;
  constexpr uint32_t kRGBA8StepBytes = kTestWidth * kRGBA8Channels * kBytesPerChannel;
  constexpr size_t kRGBA8BufferSize = kRGBA8StepBytes * kTestHeight;

  nitros::NitrosImage img;
  void * ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, kRGBA8BufferSize);
  ASSERT_EQ(err, cudaSuccess);

  // WriteHandle goes out of scope to record write event
  {
    auto write_handle = img.from_external(
      ptr,
      kRGBA8BufferSize,
      kTestWidth,
      kTestHeight,
      kRGBA8StepBytes,
      "rgba8",
      stream_);
  }

  auto read_handle = img.get_read_handle(stream_);

  auto tensor_handle = CVCUDATensorHandle<ReadHandle>::from_nitros_image(
    img,
    std::move(read_handle),
    nvcv::FMT_RGBA8,
    kRGBA8Channels,
    kBytesPerChannel);

  const auto & tensor = tensor_handle.get_tensor();
  auto shape = tensor.shape();

  EXPECT_EQ(shape[0], 1);
  EXPECT_EQ(shape[1], static_cast<int64_t>(kTestHeight));
  EXPECT_EQ(shape[2], static_cast<int64_t>(kTestWidth));
  EXPECT_EQ(shape[3], kRGBA8Channels);
}

// Test get_tensor const accessor
TEST_F(CVCUDATensorHandleTest, GetTensorConstAccessor)
{
  auto img = CreateTestNitrosImage();
  auto read_handle = img.get_read_handle(stream_);

  const auto tensor_handle = CVCUDATensorHandle<ReadHandle>::from_nitros_image(
    img,
    std::move(read_handle),
    nvcv::FMT_RGB8,
    kNumChannels,
    kBytesPerChannel);

  // Access via const reference
  const auto & tensor = tensor_handle.get_tensor();
  EXPECT_EQ(tensor.shape().rank(), 4);
}

// Test get_tensor non-const accessor
TEST_F(CVCUDATensorHandleTest, GetTensorNonConstAccessor)
{
  auto img = CreateTestNitrosImage();
  auto read_handle = img.get_read_handle(stream_);

  auto tensor_handle = CVCUDATensorHandle<ReadHandle>::from_nitros_image(
    img,
    std::move(read_handle),
    nvcv::FMT_RGB8,
    kNumChannels,
    kBytesPerChannel);

  // Access via non-const reference
  auto & tensor = tensor_handle.get_tensor();
  EXPECT_EQ(tensor.shape().rank(), 4);
}

// Test from_nitros_image with NV12 encoding expands tensor height to H*3/2 as expected by CV-CUDA.
TEST_F(CVCUDATensorHandleTest, FromNitrosImageNV12)
{
  constexpr uint32_t kNV12Step = kTestWidth;  // NV12: step = width (single Y channel)
  constexpr uint32_t kNV12Height = kTestHeight;
  constexpr int32_t kExpectedTensorHeight = kNV12Height * 3 / 2;  // Y + UV stacked
  constexpr size_t kNV12BufferSize = kNV12Step * kExpectedTensorHeight;

  nitros::NitrosImage img;
  void * ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, kNV12BufferSize);
  ASSERT_EQ(err, cudaSuccess);

  {
    auto write_handle = img.from_external(
      ptr, kNV12BufferSize, kTestWidth, kNV12Height, kNV12Step, "nv12", stream_);
  }

  auto read_handle = img.get_read_handle(stream_);
  auto tensor_handle = CVCUDATensorHandle<ReadHandle>::from_nitros_image(
    img, std::move(read_handle), nvcv::FMT_Y8, 1, 1);

  const auto & tensor = tensor_handle.get_tensor();
  auto shape = tensor.shape();

  EXPECT_EQ(shape.rank(), 4);
  EXPECT_EQ(shape[0], 1);
  EXPECT_EQ(shape[1], static_cast<int64_t>(kExpectedTensorHeight));
  EXPECT_EQ(shape[2], static_cast<int64_t>(kTestWidth));
  EXPECT_EQ(shape[3], 1);
  EXPECT_EQ(tensor.layout(), nvcv::TENSOR_NHWC);
}

// Test WrapCVCUDATensorNV12 convenience wrapper produces same result
TEST_F(CVCUDATensorHandleTest, WrapCVCUDATensorNV12FreeFunction)
{
  constexpr uint32_t kNV12Step = kTestWidth;
  constexpr uint32_t kNV12Height = kTestHeight;
  constexpr int32_t kExpectedTensorHeight = kNV12Height * 3 / 2;
  constexpr size_t kNV12BufferSize = kNV12Step * kExpectedTensorHeight;

  nitros::NitrosImage img;
  void * ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, kNV12BufferSize);
  ASSERT_EQ(err, cudaSuccess);

  {
    auto write_handle = img.from_external(
      ptr, kNV12BufferSize, kTestWidth, kNV12Height, kNV12Step, "nv12", stream_);
  }

  auto read_handle = img.get_read_handle(stream_);
  auto tensor_handle = WrapCVCUDATensorNV12(img, std::move(read_handle));

  const auto & tensor = tensor_handle.get_tensor();
  auto shape = tensor.shape();

  EXPECT_EQ(shape[0], 1);
  EXPECT_EQ(shape[1], static_cast<int64_t>(kExpectedTensorHeight));
  EXPECT_EQ(shape[2], static_cast<int64_t>(kTestWidth));
  EXPECT_EQ(shape[3], 1);
}

// Verify packed encodings do NOT get height expansion
TEST_F(CVCUDATensorHandleTest, PackedEncodingHeightUnchanged)
{
  auto img = CreateTestNitrosImage();  // rgb8
  auto read_handle = img.get_read_handle(stream_);

  auto tensor_handle = CVCUDATensorHandle<ReadHandle>::from_nitros_image(
    img, std::move(read_handle), nvcv::FMT_RGB8, kNumChannels, kBytesPerChannel);

  const auto & tensor = tensor_handle.get_tensor();
  auto shape = tensor.shape();

  EXPECT_EQ(shape[1], static_cast<int64_t>(kTestHeight));
}

}  // namespace test
}  // namespace cvcuda_utils
}  // namespace isaac_ros
}  // namespace nvidia

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
