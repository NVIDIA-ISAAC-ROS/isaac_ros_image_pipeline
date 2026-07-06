// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string>

#include "isaac_ros_cvcuda_utils/cvcuda_utilities.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/image_encodings.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace cvcuda_utils
{
namespace test
{

class CVCUDAUtilitiesTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    rclcpp::init(0, nullptr);
  }

  void TearDown() override
  {
    rclcpp::shutdown();
  }
};

// ============================================================================
// ToNVCVFormat tests
// ============================================================================

TEST_F(CVCUDAUtilitiesTest, ToNVCVFormatRGB8)
{
  auto format = ToNVCVFormat(sensor_msgs::image_encodings::RGB8);
  EXPECT_EQ(format.format, nvcv::FMT_RGB8);
  EXPECT_EQ(format.float_format, nvcv::FMT_RGBf32);
  EXPECT_EQ(format.float_encoding, sensor_msgs::image_encodings::TYPE_32FC3);
}

TEST_F(CVCUDAUtilitiesTest, ToNVCVFormatBGR8)
{
  auto format = ToNVCVFormat(sensor_msgs::image_encodings::BGR8);
  EXPECT_EQ(format.format, nvcv::FMT_BGR8);
  EXPECT_EQ(format.float_format, nvcv::FMT_BGRf32);
  EXPECT_EQ(format.float_encoding, sensor_msgs::image_encodings::TYPE_32FC3);
}

TEST_F(CVCUDAUtilitiesTest, ToNVCVFormatRGBA8)
{
  auto format = ToNVCVFormat(sensor_msgs::image_encodings::RGBA8);
  EXPECT_EQ(format.format, nvcv::FMT_RGBA8);
  EXPECT_EQ(format.float_format, nvcv::FMT_RGBAf32);
  EXPECT_EQ(format.float_encoding, sensor_msgs::image_encodings::TYPE_32FC4);
}

TEST_F(CVCUDAUtilitiesTest, ToNVCVFormatBGRA8)
{
  auto format = ToNVCVFormat(sensor_msgs::image_encodings::BGRA8);
  EXPECT_EQ(format.format, nvcv::FMT_BGRA8);
  EXPECT_EQ(format.float_format, nvcv::FMT_BGRAf32);
  EXPECT_EQ(format.float_encoding, sensor_msgs::image_encodings::TYPE_32FC4);
}

TEST_F(CVCUDAUtilitiesTest, ToNVCVFormatMONO8)
{
  auto format = ToNVCVFormat(sensor_msgs::image_encodings::MONO8);
  EXPECT_EQ(format.format, nvcv::FMT_Y8);
  EXPECT_EQ(format.float_format, nvcv::FMT_F32);
  EXPECT_EQ(format.float_encoding, sensor_msgs::image_encodings::TYPE_32FC1);
}

TEST_F(CVCUDAUtilitiesTest, ToNVCVFormat32FC3)
{
  auto format = ToNVCVFormat(sensor_msgs::image_encodings::TYPE_32FC3);
  EXPECT_EQ(format.format, nvcv::FMT_RGBf32);
  EXPECT_EQ(format.float_format, nvcv::FMT_RGBf32);
  EXPECT_EQ(format.float_encoding, sensor_msgs::image_encodings::TYPE_32FC3);
}

TEST_F(CVCUDAUtilitiesTest, ToNVCVFormatUnsupportedThrows)
{
  EXPECT_THROW(ToNVCVFormat("unsupported_encoding"), std::invalid_argument);
}

TEST_F(CVCUDAUtilitiesTest, ToNVCVFormatEmptyStringThrows)
{
  EXPECT_THROW(ToNVCVFormat(""), std::invalid_argument);
}

// ============================================================================
// ToNVCVInterpolationType tests
// ============================================================================

TEST_F(CVCUDAUtilitiesTest, ToNVCVInterpolationTypeNearest)
{
  auto interp = ToNVCVInterpolationType("nearest");
  EXPECT_EQ(interp, NVCV_INTERP_NEAREST);
}

TEST_F(CVCUDAUtilitiesTest, ToNVCVInterpolationTypeLinear)
{
  auto interp = ToNVCVInterpolationType("linear");
  EXPECT_EQ(interp, NVCV_INTERP_LINEAR);
}

TEST_F(CVCUDAUtilitiesTest, ToNVCVInterpolationTypeCubic)
{
  auto interp = ToNVCVInterpolationType("cubic");
  EXPECT_EQ(interp, NVCV_INTERP_CUBIC);
}

TEST_F(CVCUDAUtilitiesTest, ToNVCVInterpolationTypeUnsupportedThrows)
{
  EXPECT_THROW(ToNVCVInterpolationType("bilinear"), std::invalid_argument);
}

TEST_F(CVCUDAUtilitiesTest, ToNVCVInterpolationTypeEmptyStringThrows)
{
  EXPECT_THROW(ToNVCVInterpolationType(""), std::invalid_argument);
}

TEST_F(CVCUDAUtilitiesTest, ToNVCVInterpolationTypeCaseSensitive)
{
  // Should throw because "Nearest" != "nearest"
  EXPECT_THROW(ToNVCVInterpolationType("Nearest"), std::invalid_argument);
  EXPECT_THROW(ToNVCVInterpolationType("LINEAR"), std::invalid_argument);
  EXPECT_THROW(ToNVCVInterpolationType("CUBIC"), std::invalid_argument);
}

// ============================================================================
// ToNVCVColorConversionCode tests
// ============================================================================

TEST_F(CVCUDAUtilitiesTest, ToNVCVColorConversionCodeRGB2BGR)
{
  auto code = ToNVCVColorConversionCode(
    sensor_msgs::image_encodings::RGB8,
    sensor_msgs::image_encodings::BGR8);
  EXPECT_EQ(code, NVCV_COLOR_RGB2BGR);
}

TEST_F(CVCUDAUtilitiesTest, ToNVCVColorConversionCodeBGR2RGB)
{
  auto code = ToNVCVColorConversionCode(
    sensor_msgs::image_encodings::BGR8,
    sensor_msgs::image_encodings::RGB8);
  EXPECT_EQ(code, NVCV_COLOR_BGR2RGB);
}

TEST_F(CVCUDAUtilitiesTest, ToNVCVColorConversionCodeRGB2RGBA)
{
  auto code = ToNVCVColorConversionCode(
    sensor_msgs::image_encodings::RGB8,
    sensor_msgs::image_encodings::RGBA8);
  EXPECT_EQ(code, NVCV_COLOR_RGB2RGBA);
}

TEST_F(CVCUDAUtilitiesTest, ToNVCVColorConversionCodeRGB2BGRA)
{
  auto code = ToNVCVColorConversionCode(
    sensor_msgs::image_encodings::RGB8,
    sensor_msgs::image_encodings::BGRA8);
  EXPECT_EQ(code, NVCV_COLOR_RGB2BGRA);
}

TEST_F(CVCUDAUtilitiesTest, ToNVCVColorConversionCodeBGR2RGBA)
{
  auto code = ToNVCVColorConversionCode(
    sensor_msgs::image_encodings::BGR8,
    sensor_msgs::image_encodings::RGBA8);
  EXPECT_EQ(code, NVCV_COLOR_BGR2RGBA);
}

TEST_F(CVCUDAUtilitiesTest, ToNVCVColorConversionCodeBGR2BGRA)
{
  auto code = ToNVCVColorConversionCode(
    sensor_msgs::image_encodings::BGR8,
    sensor_msgs::image_encodings::BGRA8);
  EXPECT_EQ(code, NVCV_COLOR_BGR2BGRA);
}

TEST_F(CVCUDAUtilitiesTest, ToNVCVColorConversionCodeRGBA2RGB)
{
  auto code = ToNVCVColorConversionCode(
    sensor_msgs::image_encodings::RGBA8,
    sensor_msgs::image_encodings::RGB8);
  EXPECT_EQ(code, NVCV_COLOR_RGBA2RGB);
}

TEST_F(CVCUDAUtilitiesTest, ToNVCVColorConversionCodeRGBA2BGR)
{
  auto code = ToNVCVColorConversionCode(
    sensor_msgs::image_encodings::RGBA8,
    sensor_msgs::image_encodings::BGR8);
  EXPECT_EQ(code, NVCV_COLOR_RGBA2BGR);
}

TEST_F(CVCUDAUtilitiesTest, ToNVCVColorConversionCodeRGBA2BGRA)
{
  auto code = ToNVCVColorConversionCode(
    sensor_msgs::image_encodings::RGBA8,
    sensor_msgs::image_encodings::BGRA8);
  EXPECT_EQ(code, NVCV_COLOR_RGBA2BGRA);
}

TEST_F(CVCUDAUtilitiesTest, ToNVCVColorConversionCodeBGRA2RGB)
{
  auto code = ToNVCVColorConversionCode(
    sensor_msgs::image_encodings::BGRA8,
    sensor_msgs::image_encodings::RGB8);
  EXPECT_EQ(code, NVCV_COLOR_BGRA2RGB);
}

TEST_F(CVCUDAUtilitiesTest, ToNVCVColorConversionCodeBGRA2BGR)
{
  auto code = ToNVCVColorConversionCode(
    sensor_msgs::image_encodings::BGRA8,
    sensor_msgs::image_encodings::BGR8);
  EXPECT_EQ(code, NVCV_COLOR_BGRA2BGR);
}

TEST_F(CVCUDAUtilitiesTest, ToNVCVColorConversionCodeBGRA2RGBA)
{
  auto code = ToNVCVColorConversionCode(
    sensor_msgs::image_encodings::BGRA8,
    sensor_msgs::image_encodings::RGBA8);
  EXPECT_EQ(code, NVCV_COLOR_BGRA2RGBA);
}

// Grayscale conversions
TEST_F(CVCUDAUtilitiesTest, ToNVCVColorConversionCodeRGB2GRAY)
{
  auto code = ToNVCVColorConversionCode(
    sensor_msgs::image_encodings::RGB8,
    sensor_msgs::image_encodings::MONO8);
  EXPECT_EQ(code, NVCV_COLOR_RGB2GRAY);
}

TEST_F(CVCUDAUtilitiesTest, ToNVCVColorConversionCodeBGR2GRAY)
{
  auto code = ToNVCVColorConversionCode(
    sensor_msgs::image_encodings::BGR8,
    sensor_msgs::image_encodings::MONO8);
  EXPECT_EQ(code, NVCV_COLOR_BGR2GRAY);
}

TEST_F(CVCUDAUtilitiesTest, ToNVCVColorConversionCodeRGBA2GRAY)
{
  auto code = ToNVCVColorConversionCode(
    sensor_msgs::image_encodings::RGBA8,
    sensor_msgs::image_encodings::MONO8);
  EXPECT_EQ(code, NVCV_COLOR_RGBA2GRAY);
}

TEST_F(CVCUDAUtilitiesTest, ToNVCVColorConversionCodeBGRA2GRAY)
{
  auto code = ToNVCVColorConversionCode(
    sensor_msgs::image_encodings::BGRA8,
    sensor_msgs::image_encodings::MONO8);
  EXPECT_EQ(code, NVCV_COLOR_BGRA2GRAY);
}

TEST_F(CVCUDAUtilitiesTest, ToNVCVColorConversionCodeGRAY2RGB)
{
  auto code = ToNVCVColorConversionCode(
    sensor_msgs::image_encodings::MONO8,
    sensor_msgs::image_encodings::RGB8);
  EXPECT_EQ(code, NVCV_COLOR_GRAY2RGB);
}

TEST_F(CVCUDAUtilitiesTest, ToNVCVColorConversionCodeGRAY2BGR)
{
  auto code = ToNVCVColorConversionCode(
    sensor_msgs::image_encodings::MONO8,
    sensor_msgs::image_encodings::BGR8);
  EXPECT_EQ(code, NVCV_COLOR_GRAY2BGR);
}

TEST_F(CVCUDAUtilitiesTest, ToNVCVColorConversionCodeGRAY2RGBA)
{
  auto code = ToNVCVColorConversionCode(
    sensor_msgs::image_encodings::MONO8,
    sensor_msgs::image_encodings::RGBA8);
  EXPECT_EQ(code, NVCV_COLOR_GRAY2RGBA);
}

TEST_F(CVCUDAUtilitiesTest, ToNVCVColorConversionCodeGRAY2BGRA)
{
  auto code = ToNVCVColorConversionCode(
    sensor_msgs::image_encodings::MONO8,
    sensor_msgs::image_encodings::BGRA8);
  EXPECT_EQ(code, NVCV_COLOR_GRAY2BGRA);
}

// Error cases
TEST_F(CVCUDAUtilitiesTest, ToNVCVColorConversionCodeUnsupportedThrows)
{
  // Same format should throw (no conversion needed)
  EXPECT_THROW(
    ToNVCVColorConversionCode(
      sensor_msgs::image_encodings::RGB8,
      sensor_msgs::image_encodings::RGB8),
    std::invalid_argument);
}

TEST_F(CVCUDAUtilitiesTest, ToNVCVColorConversionCodeInvalidEncodingThrows)
{
  EXPECT_THROW(
    ToNVCVColorConversionCode("invalid_encoding", sensor_msgs::image_encodings::RGB8),
    std::invalid_argument);

  EXPECT_THROW(
    ToNVCVColorConversionCode(sensor_msgs::image_encodings::RGB8, "invalid_encoding"),
    std::invalid_argument);
}

TEST_F(CVCUDAUtilitiesTest, GetNV12ConversionCode)
{
  EXPECT_EQ(
    GetNV12ConversionCode(sensor_msgs::image_encodings::RGB8),
    NVCV_COLOR_YUV2RGB_NV12);
  EXPECT_EQ(
    GetNV12ConversionCode(sensor_msgs::image_encodings::BGR8),
    NVCV_COLOR_YUV2BGR_NV12);
  // RGBA8/BGRA8/MONO8 are unsupported by cvcuda::AdvCvtColor for NV12 input
  EXPECT_THROW(GetNV12ConversionCode(sensor_msgs::image_encodings::RGBA8), std::invalid_argument);
  EXPECT_THROW(GetNV12ConversionCode(sensor_msgs::image_encodings::BGRA8), std::invalid_argument);
  EXPECT_THROW(GetNV12ConversionCode(sensor_msgs::image_encodings::MONO8), std::invalid_argument);
  EXPECT_THROW(GetNV12ConversionCode("nv12"), std::invalid_argument);
  EXPECT_THROW(GetNV12ConversionCode(""), std::invalid_argument);
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
