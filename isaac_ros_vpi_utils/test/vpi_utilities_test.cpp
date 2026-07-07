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

#include <string>

#include "isaac_ros_vpi_utils/vpi_utilities.hpp"
#include "rclcpp/rclcpp.hpp"
#include "vpi/VPI.h"

namespace nvidia
{
namespace isaac_ros
{
namespace vpi_utils
{
namespace test
{

class VPIUtilitiesTest : public ::testing::Test
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
// ToVpiFormat tests
// ============================================================================

TEST_F(VPIUtilitiesTest, ToVpiFormatRgba8)
{
  auto format = ToVpiFormat("rgba8");
  EXPECT_EQ(format.image_format, VPI_IMAGE_FORMAT_RGBA8);
  ASSERT_EQ(format.pixel_type.size(), 1u);
  EXPECT_EQ(format.pixel_type[0], VPI_PIXEL_TYPE_4U8);
}

TEST_F(VPIUtilitiesTest, ToVpiFormatBgra8)
{
  auto format = ToVpiFormat("bgra8");
  EXPECT_EQ(format.image_format, VPI_IMAGE_FORMAT_BGRA8);
  ASSERT_EQ(format.pixel_type.size(), 1u);
  EXPECT_EQ(format.pixel_type[0], VPI_PIXEL_TYPE_4U8);
}

TEST_F(VPIUtilitiesTest, ToVpiFormatRgb8)
{
  auto format = ToVpiFormat("rgb8");
  EXPECT_EQ(format.image_format, VPI_IMAGE_FORMAT_RGB8);
  ASSERT_EQ(format.pixel_type.size(), 1u);
  EXPECT_EQ(format.pixel_type[0], VPI_PIXEL_TYPE_3U8);
}

TEST_F(VPIUtilitiesTest, ToVpiFormatBgr8)
{
  auto format = ToVpiFormat("bgr8");
  EXPECT_EQ(format.image_format, VPI_IMAGE_FORMAT_BGR8);
  ASSERT_EQ(format.pixel_type.size(), 1u);
  EXPECT_EQ(format.pixel_type[0], VPI_PIXEL_TYPE_3U8);
}

TEST_F(VPIUtilitiesTest, ToVpiFormatMono8)
{
  auto format = ToVpiFormat("mono8");
  EXPECT_EQ(format.image_format, VPI_IMAGE_FORMAT_U8);
  ASSERT_EQ(format.pixel_type.size(), 1u);
  EXPECT_EQ(format.pixel_type[0], VPI_PIXEL_TYPE_U8);
}

TEST_F(VPIUtilitiesTest, ToVpiFormatMono16)
{
  auto format = ToVpiFormat("mono16");
  EXPECT_EQ(format.image_format, VPI_IMAGE_FORMAT_U16);
  ASSERT_EQ(format.pixel_type.size(), 1u);
  EXPECT_EQ(format.pixel_type[0], VPI_PIXEL_TYPE_U16);
}

TEST_F(VPIUtilitiesTest, ToVpiFormatNv12)
{
  auto format = ToVpiFormat("nv12");
  EXPECT_EQ(format.image_format, VPI_IMAGE_FORMAT_NV12);
  ASSERT_EQ(format.pixel_type.size(), 2u);
  EXPECT_EQ(format.pixel_type[0], VPI_PIXEL_TYPE_U8);
  EXPECT_EQ(format.pixel_type[1], VPI_PIXEL_TYPE_2U8);
}

TEST_F(VPIUtilitiesTest, ToVpiFormatNv24)
{
  auto format = ToVpiFormat("nv24");
  EXPECT_EQ(format.image_format, VPI_IMAGE_FORMAT_NV24);
  ASSERT_EQ(format.pixel_type.size(), 2u);
  EXPECT_EQ(format.pixel_type[0], VPI_PIXEL_TYPE_U8);
  EXPECT_EQ(format.pixel_type[1], VPI_PIXEL_TYPE_2U8);
}

TEST_F(VPIUtilitiesTest, ToVpiFormat32FC1)
{
  auto format = ToVpiFormat("32FC1");
  EXPECT_EQ(format.image_format, VPI_IMAGE_FORMAT_F32);
  ASSERT_EQ(format.pixel_type.size(), 1u);
  EXPECT_EQ(format.pixel_type[0], VPI_PIXEL_TYPE_F32);
}

TEST_F(VPIUtilitiesTest, ToVpiFormatUnsupportedThrows)
{
  EXPECT_THROW(ToVpiFormat("unsupported_encoding"), std::runtime_error);
}

TEST_F(VPIUtilitiesTest, ToVpiFormatEmptyStringThrows)
{
  EXPECT_THROW(ToVpiFormat(""), std::runtime_error);
}

// ============================================================================
// ToVpiInterpolationType tests
// ============================================================================

TEST_F(VPIUtilitiesTest, ToVpiInterpolationTypeNearest)
{
  EXPECT_EQ(ToVpiInterpolationType("nearest"), VPI_INTERP_NEAREST);
}

TEST_F(VPIUtilitiesTest, ToVpiInterpolationTypeLinear)
{
  EXPECT_EQ(ToVpiInterpolationType("linear"), VPI_INTERP_LINEAR);
}

TEST_F(VPIUtilitiesTest, ToVpiInterpolationTypeCubic)
{
  EXPECT_EQ(ToVpiInterpolationType("cubic"), VPI_INTERP_CATMULL_ROM);
}

TEST_F(VPIUtilitiesTest, ToVpiInterpolationTypeUnsupportedThrows)
{
  EXPECT_THROW(ToVpiInterpolationType("bilinear"), std::runtime_error);
}

TEST_F(VPIUtilitiesTest, ToVpiInterpolationTypeEmptyStringThrows)
{
  EXPECT_THROW(ToVpiInterpolationType(""), std::runtime_error);
}

// ============================================================================
// ToVpiBorderType tests
// ============================================================================

TEST_F(VPIUtilitiesTest, ToVpiBorderTypeZero)
{
  EXPECT_EQ(ToVpiBorderType("zero"), VPI_BORDER_ZERO);
}

TEST_F(VPIUtilitiesTest, ToVpiBorderTypeClamp)
{
  EXPECT_EQ(ToVpiBorderType("clamp"), VPI_BORDER_CLAMP);
}

TEST_F(VPIUtilitiesTest, ToVpiBorderTypeReflect)
{
  EXPECT_EQ(ToVpiBorderType("reflect"), VPI_BORDER_REFLECT);
}

TEST_F(VPIUtilitiesTest, ToVpiBorderTypeMirror)
{
  EXPECT_EQ(ToVpiBorderType("mirror"), VPI_BORDER_MIRROR);
}

TEST_F(VPIUtilitiesTest, ToVpiBorderTypeLimited)
{
  EXPECT_EQ(ToVpiBorderType("limited"), VPI_BORDER_LIMITED);
}

TEST_F(VPIUtilitiesTest, ToVpiBorderTypeUnsupportedThrows)
{
  EXPECT_THROW(ToVpiBorderType("invalid"), std::runtime_error);
}

TEST_F(VPIUtilitiesTest, ToVpiBorderTypeEmptyStringThrows)
{
  EXPECT_THROW(ToVpiBorderType(""), std::runtime_error);
}

// ============================================================================
// ToVPIBackend tests
// ============================================================================

TEST_F(VPIUtilitiesTest, ToVPIBackendCPU)
{
  EXPECT_EQ(ToVPIBackend("CPU"), VPI_BACKEND_CPU);
}

TEST_F(VPIUtilitiesTest, ToVPIBackendCUDA)
{
  EXPECT_EQ(ToVPIBackend("CUDA"), VPI_BACKEND_CUDA);
}

TEST_F(VPIUtilitiesTest, ToVPIBackendPVA)
{
  EXPECT_EQ(ToVPIBackend("PVA"), VPI_BACKEND_PVA);
}

TEST_F(VPIUtilitiesTest, ToVPIBackendOFA)
{
  EXPECT_EQ(ToVPIBackend("OFA"), VPI_BACKEND_OFA);
}

TEST_F(VPIUtilitiesTest, ToVPIBackendVIC)
{
  EXPECT_EQ(ToVPIBackend("VIC"), VPI_BACKEND_VIC);
}

TEST_F(VPIUtilitiesTest, ToVPIBackendTEGRA)
{
  EXPECT_EQ(ToVPIBackend("TEGRA"), VPI_BACKEND_TEGRA);
}

TEST_F(VPIUtilitiesTest, ToVPIBackendJETSON)
{
  EXPECT_EQ(ToVPIBackend("JETSON"), VPI_BACKEND_JETSON);
}

TEST_F(VPIUtilitiesTest, ToVPIBackendALL)
{
  EXPECT_EQ(ToVPIBackend("ALL"), VPI_BACKEND_ALL);
}

TEST_F(VPIUtilitiesTest, ToVPIBackendUnsupportedThrows)
{
  EXPECT_THROW(ToVPIBackend("invalid_backend"), std::runtime_error);
}

TEST_F(VPIUtilitiesTest, ToVPIBackendEmptyStringThrows)
{
  EXPECT_THROW(ToVPIBackend(""), std::runtime_error);
}

// ============================================================================
// DeclareVPIBackendParameter tests
// ============================================================================

TEST_F(VPIUtilitiesTest, DeclareVPIBackendParameterDefaultReturnsDefaultBackends)
{
  rclcpp::NodeOptions options;
  rclcpp::Node node("test_node", options);
  const uint32_t default_backends = VPI_BACKEND_CPU;
  uint32_t result = DeclareVPIBackendParameter(&node, default_backends);
  EXPECT_EQ(result, default_backends);
}

TEST_F(VPIUtilitiesTest, DeclareVPIBackendParameterSingleBackend)
{
  rclcpp::NodeOptions options;
  options.arguments({"--ros-args", "-p", "backends:=CPU"});
  rclcpp::Node node("test_node", options);
  uint32_t result = DeclareVPIBackendParameter(&node, VPI_BACKEND_CUDA);
  EXPECT_EQ(result, VPI_BACKEND_CPU);
}

TEST_F(VPIUtilitiesTest, DeclareVPIBackendParameterMultipleBackends)
{
  rclcpp::NodeOptions options;
  options.arguments({"--ros-args", "-p", "backends:=CPU,CUDA"});
  rclcpp::Node node("test_node", options);
  uint32_t result = DeclareVPIBackendParameter(&node, VPI_BACKEND_CPU);
  EXPECT_EQ(result, VPI_BACKEND_CPU | VPI_BACKEND_CUDA);
}

TEST_F(VPIUtilitiesTest, DeclareVPIBackendParameterInvalidReturnsDefault)
{
  rclcpp::NodeOptions options;
  options.arguments({"--ros-args", "-p", "backends:=INVALID_BACKEND"});
  rclcpp::Node node("test_node", options);
  const uint32_t default_backends = VPI_BACKEND_CUDA;
  uint32_t result = DeclareVPIBackendParameter(&node, default_backends);
  EXPECT_EQ(result, default_backends);
}

}  // namespace test
}  // namespace vpi_utils
}  // namespace isaac_ros
}  // namespace nvidia

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
