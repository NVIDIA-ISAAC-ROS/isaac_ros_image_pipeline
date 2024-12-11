// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "pad_node.hpp"
#include "rclcpp/rclcpp.hpp"

// Objective: to cover code lines where exceptions are thrown
// Approach: send Invalid Arguments for node parameters to trigger the exception

class PadNodeTestSuite : public ::testing::Test
{
protected:
  void SetUp() {rclcpp::init(0, nullptr);}
  void TearDown() {(void)rclcpp::shutdown();}
};


void test_unsupported_padding_type()
{
  rclcpp::NodeOptions options;
  options.arguments(
  {
    "--ros-args",
    "-p", "padding_type:='INVALID'",
  });
  try {
    nvidia::isaac_ros::image_proc::PadNode pad_node(options);
  } catch (const std::invalid_argument & e) {
    std::string err(e.what());
    if (err.find("Unsupported padding type") != std::string::npos) {
      _exit(1);
    }
  }
  _exit(0);
}

void test_unsupported_border_type()
{
  rclcpp::NodeOptions options;
  options.arguments(
  {
    "--ros-args",
    "-p", "padding_type:='CENTER'",
    "-p", "border_type:='INVALID'",
  });
  try {
    nvidia::isaac_ros::image_proc::PadNode pad_node(options);
  } catch (const std::invalid_argument & e) {
    std::string err(e.what());
    if (err.find("Unsupported border type") != std::string::npos) {
      _exit(1);
    }
  }
  _exit(0);
}

void test_invalid_border_pixel_channel_values()
{
  rclcpp::NodeOptions options;
  options.arguments(
  {
    "--ros-args",
    "-p", "padding_type:='CENTER'",
    "-p", "border_type:='CONSTANT'",
    "-p", "border_pixel_color_value:=[0.0, 0.0, 0.0]",
  });
  try {
    nvidia::isaac_ros::image_proc::PadNode pad_node(options);
  } catch (const std::invalid_argument & e) {
    std::string err(e.what());
    if (err.find("Invalid length of border_pixel_channel_values") != std::string::npos) {
      _exit(1);
    }
  }
  _exit(0);
}


TEST_F(PadNodeTestSuite, test_unsupported_padding_type)
{
  EXPECT_EXIT(test_unsupported_padding_type(), testing::ExitedWithCode(1), "");
}

TEST_F(PadNodeTestSuite, test_unsupported_border_type)
{
  EXPECT_EXIT(test_unsupported_border_type(), testing::ExitedWithCode(1), "");
}

TEST_F(PadNodeTestSuite, test_invalid_border_pixel_channel_values)
{
  EXPECT_EXIT(test_invalid_border_pixel_channel_values(), testing::ExitedWithCode(1), "");
}


int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
  return RUN_ALL_TESTS();
}
