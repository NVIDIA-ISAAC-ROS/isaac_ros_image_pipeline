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
#include "crop_node.hpp"
#include "rclcpp/rclcpp.hpp"

// Objective: to cover code lines where exceptions are thrown
// Approach: send Invalid Arguments for node parameters to trigger the exception

class CropNodeTestSuite : public ::testing::Test
{
protected:
  void SetUp() {rclcpp::init(0, nullptr);}
  void TearDown() {(void)rclcpp::shutdown();}
};


void test_invalid_output_dimension()
{
  rclcpp::NodeOptions options;
  options.arguments(
  {
    "--ros-args",
    "-p", "input_width:=0",
    "-p", "crop_mode:='CENTER'",
  });
  try {
    nvidia::isaac_ros::image_proc::CropNode crop_node(options);
  } catch (const std::invalid_argument & e) {
    std::string err(e.what());
    if (err.find("Invalid output dimension") != std::string::npos) {
      _exit(1);
    }
  }
  _exit(0);
}

void test_empty_crop_mode()
{
  rclcpp::NodeOptions options;
  options.arguments(
  {
    "--ros-args",
    "-p", "input_width:=1",
    "-p", "input_height:=1",
    "-p", "crop_width:=1",
    "-p", "crop_height:=1",
  });
  try {
    nvidia::isaac_ros::image_proc::CropNode crop_node(options);
  } catch (const std::invalid_argument & e) {
    std::string err(e.what());
    if (err.find("Crop Mode is not set") != std::string::npos) {
      _exit(1);
    }
  }
  _exit(0);
}

void test_unsupported_crop_mode()
{
  rclcpp::NodeOptions options;
  options.arguments(
  {
    "--ros-args",
    "-p", "input_width:=1",
    "-p", "input_height:=1",
    "-p", "crop_width:=1",
    "-p", "crop_height:=1",
    "-p", "crop_mode:='INVALID'",
  });
  try {
    nvidia::isaac_ros::image_proc::CropNode crop_node(options);
  } catch (const std::invalid_argument & e) {
    std::string err(e.what());
    if (err.find("Unsupported crop mode.") != std::string::npos) {
      _exit(1);
    }
  }
  _exit(0);
}


TEST_F(CropNodeTestSuite, test_invalid_output_dimension)
{
  EXPECT_EXIT(test_invalid_output_dimension(), testing::ExitedWithCode(1), "");
}

TEST_F(CropNodeTestSuite, test_empty_crop_mode)
{
  EXPECT_EXIT(test_empty_crop_mode(), testing::ExitedWithCode(1), "");
}

TEST_F(CropNodeTestSuite, test_unsupported_crop_mode)
{
  EXPECT_EXIT(test_unsupported_crop_mode(), testing::ExitedWithCode(1), "");
}


int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
  return RUN_ALL_TESTS();
}
