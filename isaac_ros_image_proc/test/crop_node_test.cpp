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

#include <gmock/gmock.h>
#include "isaac_ros_image_proc/crop_node.hpp"
#include "rclcpp/rclcpp.hpp"

// Objective: to cover code lines where exceptions are thrown
// Approach: send Invalid Arguments for node parameters to trigger the exception

TEST(crop_node_test, test_invalid_output_dimension)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  options.append_parameter_override("crop_mode", "CENTER");
  options.append_parameter_override("input_width", 0);
  EXPECT_THROW(
  {
    try {
      nvidia::isaac_ros::image_proc::CropNode crop_node(options);
    } catch (const std::invalid_argument & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("Invalid output dimension"));
      throw;
    } catch (const rclcpp::exceptions::InvalidParameterValueException & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("No parameter value set"));
      throw;
    }
  }, std::invalid_argument);
  rclcpp::shutdown();
}

TEST(crop_node_test, test_empty_crop_mode)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  options.append_parameter_override("crop_mode", "");
  options.append_parameter_override("input_width", 1);
  options.append_parameter_override("input_height", 1);
  options.append_parameter_override("crop_width", 1);
  options.append_parameter_override("crop_height", 1);
  EXPECT_THROW(
  {
    try {
      nvidia::isaac_ros::image_proc::CropNode crop_node(options);
    } catch (const std::invalid_argument & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("Crop Mode is not set"));
      throw;
    } catch (const rclcpp::exceptions::InvalidParameterValueException & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("No parameter value set"));
      throw;
    }
  }, std::invalid_argument);
  rclcpp::shutdown();
}

TEST(crop_node_test, test_unsupported_crop_mode)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  options.append_parameter_override("crop_mode", "INVALID");
  options.append_parameter_override("input_width", 1);
  options.append_parameter_override("input_height", 1);
  options.append_parameter_override("crop_width", 1);
  options.append_parameter_override("crop_height", 1);
  EXPECT_THROW(
  {
    try {
      nvidia::isaac_ros::image_proc::CropNode crop_node(options);
    } catch (const std::invalid_argument & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("Unsupported crop mode."));
      throw;
    } catch (const rclcpp::exceptions::InvalidParameterValueException & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("No parameter value set"));
      throw;
    }
  }, std::invalid_argument);
  rclcpp::shutdown();
}


int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
