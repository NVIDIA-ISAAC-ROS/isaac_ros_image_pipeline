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

#pragma once

#include <sstream>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "vpi/VPI.h"

constexpr uint32_t VPI_BACKEND_JETSON = VPI_BACKEND_OFA | VPI_BACKEND_PVA | VPI_BACKEND_VIC;

// VPI status check macro
#define CHECK_VPI_STATUS(STMT) \
  do { \
    VPIStatus status = (STMT); \
    if (status != VPI_SUCCESS) { \
      char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH]; \
      vpiGetLastStatusMessage(buffer, sizeof(buffer)); \
      std::ostringstream ss; \
      ss << __FILE__ << ":" << __LINE__ << ": " << vpiStatusGetName(status) << ": " << buffer; \
      RCLCPP_ERROR( \
        rclcpp::get_logger( \
          "Isaac ROS VPI utilities"), "Error in VPI method. Error[%s]", ss.str().c_str()); \
      throw std::runtime_error(ss.str()); \
    } \
  } while (0);

namespace nvidia
{
namespace isaac_ros
{
namespace vpi_utils
{
/**
* @brief Declare and parse ROS 2 parameter into VPI backend flags
*
* @param node The node to declare the parameter with
* @param default_backends The default backends to use if given invalid input
* @return uint32_t The resulting VPI backend flags
*/
uint32_t DeclareVPIBackendParameter(rclcpp::Node * node, uint32_t default_backends) noexcept;

// Data structure to hold VPI format information
struct VPIFormat
{
  VPIImageFormat image_format;
  std::vector<VPIPixelType> pixel_type;
};

/**
  * @brief Convert a NitrosImage format into VPI format
  *
  * @param value Input NitrosImage format
  * @return VPIFormat Resulting VPIFormat
  */
VPIFormat ToVpiFormat(const std::string & encoding);

/**
  * @brief Convert a string interpolation type into VPI interpolation type
  *
  * @param interp_type Input string interpolation type
  * @return VPIInterpolationType Resulting VPI interpolation type
  */
VPIInterpolationType ToVpiInterpolationType(const std::string & interp_type);

/**
 * @brief Convert a string border type into VPI border type
  *
  * @param border_type Input string border type
  * @return VPIBorderType Resulting VPI border type
  */
VPIBorderExtension ToVpiBorderType(const std::string & border_type);

/**
 * @brief Convert a string to a VPI backend
 *
 * @param backend Input string backend
 * @return uint32_t Resulting VPI backend
 */
uint32_t ToVPIBackend(const std::string & backend);

/**
 * @brief VPI backend to string
 *
 * @param backend The VPI backend to convert
 * @return std::string Resulting string
 */
std::string VPIBackendToString(uint32_t backend);
}  // namespace vpi_utils
}  // namespace isaac_ros
}  // namespace nvidia
