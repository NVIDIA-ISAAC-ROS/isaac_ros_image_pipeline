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

#include "isaac_ros_vpi_utils/vpi_utilities.hpp"

#include <string>
#include <unordered_map>

#include "rclcpp/rclcpp.hpp"
#include "vpi/VPI.h"

namespace nvidia
{
namespace isaac_ros
{
namespace vpi_utils
{

namespace
{
// Map from human-friendly backend string to VPI backend as uint32_t
const std::unordered_map<std::string, uint32_t> g_str_to_vpi_backend({
          {"CPU", VPI_BACKEND_CPU},
          {"CUDA", VPI_BACKEND_CUDA},
          {"PVA", VPI_BACKEND_PVA},
          {"OFA", VPI_BACKEND_OFA},
          {"VIC", VPI_BACKEND_VIC},
          {"TEGRA", VPI_BACKEND_TEGRA},
          {"JETSON", VPI_BACKEND_JETSON},
          {"ALL", VPI_BACKEND_ALL},
        });
}  // namespace

uint32_t DeclareVPIBackendParameter(rclcpp::Node * node, uint32_t default_backends) noexcept
{
  const std::string DEFAULT_BACKENDS_STRING{""};
  const std::string backends_string{node->declare_parameter("backends", DEFAULT_BACKENDS_STRING)};

  // If the ROS 2 parameter is still at the default value, then return the default backend
  if (backends_string == DEFAULT_BACKENDS_STRING) {
    return default_backends;
  }

  // Final backend to be returned after combining all requested backends
  uint32_t backends{};

  std::stringstream stream{backends_string};
  while (stream.good()) {
    // Extract next backend delimited by commas
    std::string backend_str;
    std::getline(stream, backend_str, ',');

    // Search the map for the backend
    auto backend_it{g_str_to_vpi_backend.find(backend_str)};
    if (backend_it != g_str_to_vpi_backend.end()) {
      // If found, bitwise-or the backend to add it into the allowable backends
      backends |= backend_it->second;
    } else {
      // Otherwise, log an error with all allowable backends
      std::ostringstream os{};
      os << "Backend '" << backend_str << "' from requested backends '" << backends_string <<
        "' not recognized. Backend must be one of:" <<
        std::endl;
      std::for_each(
        g_str_to_vpi_backend.begin(), g_str_to_vpi_backend.end(), [&os](auto & entry) {
          os << entry.first << std::endl;
        });

      RCLCPP_ERROR(node->get_logger(), "%s", os.str().c_str());

      // Return default backends due to error
      return default_backends;
    }
  }

  // Once all backends have been bitwise-or'ed together, return the result
  return backends;
}

VPIFormat ToVpiFormat(const std::string & encoding)
{
  if (encoding == "rgba8") {
    return VPIFormat{VPI_IMAGE_FORMAT_RGBA8, {VPI_PIXEL_TYPE_4U8}};
  } else if (encoding == "bgra8") {
    return VPIFormat{VPI_IMAGE_FORMAT_BGRA8, {VPI_PIXEL_TYPE_4U8}};
  } else if (encoding == "rgb8") {
    return VPIFormat{VPI_IMAGE_FORMAT_RGB8, {VPI_PIXEL_TYPE_3U8}};
  } else if (encoding == "bgr8") {
    return VPIFormat{VPI_IMAGE_FORMAT_BGR8, {VPI_PIXEL_TYPE_3U8}};
  } else if (encoding == "mono8") {
    return VPIFormat{VPI_IMAGE_FORMAT_U8, {VPI_PIXEL_TYPE_U8}};
  } else if (encoding == "mono16") {
    return VPIFormat{VPI_IMAGE_FORMAT_U16, {VPI_PIXEL_TYPE_U16}};
  } else if (encoding == "nv12") {
    return VPIFormat{VPI_IMAGE_FORMAT_NV12, {VPI_PIXEL_TYPE_U8, VPI_PIXEL_TYPE_2U8}};
  } else if (encoding == "nv24") {
    return VPIFormat{VPI_IMAGE_FORMAT_NV24, {VPI_PIXEL_TYPE_U8, VPI_PIXEL_TYPE_2U8}};
  } else if (encoding == "32FC1") {
    return VPIFormat{VPI_IMAGE_FORMAT_F32, {VPI_PIXEL_TYPE_F32}};
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("Isaac ROS VPI utilities"),
        "Unsupported encoding: %s", encoding.c_str());
    throw std::runtime_error("Unsupported encoding: " + encoding);
  }
}

VPIInterpolationType ToVpiInterpolationType(const std::string & interp_type)
{
  if (interp_type == "nearest") {
    return VPI_INTERP_NEAREST;
  } else if (interp_type == "linear") {
    return VPI_INTERP_LINEAR;
  } else if (interp_type == "cubic") {
    return VPI_INTERP_CATMULL_ROM;
  } else {
    throw std::runtime_error("Unsupported interpolation type: " + interp_type);
  }
}

VPIBorderExtension ToVpiBorderType(const std::string & border_type)
{
  if (border_type == "zero") {
    return VPI_BORDER_ZERO;
  } else if (border_type == "clamp") {
    return VPI_BORDER_CLAMP;
  } else if (border_type == "reflect") {
    return VPI_BORDER_REFLECT;
  } else if (border_type == "mirror") {
    return VPI_BORDER_MIRROR;
  } else if (border_type == "limited") {
    return VPI_BORDER_LIMITED;
  } else {
    throw std::runtime_error("Unsupported border type: " + border_type);
  }
}

uint32_t ToVPIBackend(const std::string & backend)
{
  auto backend_it{g_str_to_vpi_backend.find(backend)};
  if (backend_it != g_str_to_vpi_backend.end()) {
    return backend_it->second;
  }
  throw std::runtime_error("Unsupported backend: " + backend);
}

std::string VPIBackendToString(uint32_t backend)
{
  for (const auto & [key, value] : g_str_to_vpi_backend) {
    if (value == backend) {
      return key;
    }
  }
  throw std::runtime_error("Unsupported backend: " + std::to_string(backend));
}

}  // namespace vpi_utils
}  // namespace isaac_ros
}  // namespace nvidia
