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

#include <memory>
#include <string>
#include <utility>

#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_data_type.hpp"
#include "cvcuda/Types.h"
#include "sensor_msgs/image_encodings.hpp"
#include "nvcv/BorderType.h"
#include "nvcv/Tensor.hpp"
#include "nvcv/TensorLayout.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace cvcuda_utils
{

// ROS Jazzy sensor_msgs::image_encodings defines NV21 and NV24 but not NV12.
constexpr char kEncodingNV12[] = "nv12";

struct NVCVImageFormat
{
  nvcv::ImageFormat format;
  nvcv::ImageFormat float_format;
  nvcv::ImageFormat planar_float_format;
  std::string float_encoding;
};

/**
 * @brief Convert a string encoding into a NVCVImageFormat
 *
 * @param encoding
 * @return NVCVImageFormat
 */
NVCVImageFormat ToNVCVFormat(const std::string & encoding);

/**
 * @brief Convert a string flip mode into a int32_t flip flag
 *
 * @param flip_mode
 * @return int32_t flip flag
 */
int32_t ToNVCVFlipMode(const std::string & flip_mode);

/**
 * @brief Convert a string interpolation type into a NVCVInterpolationType
 *
 * @param interp_type
 * @return NVCVInterpolationType
 */
NVCVInterpolationType ToNVCVInterpolationType(const std::string & interp_type);

/**
 * @brief Convert image encoding pair into a NVCVColorConversionCode
 *
 * @param in_encoding
 * @param out_encoding
 * @return NVCVColorConversionCode
 */
NVCVColorConversionCode ToNVCVColorConversionCode(
  const std::string & in_encoding,
  const std::string & out_encoding);

/**
 * @brief Convert a string border type into a NVCVBorderType
 *
 * @param border_type
 * @return NVCVBorderType
 */
NVCVBorderType ToNVCVBorderType(const std::string & border_type);

/**
 * @brief Convert a string remap map value type into a NVCVRemapMapValueType
 *
 * @param remap_map_value_type
 * @return NVCVRemapMapValueType
 */
NVCVRemapMapValueType ToNVCVRemapMapValueType(const std::string & remap_map_value_type);

/**
 * @brief Convert a string tensor layout into a nvcv::TensorLayout
 *
 * @param tensor_layout
 * @return nvcv::TensorLayout
 */
nvcv::TensorLayout ToNVCVTensorLayout(const std::string & tensor_layout);

/**
 * @brief Convert a string data type into a nvcv::DataType
 *
 * @param data_type
 * @return nvcv::DataType
 */
nvcv::DataType ToNVCVDataType(const nvidia::isaac_ros::nitros::NitrosDataType & data_type);

/**
 * @brief
 *
 * @param data_type
 * @return nvcv::DataType
 */
nvcv::DataType ToNVCVDataType(const  nvcv::ImageFormat & image_format);

/**
 * @brief Check whether an encoding is a semi-planar multiplanar format
 *        that requires special tensor wrapping (e.g. NV12).
 *        Currently only NV12 is supported.
 */
bool IsMultiplanarEncoding(const std::string & encoding);

/**
 * @brief Get the AdvCvtColor conversion code for NV12 input to a packed output.
 *
 * @param out_encoding Target packed encoding (rgb8, bgr8)
 * @return NVCVColorConversionCode for use with cvcuda::AdvCvtColor
 * @throws std::invalid_argument if out_encoding is not rgb8 or bgr8
 */
NVCVColorConversionCode GetNV12ConversionCode(const std::string & out_encoding);
}  // namespace cvcuda_utils
}  // namespace isaac_ros
}  // namespace nvidia
