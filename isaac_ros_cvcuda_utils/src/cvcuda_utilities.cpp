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

#include "isaac_ros_cvcuda_utils/cvcuda_utilities.hpp"

#include <map>
#include <unordered_map>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/image_encodings.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace cvcuda_utils
{
NVCVImageFormat ToNVCVFormat(const std::string & encoding)
{
  static const std::unordered_map<std::string, NVCVImageFormat>
  str_to_nvcv_format({
          {sensor_msgs::image_encodings::RGB8,
            NVCVImageFormat{nvcv::FMT_RGB8, nvcv::FMT_RGBf32, nvcv::FMT_RGBf32p,
              sensor_msgs::image_encodings::TYPE_32FC3}},
          {sensor_msgs::image_encodings::BGR8,
            NVCVImageFormat{nvcv::FMT_BGR8, nvcv::FMT_BGRf32, nvcv::FMT_BGRf32p,
              sensor_msgs::image_encodings::TYPE_32FC3}},
          {sensor_msgs::image_encodings::RGBA8,
            NVCVImageFormat{nvcv::FMT_RGBA8, nvcv::FMT_RGBAf32, nvcv::FMT_RGBAf32p,
              sensor_msgs::image_encodings::TYPE_32FC4}},
          {sensor_msgs::image_encodings::BGRA8,
            NVCVImageFormat{nvcv::FMT_BGRA8, nvcv::FMT_BGRAf32, nvcv::FMT_BGRAf32p,
              sensor_msgs::image_encodings::TYPE_32FC4}},
          {sensor_msgs::image_encodings::MONO8,
            NVCVImageFormat{nvcv::FMT_Y8, nvcv::FMT_F32, nvcv::FMT_F32,
              sensor_msgs::image_encodings::TYPE_32FC1}},
          {sensor_msgs::image_encodings::TYPE_16UC1,
            NVCVImageFormat{nvcv::FMT_Y16, nvcv::FMT_F16, nvcv::FMT_F16,
              sensor_msgs::image_encodings::TYPE_32FC1}},
          {sensor_msgs::image_encodings::MONO16,
            NVCVImageFormat{nvcv::FMT_Y16, nvcv::FMT_F32, nvcv::FMT_F32,
              sensor_msgs::image_encodings::TYPE_32FC1}},
          {sensor_msgs::image_encodings::TYPE_32FC3,
            NVCVImageFormat{nvcv::FMT_RGBf32, nvcv::FMT_RGBf32, nvcv::FMT_RGBf32p,
              sensor_msgs::image_encodings::TYPE_32FC3}},
          {sensor_msgs::image_encodings::TYPE_32FC4,
            NVCVImageFormat{nvcv::FMT_RGBAf32, nvcv::FMT_RGBAf32, nvcv::FMT_RGBAf32p,
              sensor_msgs::image_encodings::TYPE_32FC4}},
        });
  auto it = str_to_nvcv_format.find(encoding);
  if (it == str_to_nvcv_format.end()) {
    RCLCPP_ERROR(rclcpp::get_logger("cvcuda_utilities"), "Unsupported encoding: %s",
      encoding.c_str());
    throw std::invalid_argument("cvcuda_utilities: Unsupported encoding: " + encoding);
  }
  return it->second;
}

NVCVInterpolationType ToNVCVInterpolationType(const std::string & interp_type)
{
  static const std::unordered_map<std::string, NVCVInterpolationType>
  str_to_nvcv_interpolation_type({
          {"nearest", NVCV_INTERP_NEAREST},
          {"linear", NVCV_INTERP_LINEAR},
          {"cubic", NVCV_INTERP_CUBIC}
        });
  auto it = str_to_nvcv_interpolation_type.find(interp_type);
  if (it == str_to_nvcv_interpolation_type.end()) {
    RCLCPP_ERROR(rclcpp::get_logger("cvcuda_utilities"), "Unsupported interpolation type: %s",
      interp_type.c_str());
    throw std::invalid_argument("Unsupported interpolation type: " + interp_type);
  }
  return it->second;
}

int32_t ToNVCVFlipMode(const std::string & flip_mode)
{
  // CVCUDA flip_flag to specify how to flip the array:
  // 0 : flipping around the x-axis (VERTICAL)
  // 1 : flipping around the y-axis (HORIZONTAL)
  // -1 : flipping around both axes (BOTH, 180 degree rotation)
  static const std::unordered_map<std::string, int32_t>
  str_to_nvcv_flip_mode({
          {"VERTICAL", 0},
          {"HORIZONTAL", 1},
          {"BOTH", -1}
        });
  auto it = str_to_nvcv_flip_mode.find(flip_mode);
  if (it == str_to_nvcv_flip_mode.end()) {
    RCLCPP_ERROR(rclcpp::get_logger("cvcuda_utilities"), "Unsupported flip mode: %s",
      flip_mode.c_str());
    throw std::invalid_argument("Unsupported flip mode: " + flip_mode);
  }
  return it->second;
}

NVCVColorConversionCode ToNVCVColorConversionCode(
  const std::string & in_encoding,
  const std::string & out_encoding)
{
  static const std::map<std::pair<nvcv::ImageFormat, nvcv::ImageFormat>,
    NVCVColorConversionCode> str_to_nvcv_color_conversion_code({
          {std::make_pair(nvcv::FMT_RGB8, nvcv::FMT_BGR8), NVCV_COLOR_RGB2BGR},
          {std::make_pair(nvcv::FMT_RGB8, nvcv::FMT_RGBA8), NVCV_COLOR_RGB2RGBA},
          {std::make_pair(nvcv::FMT_RGB8, nvcv::FMT_BGRA8), NVCV_COLOR_RGB2BGRA},
          {std::make_pair(nvcv::FMT_BGR8, nvcv::FMT_RGB8), NVCV_COLOR_BGR2RGB},
          {std::make_pair(nvcv::FMT_BGR8, nvcv::FMT_RGBA8), NVCV_COLOR_BGR2RGBA},
          {std::make_pair(nvcv::FMT_BGR8, nvcv::FMT_BGRA8), NVCV_COLOR_BGR2BGRA},
          {std::make_pair(nvcv::FMT_RGBA8, nvcv::FMT_BGR8), NVCV_COLOR_RGBA2BGR},
          {std::make_pair(nvcv::FMT_RGBA8, nvcv::FMT_RGB8), NVCV_COLOR_RGBA2RGB},
          {std::make_pair(nvcv::FMT_RGBA8, nvcv::FMT_BGRA8), NVCV_COLOR_RGBA2BGRA},
          {std::make_pair(nvcv::FMT_BGRA8, nvcv::FMT_RGB8), NVCV_COLOR_BGRA2RGB},
          {std::make_pair(nvcv::FMT_BGRA8, nvcv::FMT_RGBA8), NVCV_COLOR_BGRA2RGBA},
          {std::make_pair(nvcv::FMT_BGRA8, nvcv::FMT_BGR8), NVCV_COLOR_BGRA2BGR},
          {std::make_pair(nvcv::FMT_RGB8, nvcv::FMT_Y8), NVCV_COLOR_RGB2GRAY},
          {std::make_pair(nvcv::FMT_BGR8, nvcv::FMT_Y8), NVCV_COLOR_BGR2GRAY},
          {std::make_pair(nvcv::FMT_RGBA8, nvcv::FMT_Y8), NVCV_COLOR_RGBA2GRAY},
          {std::make_pair(nvcv::FMT_BGRA8, nvcv::FMT_Y8), NVCV_COLOR_BGRA2GRAY},
          {std::make_pair(nvcv::FMT_Y8, nvcv::FMT_RGB8), NVCV_COLOR_GRAY2RGB},
          {std::make_pair(nvcv::FMT_Y8, nvcv::FMT_BGR8), NVCV_COLOR_GRAY2BGR},
          {std::make_pair(nvcv::FMT_Y8, nvcv::FMT_RGBA8), NVCV_COLOR_GRAY2RGBA},
          {std::make_pair(nvcv::FMT_Y8, nvcv::FMT_BGRA8), NVCV_COLOR_GRAY2BGRA},
        });

  const auto in_format = ToNVCVFormat(in_encoding);
  const auto out_format = ToNVCVFormat(out_encoding);
  auto it = str_to_nvcv_color_conversion_code.find(std::make_pair(in_format.format,
    out_format.format));
  if (it == str_to_nvcv_color_conversion_code.end()) {
    RCLCPP_ERROR(rclcpp::get_logger("cvcuda_utilities"),
      "Unsupported color conversion pair: %s -> %s",
      in_encoding.c_str(), out_encoding.c_str());
    throw std::invalid_argument("Unsupported color conversion pair: " +
      in_encoding + " -> " + out_encoding);
  }
  return it->second;
}

NVCVBorderType ToNVCVBorderType(const std::string & border_type)
{
  static const std::unordered_map<std::string, NVCVBorderType>
  str_to_nvcv_border_type({
          {"CONSTANT", NVCV_BORDER_CONSTANT},
          {"REPLICATE", NVCV_BORDER_REPLICATE},
          {"REFLECT", NVCV_BORDER_REFLECT},
          {"WRAP", NVCV_BORDER_WRAP},
          {"REFLECT101", NVCV_BORDER_REFLECT101}
        });
  auto it = str_to_nvcv_border_type.find(border_type);
  if (it == str_to_nvcv_border_type.end()) {
    RCLCPP_ERROR(rclcpp::get_logger("cvcuda_utilities"), "Unsupported border type: %s",
      border_type.c_str());
    throw std::invalid_argument("Unsupported border type: " + border_type);
  }
  return it->second;
}

NVCVRemapMapValueType ToNVCVRemapMapValueType(const std::string & remap_map_value_type)
{
  static const std::unordered_map<std::string, NVCVRemapMapValueType>
  str_to_nvcv_remap_map_value_type({
          {"ABSOLUTE", NVCV_REMAP_ABSOLUTE},
          {"ABSOLUTE_NORMALIZED", NVCV_REMAP_ABSOLUTE_NORMALIZED},
          {"RELATIVE_NORMALIZED", NVCV_REMAP_RELATIVE_NORMALIZED}
        });
  auto it = str_to_nvcv_remap_map_value_type.find(remap_map_value_type);
  if (it == str_to_nvcv_remap_map_value_type.end()) {
    RCLCPP_ERROR(rclcpp::get_logger("cvcuda_utilities"), "Unsupported remap map value type: %s",
      remap_map_value_type.c_str());
    throw std::invalid_argument("Unsupported remap map value type: " + remap_map_value_type);
  }
  return it->second;
}

nvcv::TensorLayout ToNVCVTensorLayout(const std::string & tensor_layout)
{
  static const std::unordered_map<std::string, nvcv::TensorLayout> str_to_nvcv_tensor_layout({
          {"NHWC", nvcv::TENSOR_NHWC},
          {"NCHW", nvcv::TENSOR_NCHW},
          {"HWC", nvcv::TENSOR_HWC},
          {"CHW", nvcv::TENSOR_CHW},
        });
  auto it = str_to_nvcv_tensor_layout.find(tensor_layout);
  if (it == str_to_nvcv_tensor_layout.end()) {
    RCLCPP_ERROR(rclcpp::get_logger("cvcuda_utilities"), "Unsupported tensor layout: %s",
      tensor_layout.c_str());
    throw std::invalid_argument("Unsupported tensor layout: " + tensor_layout);
  }
  return it->second;
}

nvcv::DataType ToNVCVDataType(const nvidia::isaac_ros::nitros::NitrosDataType & data_type)
{
  static const std::unordered_map<nvidia::isaac_ros::nitros::NitrosDataType,
    nvcv::DataType> data_type_map({
          {nvidia::isaac_ros::nitros::NitrosDataType::kInt8, nvcv::TYPE_S8},
          {nvidia::isaac_ros::nitros::NitrosDataType::kUnsigned8, nvcv::TYPE_U8},
          {nvidia::isaac_ros::nitros::NitrosDataType::kInt16, nvcv::TYPE_S16},
          {nvidia::isaac_ros::nitros::NitrosDataType::kUnsigned16, nvcv::TYPE_U16},
          {nvidia::isaac_ros::nitros::NitrosDataType::kInt32, nvcv::TYPE_S32},
          {nvidia::isaac_ros::nitros::NitrosDataType::kUnsigned32, nvcv::TYPE_U32},
          {nvidia::isaac_ros::nitros::NitrosDataType::kInt64, nvcv::TYPE_S64},
          {nvidia::isaac_ros::nitros::NitrosDataType::kUnsigned64, nvcv::TYPE_U64},
          {nvidia::isaac_ros::nitros::NitrosDataType::kFloat32, nvcv::TYPE_F32},
          {nvidia::isaac_ros::nitros::NitrosDataType::kFloat64, nvcv::TYPE_F64},
        });
  auto it = data_type_map.find(data_type);
  if (it == data_type_map.end()) {
    RCLCPP_ERROR(rclcpp::get_logger("cvcuda_utilities"), "Unsupported data type: %d",
      static_cast<int>(data_type));
    throw std::invalid_argument("Unsupported data type: " +
      std::to_string(static_cast<int>(data_type)));
  }
  return it->second;
}

nvcv::DataType ToNVCVDataType(const nvcv::ImageFormat & image_format)
{
  static const std::map<nvcv::ImageFormat, nvcv::DataType> data_type_map({
          {nvcv::FMT_RGB8, nvcv::TYPE_3U8},
          {nvcv::FMT_BGR8, nvcv::TYPE_3U8},
          {nvcv::FMT_RGBA8, nvcv::TYPE_4U8},
          {nvcv::FMT_BGRA8, nvcv::TYPE_4U8},
          {nvcv::FMT_Y8, nvcv::TYPE_U8},
          {nvcv::FMT_RGBf32, nvcv::TYPE_3F32},
          {nvcv::FMT_BGRf32, nvcv::TYPE_3F32},
          {nvcv::FMT_RGBAf32, nvcv::TYPE_4F32},
          {nvcv::FMT_BGRAf32, nvcv::TYPE_4F32},
          {nvcv::FMT_F32, nvcv::TYPE_F32},
        });
  auto it = data_type_map.find(image_format);
  if (it == data_type_map.end()) {
    RCLCPP_ERROR(rclcpp::get_logger("cvcuda_utilities"), "Unsupported image format: %s",
      nvcvImageFormatGetName(image_format));
    throw std::invalid_argument(std::string("Unsupported image format: ") +
      nvcvImageFormatGetName(image_format));
  }
  return it->second;
}

bool IsMultiplanarEncoding(const std::string & encoding)
{
  return encoding == kEncodingNV12;
}

NVCVColorConversionCode GetNV12ConversionCode(const std::string & out_encoding)
{
  // Only NV12->RGB8/BGR8 are accepted by cvcuda::AdvCvtColor per its
  // OpAdvCvtColor.h docs. RGBA8/BGRA8 NVCV codes exist in the enum but
  // AdvCvtColor rejects them at runtime.
  static const std::unordered_map<std::string, NVCVColorConversionCode>
  nv12_conversion_codes({
          {sensor_msgs::image_encodings::RGB8, NVCV_COLOR_YUV2RGB_NV12},
          {sensor_msgs::image_encodings::BGR8, NVCV_COLOR_YUV2BGR_NV12},
        });
  auto it = nv12_conversion_codes.find(out_encoding);
  if (it == nv12_conversion_codes.end()) {
    const std::string msg =
      "No NV12 conversion code for output encoding: " + out_encoding +
      ". Supported: rgb8, bgr8.";
    RCLCPP_ERROR(rclcpp::get_logger("cvcuda_utilities"), "%s", msg.c_str());
    throw std::invalid_argument(msg);
  }
  return it->second;
}

}  // namespace cvcuda_utils
}  // namespace isaac_ros
}  // namespace nvidia
