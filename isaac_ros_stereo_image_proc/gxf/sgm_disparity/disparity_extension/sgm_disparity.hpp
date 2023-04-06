// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NVIDIA_ISAAC_ROS_EXTENSIONS_SGM_Disparity_HPP_
#define NVIDIA_ISAAC_ROS_EXTENSIONS_SGM_Disparity_HPP_

#include <memory>
#include <queue>
#include <string>

#include "gxf/core/entity.hpp"
#include "gxf/core/gxf.h"
#include "gxf/core/parameter.hpp"
#include "gxf/std/allocator.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/parameter_parser_std.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"

#include "vpi/algo/ConvertImageFormat.h"
#include "vpi/algo/Rescale.h"
#include "vpi/algo/StereoDisparity.h"
#include "vpi/CUDAInterop.h"
#include "vpi/VPI.h"

#define CHECK_STATUS(STMT) \
  do { \
    VPIStatus status = (STMT); \
    if (status != VPI_SUCCESS) { \
      char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH]; \
      vpiGetLastStatusMessage(buffer, sizeof(buffer)); \
      std::ostringstream ss; \
      ss << __FILE__ << ":" << __LINE__ << ": " << vpiStatusGetName(status) << ": " << buffer; \
      GXF_LOG_ERROR(ss.str().c_str()); \
      return GXF_FAILURE; \
    } \
  } while (0);

constexpr uint64_t VPI_BACKEND_XAVIER = VPI_BACKEND_PVA | VPI_BACKEND_NVENC | VPI_BACKEND_VIC;
constexpr uint64_t VPI_BACKEND_ORIN = VPI_BACKEND_OFA | VPI_BACKEND_PVA | VPI_BACKEND_VIC;

namespace nvidia
{
namespace isaac_ros
{

// GXF codelet that subscribe left and right image using videobuffer,
// and publish the disparity using videobuffer format.
class SGMDisparity : public gxf::Codelet
{
public:
  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;
  gxf_result_t registerInterface(gxf::Registrar * registrar) override;

private:
  // The name of the input left image
  gxf::Parameter<std::string> left_image_name_;
  // The name of the input right image
  gxf::Parameter<std::string> right_image_name_;
  // The name of the output video buffer
  gxf::Parameter<std::string> output_name_;
  // Data allocator to create a video buffer
  gxf::Parameter<gxf::Handle<gxf::Allocator>> pool_;
  // Data receiver to get left image data
  gxf::Parameter<gxf::Handle<gxf::Receiver>> left_image_receiver_;
  // Data receiver to get right image data
  gxf::Parameter<gxf::Handle<gxf::Receiver>> right_image_receiver_;
  // Data transmitter to send the data
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> output_transmitter_;
  // The backend for compute disparity
  gxf::Parameter<std::string> vpi_backends_string_;
  // Min and Max disparity search range
  gxf::Parameter<float> min_disparity_;
  gxf::Parameter<float> max_disparity_;
  // Enable the debug message
  gxf::Parameter<bool> debug_;

  // Output size
  size_t output_width_{}, output_height_{};

  // Input size
  size_t input_width_{}, input_height_{};

  // Shared VPI stream for submitting all operations
  VPIStream vpi_stream_{};

  // Left and right VPI images in original input dimensions
  VPIImage left_input_{}, right_input_{};
  VPIImageData left_input_data_, right_input_data_;

  // Left and right VPI images in stereo algorithm-specific format
  VPIImage left_formatted_{}, right_formatted_{};

  // Left and right VPI images in stereo algorithm-specific format and size
  VPIImage left_stereo_{}, right_stereo_{};

  // Raw disparity, resized disparity, and confidence map in VPI-specific format
  VPIImage disparity_raw_{}, disparity_resized_{}, confidence_map_{};

  // Final disparity output in display-friendly format
  VPIImage disparity_{};
  VPIImageData disparity_data_{};

  // VPI algorithm parameters
  VPIConvertImageFormatParams stereo_input_scale_params_{};
  VPIStereoDisparityEstimatorCreationParams disparity_params_{};
  VPIConvertImageFormatParams disparity_scale_params_{};

  // VPI stereo calculation parameters
  VPIPayload stereo_payload_{};
  VPIImageFormat stereo_format_{};

  // VPI backends
  uint64_t vpi_backends_{};
  uint64_t vpi_flags{};

  // Output stereo image dimensions
  int32_t stereo_height_{}, stereo_width_{};

  // Cached values from previous iteration to compare against
  int32_t prev_height_{}, prev_width_{};

  // Special configuration values used for Tegra backend
  const int32_t kTegraSupportedStereoWidth{1920};
  const int32_t kTegraSupportedStereoHeight{1080};
  const int32_t kTegraSupportedScale{256};
};

}  // namespace isaac_ros
}  // namespace nvidia

#endif  // NVIDIA_ISAAC_ROS_EXTENSIONS_SGM_Disparity_HPP_
