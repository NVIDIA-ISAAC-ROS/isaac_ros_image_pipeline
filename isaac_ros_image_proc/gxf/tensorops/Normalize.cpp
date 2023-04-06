// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "Normalize.hpp"

namespace nvidia {
namespace cvcore {
namespace tensor_ops {

namespace detail {

template<::cvcore::ImageType T_IN, ::cvcore::ImageType T_OUT>
gxf_result_t NormalizeC1Impl(gxf::Entity& output, gxf::Entity& input, const ImageInfo& output_info,
                             const ImageInfo& input_info, const char* output_name, const char* input_name,
                             gxf::Handle<ImageAdapter> output_adapter, gxf::Handle<ImageAdapter> input_adapter,
                             gxf::Handle<gxf::Allocator> allocator, const std::vector<float>& scales,
                             const std::vector<float>& offsets, cudaStream_t stream) {
  if (scales.size() != 1 || offsets.size() != 1) {
    GXF_LOG_ERROR("invalid scales/offsets dimension");
    return GXF_FAILURE;
  }

  auto input_image = input_adapter->WrapImageFromMessage<T_IN>(input, input_name);
  if (!input_image) {
    return GXF_FAILURE;
  }

  auto error = output_adapter->AddImageToMessage<T_OUT>(output, output_info.width, output_info.height, allocator,
                                                        output_info.is_cpu, output_name);
  if (error != GXF_SUCCESS) {
    return GXF_FAILURE;
  }

  auto output_image = output_adapter->WrapImageFromMessage<T_OUT>(output, output_name);
  if (!output_image) {
    return GXF_FAILURE;
  }
  ::cvcore::tensor_ops::Normalize(output_image.value(), input_image.value(), scales[0], offsets[0], stream);
  return GXF_SUCCESS;
}

template<::cvcore::ImageType T_IN, ::cvcore::ImageType T_OUT>
gxf_result_t NormalizeC3Impl(gxf::Entity& output, gxf::Entity& input, const ImageInfo& output_info,
                             const ImageInfo& input_info, const char* output_name, const char* input_name,
                             gxf::Handle<ImageAdapter> output_adapter, gxf::Handle<ImageAdapter> input_adapter,
                             gxf::Handle<gxf::Allocator> allocator, const std::vector<float>& scales,
                             const std::vector<float>& offsets, cudaStream_t stream) {
  if (scales.size() != 3 || offsets.size() != 3) {
    GXF_LOG_ERROR("invalid scales/offsets dimension");
    return GXF_FAILURE;
  }

  auto input_image = input_adapter->WrapImageFromMessage<T_IN>(input, input_name);
  if (!input_image) {
    return GXF_FAILURE;
  }

  auto error = output_adapter->AddImageToMessage<T_OUT>(output, output_info.width, output_info.height, allocator,
                                                        output_info.is_cpu, output_name);
  if (error != GXF_SUCCESS) {
    return GXF_FAILURE;
  }

  auto output_image = output_adapter->WrapImageFromMessage<T_OUT>(output, output_name);
  if (!output_image) {
    return GXF_FAILURE;
  }
  const float scales_value[3]  = {scales[0], scales[1], scales[2]};
  const float offsets_value[3] = {offsets[0], offsets[1], offsets[2]};
  ::cvcore::tensor_ops::Normalize(output_image.value(), input_image.value(), scales_value, offsets_value, stream);
  return GXF_SUCCESS;
}

} // namespace detail

gxf_result_t Normalize::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;

  result &= registrar->parameter(scales_, "scales");
  result &= registrar->parameter(offsets_, "offsets");
  result &= registrar->parameter(receiver_, "receiver");
  result &= registrar->parameter(transmitter_, "transmitter");
  result &= registrar->parameter(pool_, "pool");
  result &= registrar->parameter(stream_pool_, "stream_pool", "cuda stream pool", "cuda stream pool object",
                                 gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(input_adapter_, "input_adapter");
  result &= registrar->parameter(output_adapter_, "output_adapter");
  result &= registrar->parameter(input_name_, "input_name", "input name", "input tensor name",
                                 gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(output_name_, "output_name", "output name", "output tensor name",
                                 gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);

  return gxf::ToResultCode(result);
}

gxf::Expected<ImageInfo> Normalize::doInferOutputInfo(gxf::Entity& input) {
  // Output type is F32
  ::cvcore::ImageType output_type;
  switch (input_info_.type) {
  case ::cvcore::ImageType::Y_U8:
  case ::cvcore::ImageType::Y_U16:
  case ::cvcore::ImageType::Y_F32: {
    output_type = ::cvcore::ImageType::Y_F32;
    break;
  }
  case ::cvcore::ImageType::RGB_U8:
  case ::cvcore::ImageType::RGB_U16:
  case ::cvcore::ImageType::RGB_F32: {
    output_type = ::cvcore::ImageType::RGB_F32;
    break;
  }
  case ::cvcore::ImageType::BGR_U8:
  case ::cvcore::ImageType::BGR_U16:
  case ::cvcore::ImageType::BGR_F32: {
    output_type = ::cvcore::ImageType::BGR_F32;
    break;
  }
  default: {
    GXF_LOG_ERROR("invalid input type for normalize.");
    return gxf::Unexpected{GXF_FAILURE};
  }
  }
  // Operation must be performed under any condition
  no_op_ = false;
  return ImageInfo{output_type, input_info_.width, input_info_.height, input_info_.is_cpu};
}

gxf_result_t Normalize::doUpdateCameraMessage(gxf::Handle<gxf::CameraModel>& output,
                                              gxf::Handle<gxf::CameraModel>& input) {
  *output = *input;
  return GXF_SUCCESS;
}

#define DEFINE_NORMALIZE_C1(INPUT_TYPE, OUTPUT_TYPE)                                                                 \
  if (input_info_.type == INPUT_TYPE && output_info_.type == OUTPUT_TYPE) {                                          \
    return detail::NormalizeC1Impl<INPUT_TYPE, OUTPUT_TYPE>(output, input, output_info_, input_info_, output_name,   \
                                                            input_name, output_adapter_.get(), input_adapter_.get(), \
                                                            pool_.get(), scales_.get(), offsets_.get(), stream);     \
  }

#define DEFINE_NORMALIZE_C3(INPUT_TYPE, OUTPUT_TYPE)                                                                 \
  if (input_info_.type == INPUT_TYPE && output_info_.type == OUTPUT_TYPE) {                                          \
    return detail::NormalizeC3Impl<INPUT_TYPE, OUTPUT_TYPE>(output, input, output_info_, input_info_, output_name,   \
                                                            input_name, output_adapter_.get(), input_adapter_.get(), \
                                                            pool_.get(), scales_.get(), offsets_.get(), stream);     \
  }

gxf_result_t Normalize::doExecute(gxf::Entity& output, gxf::Entity& input, cudaStream_t stream, const char* output_name,
                                  const char* input_name) {
  GXF_LOG_INFO("execute normalize");

  // Run the image normalization operation
  DEFINE_NORMALIZE_C1(::cvcore::ImageType::Y_U8, ::cvcore::ImageType::Y_F32);
  DEFINE_NORMALIZE_C1(::cvcore::ImageType::Y_U16, ::cvcore::ImageType::Y_F32);
  DEFINE_NORMALIZE_C1(::cvcore::ImageType::Y_F32, ::cvcore::ImageType::Y_F32);
  DEFINE_NORMALIZE_C3(::cvcore::ImageType::RGB_U8, ::cvcore::ImageType::RGB_F32);
  DEFINE_NORMALIZE_C3(::cvcore::ImageType::RGB_U16, ::cvcore::ImageType::RGB_F32);
  DEFINE_NORMALIZE_C3(::cvcore::ImageType::RGB_F32, ::cvcore::ImageType::RGB_F32);
  DEFINE_NORMALIZE_C3(::cvcore::ImageType::BGR_U8, ::cvcore::ImageType::BGR_F32);
  DEFINE_NORMALIZE_C3(::cvcore::ImageType::BGR_U16, ::cvcore::ImageType::BGR_F32);
  DEFINE_NORMALIZE_C3(::cvcore::ImageType::BGR_F32, ::cvcore::ImageType::BGR_F32);

  // Return error code for unsupported type
  GXF_LOG_ERROR("invalid input/output type for image normalize.");
  return GXF_FAILURE;
}

} // namespace tensor_ops
} // namespace cvcore
} // namespace nvidia
