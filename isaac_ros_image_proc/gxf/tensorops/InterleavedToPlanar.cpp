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
#include "InterleavedToPlanar.hpp"

namespace nvidia {
namespace cvcore {
namespace tensor_ops {

namespace detail {

template<::cvcore::ImageType T_IN, ::cvcore::ImageType T_OUT>
gxf_result_t InterleavedToPlanarImpl(gxf::Entity& output, gxf::Entity& input, const ImageInfo& output_info,
                                     const ImageInfo& input_info, const char* output_name, const char* input_name,
                                     gxf::Handle<ImageAdapter> output_adapter, gxf::Handle<ImageAdapter> input_adapter,
                                     gxf::Handle<gxf::Allocator> allocator, cudaStream_t stream) {
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
  ::cvcore::tensor_ops::InterleavedToPlanar(output_image.value(), input_image.value(), stream);
  return GXF_SUCCESS;
}

} // namespace detail

gxf_result_t InterleavedToPlanar::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;

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

gxf::Expected<ImageInfo> InterleavedToPlanar::doInferOutputInfo(gxf::Entity& input) {
  // Output type is planar
  ::cvcore::ImageType output_type;
  switch (input_info_.type) {
  case ::cvcore::ImageType::RGB_U8: {
    output_type = ::cvcore::ImageType::PLANAR_RGB_U8;
    break;
  }
  case ::cvcore::ImageType::RGB_U16: {
    output_type = ::cvcore::ImageType::PLANAR_RGB_U16;
    break;
  }
  case ::cvcore::ImageType::RGB_F32: {
    output_type = ::cvcore::ImageType::PLANAR_RGB_F32;
    break;
  }
  case ::cvcore::ImageType::BGR_U8: {
    output_type = ::cvcore::ImageType::PLANAR_BGR_U8;
    break;
  }
  case ::cvcore::ImageType::BGR_U16: {
    output_type = ::cvcore::ImageType::PLANAR_BGR_U16;
    break;
  }
  case ::cvcore::ImageType::BGR_F32: {
    output_type = ::cvcore::ImageType::PLANAR_BGR_F32;
    break;
  }
  case ::cvcore::ImageType::PLANAR_RGB_U8:
  case ::cvcore::ImageType::PLANAR_RGB_U16:
  case ::cvcore::ImageType::PLANAR_RGB_F32:
  case ::cvcore::ImageType::PLANAR_BGR_U8:
  case ::cvcore::ImageType::PLANAR_BGR_U16:
  case ::cvcore::ImageType::PLANAR_BGR_F32: {
    output_type = input_info_.type;
    no_op_      = true;
    break;
  }
  default: {
    GXF_LOG_ERROR("invalid input type for interleaved to planar conversion.");
    return gxf::Unexpected{GXF_FAILURE};
  }
  }
  return ImageInfo{output_type, input_info_.width, input_info_.height, input_info_.is_cpu};
}

gxf_result_t InterleavedToPlanar::doUpdateCameraMessage(gxf::Handle<gxf::CameraModel>& output,
                                                        gxf::Handle<gxf::CameraModel>& input) {
  *output = *input;
  return GXF_SUCCESS;
}

#define DEFINE_INTERLEAVED_TO_PLANAR(INPUT_TYPE, OUTPUT_TYPE)                                                       \
  if (input_info_.type == INPUT_TYPE && output_info_.type == OUTPUT_TYPE) {                                         \
    return detail::InterleavedToPlanarImpl<INPUT_TYPE, OUTPUT_TYPE>(output, input, output_info_, input_info_,       \
                                                                    output_name, input_name, output_adapter_.get(), \
                                                                    input_adapter_.get(), pool_.get(), stream);     \
  }

gxf_result_t InterleavedToPlanar::doExecute(gxf::Entity& output, gxf::Entity& input, cudaStream_t stream,
                                            const char* output_name, const char* input_name) {
  GXF_LOG_INFO("execute interleaved_to_planar conversion");
  // Run the interleaved to planar operation
  DEFINE_INTERLEAVED_TO_PLANAR(::cvcore::ImageType::RGB_U8, ::cvcore::ImageType::PLANAR_RGB_U8);
  DEFINE_INTERLEAVED_TO_PLANAR(::cvcore::ImageType::RGB_U16, ::cvcore::ImageType::PLANAR_RGB_U16);
  DEFINE_INTERLEAVED_TO_PLANAR(::cvcore::ImageType::RGB_F32, ::cvcore::ImageType::PLANAR_RGB_F32);
  DEFINE_INTERLEAVED_TO_PLANAR(::cvcore::ImageType::BGR_U8, ::cvcore::ImageType::PLANAR_BGR_U8);
  DEFINE_INTERLEAVED_TO_PLANAR(::cvcore::ImageType::BGR_U16, ::cvcore::ImageType::PLANAR_BGR_U16);
  DEFINE_INTERLEAVED_TO_PLANAR(::cvcore::ImageType::BGR_F32, ::cvcore::ImageType::PLANAR_BGR_F32);

  // Return error code for unsupported type
  GXF_LOG_ERROR("invalid input/output type for image interleaved to planar conversion.");
  return GXF_FAILURE;
}

} // namespace tensor_ops
} // namespace cvcore
} // namespace nvidia
