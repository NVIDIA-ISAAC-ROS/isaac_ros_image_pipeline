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
#include "ConvertColorFormat.hpp"

namespace nvidia {
namespace cvcore {
namespace tensor_ops {

namespace detail {

template<::cvcore::ImageType T_IN, ::cvcore::ImageType T_OUT>
gxf_result_t ConvertColorFormatImpl(gxf::Entity& output, gxf::Entity& input, const ImageInfo& output_info,
                                    const ImageInfo& input_info, const char* output_name, const char* input_name,
                                    gxf::Handle<ImageAdapter> output_adapter, gxf::Handle<ImageAdapter> input_adapter,
                                    gxf::Handle<gxf::Allocator> allocator,
                                    ::cvcore::tensor_ops::ColorConversionType type, cudaStream_t stream) {
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

  ::cvcore::tensor_ops::ConvertColorFormat(output_image.value(), input_image.value(), type, stream);
  return GXF_SUCCESS;
}

template<::cvcore::ImageType T_IN, ::cvcore::ImageType T_OUT>
gxf_result_t ConvertColorFormatStreamImpl(gxf::Entity& output, gxf::Entity& input, const ImageInfo& output_info,
                                          const ImageInfo& input_info, const char* output_name, const char* input_name,
                                          gxf::Handle<TensorStream> stream, gxf::Handle<ImageAdapter> output_adapter,
                                          gxf::Handle<ImageAdapter> input_adapter,
                                          gxf::Handle<gxf::Allocator> allocator) {
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

  auto err_code = stream->getStream()->ColorConvert(output_image.value(), input_image.value());
  if (err_code != ::cvcore::make_error_condition(::cvcore::ErrorCode::SUCCESS)) {
    GXF_LOG_ERROR("color conversion operation failed.");
    return GXF_FAILURE;
  }

  return GXF_SUCCESS;
}

} // namespace detail

template<bool USE_TENSOR_STREAM>
gxf_result_t ConvertColorFormatBase<USE_TENSOR_STREAM>::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;

  result &= registrar->parameter(output_type_, "output_type");
  result &= registrar->parameter(receiver_, "receiver");
  result &= registrar->parameter(transmitter_, "transmitter");
  result &= registrar->parameter(pool_, "pool");
  result &= registrar->parameter(stream_, "stream", "tensor stream", "tensor stream object",
                                 gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
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

template<bool USE_TENSOR_STREAM>
gxf::Expected<ImageInfo> ConvertColorFormatBase<USE_TENSOR_STREAM>::doInferOutputInfo(gxf::Entity& input) {
  // Set output type
  auto output_type = GetImageTypeFromString(output_type_);
  if (!output_type) {
    return gxf::Unexpected{GXF_FAILURE};
  }
  // Check if no-op is needed
  no_op_ = output_type.value() == input_info_.type;
  return ImageInfo{output_type.value(), input_info_.width, input_info_.height, input_info_.is_cpu};
}

template<bool USE_TENSOR_STREAM>
gxf_result_t ConvertColorFormatBase<USE_TENSOR_STREAM>::doUpdateCameraMessage(gxf::Handle<gxf::CameraModel>& output,
                                                                              gxf::Handle<gxf::CameraModel>& input) {
  *output = *input;
  return GXF_SUCCESS;
}

#define DEFINE_CONVERT_COLOR_FORMAT(INPUT_TYPE, OUTPUT_TYPE, CONVERSION_TYPE)                                         \
  if (input_info_.type == INPUT_TYPE && output_info_.type == OUTPUT_TYPE) {                                           \
    return detail::ConvertColorFormatImpl<INPUT_TYPE, OUTPUT_TYPE>(                                                   \
      output, input, output_info_, input_info_, output_name, input_name, output_adapter_.get(), input_adapter_.get(), \
      pool_.get(), CONVERSION_TYPE, stream);                                                                          \
  }

#define DEFINE_STREAM_CONVERT_COLOR_FORMAT(INPUT_TYPE, OUTPUT_TYPE)                                 \
  if (input_info_.type == INPUT_TYPE && output_info_.type == OUTPUT_TYPE) {                         \
    return detail::ConvertColorFormatStreamImpl<INPUT_TYPE, OUTPUT_TYPE>(                           \
      output, input, output_info_, input_info_, output_name, input_name, stream_.try_get().value(), \
      output_adapter_.get(), input_adapter_.get(), pool_.get());                                    \
  }

template<>
gxf_result_t ConvertColorFormatBase<true>::doExecute(gxf::Entity& output, gxf::Entity& input, cudaStream_t stream,
                                                     const char* output_name, const char* input_name) {
  GXF_LOG_INFO("execute convert color format");

  // Run the color conversion operation
  DEFINE_STREAM_CONVERT_COLOR_FORMAT(::cvcore::ImageType::RGB_U8, ::cvcore::ImageType::BGR_U8);
  DEFINE_STREAM_CONVERT_COLOR_FORMAT(::cvcore::ImageType::BGR_U8, ::cvcore::ImageType::RGB_U8);
  DEFINE_STREAM_CONVERT_COLOR_FORMAT(::cvcore::ImageType::NV12, ::cvcore::ImageType::BGR_U8);
  DEFINE_STREAM_CONVERT_COLOR_FORMAT(::cvcore::ImageType::BGR_U8, ::cvcore::ImageType::NV12);
  DEFINE_STREAM_CONVERT_COLOR_FORMAT(::cvcore::ImageType::NV12, ::cvcore::ImageType::RGB_U8);
  DEFINE_STREAM_CONVERT_COLOR_FORMAT(::cvcore::ImageType::RGB_U8, ::cvcore::ImageType::NV12);
  DEFINE_STREAM_CONVERT_COLOR_FORMAT(::cvcore::ImageType::NV24, ::cvcore::ImageType::BGR_U8);
  DEFINE_STREAM_CONVERT_COLOR_FORMAT(::cvcore::ImageType::BGR_U8, ::cvcore::ImageType::NV24);
  DEFINE_STREAM_CONVERT_COLOR_FORMAT(::cvcore::ImageType::NV24, ::cvcore::ImageType::RGB_U8);
  DEFINE_STREAM_CONVERT_COLOR_FORMAT(::cvcore::ImageType::RGB_U8, ::cvcore::ImageType::NV24);

  // Return error code for unsupported type
  GXF_LOG_ERROR("invalid input/output type for image color conversion.");
  return GXF_FAILURE;
}

template<>
gxf_result_t ConvertColorFormatBase<false>::doExecute(gxf::Entity& output, gxf::Entity& input, cudaStream_t stream,
                                                      const char* output_name, const char* input_name) {
  GXF_LOG_INFO("execute convert color format");

  // Run the color conversion operation
  DEFINE_CONVERT_COLOR_FORMAT(::cvcore::ImageType::RGB_U8, ::cvcore::ImageType::BGR_U8,
                              ::cvcore::tensor_ops::ColorConversionType::RGB2BGR);
  DEFINE_CONVERT_COLOR_FORMAT(::cvcore::ImageType::RGB_U16, ::cvcore::ImageType::BGR_U16,
                              ::cvcore::tensor_ops::ColorConversionType::RGB2BGR);
  DEFINE_CONVERT_COLOR_FORMAT(::cvcore::ImageType::RGB_F32, ::cvcore::ImageType::BGR_F32,
                              ::cvcore::tensor_ops::ColorConversionType::RGB2BGR);
  DEFINE_CONVERT_COLOR_FORMAT(::cvcore::ImageType::BGR_U8, ::cvcore::ImageType::RGB_U8,
                              ::cvcore::tensor_ops::ColorConversionType::BGR2RGB);
  DEFINE_CONVERT_COLOR_FORMAT(::cvcore::ImageType::BGR_U16, ::cvcore::ImageType::RGB_U16,
                              ::cvcore::tensor_ops::ColorConversionType::BGR2RGB);
  DEFINE_CONVERT_COLOR_FORMAT(::cvcore::ImageType::BGR_F32, ::cvcore::ImageType::RGB_F32,
                              ::cvcore::tensor_ops::ColorConversionType::BGR2RGB);
  DEFINE_CONVERT_COLOR_FORMAT(::cvcore::ImageType::RGB_U8, ::cvcore::ImageType::Y_U8,
                              ::cvcore::tensor_ops::ColorConversionType::RGB2GRAY);
  DEFINE_CONVERT_COLOR_FORMAT(::cvcore::ImageType::RGB_U16, ::cvcore::ImageType::Y_U16,
                              ::cvcore::tensor_ops::ColorConversionType::RGB2GRAY);
  DEFINE_CONVERT_COLOR_FORMAT(::cvcore::ImageType::RGB_F32, ::cvcore::ImageType::Y_F32,
                              ::cvcore::tensor_ops::ColorConversionType::RGB2GRAY);
  DEFINE_CONVERT_COLOR_FORMAT(::cvcore::ImageType::BGR_U8, ::cvcore::ImageType::Y_U8,
                              ::cvcore::tensor_ops::ColorConversionType::BGR2GRAY);
  DEFINE_CONVERT_COLOR_FORMAT(::cvcore::ImageType::BGR_U16, ::cvcore::ImageType::Y_U16,
                              ::cvcore::tensor_ops::ColorConversionType::BGR2GRAY);
  DEFINE_CONVERT_COLOR_FORMAT(::cvcore::ImageType::BGR_F32, ::cvcore::ImageType::Y_F32,
                              ::cvcore::tensor_ops::ColorConversionType::BGR2GRAY);
  DEFINE_CONVERT_COLOR_FORMAT(::cvcore::ImageType::Y_U8, ::cvcore::ImageType::RGB_U8,
                              ::cvcore::tensor_ops::ColorConversionType::GRAY2RGB);
  DEFINE_CONVERT_COLOR_FORMAT(::cvcore::ImageType::Y_U16, ::cvcore::ImageType::RGB_U16,
                              ::cvcore::tensor_ops::ColorConversionType::GRAY2RGB);
  DEFINE_CONVERT_COLOR_FORMAT(::cvcore::ImageType::Y_F32, ::cvcore::ImageType::RGB_F32,
                              ::cvcore::tensor_ops::ColorConversionType::GRAY2RGB);
  DEFINE_CONVERT_COLOR_FORMAT(::cvcore::ImageType::Y_U8, ::cvcore::ImageType::BGR_U8,
                              ::cvcore::tensor_ops::ColorConversionType::GRAY2BGR);
  DEFINE_CONVERT_COLOR_FORMAT(::cvcore::ImageType::Y_U16, ::cvcore::ImageType::BGR_U16,
                              ::cvcore::tensor_ops::ColorConversionType::GRAY2BGR);
  DEFINE_CONVERT_COLOR_FORMAT(::cvcore::ImageType::Y_F32, ::cvcore::ImageType::BGR_F32,
                              ::cvcore::tensor_ops::ColorConversionType::GRAY2BGR);

  // Return error code for unsupported type
  GXF_LOG_ERROR("invalid input/output type for image color conversion.");
  return GXF_FAILURE;
}

template class ConvertColorFormatBase<true>;
template class ConvertColorFormatBase<false>;

} // namespace tensor_ops
} // namespace cvcore
} // namespace nvidia
