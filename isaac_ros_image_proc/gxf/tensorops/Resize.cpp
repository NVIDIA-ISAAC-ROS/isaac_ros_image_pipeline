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
#include "Resize.hpp"

namespace nvidia {
namespace cvcore {
namespace tensor_ops {

namespace detail {

template<::cvcore::ImageType T>
gxf_result_t ResizeImpl(gxf::Entity& output, gxf::Entity& input, const ImageInfo& output_info,
                        const ImageInfo& input_info, const char* output_name, const char* input_name,
                        gxf::Handle<ImageAdapter> output_adapter, gxf::Handle<ImageAdapter> input_adapter,
                        gxf::Handle<gxf::Allocator> allocator, bool keep_aspect_ratio,
                        ::cvcore::tensor_ops::InterpolationType interp_type, cudaStream_t stream) {
  auto input_image = input_adapter->WrapImageFromMessage<T>(input, input_name);
  if (!input_image) {
    return GXF_FAILURE;
  }

  auto error = output_adapter->AddImageToMessage<T>(output, output_info.width, output_info.height, allocator,
                                                    output_info.is_cpu, output_name);
  if (error != GXF_SUCCESS) {
    return GXF_FAILURE;
  }

  auto output_image = output_adapter->WrapImageFromMessage<T>(output, output_name);
  if (!output_image) {
    return GXF_FAILURE;
  }
  ::cvcore::tensor_ops::Resize(output_image.value(), input_image.value(), keep_aspect_ratio, interp_type, stream);
  return GXF_SUCCESS;
}

template<::cvcore::ImageType T>
gxf_result_t ResizeStreamImpl(gxf::Entity& output, gxf::Entity& input, const ImageInfo& output_info,
                              const ImageInfo& input_info, const char* output_name, const char* input_name,
                              gxf::Handle<TensorStream> stream, gxf::Handle<ImageAdapter> output_adapter,
                              gxf::Handle<ImageAdapter> input_adapter, gxf::Handle<gxf::Allocator> allocator,
                              ::cvcore::tensor_ops::InterpolationType interp_type,
                              ::cvcore::tensor_ops::BorderType border_type) {
  auto input_image = input_adapter->WrapImageFromMessage<T>(input, input_name);
  if (!input_image) {
    return GXF_FAILURE;
  }

  auto error = output_adapter->AddImageToMessage<T>(output, output_info.width, output_info.height, allocator,
                                                    output_info.is_cpu, output_name);
  if (error != GXF_SUCCESS) {
    return GXF_FAILURE;
  }

  auto output_image = output_adapter->WrapImageFromMessage<T>(output, output_name);
  if (!output_image) {
    return GXF_FAILURE;
  }

  auto err_code = stream->getStream()->Resize(output_image.value(), input_image.value(), interp_type, border_type);
  if (err_code != ::cvcore::make_error_condition(::cvcore::ErrorCode::SUCCESS)) {
    GXF_LOG_ERROR("resize operation failed.");
    return GXF_FAILURE;
  }

  return GXF_SUCCESS;
}

} // namespace detail

template<bool USE_TENSOR_STREAM>
gxf_result_t ResizeBase<USE_TENSOR_STREAM>::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;

  result &= registrar->parameter(output_width_, "output_width");
  result &= registrar->parameter(output_height_, "output_height");
  result &= registrar->parameter(interp_type_, "interp_type");
  result &= registrar->parameter(border_type_, "border_type");
  result &= registrar->parameter(keep_aspect_ratio_, "keep_aspect_ratio");
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
gxf::Expected<ImageInfo> ResizeBase<USE_TENSOR_STREAM>::doInferOutputInfo(gxf::Entity& input) {
  // Check if no-op is needed
  no_op_ = output_width_.get() == input_info_.width && output_height_.get() == input_info_.height;
  return ImageInfo{input_info_.type, output_width_.get(), output_height_.get(), input_info_.is_cpu};
}

template<bool USE_TENSOR_STREAM>
gxf_result_t ResizeBase<USE_TENSOR_STREAM>::doUpdateCameraMessage(gxf::Handle<gxf::CameraModel>& output,
                                                                  gxf::Handle<gxf::CameraModel>& input) {
  *output = GetScaledCameraModel(*input, output_info_.width, output_info_.height, keep_aspect_ratio_.get()).value();
  return GXF_SUCCESS;
}

#define DEFINE_RESIZE(INPUT_TYPE)                                                                            \
  if (input_info_.type == INPUT_TYPE) {                                                                      \
    return detail::ResizeImpl<INPUT_TYPE>(output, input, output_info_, input_info_, output_name, input_name, \
                                          output_adapter_.get(), input_adapter_.get(), pool_.get(),          \
                                          keep_aspect_ratio_.get(), interp.value(), stream);                 \
  }

#define DEFINE_STREAM_RESIZE(INPUT_TYPE)                                                                            \
  if (input_info_.type == INPUT_TYPE) {                                                                             \
    return detail::ResizeStreamImpl<INPUT_TYPE>(output, input, output_info_, input_info_, output_name, input_name,  \
                                                stream_.try_get().value(), output_adapter_.get(),                   \
                                                input_adapter_.get(), pool_.get(), interp.value(), border.value()); \
  }

template<>
gxf_result_t ResizeBase<true>::doExecute(gxf::Entity& output, gxf::Entity& input, cudaStream_t stream,
                                         const char* output_name, const char* input_name) {
  GXF_LOG_INFO("execute resize.");
  // Check if interpolation type is valid
  auto interp = GetInterpolationType(interp_type_);
  if (!interp) {
    return interp.error();
  }
  auto border = GetBorderType(border_type_);
  if (!border) {
    return border.error();
  }

  // Run the image resizing operation
  DEFINE_STREAM_RESIZE(::cvcore::ImageType::RGB_U8);
  DEFINE_STREAM_RESIZE(::cvcore::ImageType::BGR_U8);
  DEFINE_STREAM_RESIZE(::cvcore::ImageType::NV12);
  DEFINE_STREAM_RESIZE(::cvcore::ImageType::NV24);

  // Return error code for unsupported type
  GXF_LOG_ERROR("invalid input/output type for image resize.");
  return GXF_FAILURE;
}

template<>
gxf_result_t ResizeBase<false>::doExecute(gxf::Entity& output, gxf::Entity& input, cudaStream_t stream,
                                          const char* output_name, const char* input_name) {
  GXF_LOG_INFO("execute resize.");
  // Check if interpolation type is valid
  auto interp = GetInterpolationType(interp_type_);
  if (!interp) {
    return interp.error();
  }

  // Run the image resizing operation
  DEFINE_RESIZE(::cvcore::ImageType::Y_U8);
  DEFINE_RESIZE(::cvcore::ImageType::Y_U16);
  DEFINE_RESIZE(::cvcore::ImageType::Y_F32);
  DEFINE_RESIZE(::cvcore::ImageType::RGB_U8);
  DEFINE_RESIZE(::cvcore::ImageType::RGB_U16);
  DEFINE_RESIZE(::cvcore::ImageType::RGB_F32);
  DEFINE_RESIZE(::cvcore::ImageType::BGR_U8);
  DEFINE_RESIZE(::cvcore::ImageType::BGR_U16);
  DEFINE_RESIZE(::cvcore::ImageType::BGR_F32);

  // Return error code for unsupported type
  GXF_LOG_ERROR("invalid input/output type for image resize.");
  return GXF_FAILURE;
}

template class ResizeBase<true>;
template class ResizeBase<false>;

} // namespace tensor_ops
} // namespace cvcore
} // namespace nvidia
