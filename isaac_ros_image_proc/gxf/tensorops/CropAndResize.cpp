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
#include "CropAndResize.hpp"
#include "Resize.hpp"

namespace nvidia {
namespace cvcore {
namespace tensor_ops {

namespace detail {

template<::cvcore::ImageType T>
gxf_result_t CropAndResizeImpl(gxf::Entity& output, gxf::Entity& input, const ImageInfo& output_info,
                               const ImageInfo& input_info, const char* output_name, const char* input_name,
                               gxf::Handle<ImageAdapter> output_adapter, gxf::Handle<ImageAdapter> input_adapter,
                               gxf::Handle<gxf::Allocator> allocator, const std::vector<::cvcore::BBox>& src_rois,
                               ::cvcore::tensor_ops::InterpolationType interp_type, cudaStream_t stream) {
  auto input_image = input_adapter->WrapImageFromMessage<T>(input, input_name);
  if (!input_image) {
    return GXF_FAILURE;
  }

  const size_t num_output = src_rois.size();

  for (size_t i = 0; i < num_output; i++) {
    const std::string output_name_i = std::string(output_name) + "_" + std::to_string(i);
    auto error = output_adapter->AddImageToMessage<T>(output, output_info.width, output_info.height, allocator,
                                                      output_info.is_cpu, output_name_i.c_str());
    if (error != GXF_SUCCESS) {
      return GXF_FAILURE;
    }
    auto output_image = output_adapter->WrapImageFromMessage<T>(output, output_name_i.c_str());
    if (!output_image) {
      return GXF_FAILURE;
    }
    ::cvcore::tensor_ops::CropAndResize(output_image.value(), input_image.value(), src_rois[i], interp_type, stream);
  }

  return GXF_SUCCESS;
}

} // namespace detail

gxf_result_t CropAndResize::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;

  result &= registrar->parameter(output_width_, "output_width");
  result &= registrar->parameter(output_height_, "output_height");
  result &= registrar->parameter(interp_type_, "interp_type");
  result &= registrar->parameter(keep_aspect_ratio_, "keep_aspect_ratio");
  result &= registrar->parameter(receiver_bbox_, "receiver_bbox");
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

gxf::Expected<ImageInfo> CropAndResize::doInferOutputInfo(gxf::Entity& input) {
  // Set crop regions
  auto input_bbox_message = receiver_bbox_->receive();
  if (!input_bbox_message) {
    return gxf::Unexpected{GXF_FAILURE};
  }
  auto bbox_tensor = input_bbox_message.value().get<gxf::Tensor>();
  if (!bbox_tensor) {
    return gxf::Unexpected{GXF_FAILURE};
  }
  const gxf::Shape bbox_shape = bbox_tensor.value()->shape();
  if (bbox_shape.rank() != 2 || bbox_shape.dimension(1) != 4) {
    GXF_LOG_ERROR("invalid input bbox dimension.");
    return gxf::Unexpected{GXF_FAILURE};
  }
  const size_t num_bbox = bbox_shape.dimension(0);
  auto bbox_pointer     = bbox_tensor.value()->data<int32_t>();
  if (!bbox_pointer) {
    GXF_LOG_ERROR("empty bbox input.");
    return gxf::Unexpected{GXF_FAILURE};
  }
  std::vector<::cvcore::BBox> rois;
  for (size_t i = 0; i < num_bbox; i++) {
    const int index = i * 4;
    rois.push_back({bbox_pointer.value()[index], bbox_pointer.value()[index + 1], bbox_pointer.value()[index + 2],
                    bbox_pointer.value()[index + 3]});
  }
  input_rois_ = std::move(rois);
  // Check if no-op is needed
  no_op_ = input_rois_.size() == 1 && input_rois_[0].xmin == 0 &&
           input_rois_[0].xmax == static_cast<int>(input_info_.width) && input_rois_[0].ymin == 0 &&
           input_rois_[0].ymax == static_cast<int>(input_info_.height);

  return ImageInfo{input_info_.type, output_width_.get(), output_height_.get(), input_info_.is_cpu};
}

gxf_result_t CropAndResize::doUpdateCameraMessage(gxf::Handle<gxf::CameraModel>& output,
                                                  gxf::Handle<gxf::CameraModel>& input) {
  auto crop_result = GetCroppedCameraModel(*input, input_rois_[0]);
  if (!crop_result) {
    return GXF_FAILURE;
  }
  *output = GetScaledCameraModel(crop_result.value(), output_info_.width, output_info_.height, false).value();
  return GXF_SUCCESS;
}

#define DEFINE_CROP_AND_RESIZE(INPUT_TYPE)                                                                          \
  if (input_info_.type == INPUT_TYPE) {                                                                             \
    return detail::CropAndResizeImpl<INPUT_TYPE>(output, input, output_info_, input_info_, output_name, input_name, \
                                                 output_adapter_.get(), input_adapter_.get(), pool_.get(),          \
                                                 input_rois_, interp.value(), stream);                              \
  }

gxf_result_t CropAndResize::doExecute(gxf::Entity& output, gxf::Entity& input, cudaStream_t stream,
                                      const char* output_name, const char* input_name) {
  GXF_LOG_INFO("execute crop_and_resize.");
  // Check if interpolation type is valid
  auto interp = GetInterpolationType(interp_type_);
  if (!interp) {
    return interp.error();
  }

  // Run the image resizing operation
  DEFINE_CROP_AND_RESIZE(::cvcore::ImageType::Y_U8);
  DEFINE_CROP_AND_RESIZE(::cvcore::ImageType::Y_U16);
  DEFINE_CROP_AND_RESIZE(::cvcore::ImageType::Y_F32);
  DEFINE_CROP_AND_RESIZE(::cvcore::ImageType::RGB_U8);
  DEFINE_CROP_AND_RESIZE(::cvcore::ImageType::RGB_U16);
  DEFINE_CROP_AND_RESIZE(::cvcore::ImageType::RGB_F32);
  DEFINE_CROP_AND_RESIZE(::cvcore::ImageType::BGR_U8);
  DEFINE_CROP_AND_RESIZE(::cvcore::ImageType::BGR_U16);
  DEFINE_CROP_AND_RESIZE(::cvcore::ImageType::BGR_F32);

  // Return error code for unsupported type
  GXF_LOG_ERROR("invalid input/output type for image crop_and_resize.");
  return GXF_FAILURE;
}

} // namespace tensor_ops
} // namespace cvcore
} // namespace nvidia
