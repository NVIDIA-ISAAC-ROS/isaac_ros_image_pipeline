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

#include <numeric>

#include "ImageUtils.hpp"
#include "Undistort.hpp"
#include "gxf/multimedia/camera.hpp"

namespace nvidia {
namespace cvcore {
namespace tensor_ops {

namespace detail {

template<::cvcore::ImageType T>
gxf_result_t UndistortImpl(gxf::Entity& output, gxf::Entity& input, const ImageInfo& output_info,
                           const ImageInfo& input_info, const char* output_name, const char* input_name,
                           gxf::Handle<TensorStream> stream, gxf::Handle<ImageAdapter> output_adapter,
                           gxf::Handle<ImageAdapter> input_adapter, gxf::Handle<gxf::Allocator> allocator,
                           ::cvcore::tensor_ops::ImageWarp warp, ::cvcore::tensor_ops::InterpolationType interp_type,
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

  auto err_code = stream->getStream()->Remap(output_image.value(), input_image.value(), warp, interp_type, border_type);
  if (err_code != ::cvcore::make_error_condition(::cvcore::ErrorCode::SUCCESS)) {
    GXF_LOG_ERROR("undistort operation failed.");
    return GXF_FAILURE;
  }

  return GXF_SUCCESS;
}

gxf::Expected<::cvcore::CameraIntrinsics> GetIntrinsicsFromMessage(gxf::Handle<gxf::CameraModel>& camera_model) {
  return ::cvcore::CameraIntrinsics(camera_model->focal_length.x, camera_model->focal_length.y,
                                    camera_model->principal_point.x, camera_model->principal_point.y,
                                    camera_model->skew_value);
}

gxf::Expected<::cvcore::CameraExtrinsics> GetExtrinsicsFromMessage(gxf::Handle<gxf::Pose3D>& pose) {
  float raw_matrix[3][4];
  for (size_t i = 0; i < 9; i++) {
    raw_matrix[i / 3][i % 3] = pose->rotation[i];
  }
  for (size_t i = 0; i < 3; i++) {
    raw_matrix[i][3] = pose->translation[i];
  }
  return ::cvcore::CameraExtrinsics(raw_matrix);
}

gxf::Expected<::cvcore::CameraDistortionModel> GetDistortionsFromMessage(gxf::Handle<gxf::CameraModel>& camera_model) {
  auto distortion_type = GetCameraDistortionType(camera_model->distortion_type);
  if (!distortion_type) {
    return gxf::Unexpected{GXF_FAILURE};
  }
  auto distortion_model = ::cvcore::CameraDistortionModel();
  for (size_t i = 0; i < 8; i++) {
    distortion_model.coefficients[i] = camera_model->distortion_coefficients[i];
  }
  distortion_model.type = distortion_type.value();
  return distortion_model;
}

} // namespace detail

gxf_result_t UndistortBase::start() {
  // Load grid object
  image_grid_.numHorizRegions = regions_width_.get().size();
  image_grid_.numVertRegions  = regions_height_.get().size();
  if (regions_width_.get().size() != horizontal_intervals_.get().size() ||
      regions_height_.get().size() != vertical_intervals_.get().size()) {
    GXF_LOG_ERROR("invalid image grid.");
    return GXF_FAILURE;
  }
  std::copy(regions_width_.get().begin(), regions_width_.get().end(), image_grid_.regionWidth.begin());
  std::copy(regions_height_.get().begin(), regions_height_.get().end(), image_grid_.regionHeight.begin());
  std::copy(horizontal_intervals_.get().begin(), horizontal_intervals_.get().end(), image_grid_.horizInterval.begin());
  std::copy(vertical_intervals_.get().begin(), vertical_intervals_.get().end(), image_grid_.vertInterval.begin());
  output_shape_.x = static_cast<decltype(output_shape_.x)>(
    std::accumulate(image_grid_.regionWidth.begin(), image_grid_.regionWidth.end(), 0));
  output_shape_.y = static_cast<decltype(output_shape_.y)>(
    std::accumulate(image_grid_.regionHeight.begin(), image_grid_.regionHeight.end(), 0));

  // Generate Image Warp if possible
  if (input_camera_model_.try_get() && reference_frame_.try_get()) {
    input_camera_info_ = {input_camera_model_.try_get().value()->getCameraIntrinsics(),
                          reference_frame_.try_get().value()->getCameraExtrinsics(),
                          input_camera_model_.try_get().value()->getDistortionModel()};

    output_camera_intrinsics_ = output_camera_model_.try_get()
                                  ? output_camera_model_.try_get().value()->getCameraIntrinsics()
                                  : input_camera_info_.intrinsic;

    auto err_code = stream_->getStream()->GenerateWarpFromCameraModel(image_warp_, image_grid_, input_camera_info_,
                                                                      output_camera_intrinsics_);
    if (err_code != ::cvcore::make_error_condition(::cvcore::ErrorCode::SUCCESS)) {
      GXF_LOG_ERROR("image warp creation failed.");
      return GXF_FAILURE;
    }
  }

  return GXF_SUCCESS;
}

gxf_result_t UndistortBase::stop() {
  auto err_code = stream_->getStream()->DestroyWarp(image_warp_);
  if (err_code != ::cvcore::make_error_condition(::cvcore::ErrorCode::SUCCESS)) {
    GXF_LOG_ERROR("image warp de-allocation failed.");
    return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}

gxf_result_t UndistortBase::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;

  result &= registrar->parameter(input_camera_model_, "input_camera_model", "", "",
                                 gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(reference_frame_, "reference_frame", "", "", gxf::Registrar::NoDefaultParameter(),
                                 GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(output_camera_model_, "output_camera_model", "", "",
                                 gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(regions_width_, "regions_width");
  result &= registrar->parameter(regions_height_, "regions_height");
  result &= registrar->parameter(horizontal_intervals_, "horizontal_intervals");
  result &= registrar->parameter(vertical_intervals_, "vertical_intervals");
  result &= registrar->parameter(interp_type_, "interp_type");
  result &= registrar->parameter(border_type_, "border_type");
  result &= registrar->parameter(receiver_, "receiver");
  result &= registrar->parameter(transmitter_, "transmitter");
  result &= registrar->parameter(pool_, "pool");
  result &= registrar->parameter(stream_, "stream");
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

gxf::Expected<ImageInfo> UndistortBase::doInferOutputInfo(gxf::Entity& input) {
  // Check if the input distortion type is Perpective
  auto maybe_camera_message = input.get<gxf::CameraModel>();
  if (maybe_camera_message) {
    no_op_ = maybe_camera_message.value()->distortion_type == gxf::DistortionType::Perspective;
  }
  // Output size may vary, but the format must be the same
  return ImageInfo{input_info_.type, static_cast<size_t>(output_shape_.x), static_cast<size_t>(output_shape_.y),
                   input_info_.is_cpu};
}

gxf_result_t UndistortBase::doUpdateCameraMessage(gxf::Handle<gxf::CameraModel>& output,
                                                  gxf::Handle<gxf::CameraModel>& input) {
  *output                   = *input;
  (*output).distortion_type = gxf::DistortionType::Perspective;
  for (size_t i = 0; i < gxf::CameraModel::kMaxDistortionCoefficients; i++) {
    (*output).distortion_coefficients[i] = 0.;
  }
  (*output).dimensions        = output_shape_;
  (*output).focal_length.x    = output_camera_intrinsics_.fx();
  (*output).focal_length.y    = output_camera_intrinsics_.fy();
  (*output).principal_point.x = output_camera_intrinsics_.cx();
  (*output).principal_point.y = output_camera_intrinsics_.cy();
  (*output).skew_value        = output_camera_intrinsics_.skew();
  return GXF_SUCCESS;
}

#define DEFINE_UNDISTORT(INPUT_TYPE)                                                                                  \
  if (input_info_.type == INPUT_TYPE) {                                                                               \
    return detail::UndistortImpl<INPUT_TYPE>(output, input, output_info_, input_info_, output_name, input_name,       \
                                             stream_.get(), output_adapter_.get(), input_adapter_.get(), pool_.get(), \
                                             image_warp_, interp.value(), border.value());                            \
  }

gxf_result_t UndistortBase::doExecute(gxf::Entity& output, gxf::Entity& input, cudaStream_t stream,
                                      const char* output_name, const char* input_name) {
  GXF_LOG_INFO("execute undistort.");
  auto maybe_camera_message = input.get<gxf::CameraModel>();
  auto maybe_pose3d_message = input.get<gxf::Pose3D>();
  if (!maybe_camera_message) {
    if (image_warp_ == nullptr) {
      GXF_LOG_ERROR("no camera information found.");
      return GXF_FAILURE;
    }
  } else {
    auto maybe_intrinsics  = detail::GetIntrinsicsFromMessage(maybe_camera_message.value());
    auto maybe_distortions = detail::GetDistortionsFromMessage(maybe_camera_message.value());
    if (!maybe_intrinsics || !maybe_distortions) {
      return GXF_FAILURE;
    }
    const auto& new_intrinsics  = maybe_intrinsics.value();
    const auto& new_distortions = maybe_distortions.value();

    auto new_extrinsics = maybe_pose3d_message ? detail::GetExtrinsicsFromMessage(maybe_pose3d_message.value()).value()
                                               : ::cvcore::CameraExtrinsics();

    const bool reset = image_warp_ == nullptr || new_intrinsics != input_camera_info_.intrinsic ||
                       new_distortions != input_camera_info_.distortion ||
                       new_extrinsics != input_camera_info_.extrinsic;

    if (reset) {
      auto new_width  = static_cast<float>(output_shape_.x);
      auto new_height = static_cast<float>(output_shape_.y);

      // These two parameters (width_scale and height_scale) can be
      // used to determine a crop or pad regime depending on which dimension to
      // preserve in the case of keep_aspect ratio. In this case, we assume
      // always_crop=True, or that we will always use the largest dimension
      // change.
      auto width_scale  = new_width / static_cast<float>(input_info_.width);
      auto height_scale = new_height / static_cast<float>(input_info_.height);
      auto scale        = std::max({width_scale, height_scale}); // Always crop

      input_camera_info_                           = {new_intrinsics, new_extrinsics, new_distortions};
      output_camera_intrinsics_                    = new_intrinsics;
      output_camera_intrinsics_.m_intrinsics[0][0] = scale * new_intrinsics.m_intrinsics[0][0];
      output_camera_intrinsics_.m_intrinsics[0][1] = scale * new_intrinsics.m_intrinsics[0][1];
      output_camera_intrinsics_.m_intrinsics[1][1] = scale * new_intrinsics.m_intrinsics[1][1];
      output_camera_intrinsics_.m_intrinsics[0][2] = scale * new_intrinsics.m_intrinsics[0][2];
      output_camera_intrinsics_.m_intrinsics[1][2] = scale * new_intrinsics.m_intrinsics[1][2];

      auto err_code = stream_->getStream()->GenerateWarpFromCameraModel(image_warp_, image_grid_, input_camera_info_,
                                                                        output_camera_intrinsics_);
      if (err_code != ::cvcore::make_error_condition(::cvcore::ErrorCode::SUCCESS)) {
        GXF_LOG_ERROR("image warp creation failed.");
        return GXF_FAILURE;
      }
    }
  }

  auto interp = GetInterpolationType(interp_type_);
  if (!interp) {
    return interp.error();
  }
  auto border = GetBorderType(border_type_);
  if (!border) {
    return border.error();
  }

  // Run the image undistortion operation
  DEFINE_UNDISTORT(::cvcore::ImageType::BGR_U8);
  DEFINE_UNDISTORT(::cvcore::ImageType::RGB_U8);
  DEFINE_UNDISTORT(::cvcore::ImageType::NV12);
  DEFINE_UNDISTORT(::cvcore::ImageType::NV24);

  // Return error code for unsupported type
  GXF_LOG_ERROR("invalid input/output type for image undistort.");
  return GXF_FAILURE;
}

} // namespace tensor_ops
} // namespace cvcore
} // namespace nvidia
