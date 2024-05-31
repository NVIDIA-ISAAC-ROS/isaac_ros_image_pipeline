// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "BBoxGenerator.hpp"

#include <random>

#include "gxf/std/tensor.hpp"

namespace nvidia {
namespace isaac {
namespace tensor_ops {

gxf_result_t BBoxGenerator::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;

  result &= registrar->parameter(
    image_width_, "image_width",
    "The width of the image", "");
  result &= registrar->parameter(
    image_height_, "image_height",
    "The height of the image", "");
  result &= registrar->parameter(
    bbox_count_, "bbox_count",
    "The number of the output bboxes", "");
  result &= registrar->parameter(
    bbox_width_, "bbox_width",
    "Width of the output bbox", "",
    gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(
    bbox_height_, "bbox_height",
    "Height of the output bbox", "",
    gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(
    bbox_loc_x_, "bbox_loc_x",
    "x coordinates of left top corner", "",
    gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(
    bbox_loc_y_, "bbox_loc_y",
    "y coordinates of left top corner", "",
    gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(
    pool_, "pool",
    "Data pool to allocate memory for gxf tensors", "");
  result &= registrar->parameter(
    transmitter_, "transmitter",
    "Transmitter to send the data", "");

  return gxf::ToResultCode(result);
}

gxf_result_t BBoxGenerator::start() {
  srand(1);
  return GXF_SUCCESS;
}

gxf_result_t BBoxGenerator::tick() {
  // Creating GXF tensor to hold the output tensor
  const size_t num_bbox                    = bbox_count_;
  gxf::Expected<gxf::Entity> outputMessage = gxf::CreateTensorMap(
      context(), pool_,
      {{"", gxf::MemoryStorageType::kHost,
      {static_cast<int>(num_bbox), 4},
      gxf::PrimitiveType::kInt32}});
  // Check for errors
  if (!outputMessage)
    return GXF_FAILURE;

  // Getting the pointer to the GXF tensor
  auto output_tensor = outputMessage.value().get<gxf::Tensor>();
  // Check for errors
  if (!output_tensor)
    return GXF_FAILURE;

  // Fill in random bboxes
  const size_t image_width  = image_width_;
  const size_t image_height = image_height_;
  auto bbox_pointer         = output_tensor.value()->data<int32_t>();

  if (!bbox_pointer) {
    GXF_LOG_ERROR("empty bbox input.");
    return GXF_FAILURE;
  }

  auto bbox_width       = bbox_width_.try_get();
  auto bbox_height      = bbox_height_.try_get();
  auto bbox_x           = bbox_loc_x_.try_get();
  auto bbox_y           = bbox_loc_y_.try_get();
  const bool fixed_bbox = bbox_width && bbox_height && bbox_x && bbox_y;

  for (size_t i = 0; i < num_bbox; i++) {
    const int index = i * 4;
    int w           = fixed_bbox ? bbox_width.value() : rand() % image_width + 1;
    int h           = fixed_bbox ? bbox_height.value() : rand() % image_height + 1;
    int xmin        = fixed_bbox ? bbox_x.value() : rand() % (image_width - w);
    int ymin        = fixed_bbox ? bbox_y.value() : rand() % (image_height - h);

    bbox_pointer.value()[index]     = xmin;
    bbox_pointer.value()[index + 1] = ymin;
    bbox_pointer.value()[index + 2] = xmin + w;
    bbox_pointer.value()[index + 3] = ymin + h;
  }

  // Sending the data
  transmitter_->publish(outputMessage.value());

  return GXF_SUCCESS;
}

}  // namespace tensor_ops
}  // namespace isaac
}  // namespace nvidia
