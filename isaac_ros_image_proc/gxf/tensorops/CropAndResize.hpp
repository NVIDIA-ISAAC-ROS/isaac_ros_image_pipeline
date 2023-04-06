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
#ifndef NVIDIA_CVCORE_CROP_AND_RESIZE_HPP
#define NVIDIA_CVCORE_CROP_AND_RESIZE_HPP

#include "TensorOperator.hpp"
#include "cv/core/BBox.h"

namespace nvidia {
namespace cvcore {
namespace tensor_ops {

// CropAndResize operator.
class CropAndResize : public TensorOperator {
public:
  virtual ~CropAndResize() {}

  gxf_result_t registerInterface(gxf::Registrar* registrar) override;

private:
  gxf::Expected<ImageInfo> doInferOutputInfo(gxf::Entity& input) override final;
  gxf_result_t doUpdateCameraMessage(gxf::Handle<gxf::CameraModel>& output,
                                     gxf::Handle<gxf::CameraModel>& input) override final;
  gxf_result_t doExecute(gxf::Entity& output, gxf::Entity& input, cudaStream_t stream, const char* output_name,
                         const char* input_name) override final;

  gxf::Parameter<size_t> output_width_;
  gxf::Parameter<size_t> output_height_;
  gxf::Parameter<std::string> interp_type_;
  gxf::Parameter<bool> keep_aspect_ratio_;
  gxf::Parameter<gxf::Handle<gxf::Receiver>> receiver_bbox_;
  std::vector<::cvcore::BBox> input_rois_;
};

} // namespace tensor_ops
} // namespace cvcore
} // namespace nvidia

#endif
