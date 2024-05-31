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

#pragma once

#include <vpi/Image.h>
#include "extensions/tensorops/core/ImageUtils.h"
#include "extensions/tensorops/core/ITensorOperatorStream.h"
#include "extensions/tensorops/core/VPITensorOperators.h"

namespace cvcore {
namespace tensor_ops {

/**
 * Color convert implementation for VPI backend.
 */
class VPITensorStream::VPIColorConvertImpl {
 public:
    /**
    * Image color conversion constructor.
    */
    VPIColorConvertImpl();

    /**
    * Image color conversion a given image type.
    * @param outputImage Output image.
    * @param inputImage Input image.
    * @param stream specified VPI stream.
    * @param backend specified VPI backend.
    */
    template<ImageType T_OUT, ImageType T_IN>
    std::error_code execute(Image<T_OUT> & outputImage, const Image<T_IN> & inputImage,
        VPIStream & stream, VPIBackend backend);

    /**
    * Image color conversion destroy function to deallocate resources.
    */
    ~VPIColorConvertImpl();

 private:
    VPIImage m_inputImage;
    VPIImage m_outputImage;
    VPIImageData m_inputImageData;
    VPIImageData m_outputImageData;
};

}  // namespace tensor_ops
}  // namespace cvcore
