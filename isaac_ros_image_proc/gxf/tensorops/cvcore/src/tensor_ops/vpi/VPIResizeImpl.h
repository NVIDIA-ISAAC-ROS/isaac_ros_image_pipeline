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

#ifndef CVCORE_VPIRESIZEIMPL_H
#define CVCORE_VPIRESIZEIMPL_H

#include "VPITensorOperators.h"

#include <vpi/Image.h>

#include "cv/tensor_ops/ITensorOperatorStream.h"
#include "cv/tensor_ops/ImageUtils.h"

namespace cvcore { namespace tensor_ops {

/**
 * Remap implementation used for Lens Distortion.
 */
class VPITensorStream::VPIResizeImpl
{
public:
    /**
     * Initialization for Image resizing.
     */
    VPIResizeImpl();

    /**
     * Image resizing a given image type.
     * @param outputImage Resize output image of type .RGB_U8
     * @param inputImage Resize input image of type RGB_U8.
     * @param type Interpolation type used for resize.
     * @param border Image border extension used for resize
     */
    template<ImageType T>
    std::error_code execute(Image<T> &outputImage, const Image<T> &inputImage, InterpolationType interpolation,
                            BorderType border, VPIStream &stream, VPIBackend backend);

    /**
      * Image resizing destroy function to deallocate resources.
     */
    ~VPIResizeImpl();

private:
    VPIImage m_inputImage;
    VPIImage m_outputImage;
    VPIImageData m_inputImageData;
    VPIImageData m_outputImageData;
};

}} // namespace cvcore::tensor_ops

#endif //CVCORE_VPIRESIZEIMPL_H
