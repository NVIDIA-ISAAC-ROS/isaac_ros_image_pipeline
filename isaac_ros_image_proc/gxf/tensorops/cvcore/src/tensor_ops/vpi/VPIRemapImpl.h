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

#ifndef CVCORE_VPIREMAPIMPL_H
#define CVCORE_VPIREMAPIMPL_H

#include <vpi/WarpMap.h>
#include <vpi/Image.h>

#include "VPITensorOperators.h"
#include "VPIImageWarp.h"

namespace cvcore { namespace tensor_ops {
/**
 * Remap implementation used for Lens Distortion.
 */
class VPITensorStream::VPIRemapImpl
{
    public:
        /* VPIRemapImpl constructor */
        VPIRemapImpl();

        /**
        * Remap Intialization.
        * @param outputImage Remap output image of Type
        * @param inputImage Remap input image of Type
        * @param backend Compute backend
        * @return Success if intialization is done successfully, otherwise error is returned
        */
        template<ImageType Type>
        std::error_code initialize(Image<Type> & outImage,
                                   const Image<Type> & inImage,
                                   VPIBackend backend);

        /**
        * Remap execution function(non-blocking)
        * Application is reqiured to call Sync() before accessing the generated Image.
        * @param outImage Remap output image of type NV12
        * @param inImage Remap input image of type NV12
        * @param warp Remap warp pointer
        * @param interpolation Interpolation type used for remap
        * @param border Border type used for remap
        * @param stream VPI stream used for remap
        * @param backend VPI backend used for remap
        * @return Success if remap is submitted successfully, otherwise error is returned
        */
        template<ImageType Type>
        std::error_code execute(Image<Type> & outImage,
                                const Image<Type> & inImage,
                                const VPIImageWarp * warp,
                                InterpolationType interpolation,
                                BorderType border,
                                VPIStream & stream,
                                VPIBackend backend);

        /* VPIRemapImpl destructor to release resources */
        ~VPIRemapImpl();

    private:
        VPIImage m_inputImage;
        VPIImage m_outputImage;
        VPIImageData m_inputImageData;
        VPIImageData m_outputImageData;
};

}} // namespace cvcore::tensor_ops

#endif //CVCORE_VPIREMAPIMPL_H
