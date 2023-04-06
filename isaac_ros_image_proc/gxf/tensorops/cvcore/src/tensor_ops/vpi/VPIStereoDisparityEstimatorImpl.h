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

#ifndef CVCORE_VPISTEREODISPARITYESTIMATORIMPL_H
#define CVCORE_VPISTEREODISPARITYESTIMATORIMPL_H

#include <vpi/Image.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/StereoDisparity.h>

#include "VPITensorOperators.h"

namespace cvcore { namespace tensor_ops {
/**
 * StereoDisparityEstimator implementation used for stereo dispaity estimate.
 */
class VPITensorStream::VPIStereoDisparityEstimatorImpl
{
public:
    /* VPIStereoDisparityEstimatorImpl constructor */
    VPIStereoDisparityEstimatorImpl();

    /**
        * StereoDisparityEstimator Intialization.
        * @param outputImage StereoDisparityEstimator output image
        * @param leftImage StereoDisparityEstimator input left image
        * @param rightImage StereoDisparityEstimator input right image
        * @param backend VPI backend used for StereoDisparityEstimator
        * @return Success if intialization is done successfully, otherwise error is returned
        */
    template<ImageType T_OUT, ImageType T_IN>
    std::error_code initialize(Image<T_OUT> &outImage, const Image<T_IN> &leftImage, const Image<T_IN> &rightImage,
                               VPIBackend backend);

    /**
        * StereoDisparityEstimator execution function(non-blocking)
        * Application is reqiured to call Sync() before accessing the generated Image.
        * @param outImage StereoDisparityEstimator output image
        * @param leftImage StereoDisparityEstimator input left image
        * @param rightImage StereoDisparityEstimator input right image
        * @param windowSize Represents the median filter size (on PVA+NVENC_VIC backend) or census transform window size (other backends) used in the algorithm
        * @param maxDisparity Maximum disparity for matching search
        * @param stream VPI stream used for StereoDisparityEstimator
        * @param backend VPI backend used for StereoDisparityEstimator
        * @return Success if StereoDisparityEstimator is submitted successfully, otherwise error is returned
        */
    template<ImageType T_OUT, ImageType T_IN>
    std::error_code execute(Image<T_OUT> &outImage, const Image<T_IN> &leftImage, const Image<T_IN> &rightImage,
                            size_t windowSize, size_t maxDisparity, VPIStream &stream, VPIBackend backend);

    /* VPIStereoDisparityEstimatorImpl destructor to release resources */
    ~VPIStereoDisparityEstimatorImpl();

private:
    VPIImage m_inputLeftImage;
    VPIImage m_inputRightImage;
    VPIImage m_outputImage;
    VPIImage m_tempImage;
    VPIImageData m_inputLeftImageData;
    VPIImageData m_inputRightImageData;
    VPIImageData m_outputImageData;
    VPIPayload m_payload;
    VPIStereoDisparityEstimatorParams m_stereoParams;
    VPIConvertImageFormatParams m_cvtParams;
};

}} // namespace cvcore::tensor_ops

#endif //CVCORE_VPISTEREODISPARITYESTIMATORIMPL_H
