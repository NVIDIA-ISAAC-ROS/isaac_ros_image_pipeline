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

#include <stdio.h>
#include <cstring>

#include "VPIStereoDisparityEstimatorImpl.h"
#include "VPIEnumMapping.h"
#include "VPIStatusMapping.h"

#include "cv/core/CameraModel.h"
#include "cv/core/Image.h"

#include <vpi/Status.h>

#ifdef NVBENCH_ENABLE
#include <nvbench/VPI.h>
#endif

namespace cvcore { namespace tensor_ops {

VPITensorStream::VPIStereoDisparityEstimatorImpl::VPIStereoDisparityEstimatorImpl()
    : m_inputLeftImage(nullptr)
    , m_inputRightImage(nullptr)
    , m_outputImage(nullptr)
    , m_tempImage(nullptr)
    , m_payload(nullptr)
    , m_stereoParams()
{
    std::memset(reinterpret_cast<void *>(&m_inputLeftImageData), 0, sizeof(VPIImageData));
    std::memset(reinterpret_cast<void *>(&m_inputRightImageData), 0, sizeof(VPIImageData));
    std::memset(reinterpret_cast<void *>(&m_outputImageData), 0, sizeof(VPIImageData));
    // Disparity values returned from VPI are in Q10.5 format, i.e., signed fixed point with 5 fractional bits. Divide it by 32.0f to convert it to floating point.
    vpiInitConvertImageFormatParams(&m_cvtParams);
    m_cvtParams.scale = 1.0f / 32;
}

template<ImageType T_OUT, ImageType T_IN>
std::error_code VPITensorStream::VPIStereoDisparityEstimatorImpl::initialize(Image<T_OUT> &outImage,
                                                                             const Image<T_IN> &leftImage,
                                                                             const Image<T_IN> &rightImage,
                                                                             VPIBackend backend)
{
    std::error_code status;
    const std::error_code success = make_error_code(VPI_SUCCESS);
    status                        = CreateVPIImageWrapper(m_inputLeftImage, m_inputLeftImageData, leftImage, backend);
    if (status == success)
    {
        status = CreateVPIImageWrapper(m_inputRightImage, m_inputRightImageData, rightImage, backend);
    }
    if (status == success)
    {
        status = CreateVPIImageWrapper(m_outputImage, m_outputImageData, outImage, backend);
    }
    if (status == success)
    {
        status = make_error_code(
            vpiImageCreate(outImage.getWidth(), outImage.getHeight(), VPI_IMAGE_FORMAT_S16, 0, &m_tempImage));
    }
    if (status == success)
    {
        status = make_error_code(vpiCreateStereoDisparityEstimator(backend, outImage.getWidth(), outImage.getHeight(),
                                                                   ToVpiImageFormat(T_IN), NULL, &m_payload));
    }
    return status;
}

template std::error_code VPITensorStream::VPIStereoDisparityEstimatorImpl::initialize(Image<Y_F32> &outImage,
                                                                                      const Image<Y_U8> &leftImage,
                                                                                      const Image<Y_U8> &rightImage,
                                                                                      VPIBackend backend);
template std::error_code VPITensorStream::VPIStereoDisparityEstimatorImpl::initialize(Image<Y_F32> &outImage,
                                                                                      const Image<NV12> &leftImage,
                                                                                      const Image<NV12> &rightImage,
                                                                                      VPIBackend backend);
template std::error_code VPITensorStream::VPIStereoDisparityEstimatorImpl::initialize(Image<Y_F32> &outImage,
                                                                                      const Image<NV24> &leftImage,
                                                                                      const Image<NV24> &rightImage,
                                                                                      VPIBackend backend);

// -----------------------------------------------------------------------------

template<ImageType T_OUT, ImageType T_IN>
std::error_code VPITensorStream::VPIStereoDisparityEstimatorImpl::execute(Image<T_OUT> &outImage,
                                                                          const Image<T_IN> &leftImage,
                                                                          const Image<T_IN> &rightImage,
                                                                          size_t windowSize, size_t maxDisparity,
                                                                          VPIStream &stream, VPIBackend backend)
{
    std::error_code status      = make_error_code(VPI_SUCCESS);
    m_stereoParams.windowSize   = static_cast<int32_t>(windowSize);
    m_stereoParams.maxDisparity = static_cast<int32_t>(maxDisparity);

    bool paramsChanged = m_inputLeftImage == nullptr || m_inputRightImage == nullptr || m_outputImage == nullptr ||
                         CheckParamsChanged(m_inputLeftImageData, leftImage) ||
                         CheckParamsChanged(m_inputRightImageData, rightImage) ||
                         CheckParamsChanged(m_outputImageData, outImage);

    if (paramsChanged)
    {
        if (m_payload != nullptr)
        {
            vpiPayloadDestroy(m_payload);
        }
        if (m_tempImage != nullptr)
        {
            vpiImageDestroy(m_tempImage);
        }
        DestroyVPIImageWrapper(m_inputLeftImage, m_inputLeftImageData);
        DestroyVPIImageWrapper(m_inputRightImage, m_inputRightImageData);
        DestroyVPIImageWrapper(m_outputImage, m_outputImageData);

        status = initialize(outImage, leftImage, rightImage, backend);
    }

    if (status == make_error_code(VPI_SUCCESS))
    {
        status = UpdateImage(m_inputLeftImage, m_inputLeftImageData, leftImage);
    }

    if (status == make_error_code(VPI_SUCCESS))
    {
        status = UpdateImage(m_inputRightImage, m_inputRightImageData, rightImage);
    }

    if (status == make_error_code(VPI_SUCCESS))
    {
        status = UpdateImage(m_outputImage, m_outputImageData, outImage);
    }

    if (status == make_error_code(VPI_SUCCESS))
    {
#ifdef NVBENCH_ENABLE
        std::string tag = "VPISubmitStereoDisparityEstimator_" + GetMemoryTypeAsString(leftImage.isCPU()) + "Input_" +
                          GetMemoryTypeAsString(outImage.isCPU()) + "Output_" + getVPIBackendString(backend) +
                          "Backend";
        nv::bench::Timer timerFunc = nv::bench::VPI(tag.c_str(), nv::bench::Flag::DEFAULT, stream);
#endif
        // Submit SGM task for Stereo Disparity Estimator
        status = make_error_code(vpiSubmitStereoDisparityEstimator(
            stream, backend, m_payload, m_inputLeftImage, m_inputRightImage, m_tempImage, NULL, &m_stereoParams));
    }

    if (status == make_error_code(VPI_SUCCESS))
    {
#ifdef NVBENCH_ENABLE
        std::string tag = "VPISubmitConvertImageFormat_" + GetMemoryTypeAsString(leftImage.isCPU()) + "Input_" +
                          GetMemoryTypeAsString(outImage.isCPU()) + "Output_" + getVPIBackendString(backend) +
                          "Backend";
        nv::bench::Timer timerFunc = nv::bench::VPI(tag.c_str(), nv::bench::Flag::DEFAULT, stream);
#endif
        // Submit SGM task for Stereo Disparity Estimator
        status =
            make_error_code(vpiSubmitConvertImageFormat(stream, backend, m_tempImage, m_outputImage, &m_cvtParams));
    }

    if (status == make_error_code(VPI_SUCCESS))
    {
        // Wait for stereo disparity estimator to complete
        status = make_error_code(vpiStreamSync(stream));
    }

    if (status != make_error_code(VPI_SUCCESS))
    {
        return status;
    }
    return make_error_code(ErrorCode::SUCCESS);
}

template std::error_code VPITensorStream::VPIStereoDisparityEstimatorImpl::execute(
    Image<Y_F32> &outImage, const Image<Y_U8> &leftImage, const Image<Y_U8> &rightImage, size_t windowSize,
    size_t maxDisparity, VPIStream &stream, VPIBackend backend);
template std::error_code VPITensorStream::VPIStereoDisparityEstimatorImpl::execute(
    Image<Y_F32> &outImage, const Image<NV12> &leftImage, const Image<NV12> &rightImage, size_t windowSize,
    size_t maxDisparity, VPIStream &stream, VPIBackend backend);
template std::error_code VPITensorStream::VPIStereoDisparityEstimatorImpl::execute(
    Image<Y_F32> &outImage, const Image<NV24> &leftImage, const Image<NV24> &rightImage, size_t windowSize,
    size_t maxDisparity, VPIStream &stream, VPIBackend backend);
// -----------------------------------------------------------------------------

VPITensorStream::VPIStereoDisparityEstimatorImpl::~VPIStereoDisparityEstimatorImpl()
{
    if (m_payload != nullptr)
    {
        vpiPayloadDestroy(m_payload);
    }
    if (m_tempImage != nullptr)
    {
        vpiImageDestroy(m_tempImage);
    }
    DestroyVPIImageWrapper(m_inputLeftImage, m_inputLeftImageData);
    DestroyVPIImageWrapper(m_inputRightImage, m_inputRightImageData);
    DestroyVPIImageWrapper(m_outputImage, m_outputImageData);
}
// -----------------------------------------------------------------------------

}} // namespace cvcore::tensor_ops
