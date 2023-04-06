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
#include <string.h>
#include <cstring>
#include <iostream>

#include "cv/core/CameraModel.h"
#include "cv/core/Image.h"
#include "cv/core/Memory.h"
#include "cv/core/ProfileUtils.h"
#include "cv/core/Tensor.h"
#include "cv/tensor_ops/ImageUtils.h"

#include "VPIColorConvertImpl.h"
#include "VPIEnumMapping.h"
#include "VPIStatusMapping.h"

// VPI includes
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>

#ifdef NVBENCH_ENABLE
#include <nvbench/VPI.h>
#endif

namespace cvcore { namespace tensor_ops {

VPITensorStream::VPIColorConvertImpl::VPIColorConvertImpl()
    : m_inputImage(nullptr)
    , m_outputImage(nullptr)
{
    std::memset(reinterpret_cast<void *>(&m_inputImageData), 0, sizeof(VPIImageData));
    std::memset(reinterpret_cast<void *>(&m_outputImageData), 0, sizeof(VPIImageData));
}

template<ImageType T_OUT, ImageType T_IN>
std::error_code VPITensorStream::VPIColorConvertImpl::execute(Image<T_OUT> &outputImage, const Image<T_IN> &inputImage,
                                                              VPIStream &stream, VPIBackend backend)
{
    std::error_code errCode = make_error_code(VPIStatus::VPI_SUCCESS);

    bool paramsChanged = m_inputImage == nullptr || m_outputImage == nullptr ||
                         CheckParamsChanged(m_inputImageData, inputImage) ||
                         CheckParamsChanged(m_outputImageData, outputImage);
    if (paramsChanged)
    {
        DestroyVPIImageWrapper(m_inputImage, m_inputImageData);
        DestroyVPIImageWrapper(m_outputImage, m_outputImageData);
        errCode = CreateVPIImageWrapper(m_inputImage, m_inputImageData, inputImage, backend);
        if (errCode == make_error_code(VPI_SUCCESS))
        {
            errCode = CreateVPIImageWrapper(m_outputImage, m_outputImageData, outputImage, backend);
        }
    }

    if (errCode == make_error_code(VPIStatus::VPI_SUCCESS))
    {
        errCode = UpdateImage(m_inputImage, m_inputImageData, inputImage);
    }
    if (errCode == make_error_code(VPIStatus::VPI_SUCCESS))
    {
        errCode = UpdateImage(m_outputImage, m_outputImageData, outputImage);
    }

    if (errCode == make_error_code(VPIStatus::VPI_SUCCESS))
    {
#ifdef NVBENCH_ENABLE
        std::string tag = "VPIColorConvert_" + GetMemoryTypeAsString(inputImage.isCPU()) +"Input_" + GetMemoryTypeAsString(outputImage.isCPU()) +"Output_" + getVPIBackendString(backend) + "Backend";
        nv::bench::Timer timerFunc =
        nv::bench::VPI(tag.c_str(), nv::bench::Flag::DEFAULT, stream);
#endif
        errCode = make_error_code(vpiSubmitConvertImageFormat(stream, backend, m_inputImage, m_outputImage, nullptr));
    }

    if (errCode == make_error_code(VPIStatus::VPI_SUCCESS))
    {
        errCode = make_error_code(vpiStreamSync(stream));
    }

    if (errCode != make_error_code(VPIStatus::VPI_SUCCESS))
    {
        return errCode;
    }

    return make_error_code(ErrorCode::SUCCESS);
}

VPITensorStream::VPIColorConvertImpl::~VPIColorConvertImpl()
{
    // Destroy Input VPIImage
    DestroyVPIImageWrapper(m_inputImage, m_inputImageData);

    // Destroy Output VPIImage
    DestroyVPIImageWrapper(m_outputImage, m_outputImageData);
}

template std::error_code VPITensorStream::VPIColorConvertImpl::execute(Image<BGR_U8> &, const Image<RGB_U8> &,
                                                                       VPIStream &, VPIBackend);
template std::error_code VPITensorStream::VPIColorConvertImpl::execute(Image<RGB_U8> &, const Image<BGR_U8> &,
                                                                       VPIStream &, VPIBackend);
template std::error_code VPITensorStream::VPIColorConvertImpl::execute(Image<BGR_U8> &, const Image<NV12> &,
                                                                       VPIStream &, VPIBackend);
template std::error_code VPITensorStream::VPIColorConvertImpl::execute(Image<BGR_U8> &, const Image<NV24> &,
                                                                       VPIStream &, VPIBackend);
template std::error_code VPITensorStream::VPIColorConvertImpl::execute(Image<NV12> &, const Image<BGR_U8> &,
                                                                       VPIStream &, VPIBackend);
template std::error_code VPITensorStream::VPIColorConvertImpl::execute(Image<NV24> &, const Image<BGR_U8> &,
                                                                       VPIStream &, VPIBackend);
template std::error_code VPITensorStream::VPIColorConvertImpl::execute(Image<RGB_U8> &, const Image<NV12> &,
                                                                       VPIStream &, VPIBackend);
template std::error_code VPITensorStream::VPIColorConvertImpl::execute(Image<RGB_U8> &, const Image<NV24> &,
                                                                       VPIStream &, VPIBackend);
template std::error_code VPITensorStream::VPIColorConvertImpl::execute(Image<NV12> &, const Image<RGB_U8> &,
                                                                       VPIStream &, VPIBackend);
template std::error_code VPITensorStream::VPIColorConvertImpl::execute(Image<NV24> &, const Image<RGB_U8> &,
                                                                       VPIStream &, VPIBackend);

}} // namespace cvcore::tensor_ops
