// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "extensions/tensorops/core/VPIResizeImpl.h"

#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/Rescale.h>
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>

#include <cstring>
#include <iostream>
#include <string>

#include "extensions/tensorops/core/CameraModel.h"
#include "extensions/tensorops/core/Image.h"
#include "extensions/tensorops/core/VPIEnumMapping.h"
#include "extensions/tensorops/core/VPIStatusMapping.h"

namespace cvcore {
namespace tensor_ops {

template<ImageType T>
std::error_code VPITensorStream::VPIResizeImpl::execute(Image<T>& outputImage,
    const Image<T>& inputImage, InterpolationType interpolation, BorderType border,
    VPIStream& stream, VPIBackend backend) {
    std::error_code errCode = ErrorCode::SUCCESS;
    VPIStatus status        = VPIStatus::VPI_SUCCESS;
    VPIInterpolationType interpType;
    VPIBorderExtension borderExt;
    interpType = ToVpiInterpolationType(interpolation);
    borderExt  = ToVpiBorderType(border);

    bool paramsChanged = m_inputImage == nullptr || m_outputImage == nullptr ||
                         CheckParamsChanged(m_inputImageData, inputImage) ||
                         CheckParamsChanged(m_outputImageData, outputImage);
    if (paramsChanged) {
        DestroyVPIImageWrapper(m_inputImage, m_inputImageData);
        DestroyVPIImageWrapper(m_outputImage, m_outputImageData);
        errCode = CreateVPIImageWrapper(m_inputImage, m_inputImageData,
            inputImage, backend);
        if (errCode == make_error_code(VPI_SUCCESS)) {
            errCode = CreateVPIImageWrapper(m_outputImage, m_outputImageData,
                outputImage, backend);
        }
    } else {
        errCode = UpdateImage(m_inputImage, m_inputImageData, inputImage);
        if (errCode == make_error_code(VPIStatus::VPI_SUCCESS)) {
            errCode = UpdateImage(m_outputImage, m_outputImageData, outputImage);
        }
    }

    if (status == VPIStatus::VPI_SUCCESS) {
        status = vpiSubmitRescale(stream, backend, m_inputImage, m_outputImage,
            interpType, borderExt, 0);
    }

    if (status != VPIStatus::VPI_SUCCESS) {
        return make_error_code(status);
    }
    return make_error_code(ErrorCode::SUCCESS);
}

VPITensorStream::VPIResizeImpl::VPIResizeImpl()
    : m_inputImage(nullptr)
    , m_outputImage(nullptr) {
    std::memset(reinterpret_cast<void *>(&m_inputImageData), 0, sizeof(VPIImageData));
    std::memset(reinterpret_cast<void *>(&m_outputImageData), 0, sizeof(VPIImageData));
}

/**
* Image resizing destroy function to deallocate resources.
*/
VPITensorStream::VPIResizeImpl::~VPIResizeImpl() {
    // Destroy Input VPIImage
    DestroyVPIImageWrapper(m_inputImage, m_inputImageData);
    // Destroy Output VPIImage
    DestroyVPIImageWrapper(m_outputImage, m_outputImageData);
}

template std::error_code VPITensorStream::VPIResizeImpl::execute(Image<RGB_U8>& outputImage,
    const Image<RGB_U8>& inputImage, InterpolationType interpolation, BorderType border,
    VPIStream& stream, VPIBackend backend);
template std::error_code VPITensorStream::VPIResizeImpl::execute(Image<RGBA_U8>& outputImage,
    const Image<RGBA_U8>& inputImage,
    InterpolationType interpolation, BorderType border,
    VPIStream& stream, VPIBackend backend);
template std::error_code VPITensorStream::VPIResizeImpl::execute(Image<BGRA_U8>& outputImage,
    const Image<BGRA_U8>& inputImage,
    InterpolationType interpolation, BorderType border,
    VPIStream& stream, VPIBackend backend);
template std::error_code VPITensorStream::VPIResizeImpl::execute(Image<BGR_U8>& outputImage,
    const Image<BGR_U8>& inputImage,
    InterpolationType interpolation, BorderType border,
    VPIStream& stream, VPIBackend backend);
template std::error_code VPITensorStream::VPIResizeImpl::execute(Image<NV24>& outputImage,
    const Image<NV24>& inputImage,
    InterpolationType interpolation, BorderType border,
    VPIStream& stream, VPIBackend backend);
template std::error_code VPITensorStream::VPIResizeImpl::execute(Image<NV12>& outputImage,
    const Image<NV12>& inputImage,
    InterpolationType interpolation, BorderType border,
    VPIStream& stream, VPIBackend backend);
template std::error_code VPITensorStream::VPIResizeImpl::execute(Image<Y_U8>& outputImage,
    const Image<Y_U8>& inputImage,
    InterpolationType interpolation, BorderType border,
    VPIStream& stream, VPIBackend backend);

}  // namespace tensor_ops
}  // namespace cvcore
