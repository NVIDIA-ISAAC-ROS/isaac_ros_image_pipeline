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

#include "extensions/tensorops/core/VPIRemapImpl.h"
#include <vpi/algo/Remap.h>
#include <vpi/Status.h>

#include <cstring>

#include "extensions/tensorops/core/CameraModel.h"
#include "extensions/tensorops/core/Image.h"
#include "extensions/tensorops/core/VPIEnumMapping.h"
#include "extensions/tensorops/core/VPIStatusMapping.h"

namespace cvcore {
namespace tensor_ops {

VPITensorStream::VPIRemapImpl::VPIRemapImpl()
    : m_inputImage(nullptr)
    , m_outputImage(nullptr) {
    std::memset(reinterpret_cast<void*>(&m_inputImageData), 0, sizeof(VPIImageData));
    std::memset(reinterpret_cast<void*>(&m_outputImageData), 0, sizeof(VPIImageData));
}

template<ImageType Type>
std::error_code VPITensorStream::VPIRemapImpl::initialize(Image<Type> & outImage,
                                                          const Image<Type> & inImage,
                                                          VPIBackend backend) {
    std::error_code status;
    status = CreateVPIImageWrapper(m_inputImage, m_inputImageData, inImage, backend);
    if (status == make_error_code(VPI_SUCCESS)) {
        status = CreateVPIImageWrapper(m_outputImage, m_outputImageData, outImage, backend);
    }

    return status;
}
template std::error_code VPITensorStream::VPIRemapImpl::initialize(Image<RGB_U8>& outImage,
    const Image<RGB_U8>& inImage, VPIBackend);
template std::error_code VPITensorStream::VPIRemapImpl::initialize(Image<BGR_U8>& outImage,
    const Image<BGR_U8>& inImage, VPIBackend);
template std::error_code VPITensorStream::VPIRemapImpl::initialize(Image<NV12>& outImage,
    const Image<NV12>& inImage, VPIBackend);
template std::error_code VPITensorStream::VPIRemapImpl::initialize(Image<NV24>& outImage,
    const Image<NV24>& inImage, VPIBackend);

template<ImageType Type>
std::error_code VPITensorStream::VPIRemapImpl::execute(Image<Type> & outImage,
                                                       const Image<Type> & inImage,
                                                       const VPIImageWarp * warp,
                                                       InterpolationType interpolation,
                                                       BorderType border,
                                                       VPIStream & stream,
                                                       VPIBackend backend) {
    std::error_code status = make_error_code(VPI_SUCCESS);
    VPIInterpolationType vpiInterpolationType = ToVpiInterpolationType(interpolation);
    VPIBorderExtension vpiBorderExt = ToVpiBorderType(border);

    bool paramsChanged = m_inputImage == nullptr || m_outputImage == nullptr ||
                         CheckParamsChanged(m_inputImageData, inImage) ||
                         CheckParamsChanged(m_outputImageData, outImage);

    if (paramsChanged) {
        DestroyVPIImageWrapper(m_inputImage, m_inputImageData);
        DestroyVPIImageWrapper(m_outputImage, m_outputImageData);
        status = initialize(outImage, inImage, backend);
    }

    if (status == make_error_code(VPI_SUCCESS)) {
        status = UpdateImage(m_inputImage, m_inputImageData, inImage);
    }

    if (status == make_error_code(VPI_SUCCESS)) {
        status = UpdateImage(m_outputImage, m_outputImageData, outImage);
    }

    if (status == make_error_code(VPI_SUCCESS)) {
        // Submit remap task for Lens Distortion Correction
        status = make_error_code(vpiSubmitRemap(stream, backend, warp->payload,
             m_inputImage, m_outputImage, vpiInterpolationType, vpiBorderExt, 0));
    }

    return status;
}
template std::error_code VPITensorStream::VPIRemapImpl::execute(Image<RGB_U8> & outImage,
                                                       const Image<RGB_U8> & inImage,
                                                       const VPIImageWarp * warp,
                                                       InterpolationType interpolation,
                                                       BorderType border,
                                                       VPIStream & stream,
                                                       VPIBackend backend);
template std::error_code VPITensorStream::VPIRemapImpl::execute(Image<BGR_U8> & outImage,
                                                       const Image<BGR_U8> & inImage,
                                                       const VPIImageWarp * warp,
                                                       InterpolationType interpolation,
                                                       BorderType border,
                                                       VPIStream & stream,
                                                       VPIBackend backend);
template std::error_code VPITensorStream::VPIRemapImpl::execute(Image<NV12> & outImage,
                                                       const Image<NV12> & inImage,
                                                       const VPIImageWarp * warp,
                                                       InterpolationType interpolation,
                                                       BorderType border,
                                                       VPIStream & stream,
                                                       VPIBackend backend);
template std::error_code VPITensorStream::VPIRemapImpl::execute(Image<NV24> & outImage,
                                                       const Image<NV24> & inImage,
                                                       const VPIImageWarp * warp,
                                                       InterpolationType interpolation,
                                                       BorderType border,
                                                       VPIStream & stream,
                                                       VPIBackend backend);

VPITensorStream::VPIRemapImpl::~VPIRemapImpl() {
    DestroyVPIImageWrapper(m_inputImage, m_inputImageData);
    DestroyVPIImageWrapper(m_outputImage, m_outputImageData);
}

}  // namespace tensor_ops
}  // namespace cvcore
