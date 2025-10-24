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

#include <vpi/algo/Remap.h>
#include <vpi/Context.h>
#include <vpi/CUDAInterop.h>
#include <vpi/Image.h>
#include <vpi/Stream.h>

#include <cstring>
#include <iostream>

#include "extensions/tensorops/core/CameraModel.h"
#include "extensions/tensorops/core/Image.h"
#include "extensions/tensorops/core/VPIColorConvertImpl.h"
#include "extensions/tensorops/core/VPIEnumMapping.h"
#include "extensions/tensorops/core/VPIImageWarp.h"
#include "extensions/tensorops/core/VPIRemapImpl.h"
#include "extensions/tensorops/core/VPIResizeImpl.h"
#include "extensions/tensorops/core/VPIStatusMapping.h"
#include "extensions/tensorops/core/VPIStereoDisparityEstimatorImpl.h"
#include "extensions/tensorops/core/VPITensorOperators.h"

namespace cvcore {
namespace tensor_ops {

namespace detail {

// helper function to wrap VPI image for NV12 / NV24 image types
template<ImageType T, typename std::enable_if<IsCompositeImage<T>::value>::type * = nullptr>
std::error_code CreateVPIImageWrapperImpl(VPIImage& vpiImg, VPIImageData& imgdata,
    const Image<T>& cvcoreImage, VPIBackend backend) {
    std::memset(reinterpret_cast<void *>(&imgdata), 0, sizeof(VPIImageData));

    imgdata.bufferType = cvcoreImage.isCPU() ? VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR
                                             : VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;
    imgdata.buffer.pitch.format               = ToVpiImageFormat(T);
    imgdata.buffer.pitch.numPlanes            = 2;
    imgdata.buffer.pitch.planes[0].pBase      = const_cast<uint8_t *>(cvcoreImage.getLumaData());
    imgdata.buffer.pitch.planes[0].height     = cvcoreImage.getLumaHeight();
    imgdata.buffer.pitch.planes[0].width      = cvcoreImage.getLumaWidth();
    imgdata.buffer.pitch.planes[0].pixelType  = VPI_PIXEL_TYPE_U8;
    imgdata.buffer.pitch.planes[0].offsetBytes = 0;
    imgdata.buffer.pitch.planes[0].pitchBytes = cvcoreImage.getLumaStride(
                                                    TensorDimension::HEIGHT) * sizeof(uint8_t);
    imgdata.buffer.pitch.planes[1].pBase      = const_cast<uint8_t *>(cvcoreImage.getChromaData());
    imgdata.buffer.pitch.planes[1].height     = cvcoreImage.getChromaHeight();
    imgdata.buffer.pitch.planes[1].width      = cvcoreImage.getChromaWidth();
    imgdata.buffer.pitch.planes[1].pixelType  = VPI_PIXEL_TYPE_2U8;
    imgdata.buffer.pitch.planes[1].offsetBytes = 0;
    imgdata.buffer.pitch.planes[1].pitchBytes = cvcoreImage.getChromaStride(
                                                    TensorDimension::HEIGHT) * sizeof(uint8_t);
    VPIStatus vpiStatus;
    vpiStatus = vpiImageCreateWrapper(&imgdata, nullptr, backend, &vpiImg);

    return make_error_code(vpiStatus);
}

// helper function to wrap VPI image for interleaved image types
template<ImageType T, typename std::enable_if<IsInterleavedImage<T>::value>::type * = nullptr>
std::error_code CreateVPIImageWrapperImpl(VPIImage& vpiImg, VPIImageData& imgdata,
    const Image<T>& cvcoreImage, VPIBackend backend) {
    std::memset(reinterpret_cast<void *>(&imgdata), 0, sizeof(VPIImageData));

    using D            = typename Image<T>::DataType;
    imgdata.bufferType = cvcoreImage.isCPU() ? VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR
                                             : VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;
    imgdata.buffer.pitch.format               = ToVpiImageFormat(T);
    imgdata.buffer.pitch.numPlanes            = 1;
    D* data = const_cast<D *>(cvcoreImage.getData());
    imgdata.buffer.pitch.planes[0].pBase      = reinterpret_cast<unsigned char *>(data);
    imgdata.buffer.pitch.planes[0].height     = cvcoreImage.getHeight();
    imgdata.buffer.pitch.planes[0].width      = cvcoreImage.getWidth();
    imgdata.buffer.pitch.planes[0].pixelType  = ToVpiPixelType(T);
    imgdata.buffer.pitch.planes[0].offsetBytes = 0;
    imgdata.buffer.pitch.planes[0].pitchBytes = cvcoreImage.getStride(
        TensorDimension::HEIGHT) * GetImageElementSize(T);
    VPIStatus vpiStatus;
    vpiStatus = vpiImageCreateWrapper(&imgdata, nullptr, backend, &vpiImg);

    return make_error_code(vpiStatus);
}

// helper function to wrap VPI image for planar image types (TODO: not supported in vpi so far)
template<ImageType T, typename std::enable_if<IsPlanarImage<T>::value>::type * = nullptr>
std::error_code CreateVPIImageWrapperImpl(VPIImage& vpiImg, VPIImageData& imgdata,
    const Image<T>& cvcoreImage, VPIBackend backend) {
    return make_error_code(VPI_ERROR_INVALID_IMAGE_FORMAT);
}

}  // namespace detail

std::error_code VPITensorContext::CreateStream(TensorOperatorStream& tensorStream,
    const ComputeEngine& computeEngine) {
    tensorStream = nullptr;

    if (!IsComputeEngineCompatible(computeEngine)) {
        return ErrorCode::INVALID_ENGINE_TYPE;
    }

    try {
        tensorStream = new VPITensorStream(computeEngine);
    }
    catch (std::error_code &e) {
        return e;
    }
    catch (...) {
        return ErrorCode::INVALID_OPERATION;
    }

    return ErrorCode::SUCCESS;
}

VPITensorStream::VPITensorStream(const ComputeEngine& computeEngine)
    : m_resizer(new VPIResizeImpl())
    , m_remapper(new VPIRemapImpl())
    , m_colorConverter(new VPIColorConvertImpl())
    , m_stereoDisparityEstimator(new VPIStereoDisparityEstimatorImpl()) {
    VPIBackend backend = ToVpiBackendType(computeEngine);
    VPIStatus status   = vpiStreamCreate(backend, &m_stream);
    if (status != VPI_SUCCESS) {
        throw make_error_code(status);
    }
    m_backend = backend;
}

VPITensorStream::~VPITensorStream() {
    vpiStreamDestroy(m_stream);
}

std::error_code VPITensorContext::DestroyStream(TensorOperatorStream& inputStream) {
    if (inputStream != nullptr) {
        delete inputStream;
        inputStream = nullptr;
    }
    return ErrorCode::SUCCESS;
}

bool VPITensorContext::IsComputeEngineCompatible(
    const ComputeEngine& computeEngine) const noexcept {
    VPIBackend vpibackend = ToVpiBackendType(computeEngine);
    if (vpibackend == VPIBackend::VPI_BACKEND_INVALID) {
        return false;
    }
    return true;
}

template<ImageType T>
std::error_code CreateVPIImageWrapper(VPIImage& vpiImg, VPIImageData& imgdata,
    const Image<T>& cvcoreImage, VPIBackend backend) {
    return detail::CreateVPIImageWrapperImpl(vpiImg, imgdata, cvcoreImage, backend);
}

template std::error_code CreateVPIImageWrapper(VPIImage&,
    VPIImageData&, const Image<Y_U8>&, VPIBackend);
template std::error_code CreateVPIImageWrapper(VPIImage&,
    VPIImageData&, const Image<Y_U16>&, VPIBackend);
template std::error_code CreateVPIImageWrapper(VPIImage&,
    VPIImageData&, const Image<Y_S8>&, VPIBackend);
template std::error_code CreateVPIImageWrapper(VPIImage&,
    VPIImageData&, const Image<Y_S16>&, VPIBackend);
template std::error_code CreateVPIImageWrapper(VPIImage&,
    VPIImageData&, const Image<Y_F32>&, VPIBackend);
template std::error_code CreateVPIImageWrapper(VPIImage&,
    VPIImageData&, const Image<BGR_U8>& , VPIBackend);
template std::error_code CreateVPIImageWrapper(VPIImage&,
    VPIImageData&, const Image<RGB_U8>&, VPIBackend);
template std::error_code CreateVPIImageWrapper(VPIImage&,
    VPIImageData&, const Image<RGBA_U8>&, VPIBackend);
template std::error_code CreateVPIImageWrapper(VPIImage&,
    VPIImageData&, const Image<BGRA_U8>&, VPIBackend);
template std::error_code CreateVPIImageWrapper(VPIImage&,
    VPIImageData&, const Image<NV12>&, VPIBackend);
template std::error_code CreateVPIImageWrapper(VPIImage&,
    VPIImageData&, const Image<NV24>&, VPIBackend);

std::error_code UpdateVPIImageWrapper(VPIImage& image, VPIImageData& imageWrap, bool isCPU) {
    VPIStatus status = VPI_SUCCESS;
    status           = vpiImageSetWrapper(image, &imageWrap);
    return make_error_code(status);
}

std::error_code DestroyVPIImageWrapper(VPIImage& image, VPIImageData& imageWrap) {
    std::memset(reinterpret_cast<void *>(&imageWrap), 0, sizeof(VPIImageData));
    if (image != nullptr) {
        vpiImageDestroy(image);
    }

    image = nullptr;

    return ErrorCode::SUCCESS;
}
std::error_code VPITensorStream::Status() noexcept {
    return ErrorCode::SUCCESS;
}

std::error_code VPITensorStream::SyncStream() noexcept {
    if (m_stream) {
        return make_error_code(vpiStreamSync(m_stream));
    } else {
        return ErrorCode::INVALID_ARGUMENT;
    }
}

std::error_code VPITensorStream::Resize(Image<RGB_U8>& outputImage,
    const Image<RGB_U8>& inputImage, InterpolationType interpolation, BorderType border) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    std::error_code err_code;
    err_code = m_resizer->execute<RGB_U8>(outputImage, inputImage, interpolation,
        border, m_stream, m_backend);
    return err_code;
}

std::error_code VPITensorStream::Resize(Image<NV12>& outputImage, const Image<NV12>& inputImage,
    InterpolationType interpolation, BorderType border) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    std::error_code err_code;
    err_code = m_resizer->execute<NV12>(outputImage, inputImage, interpolation,
        border, m_stream, m_backend);
    return err_code;
}

std::error_code VPITensorStream::Resize(Image<RGBA_U8>& outputImage,
    const Image<RGBA_U8>& inputImage, InterpolationType interpolation, BorderType border) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    std::error_code err_code;
    err_code = m_resizer->execute<RGBA_U8>(outputImage, inputImage,
        interpolation, border, m_stream, m_backend);
    return err_code;
}

std::error_code VPITensorStream::Resize(Image<BGR_U8>& outputImage,
    const Image<BGR_U8>& inputImage, InterpolationType interpolation, BorderType border) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};

    std::error_code err_code;
    err_code = m_resizer->execute<BGR_U8>(outputImage, inputImage, interpolation,
        border, m_stream, m_backend);
    return err_code;
}

std::error_code VPITensorStream::Resize(Image<BGRA_U8>& outputImage,
    const Image<BGRA_U8>& inputImage, InterpolationType interpolation, BorderType border) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    std::error_code err_code;
    err_code = m_resizer->execute<BGRA_U8>(outputImage, inputImage,
        interpolation, border, m_stream, m_backend);
    return err_code;
}

std::error_code VPITensorStream::Resize(Image<NV24>& outputImage,
    const Image<NV24>& inputImage, InterpolationType interpolation, BorderType border) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    std::error_code err_code;
    err_code = m_resizer->execute<NV24>(outputImage, inputImage,
        interpolation, border, m_stream, m_backend);
    return err_code;
}

std::error_code VPITensorStream::Resize(Image<Y_U8>& outputImage,
    const Image<Y_U8>& inputImage, InterpolationType interpolation, BorderType border) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    std::error_code err_code;
    err_code = m_resizer->execute<Y_U8>(outputImage, inputImage,
        interpolation, border, m_stream, m_backend);
    return err_code;
}

std::error_code VPITensorStream::Remap(Image<RGB_U8>& outputImage,
    const Image<RGB_U8>& inputImage, const ImageWarp warp,
    InterpolationType interpolation, BorderType border) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    return m_remapper->execute(outputImage, inputImage,
        reinterpret_cast<VPIImageWarp *>(warp),
        interpolation, border, m_stream, m_backend);
}

std::error_code VPITensorStream::Remap(Image<BGR_U8>& outputImage,
    const Image<BGR_U8>& inputImage, const ImageWarp warp,
    InterpolationType interpolation, BorderType border) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    return m_remapper->execute(outputImage, inputImage,
        reinterpret_cast<VPIImageWarp *>(warp),
        interpolation, border, m_stream, m_backend);
}

std::error_code VPITensorStream::Remap(Image<NV12>& outputImage, const Image<NV12>& inputImage,
    const ImageWarp warp, InterpolationType interpolation, BorderType border) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    return m_remapper->execute(outputImage, inputImage,
        reinterpret_cast<VPIImageWarp *>(warp),
        interpolation, border, m_stream, m_backend);
}

std::error_code VPITensorStream::Remap(Image<NV24>& outputImage, const Image<NV24>& inputImage,
    const ImageWarp warp, InterpolationType interpolation, BorderType border) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    return m_remapper->execute(outputImage, inputImage,
        reinterpret_cast<VPIImageWarp *>(warp),
        interpolation, border, m_stream, m_backend);
}

std::error_code VPITensorStream::ColorConvert(Image<BGR_U8>& outputImage,
    const Image<RGB_U8>& inputImage) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    return m_colorConverter->execute(outputImage, inputImage, m_stream, m_backend);
}

std::error_code VPITensorStream::ColorConvert(Image<RGB_U8>& outputImage,
    const Image<BGR_U8>& inputImage) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    return m_colorConverter->execute(outputImage, inputImage, m_stream, m_backend);
}

std::error_code VPITensorStream::ColorConvert(Image<NV12>& outputImage,
    const Image<BGR_U8>& inputImage) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    return m_colorConverter->execute(outputImage, inputImage, m_stream, m_backend);
}

std::error_code VPITensorStream::ColorConvert(Image<NV24>& outputImage,
    const Image<BGR_U8>& inputImage) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    return m_colorConverter->execute(outputImage, inputImage, m_stream, m_backend);
}

std::error_code VPITensorStream::ColorConvert(Image<BGR_U8>& outputImage,
    const Image<NV12>& inputImage) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    return m_colorConverter->execute(outputImage, inputImage, m_stream, m_backend);
}

std::error_code VPITensorStream::ColorConvert(Image<BGR_U8>& outputImage,
    const Image<NV24>& inputImage) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};

    return m_colorConverter->execute(outputImage, inputImage, m_stream, m_backend);
}

std::error_code VPITensorStream::ColorConvert(Image<NV12>& outputImage,
    const Image<RGB_U8>& inputImage) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    return m_colorConverter->execute(outputImage, inputImage, m_stream, m_backend);
}

std::error_code VPITensorStream::ColorConvert(Image<NV24>& outputImage,
    const Image<RGB_U8>& inputImage) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    return m_colorConverter->execute(outputImage, inputImage, m_stream, m_backend);
}

std::error_code VPITensorStream::ColorConvert(Image<RGB_U8>& outputImage,
    const Image<NV12>& inputImage) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    return m_colorConverter->execute(outputImage, inputImage, m_stream, m_backend);
}

std::error_code VPITensorStream::ColorConvert(Image<RGB_U8>& outputImage,
    const Image<NV24>& inputImage) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    return m_colorConverter->execute(outputImage, inputImage, m_stream, m_backend);
}

std::error_code VPITensorStream::ColorConvert(Image<Y_U8>& outputImage,
    const Image<RGB_U8>& inputImage) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    return m_colorConverter->execute(outputImage, inputImage, m_stream, m_backend);
}

std::error_code VPITensorStream::ColorConvert(Image<Y_U8>& outputImage,
    const Image<BGR_U8>& inputImage) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    return m_colorConverter->execute(outputImage, inputImage, m_stream, m_backend);
}

std::error_code VPITensorStream::ColorConvert(Image<RGB_U8>& outputImage,
    const Image<Y_U8>& inputImage) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    return m_colorConverter->execute(outputImage, inputImage, m_stream, m_backend);
}

std::error_code VPITensorStream::ColorConvert(Image<BGR_U8>& outputImage,
    const Image<Y_U8>& inputImage) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    return m_colorConverter->execute(outputImage, inputImage, m_stream, m_backend);
}

std::error_code VPITensorStream::ColorConvert(Image<Y_U8>& outputImage,
    const Image<NV12>& inputImage) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    return m_colorConverter->execute(outputImage, inputImage, m_stream, m_backend);
}

std::error_code VPITensorStream::ColorConvert(Image<NV12>& outputImage,
    const Image<Y_U8>& inputImage) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    return m_colorConverter->execute(outputImage, inputImage, m_stream, m_backend);
}

std::error_code VPITensorStream::ColorConvert(Image<Y_U8>& outputImage,
    const Image<NV24>& inputImage) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    return m_colorConverter->execute(outputImage, inputImage, m_stream, m_backend);
}

std::error_code VPITensorStream::ColorConvert(Image<NV24>& outputImage,
    const Image<Y_U8>& inputImage) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    return m_colorConverter->execute(outputImage, inputImage, m_stream, m_backend);
}

std::error_code VPITensorStream::ColorConvert(Image<BGR_U8>& outputImage,
    const Image<BGRA_U8>& inputImage) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    return m_colorConverter->execute(outputImage, inputImage, m_stream, m_backend);
}

std::error_code VPITensorStream::ColorConvert(Image<RGB_U8>& outputImage,
    const Image<RGBA_U8>& inputImage) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    return m_colorConverter->execute(outputImage, inputImage, m_stream, m_backend);
}

std::error_code VPITensorStream::ColorConvert(Image<BGR_U8>& outputImage,
    const Image<RGBA_U8>& inputImage) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    return m_colorConverter->execute(outputImage, inputImage, m_stream, m_backend);
}

std::error_code VPITensorStream::ColorConvert(Image<RGB_U8>& outputImage,
    const Image<BGRA_U8>& inputImage) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    return m_colorConverter->execute(outputImage, inputImage, m_stream, m_backend);
}

std::error_code VPITensorStream::ColorConvert(Image<BGRA_U8>& outputImage,
    const Image<BGR_U8>& inputImage) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    return m_colorConverter->execute(outputImage, inputImage, m_stream, m_backend);
}

std::error_code VPITensorStream::ColorConvert(Image<RGBA_U8>& outputImage,
    const Image<RGB_U8>& inputImage) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    return m_colorConverter->execute(outputImage, inputImage, m_stream, m_backend);
}

std::error_code VPITensorStream::ColorConvert(Image<BGRA_U8>& outputImage,
    const Image<RGB_U8>& inputImage) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    return m_colorConverter->execute(outputImage, inputImage, m_stream, m_backend);
}

std::error_code VPITensorStream::ColorConvert(Image<RGBA_U8>& outputImage,
    const Image<BGR_U8>& inputImage) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    return m_colorConverter->execute(outputImage, inputImage, m_stream, m_backend);
}

std::error_code VPITensorStream::ColorConvert(Image<NV12>& outputImage,
    const Image<BGRA_U8>& inputImage) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    return m_colorConverter->execute(outputImage, inputImage, m_stream, m_backend);
}

std::error_code VPITensorStream::ColorConvert(Image<NV12>& outputImage,
    const Image<RGBA_U8>& inputImage) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    return m_colorConverter->execute(outputImage, inputImage, m_stream, m_backend);
}

std::error_code VPITensorStream::ColorConvert(Image<BGRA_U8>& outputImage,
    const Image<NV12>& inputImage) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    return m_colorConverter->execute(outputImage, inputImage, m_stream, m_backend);
}

std::error_code VPITensorStream::ColorConvert(Image<RGBA_U8>& outputImage,
    const Image<NV12>& inputImage) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    return m_colorConverter->execute(outputImage, inputImage, m_stream, m_backend);
}

std::error_code VPITensorStream::StereoDisparityEstimator(Image<Y_F32>& outputImage,
    const Image<Y_U8>& inputLeftImage, const Image<Y_U8>& inputRightImage,
    size_t windowSize, size_t maxDisparity) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    return m_stereoDisparityEstimator->execute(outputImage, inputLeftImage,
        inputRightImage, windowSize, maxDisparity, m_stream, m_backend);
}

std::error_code VPITensorStream::StereoDisparityEstimator(Image<Y_F32>& outputImage,
    const Image<NV12>& inputLeftImage, const Image<NV12>& inputRightImage,
    size_t windowSize, size_t maxDisparity) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    return m_stereoDisparityEstimator->execute(outputImage, inputLeftImage,
        inputRightImage, windowSize, maxDisparity,
        m_stream, m_backend);
}

std::error_code VPITensorStream::StereoDisparityEstimator(Image<Y_F32>& outputImage,
    const Image<NV24>& inputLeftImage, const Image<NV24>& inputRightImage,
    size_t windowSize, size_t maxDisparity) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    return m_stereoDisparityEstimator->execute(outputImage, inputLeftImage,
        inputRightImage, windowSize, maxDisparity, m_stream, m_backend);
}

TensorBackend VPITensorContext::Backend() const noexcept {
    return TensorBackend::VPI;
}

std::error_code VPITensorStream::GenerateWarpFromCameraModel(ImageWarp& warp,
    const ImageGrid& grid, const CameraModel& source, const CameraIntrinsics& target) {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    VPIStatus status = VPI_SUCCESS;

    VPIWarpMap map           = {0};
    map.grid.numHorizRegions = grid.numHorizRegions;
    for (std::size_t i = 0; i < static_cast<std::size_t>(grid.numHorizRegions); i++) {
        map.grid.regionWidth[i]   = grid.regionWidth[i];
        map.grid.horizInterval[i] = grid.horizInterval[i];
    }
    map.grid.numVertRegions = grid.numVertRegions;
    for (std::size_t i = 0; i < static_cast<std::size_t>(grid.numVertRegions); i++) {
        map.grid.regionHeight[i] = grid.regionHeight[i];
        map.grid.vertInterval[i] = grid.vertInterval[i];
    }
    status = vpiWarpMapAllocData(&map);

    if ((status == VPI_SUCCESS) && (map.keypoints)) {
        switch (source.distortion.type) {
            case CameraDistortionType::Polynomial: {
                VPIPolynomialLensDistortionModel distortion;
                distortion.k1 = source.distortion.k1;
                distortion.k2 = source.distortion.k2;
                distortion.k3 = source.distortion.k3;
                distortion.k4 = source.distortion.k4;
                distortion.k5 = source.distortion.k5;
                distortion.k6 = source.distortion.k6;
                distortion.p1 = source.distortion.p1;
                distortion.p2 = source.distortion.p2;
                status        = vpiWarpMapGenerateFromPolynomialLensDistortionModel(
                    source.intrinsic.m_intrinsics, source.extrinsic.m_extrinsics,
                    target.m_intrinsics, &distortion, &map);
                break;
            }
            case CameraDistortionType::FisheyeEquidistant: {
                VPIFisheyeLensDistortionModel distortion;
                distortion.k1      = source.distortion.k1;
                distortion.k2      = source.distortion.k2;
                distortion.k3      = source.distortion.k3;
                distortion.k4      = source.distortion.k4;
                distortion.mapping = VPI_FISHEYE_EQUIDISTANT;
                status             = vpiWarpMapGenerateFromFisheyeLensDistortionModel(
                    source.intrinsic.m_intrinsics, source.extrinsic.m_extrinsics,
                    target.m_intrinsics, &distortion, &map);
                break;
            }
            case CameraDistortionType::FisheyeEquisolid: {
                VPIFisheyeLensDistortionModel distortion;
                distortion.k1      = source.distortion.k1;
                distortion.k2      = source.distortion.k2;
                distortion.k3      = source.distortion.k3;
                distortion.k4      = source.distortion.k4;
                distortion.mapping = VPI_FISHEYE_EQUISOLID;
                status             = vpiWarpMapGenerateFromFisheyeLensDistortionModel(
                    source.intrinsic.m_intrinsics, source.extrinsic.m_extrinsics,
                    target.m_intrinsics, &distortion, &map);
                break;
            }
            case CameraDistortionType::FisheyeOrthoGraphic: {
                VPIFisheyeLensDistortionModel distortion;
                distortion.k1      = source.distortion.k1;
                distortion.k2      = source.distortion.k2;
                distortion.k3      = source.distortion.k3;
                distortion.k4      = source.distortion.k4;
                distortion.mapping = VPI_FISHEYE_ORTHOGRAPHIC;
                status             = vpiWarpMapGenerateFromFisheyeLensDistortionModel(
                    source.intrinsic.m_intrinsics, source.extrinsic.m_extrinsics,
                    target.m_intrinsics, &distortion, &map);
                break;
            }
            case CameraDistortionType::FisheyeStereographic: {
                VPIFisheyeLensDistortionModel distortion;
                distortion.k1      = source.distortion.k1;
                distortion.k2      = source.distortion.k2;
                distortion.k3      = source.distortion.k3;
                distortion.k4      = source.distortion.k4;
                distortion.mapping = VPI_FISHEYE_STEREOGRAPHIC;
                status             = vpiWarpMapGenerateFromFisheyeLensDistortionModel(
                    source.intrinsic.m_intrinsics, source.extrinsic.m_extrinsics,
                    target.m_intrinsics, &distortion, &map);
                break;
            }
            default:
                status = VPI_ERROR_INVALID_ARGUMENT;
                break;
        }
    }

    if ((status == VPI_SUCCESS) && (map.keypoints)) {
        if (warp != nullptr) {
            vpiPayloadDestroy(reinterpret_cast<VPIImageWarp *>(warp)->payload);
            delete warp;
        }
        warp   = new VPIImageWarp;
        status = vpiCreateRemap(m_backend, &map,
            &(reinterpret_cast<VPIImageWarp *>(warp)->payload));
    }

    // Delete map after payload is generated
    vpiWarpMapFreeData(&map);

    return make_error_code(status);
}

std::error_code VPITensorStream::DestroyWarp(ImageWarp& warp) noexcept {
    std::unique_lock<decltype(m_fence)> scopedLock{m_fence};
    if (warp != nullptr) {
        try {
            vpiPayloadDestroy(reinterpret_cast<VPIImageWarp *>(warp)->payload);
        }
        catch (std::error_code &e) {
            return e;
        }

        delete reinterpret_cast<VPIImageWarp *>(warp);
        warp = nullptr;
    }
    return make_error_code(ErrorCode::SUCCESS);
}

}  // namespace tensor_ops
}  // namespace cvcore
