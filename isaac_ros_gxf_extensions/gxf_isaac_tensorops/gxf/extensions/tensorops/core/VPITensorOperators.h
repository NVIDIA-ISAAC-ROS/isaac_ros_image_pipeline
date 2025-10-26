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

#pragma once

#include <memory>
#include <mutex>
#include <type_traits>
#include "extensions/tensorops/core/CameraModel.h"
#include "extensions/tensorops/core/ITensorOperatorContext.h"
#include "extensions/tensorops/core/ITensorOperatorStream.h"
#include "extensions/tensorops/core/VPIEnumMapping.h"
#include "vpi/CUDAInterop.h"
#include "vpi/Image.h"
#include "vpi/Stream.h"
#include "vpi/Types.h"

namespace cvcore {
namespace tensor_ops {

/**
 * Returns the corresponding VPI backend given the cvcore compute engine.
 * @param computeEngine Compute engine used.
 * @return Returns the VPIbackend.
 */
VPIBackend getVPIBackend(const ComputeEngine & computeEngine);

/**
 * Wraps a CVCore Image type into a VPIImage
 * @param vpiImage VPIImage
 * @param imgdata VPIImage data
 * @param cvcoreImage CVCore image
 * @param backend Compute backend
 * @return error code
 */
template<ImageType T>
std::error_code CreateVPIImageWrapper(VPIImage & vpiImg, VPIImageData & imgdata,
    const Image<T> & cvcoreImage, VPIBackend backend);

/**
 * Update VPI Image data pointer
 * @param vpiImage VPIImage
 * @param imgdata VPIImage data
 * @param isCPU data is on CPU or GPU
 * @return error code
 */
std::error_code UpdateVPIImageWrapper(VPIImage & image, VPIImageData & imageWrap, bool isCPU);

/**
 * Destory Wrapped VPI Image
 * @param vpiImage VPIImage
 * @param imgdata VPIImage data
 * @return error code
 */
std::error_code DestroyVPIImageWrapper(VPIImage & image, VPIImageData & imageWrap);

/**
 * Update VPI Image given CVCORE Image
 * @param vpiImage VPIImage
 * @param vpiImageData VPIImage data
 * @param image CVCORE Image
 * @return error code
 */
template<ImageType T,
         typename std::enable_if<IsInterleavedImage<T>::value>::type * = nullptr>
std::error_code UpdateImage(VPIImage & vpiImage, VPIImageData & vpiImageData,
    const Image<T> & image) {
    using D = typename Image<T>::DataType;
    D* data = const_cast<D *>(image.getData());
    vpiImageData.buffer.pitch.planes[0].pBase = reinterpret_cast<uint8_t *>(data);
    return UpdateVPIImageWrapper(vpiImage, vpiImageData, image.isCPU());
}

/**
 * Update VPI Image given CVCORE Image
 * @param vpiImage VPIImage
 * @param vpiImageData VPIImage data
 * @param image CVCORE Image
 * @return error code
 */
template<ImageType T, typename std::enable_if<IsCompositeImage<T>::value>::type * = nullptr>
std::error_code UpdateImage(VPIImage & vpiImage, VPIImageData & vpiImageData,
    const Image<T> & image) {
    vpiImageData.buffer.pitch.planes[0].pBase = const_cast<uint8_t *>(image.getLumaData());
    vpiImageData.buffer.pitch.planes[1].pBase = const_cast<uint8_t *>(image.getChromaData());
    return UpdateVPIImageWrapper(vpiImage, vpiImageData, image.isCPU());
}

/**
 * Check if params of VPIImageData is consistent with the given CVCORE Image
 * @param vpiImageData VPIImage data
 * @param image CVCORE Image
 * @return whether param changed
 */
template<ImageType T,
    typename std::enable_if<!IsCompositeImage<T>::value &&
    !IsPlanarImage<T>::value>::type * = nullptr>
bool CheckParamsChanged(VPIImageData & vpiImageData, const Image<T> & image) {
    bool paramsChanged = false;
    // Did format change
    paramsChanged = paramsChanged || vpiImageData.buffer.pitch.format != ToVpiImageFormat(T);
    // Did image dimension change
    paramsChanged =
        paramsChanged || (vpiImageData.buffer.pitch.planes[0].height !=
                              static_cast<std::int32_t>(image.getHeight()) ||
                          vpiImageData.buffer.pitch.planes[0].width !=
                              static_cast<std::int32_t>(image.getWidth()));
    return paramsChanged;
}

/**
 * Check if params of VPIImageData is consistent with the given CVCORE Image
 * @param vpiImageData VPIImage data
 * @param image CVCORE Image
 * @return whether param changed
 */
template<ImageType T, typename std::enable_if<IsCompositeImage<T>::value>::type * = nullptr>
bool CheckParamsChanged(VPIImageData & vpiImageData, const Image<T> & image) {
    bool paramsChanged = false;

    // Did format change
    paramsChanged = paramsChanged || vpiImageData.buffer.pitch.format != ToVpiImageFormat(T);

    // Did image dimension change
    paramsChanged = paramsChanged ||
                    (vpiImageData.buffer.pitch.planes[0].height !=
                        static_cast<std::int32_t>(image.getLumaHeight()) ||
                     vpiImageData.buffer.pitch.planes[0].width !=
                         static_cast<std::int32_t>(image.getLumaWidth()) ||
                     vpiImageData.buffer.pitch.planes[1].height !=
                         static_cast<std::int32_t>(image.getChromaHeight()) ||
                     vpiImageData.buffer.pitch.planes[1].width !=
                         static_cast<std::int32_t>(image.getChromaWidth()));
    return paramsChanged;
}

/**
 * Implementation of VPITensorContext
 */
class VPITensorContext : public ITensorOperatorContext {
 public:
    /**
    * Default Constructor for VPI Context.
    */
    VPITensorContext() = default;

    /**
    * Default Destructor for VPI Context.
    */
    ~VPITensorContext() = default;

    /**
     * Creates a stream based on compute engine
     * @param computeEngine CVCore Compute engine
     * @return Pointer to ITensorOperatorStream object.
     */
    std::error_code CreateStream(TensorOperatorStream&,
        const ComputeEngine & computeEngine) override;

    /**
    * Destroy stream creates.
    * @param inputStream Input stream to be deleted
    * @return Error code
    */
    std::error_code DestroyStream(TensorOperatorStream & inputStream) override;

    /**
    * Checks if stream type is supported for a given backend.
    * @param computeEngine CVCore Compute engine
    * @return true if stream type is available.
    */
    bool IsComputeEngineCompatible(const ComputeEngine & computeEngine) const noexcept override;

    /**
    * Returns the backend type
    */
    TensorBackend Backend() const noexcept override;

 private:
};

/**
 * Implementation of VPITensorStream
 */
class VPITensorStream : public ITensorOperatorStream {
 public:
    std::error_code Status() noexcept override;

    std::error_code SyncStream() noexcept override;

    std::error_code GenerateWarpFromCameraModel(ImageWarp & warp, const ImageGrid & grid,
        const CameraModel & source, const CameraIntrinsics & target) override;
    std::error_code DestroyWarp(ImageWarp & warp) noexcept override;

     std::error_code Remap(Image<RGB_U8> & outputImage, const Image<RGB_U8> & inputImage,
        const ImageWarp warp, InterpolationType interpolation, BorderType border) override;

    std::error_code Remap(Image<BGR_U8> & outputImage, const Image<BGR_U8> & inputImage,
        const ImageWarp warp, InterpolationType interpolation, BorderType border) override;

    std::error_code Remap(Image<NV12> & outputImage, const Image<NV12> & inputImage,
        const ImageWarp warp, InterpolationType interpolation, BorderType border) override;

    std::error_code Remap(Image<NV24> & outputImage, const Image<NV24> & inputImage,
        const ImageWarp warp, InterpolationType interpolation, BorderType border) override;

    std::error_code Resize(Image<RGB_U8> & outputImage, const Image<RGB_U8> & inputImage,
        InterpolationType interpolation, BorderType border) override;
    std::error_code Resize(Image<RGBA_U8> & outputImage, const Image<RGBA_U8> & inputImage,
        InterpolationType interpolation, BorderType border) override;
    std::error_code Resize(Image<NV24> & outputImage, const Image<NV24> & inputImage,
        InterpolationType interpolation, BorderType border) override;
    std::error_code Resize(Image<NV12> & outputImage, const Image<NV12> & inputImage,
        InterpolationType interpolation, BorderType border) override;
    std::error_code Resize(Image<BGR_U8> & outputImage, const Image<BGR_U8> & inputImage,
        InterpolationType interpolation, BorderType border) override;
    std::error_code Resize(Image<BGRA_U8> & outputImage, const Image<BGRA_U8> & inputImage,
        InterpolationType interpolation, BorderType border) override;
    std::error_code Resize(Image<Y_U8> & outputImage, const Image<Y_U8> & inputImage,
        InterpolationType interpolation, BorderType border) override;

    std::error_code ColorConvert(Image<BGR_U8> & outputImage,
        const Image<RGB_U8> & inputImage) override;
    std::error_code ColorConvert(Image<RGB_U8> & outputImage,
        const Image<BGR_U8> & inputImage) override;
    std::error_code ColorConvert(Image<NV12> & outputImage,
        const Image<BGR_U8> & inputImage) override;
    std::error_code ColorConvert(Image<NV24> & outputImage,
        const Image<BGR_U8> & inputImage) override;
    std::error_code ColorConvert(Image<BGR_U8> & outputImage,
        const Image<NV12> & inputImage) override;
    std::error_code ColorConvert(Image<BGR_U8> & outputImage,
        const Image<NV24> & inputImage) override;
    std::error_code ColorConvert(Image<NV12> & outputImage,
        const Image<RGB_U8> & inputImage) override;
    std::error_code ColorConvert(Image<NV24> & outputImage,
        const Image<RGB_U8> & inputImage) override;
    std::error_code ColorConvert(Image<RGB_U8> & outputImage,
        const Image<NV12> & inputImage) override;
    std::error_code ColorConvert(Image<RGB_U8> & outputImage,
        const Image<NV24> & inputImage) override;
    std::error_code ColorConvert(Image<Y_U8> & outputImage,
        const Image<BGR_U8> & inputImage) override;
    std::error_code ColorConvert(Image<Y_U8> & outputImage,
        const Image<RGB_U8> & inputImage) override;
    std::error_code ColorConvert(Image<BGR_U8> & outputImage,
        const Image<Y_U8> & inputImage) override;
    std::error_code ColorConvert(Image<RGB_U8> & outputImage,
        const Image<Y_U8> & inputImage) override;
    std::error_code ColorConvert(Image<NV12> & outputImage,
        const Image<Y_U8> & inputImage) override;
    std::error_code ColorConvert(Image<Y_U8> & outputImage,
        const Image<NV12> & inputImage) override;
    std::error_code ColorConvert(Image<NV24> & outputImage,
        const Image<Y_U8> & inputImage) override;
    std::error_code ColorConvert(Image<Y_U8> & outputImage,
        const Image<NV24> & inputImage) override;
    std::error_code ColorConvert(Image<BGR_U8> & outputImage,
        const Image<BGRA_U8> & inputImage) override;
    std::error_code ColorConvert(Image<RGB_U8> & outputImage,
        const Image<RGBA_U8> & inputImage) override;
    std::error_code ColorConvert(Image<BGR_U8> & outputImage,
        const Image<RGBA_U8> & inputImage) override;
    std::error_code ColorConvert(Image<RGB_U8> & outputImage,
        const Image<BGRA_U8> & inputImage) override;
    std::error_code ColorConvert(Image<BGRA_U8> & outputImage,
        const Image<BGR_U8> & inputImage) override;
    std::error_code ColorConvert(Image<RGBA_U8> & outputImage,
        const Image<RGB_U8> & inputImage) override;
    std::error_code ColorConvert(Image<BGRA_U8> & outputImage,
        const Image<RGB_U8> & inputImage) override;
    std::error_code ColorConvert(Image<RGBA_U8> & outputImage,
        const Image<BGR_U8> & inputImage) override;
    std::error_code ColorConvert(Image<BGRA_U8> & outputImage,
        const Image<NV12> & inputImage) override;
    std::error_code ColorConvert(Image<RGBA_U8> & outputImage,
        const Image<NV12> & inputImage) override;
    std::error_code ColorConvert(Image<NV12> & outputImage,
        const Image<BGRA_U8> & inputImage) override;
    std::error_code ColorConvert(Image<NV12> & outputImage,
        const Image<RGBA_U8> & inputImage) override;

    std::error_code StereoDisparityEstimator(Image<Y_F32> & outputImage,
        const Image<Y_U8> & inputLeftImage,
        const Image<Y_U8> & inputRightImage, size_t windowSize,
        size_t maxDisparity) override;
    std::error_code StereoDisparityEstimator(Image<Y_F32> & outputImage,
        const Image<NV12> & inputLeftImage,
        const Image<NV12> & inputRightImage, size_t windowSize,
        size_t maxDisparity) override;
    std::error_code StereoDisparityEstimator(Image<Y_F32> & outputImage,
        const Image<NV24> & inputLeftImage,
        const Image<NV24> & inputRightImage, size_t windowSize,
        size_t maxDisparity) override;

 protected:
    friend class VPITensorContext;

    VPITensorStream(const ComputeEngine & computeEngine);
    ~VPITensorStream();

 private:
    class VPIResizeImpl;
    class VPIRemapImpl;
    class VPIColorConvertImpl;
    class VPIStereoDisparityEstimatorImpl;

    mutable std::mutex m_fence;

    std::unique_ptr<VPIResizeImpl> m_resizer;
    std::unique_ptr<VPIRemapImpl> m_remapper;
    std::unique_ptr<VPIColorConvertImpl> m_colorConverter;
    std::unique_ptr<VPIStereoDisparityEstimatorImpl> m_stereoDisparityEstimator;

    VPIStream m_stream;
    VPIBackend m_backend;
};

}  // namespace tensor_ops
}  // namespace cvcore
