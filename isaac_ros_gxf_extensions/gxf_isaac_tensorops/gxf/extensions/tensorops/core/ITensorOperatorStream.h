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

#include <exception>
#include <memory>
#include <system_error>
#include <vector>

#include "extensions/tensorops/core/CameraModel.h"
#include "extensions/tensorops/core/ComputeEngine.h"
#include "extensions/tensorops/core/Errors.h"
#include "extensions/tensorops/core/IImageWarp.h"
#include "extensions/tensorops/core/Image.h"
#include "extensions/tensorops/core/ImageUtils.h"

namespace cvcore {
namespace tensor_ops {

class NotImplementedException : public std::logic_error {
 public:
    NotImplementedException()
        : std::logic_error("Method not yet implemented.") {
    }
};

class ITensorOperatorStream {
 public:
    // Public Constructor(s)/Destructor
    virtual ~ITensorOperatorStream() noexcept = default;

    // Public Accessor Method(s)
    virtual std::error_code Status() noexcept = 0;

    virtual std::error_code GenerateWarpFromCameraModel(ImageWarp& warp,
                                                        const ImageGrid& grid,
                                                        const CameraModel& source,
                                                        const CameraIntrinsics& target) = 0;

    virtual std::error_code DestroyWarp(ImageWarp & warp) noexcept = 0;

    // Public Mutator Method(s)
    virtual std::error_code Remap(Image<RGB_U8> & outputImage, const Image<RGB_U8> & inputImage,
                                  const ImageWarp warp,
                                  InterpolationType interpolation = INTERP_LINEAR,
                                  BorderType border = BORDER_ZERO) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }
    virtual std::error_code Remap(Image<BGR_U8> & outputImage, const Image<BGR_U8> & inputImage,
                                  const ImageWarp warp,
                                  InterpolationType interpolation = INTERP_LINEAR,
                                  BorderType border = BORDER_ZERO) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

    virtual std::error_code Remap(Image<NV12> & outputImage,
        const Image<NV12> & inputImage,
        const ImageWarp warp,
        InterpolationType interpolation = INTERP_LINEAR,
        BorderType border = BORDER_ZERO) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

    virtual std::error_code Remap(Image<NV24> & outputImage,
        const Image<NV24> & inputImage,
        const ImageWarp warp,
        InterpolationType interpolation = INTERP_LINEAR,
        BorderType border = BORDER_ZERO) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

    virtual std::error_code Resize(Image<RGB_U8> & outputImage,
        const Image<RGB_U8> & inputImage,
        InterpolationType interpolation = INTERP_LINEAR,
        BorderType border = BORDER_ZERO) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }
    virtual std::error_code Resize(Image<BGR_U8> & outputImage,
        const Image<BGR_U8> & inputImage,
        InterpolationType interpolation = INTERP_LINEAR,
        BorderType border = BORDER_ZERO) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }
    virtual std::error_code Resize(Image<NV12> & outputImage,
        const Image<NV12> & inputImage,
        InterpolationType interpolation = INTERP_LINEAR,
        BorderType border = BORDER_ZERO) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }
    virtual std::error_code Resize(Image<RGBA_U8> & outputImage,
        const Image<RGBA_U8> & inputImage,
        InterpolationType interpolation = INTERP_LINEAR,
        BorderType border = BORDER_ZERO) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }
    virtual std::error_code Resize(Image<BGRA_U8> & outputImage,
        const Image<BGRA_U8> & inputImage,
        InterpolationType interpolation = INTERP_LINEAR,
        BorderType border = BORDER_ZERO) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }
    virtual std::error_code Resize(Image<NV24> & outputImage,
        const Image<NV24> & inputImage,
        InterpolationType interpolation = INTERP_LINEAR,
        BorderType border = BORDER_ZERO) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }
    virtual std::error_code Resize(Image<Y_U8> & outputImage,
        const Image<Y_U8> & inputImage,
        InterpolationType interpolation = INTERP_LINEAR,
        BorderType border = BORDER_ZERO) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

    virtual std::error_code Normalize(Image<RGB_U8> & outputImage,
        const Image<RGB_U8> & inputImage) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }
    virtual std::error_code Normalize(Image<BGR_U8> & outputImage,
        const Image<BGR_U8> & inputImage) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }
    virtual std::error_code Normalize(Image<NV12> & outputImage,
        const Image<NV12> & inputImage) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }
    virtual std::error_code Normalize(Image<NV24> & outputImage,
        const Image<NV24> & inputImage) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

    virtual std::error_code ColorConvert(Image<BGR_U8> & outputImage,
        const Image<RGB_U8> & inputImage) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

    virtual std::error_code ColorConvert(Image<RGB_U8> & outputImage,
        const Image<BGR_U8> & inputImage) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

    virtual std::error_code ColorConvert(Image<BGR_U8> & outputImage,
        const Image<NV12> & inputImage) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

    virtual std::error_code ColorConvert(Image<BGR_U8> & outputImage,
        const Image<NV24> & inputImage) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

    virtual std::error_code ColorConvert(Image<NV12> & outputImage,
        const Image<BGR_U8> & inputImage) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

    virtual std::error_code ColorConvert(Image<NV24> & outputImage,
        const Image<BGR_U8> & inputImage) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

    virtual std::error_code ColorConvert(Image<RGB_U8> & outputImage,
        const Image<NV12> & inputImage) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

    virtual std::error_code ColorConvert(Image<RGB_U8> & outputImage,
        const Image<NV24> & inputImage) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

    virtual std::error_code ColorConvert(Image<NV12> & outputImage,
        const Image<RGB_U8> & inputImage) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

    virtual std::error_code ColorConvert(Image<NV24> & outputImage,
        const Image<RGB_U8> & inputImage) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

    virtual std::error_code ColorConvert(Image<Y_U8> & outputImage,
        const Image<RGB_U8> & inputImage) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

    virtual std::error_code ColorConvert(Image<RGB_U8> & outputImage,
        const Image<Y_U8> & inputImage) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

    virtual std::error_code ColorConvert(Image<Y_U8> & outputImage,
        const Image<BGR_U8> & inputImage) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

    virtual std::error_code ColorConvert(Image<BGR_U8> & outputImage,
        const Image<Y_U8> & inputImage) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

    virtual std::error_code ColorConvert(Image<Y_U8> & outputImage,
        const Image<NV12> & inputImage) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

    virtual std::error_code ColorConvert(Image<NV12> & outputImage,
        const Image<Y_U8> & inputImage) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

    virtual std::error_code ColorConvert(Image<Y_U8> & outputImage,
        const Image<NV24> & inputImage) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

    virtual std::error_code ColorConvert(Image<NV24> & outputImage,
        const Image<Y_U8> & inputImage) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

    virtual std::error_code ColorConvert(Image<BGR_U8> & outputImage,
        const Image<BGRA_U8> & inputImage) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

    virtual std::error_code ColorConvert(Image<RGB_U8> & outputImage,
        const Image<RGBA_U8> & inputImage) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

    virtual std::error_code ColorConvert(Image<BGR_U8> & outputImage,
        const Image<RGBA_U8> & inputImage) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

    virtual std::error_code ColorConvert(Image<RGB_U8> & outputImage,
        const Image<BGRA_U8> & inputImage) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

    virtual std::error_code ColorConvert(Image<BGRA_U8> & outputImage,
        const Image<BGR_U8> & inputImage) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

    virtual std::error_code ColorConvert(Image<RGBA_U8> & outputImage,
        const Image<RGB_U8> & inputImage) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

    virtual std::error_code ColorConvert(Image<BGRA_U8> & outputImage,
        const Image<RGB_U8> & inputImage) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

    virtual std::error_code ColorConvert(Image<RGBA_U8> & outputImage,
        const Image<BGR_U8> & inputImage) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

    virtual std::error_code ColorConvert(Image<BGRA_U8> & outputImage,
        const Image<NV12> & inputImage) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

    virtual std::error_code ColorConvert(Image<RGBA_U8> & outputImage,
        const Image<NV12> & inputImage) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

    virtual std::error_code ColorConvert(Image<NV12> & outputImage,
        const Image<BGRA_U8> & inputImage) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

    virtual std::error_code ColorConvert(Image<NV12> & outputImage,
        const Image<RGBA_U8> & inputImage) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

    virtual std::error_code StereoDisparityEstimator(Image<Y_F32> & outputImage,
        const Image<Y_U8> & inputLeftImage, const Image<Y_U8> & inputRightImage,
        size_t windowSize, size_t maxDisparity) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

    virtual std::error_code StereoDisparityEstimator(Image<Y_F32> & outputImage,
        const Image<NV12> & inputLeftImage, const Image<NV12> & inputRightImage,
        size_t windowSize, size_t maxDisparity) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

    virtual std::error_code StereoDisparityEstimator(Image<Y_F32> & outputImage,
        const Image<NV24> & inputLeftImage, const Image<NV24> & inputRightImage,
        size_t windowSize, size_t maxDisparity) {
        throw NotImplementedException();
        return make_error_code(ErrorCode::NOT_IMPLEMENTED);
    }

 protected:
    // Protected Constructor(s)
    ITensorOperatorStream()                                  = default;
    ITensorOperatorStream(const ITensorOperatorStream&)     = default;
    ITensorOperatorStream(ITensorOperatorStream &&) noexcept = default;

    // Protected Operator(s)
    ITensorOperatorStream& operator=(const ITensorOperatorStream&) = default;
    ITensorOperatorStream& operator=(ITensorOperatorStream&&) noexcept = default;
};

using TensorOperatorStream = ITensorOperatorStream *;

}  // namespace tensor_ops
}  // namespace cvcore
