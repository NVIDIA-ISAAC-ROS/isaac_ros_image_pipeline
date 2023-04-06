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

#ifndef CVCORE_CAMERAMODEL_H
#define CVCORE_CAMERAMODEL_H

#include <array>

#include "cv/core/Array.h"
#include "cv/core/MathTypes.h"

namespace cvcore {

/**
 * An enum.
 * Enum type for Camera Distortion type.
 */
enum class CameraDistortionType : uint8_t
{
    UNKNOWN,                    /**< Unknown arbitrary distortion model. */
    NONE,                       /**< No distortion applied. */
    Polynomial,                 /**< Polynomial distortion model. */
    FisheyeEquidistant,         /**< Equidistant Fisheye distortion model. */
    FisheyeEquisolid,           /**< Equisolid Fisheye distortion model. */
    FisheyeOrthoGraphic,        /**< Orthographic Fisheye distortion model. */
    FisheyeStereographic        /**< Stereographic Fisheye distortion model. */
};

/**
 * Struct type used to store Camera Distortion model type and coefficients.
 */
struct CameraDistortionModel
{
    CameraDistortionType type;  /**< Camera distortion model type. */
    union                       /**< Camera distortion model coefficients. */
    {
        float coefficients[8];
        struct
        {
            float k1, k2, k3, k4, k5, k6;
            float p1, p2;
        };
    };

    CameraDistortionModel()
        : type(CameraDistortionType::UNKNOWN),
          k1(0.0), k2(0.0), k3(0.0), k4(0.0), k5(0.0), k6(0.0),
          p1(0.0), p2(0.0) {}

    /**
    * Camera Distortion Model creation using array of coefficients.
    * @param distortionType Camera distortion model type
    * @param distortionCoefficients An array of camera distortion model coefficients
    * @return Camera Distortion Model
    */
    CameraDistortionModel(CameraDistortionType distortionType, std::array<float, 8> & distortionCoefficients)
        : type(distortionType)
    {
        std::copy(distortionCoefficients.begin(), distortionCoefficients.end(), std::begin(coefficients));
    }

    /**
    * Camera Distortion Model creation using individual coefficients.
    * @param distortionType Camera distortion model type
    * @param k1 Camera distortion model coefficient - k1
    * @param k2 Camera distortion model coefficient - k2
    * @param k3 Camera distortion model coefficient - k3
    * @param k4 Camera distortion model coefficient - k4
    * @param k5 Camera distortion model coefficient - k5
    * @param k6 Camera distortion model coefficient - k6
    * @param p1 Camera distortion model coefficient - p1
    * @param p2 Camera distortion model coefficient - p2
    * @return Camera Distortion Model
    */
    CameraDistortionModel(CameraDistortionType distortionType, float k1, float k2, float k3, \
                         float k4, float k5, float k6, float p1, float p2)
        : type(distortionType)
        , k1(k1)
        , k2(k2)
        , k3(k3)
        , k4(k4)
        , k5(k5)
        , k6(k6)
        , p1(p1)
        , p2(p2)
    {

    }

    /**
    * Get camera distortion model type.
    * @return Camera distortion model type
    */
    CameraDistortionType getDistortionType() const
    {
        return type;
    }

    /**
    * Get camera distortion model coefficients.
    * @return Camera distortion model coefficients array
    */
    const float * getCoefficients() const
    {
        return &coefficients[0];
    }

    inline bool operator==(const CameraDistortionModel & other) const noexcept
    {
        return this->k1 == other.k1 &&
               this->k2 == other.k2 && 
               this->k3 == other.k3 && 
               this->k4 == other.k4 && 
               this->k5 == other.k5 && 
               this->k6 == other.k6 && 
               this->p1 == other.p1 && 
               this->p2 == other.p2;
    }

    inline bool operator!=(const CameraDistortionModel & other) const noexcept
    {
        return !(*this == other);
    }
};

/**
 * Struct type used to store Camera Intrinsics.
 */
struct CameraIntrinsics
{
    CameraIntrinsics() = default;

    /**
    * Camera Instrinsics creation with given intrinsics values
    * @param fx Camera axis x focal length in pixels
    * @param fy Camera axis y focal length in pixels
    * @param cx Camera axis x principal point in pixels
    * @param cy Camera axis y principal point in pixels
    * @param s Camera slanted pixel
    * @return Camera Intrinsics
    */
    CameraIntrinsics(float fx, float fy, float cx, float cy, float s = 0.0)
    {
        m_intrinsics[0][0] = fx;
        m_intrinsics[0][1] = s;
        m_intrinsics[0][2] = cx;
        m_intrinsics[1][0] = 0.0;
        m_intrinsics[1][1] = fy;
        m_intrinsics[1][2] = cy;
    }

    /**
    * Get camera intrinsics x focal length.
    * @return Camera x focal length
    */
    float fx() const
    {
        return m_intrinsics[0][0];
    }

    /**
    * Get camera intrinsics y focal length.
    * @return Camera y focal length
    */
    float fy() const
    {
        return m_intrinsics[1][1];
    }

    /**
    * Get camera intrinsics x principal point.
    * @return Camera x principal point
    */
    float cx() const
    {
        return m_intrinsics[0][2];
    }

    /**
    * Get camera intrinsics y principal point.
    * @return Camera y principal point
    */
    float cy() const
    {
        return m_intrinsics[1][2];
    }

    /**
    * Get camera intrinsics slanted pixels.
    * @return Camera slanted pixels
    */
    float skew() const
    {
        return m_intrinsics[0][1];
    }

    /**
    * Get camera intrinsics 2D array.
    * @return Camera intrisics array
    */
    const float * getMatrix23() const
    {
        return &m_intrinsics[0][0];
    }

    inline bool operator==(const CameraIntrinsics & other) const noexcept
    {
        return m_intrinsics[0][0] == other.m_intrinsics[0][0] &&
               m_intrinsics[0][1] == other.m_intrinsics[0][1] &&
               m_intrinsics[0][2] == other.m_intrinsics[0][2] &&
               m_intrinsics[1][0] == other.m_intrinsics[1][0] &&
               m_intrinsics[1][1] == other.m_intrinsics[1][1] &&
               m_intrinsics[1][2] == other.m_intrinsics[1][2];
    }

    inline bool operator!=(const CameraIntrinsics & other) const noexcept
    {
        return !(*this == other);
    }

    float m_intrinsics[2][3] {{1.0, 0.0, 0.0},{0.0, 1.0, 0.0}};           /**< Camera intrinsics 2D arrat. */
};

/**
 * Struct type used to store Camera Extrinsics.
 */
struct CameraExtrinsics
{
    using RawMatrixType = float[3][4];

    CameraExtrinsics() = default;

    /**
    * Camera Extrinsics creation with given extrinsics as raw 2D [3 x 4] array
    * @param extrinsics Camera extrinsics as raw 2D array
    * @return Camera Extrinsics
    */
    explicit CameraExtrinsics(const RawMatrixType & extrinsics)
    {
        std::copy(&extrinsics[0][0], &extrinsics[0][0] + 3 * 4, &m_extrinsics[0][0]);
    }

    inline bool operator==(const CameraExtrinsics & other) const noexcept
    {
        return m_extrinsics[0][0] == other.m_extrinsics[0][0] &&
                m_extrinsics[0][1] == other.m_extrinsics[0][1] &&
                m_extrinsics[0][2] == other.m_extrinsics[0][2] &&
                m_extrinsics[0][3] == other.m_extrinsics[0][3] &&
                m_extrinsics[1][0] == other.m_extrinsics[1][0] &&
                m_extrinsics[1][1] == other.m_extrinsics[1][1] &&
                m_extrinsics[1][2] == other.m_extrinsics[1][2] &&
                m_extrinsics[1][3] == other.m_extrinsics[1][3] &&
                m_extrinsics[2][0] == other.m_extrinsics[2][0] &&
                m_extrinsics[2][1] == other.m_extrinsics[2][1] &&
                m_extrinsics[2][2] == other.m_extrinsics[2][2] &&
                m_extrinsics[2][3] == other.m_extrinsics[2][3];
    }

    inline bool operator!=(const CameraExtrinsics & other) const noexcept
    {
        return !(*this == other);
    }

    RawMatrixType m_extrinsics {{1.0, 0.0, 0.0, 0.0},
                                {0.0, 1.0, 0.0, 0.0},
                                {0.0, 0.0, 1.0, 0.0}};
};

struct CameraModel 
{
    CameraIntrinsics intrinsic;
    CameraExtrinsics extrinsic;
    CameraDistortionModel distortion;
};

} // namespace cvcore

#endif // CVCORE_CAMERAMODEL_H
