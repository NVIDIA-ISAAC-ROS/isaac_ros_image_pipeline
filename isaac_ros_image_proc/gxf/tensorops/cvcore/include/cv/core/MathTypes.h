// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CVCORE_Math_H
#define CVCORE_Math_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

#include "Tensor.h"

namespace cvcore {

/** Matrix types using Tensor backend */
using Matrixf = Tensor<CHW, C1, F32>;
using Matrixd = Tensor<CHW, C1, F64>;

/**
 * A struct.
 * Structure used to store Vector2 Values.
 */
template<class T>
struct Vector2
{
    T x; /**< point x coordinate. */
    T y; /**< point y coordinate. */

    inline T &operator[](size_t i)
    {
        if (i == 0)
        {
            return x;
        }
        else if (i == 1)
        {
            return y;
        }
        else
        {
            throw std::out_of_range("cvcore::Vector2 ==> Requested index is out of bounds");
        }
    }
};

/**
 * A struct.
 * Structure used to store Vector3 Values.
 */
template<class T>
struct Vector3
{
    T x; /**< point x coordinate. */
    T y; /**< point y coordinate. */
    T z; /**< point z coordinate. */

    inline T &operator[](size_t i)
    {
        if (i == 0)
        {
            return x;
        }
        else if (i == 1)
        {
            return y;
        }
        else if (i == 2)
        {
            return z;
        }
        else
        {
            throw std::out_of_range("cvcore::Vector3 ==> Requested index is out of bounds");
        }
    }
};

using Vector2i = Vector2<int>;
using Vector3i = Vector3<int>;

using Vector2f = Vector2<float>;
using Vector3f = Vector3<float>;

using Vector2d = Vector2<double>;
using Vector3d = Vector3<double>;

/**
 * A struct
 * Structure used to store AxisAngle Rotation parameters.
 */
struct AxisAngleRotation
{
    double angle;  /** Counterclockwise rotation angle [0, 2PI]. */
    Vector3d axis; /** 3d axis of rotation. */

    AxisAngleRotation()
        : angle(0.0)
        , axis{0, 0, 0}
    {
    }

    AxisAngleRotation(double angleinput, Vector3d axisinput)
        : angle(angleinput)
        , axis(axisinput)
    {
    }
};

/**
 * A struct.
 * Structure used to store quaternion rotation representation.
 * A rotation of unit vector u with rotation theta can be represented in quaternion as:
 * q={cos(theta/2)+ i(u*sin(theta/2))}
*/
struct Quaternion
{
    double qx, qy, qz; /** Axis or imaginary component of the quaternion representation. */
    double qw;         /** Angle or real component of the quaternion representation. */

    Quaternion()
        : qx(0.0)
        , qy(0.0)
        , qz(0.0)
        , qw(0.0)
    {
    }

    Quaternion(double qxinput, double qyinput, double qzinput, double qwinput)
        : qx(qxinput)
        , qy(qyinput)
        , qz(qzinput)
        , qw(qwinput)
    {
    }
};

/**
 * Convert rotation matrix to rotation vector.
 * @param rotMatrix Rotation matrix of 3x3 values.
 * @return 3D Rotation vector {theta * xaxis, theta * yaxis, theta * zaxis}
 * where theta is the angle of rotation in radians
 */
Vector3d RotationMatrixToRotationVector(const std::vector<double> &rotMatrix);

/**
 * Convert rotation matrix to axis angle representation.
 * @param rotMatrix Rotation matrix of 3x3 values.
 * @return Axis angle rotation
 */
AxisAngleRotation RotationMatrixToAxisAngleRotation(const std::vector<double> &rotMatrix);

/**
 * Convert axis angle representation to rotation matrix.
 * @param axisangle  Axis angle rotation.
 * @return Rotation matrix of 3x3 values.
 */
std::vector<double> AxisAngleToRotationMatrix(const AxisAngleRotation &axisangle);

/**
 * Convert axis angle representation to 3d rotation vector.
 * Rotation vector is  {theta * xaxis, theta * yaxis, theta * zaxis}
 * where theta is the angle of rotation in radians.
 * @param axisangle  Axis angle rotation.
 * @return 3D Rotation Vector
 */
Vector3d AxisAngleRotationToRotationVector(const AxisAngleRotation &axisangle);

/**
 * Convert rotation vector to axis angle representation.
 * @param rotVector 3D rotation vector.
 * @return Axis angle rotation.
 */
AxisAngleRotation RotationVectorToAxisAngleRotation(const Vector3d &rotVector);

/**
 * Convert axis angle representation to quaternion.
 * @param axisangle Axis angle representation.
 * @return Quaternion rotation.
 */
Quaternion AxisAngleRotationToQuaternion(const AxisAngleRotation &axisangle);

/**
 * Convert quaternion rotation to axis angle rotation.
 * @param qrotation Quaternion rotation representation.
 * @return Axis angle rotation.
 */
AxisAngleRotation QuaternionToAxisAngleRotation(const Quaternion &qrotation);

/**
 * Convert quaternion rotation to rotation matrix.
 * @param qrotation Quaternion rotation representation.
 * @return Rotation matrix.
 */
std::vector<double> QuaternionToRotationMatrix(const Quaternion &qrotation);

/**
 * Convert rotation matrix to Quaternion.
 * @param rotMatrix Rotation matrix
 * @return Quaternion rotation.
 */
Quaternion RotationMatrixToQuaternion(const std::vector<double> &rotMatrix);

/**
 * A struct.
 * Structure used to store Pose3D parameters.
 */
template<class T>
struct Pose3
{
    AxisAngleRotation rotation; /**Rotation expressed in axis angle notation.*/
    Vector3<T> translation;     /*Translation expressed as x,y,z coordinates.*/
};

using Pose3d = Pose3<double>;
using Pose3f = Pose3<float>;

} // namespace cvcore

#endif // CVCORE_Math_H
