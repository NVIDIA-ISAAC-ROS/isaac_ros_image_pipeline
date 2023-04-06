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

#include "cv/core/MathTypes.h"

namespace cvcore {

namespace {

AxisAngleRotation RotationMatrixToAxisAngleUtil(const std::vector<double> &rotMatrix)
{
    assert(rotMatrix.size() == 9);
    AxisAngleRotation axisangle;
    double row0 = 0.5 * (rotMatrix[7] - rotMatrix[5]);
    double row1 = 0.5 * (rotMatrix[2] - rotMatrix[6]);
    double row2 = 0.5 * (rotMatrix[3] - rotMatrix[1]);

    double sin_angle = std::sqrt(row0 * row0 + row1 * row1 + row2 * row2);
    double cos_angle = 0.5 * (rotMatrix[0] + rotMatrix[4] + rotMatrix[8] - 1.0);
    sin_angle        = sin_angle > 1.0 ? 1.0 : sin_angle;
    cos_angle        = cos_angle > 1.0 ? 1.0 : (cos_angle < -1.0 ? -1.0 : cos_angle);

    if (sin_angle == 0.0)
    {
        axisangle.angle  = 0;
        axisangle.axis.x = 0;
        axisangle.axis.y = 0;
        axisangle.axis.z = 0;
    }
    else
    {
        axisangle.angle  = std::atan2(sin_angle, cos_angle);
        axisangle.axis.x = row0 / sin_angle;
        axisangle.axis.y = row1 / sin_angle;
        axisangle.axis.z = row2 / sin_angle;
    }
    return axisangle;
}
} // namespace

Vector3d RotationMatrixToRotationVector(const std::vector<double> &rotMatrix)
{
    AxisAngleRotation axisangle = RotationMatrixToAxisAngleUtil(rotMatrix);
    Vector3d rotVector;
    rotVector.x = axisangle.angle * axisangle.axis.x;
    rotVector.y = axisangle.angle * axisangle.axis.y;
    rotVector.z = axisangle.angle * axisangle.axis.z;
    return rotVector;
}

AxisAngleRotation RotationMatrixToAxisAngleRotation(const std::vector<double> &rotMatrix)
{
    AxisAngleRotation axisangle = RotationMatrixToAxisAngleUtil(rotMatrix);
    return axisangle;
}

std::vector<double> AxisAngleToRotationMatrix(const AxisAngleRotation &axisangle)
{
    std::vector<double> rotMatrix;
    rotMatrix.resize(9);
    double cosangle = std::cos(axisangle.angle);
    double sinagle  = std::sin(axisangle.angle);
    double temp     = 1.0 - cosangle;

    rotMatrix[0] = cosangle + axisangle.axis.x * axisangle.axis.x * temp;
    rotMatrix[4] = cosangle + axisangle.axis.y * axisangle.axis.y * temp;
    rotMatrix[8] = cosangle + axisangle.axis.z * axisangle.axis.z * temp;

    double value1 = axisangle.axis.x * axisangle.axis.y * temp;
    double value2 = axisangle.axis.z * sinagle;
    rotMatrix[3]  = value1 + value2;
    rotMatrix[1]  = value1 - value2;
    value1        = axisangle.axis.x * axisangle.axis.z * temp;
    value2        = axisangle.axis.y * sinagle;
    rotMatrix[6]  = value1 - value2;
    rotMatrix[2]  = value1 + value2;
    value1        = axisangle.axis.y * axisangle.axis.z * temp;
    value2        = axisangle.axis.x * sinagle;
    rotMatrix[7]  = value1 + value2;
    rotMatrix[5]  = value1 - value2;
    return rotMatrix;
}

Vector3d AxisAngleRotationToRotationVector(const AxisAngleRotation &axisangle)
{
    double angle = axisangle.angle;
    Vector3d rotVector;
    rotVector.x = angle * axisangle.axis.x;
    rotVector.y = angle * axisangle.axis.y;
    rotVector.z = angle * axisangle.axis.z;
    return rotVector;
}

AxisAngleRotation RotationVectorToAxisAngleRotation(const Vector3d &rotVector)
{
    double normVector = rotVector.x * rotVector.x + rotVector.y * rotVector.y + rotVector.z * rotVector.z;
    normVector        = std::sqrt(normVector);
    AxisAngleRotation axisangle;
    if (normVector == 0)
    {
        axisangle.angle  = 0;
        axisangle.axis.x = 0;
        axisangle.axis.y = 0;
        axisangle.axis.z = 0;
    }
    else
    {
        axisangle.angle  = normVector;
        axisangle.axis.x = rotVector.x / normVector;
        axisangle.axis.y = rotVector.y / normVector;
        axisangle.axis.z = rotVector.z / normVector;
    }
    return axisangle;
}

Quaternion AxisAngleRotationToQuaternion(const AxisAngleRotation &axisangle)
{
    Quaternion qrotation;
    qrotation.qx = axisangle.axis.x * sin(axisangle.angle / 2);
    qrotation.qy = axisangle.axis.y * sin(axisangle.angle / 2);
    qrotation.qz = axisangle.axis.z * sin(axisangle.angle / 2);
    qrotation.qw = std::cos(axisangle.angle / 2);
    return qrotation;
}

AxisAngleRotation QuaternionToAxisAngleRotation(const Quaternion &qrotation)
{
    Quaternion qtemp(qrotation.qx, qrotation.qy, qrotation.qz, qrotation.qw);
    if (qrotation.qw > 1)
    {
        double qnorm = qrotation.qx * qrotation.qx + qrotation.qy * qrotation.qy + qrotation.qz * qrotation.qz +
                       qrotation.qw * qrotation.qw;
        qnorm    = std::sqrt(qnorm);
        qtemp.qx = qrotation.qx / qnorm;
        qtemp.qy = qrotation.qy / qnorm;
        qtemp.qz = qrotation.qz / qnorm;
        qtemp.qw = qrotation.qw / qnorm;
    }
    AxisAngleRotation axisangle;
    axisangle.angle = 2 * std::acos(qtemp.qw);
    double normaxis = std::sqrt(1 - qtemp.qw * qtemp.qw);
    if (normaxis < 0.001)
    {
        axisangle.axis.x = qtemp.qx;
        axisangle.axis.y = qtemp.qy;
        axisangle.axis.z = qtemp.qz;
    }
    else
    {
        axisangle.axis.x = qtemp.qx / normaxis;
        axisangle.axis.y = qtemp.qy / normaxis;
        axisangle.axis.z = qtemp.qz / normaxis;
    }
    return axisangle;
}

std::vector<double> QuaternionToRotationMatrix(const Quaternion &qrotation)
{
    std::vector<double> rotMatrix;
    rotMatrix.resize(9);
    double qxsquare = qrotation.qx * qrotation.qx;
    double qysquare = qrotation.qy * qrotation.qy;
    double qzsquare = qrotation.qz * qrotation.qz;
    double qwsquare = qrotation.qw * qrotation.qw;

    // Ensure quaternion is normalized
    double invsersenorm = 1 / (qxsquare + qysquare + qzsquare + qwsquare);
    rotMatrix[0]        = (qxsquare - qysquare - qzsquare + qwsquare) * invsersenorm;
    rotMatrix[4]        = (-qxsquare + qysquare - qzsquare + qwsquare) * invsersenorm;
    rotMatrix[8]        = (-qxsquare - qysquare + qzsquare + qwsquare) * invsersenorm;

    double value1 = qrotation.qx * qrotation.qy;
    double value2 = qrotation.qz * qrotation.qw;
    rotMatrix[3]  = 2.0 * (value1 + value2) * invsersenorm;
    rotMatrix[1]  = 2.0 * (value1 - value2) * invsersenorm;

    value1       = qrotation.qx * qrotation.qz;
    value2       = qrotation.qy * qrotation.qw;
    rotMatrix[6] = 2.0 * (value1 - value2) * invsersenorm;
    rotMatrix[2] = 2.0 * (value1 + value2) * invsersenorm;
    value1       = qrotation.qz * qrotation.qy;
    value2       = qrotation.qx * qrotation.qw;
    rotMatrix[7] = 2.0 * (value1 + value2) * invsersenorm;
    rotMatrix[5] = 2.0 * (value1 - value2) * invsersenorm;
    return rotMatrix;
}

Quaternion RotationMatrixToQuaternion(const std::vector<double> &rotMatrix)
{
    Quaternion qrotation;
    double diagsum = rotMatrix[0] + rotMatrix[4] + rotMatrix[8];
    if (diagsum > 0)
    {
        double temp  = 1 / (2 * std::sqrt(diagsum + 1.0));
        qrotation.qw = 0.25 / temp;
        qrotation.qx = (rotMatrix[7] - rotMatrix[5]) * temp;
        qrotation.qy = (rotMatrix[2] - rotMatrix[6]) * temp;
        qrotation.qz = (rotMatrix[3] - rotMatrix[1]) * temp;
    }
    else
    {
        if (rotMatrix[0] > rotMatrix[4] && rotMatrix[0] > rotMatrix[8])
        {
            double temp  = 2 * std::sqrt(rotMatrix[0] - rotMatrix[4] - rotMatrix[8] + 1.0);
            qrotation.qx = 0.25 * temp;
            qrotation.qw = (rotMatrix[7] - rotMatrix[5]) * temp;
            qrotation.qy = (rotMatrix[2] - rotMatrix[6]) * temp;
            qrotation.qz = (rotMatrix[3] - rotMatrix[1]) * temp;
        }
        else if (rotMatrix[0] > rotMatrix[8])
        {
            double temp  = 2 * std::sqrt(rotMatrix[4] - rotMatrix[0] - rotMatrix[8] + 1.0);
            qrotation.qy = 0.25 * temp;
            qrotation.qx = (rotMatrix[7] - rotMatrix[5]) * temp;
            qrotation.qw = (rotMatrix[2] - rotMatrix[6]) * temp;
            qrotation.qz = (rotMatrix[3] - rotMatrix[1]) * temp;
        }
        else
        {
            double temp  = 2 * std::sqrt(rotMatrix[8] - rotMatrix[4] - rotMatrix[0] + 1.0);
            qrotation.qz = 0.25 * temp;
            qrotation.qx = (rotMatrix[7] - rotMatrix[5]) * temp;
            qrotation.qy = (rotMatrix[2] - rotMatrix[6]) * temp;
            qrotation.qw = (rotMatrix[3] - rotMatrix[1]) * temp;
        }
    }
    return qrotation;
}

} // namespace cvcore
