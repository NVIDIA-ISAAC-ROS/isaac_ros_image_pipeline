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

#ifndef CVCORE_BBOX_UTILS_H
#define CVCORE_BBOX_UTILS_H

#include "cv/core/BBox.h"

namespace cvcore { namespace tensor_ops {

/**
 * An enum.
 * Enum type for Bounding Box interpolation
 */
enum BBoxInterpolationType
{
    CONST_INTERPOLATION, /**< Uses constant value interpolation */
    IMAGE_INTERPOLATION, /**< Interpolates based on image size */
};

/**
 * An enum.
 * Enum type for Bounding Box scaling operation.
 */
enum BBoxScaleType
{
    NORMAL, /**< Scale box without fixing center point. */
    CENTER  /**< Scale box with center fixed. */
};

/**
 * Function to calculate the intersection of two bounding boxes.
 * @param a one of the BBox.
 * @param b the other BBox.
 * @return intersection area of the two bounding boxes.
 */
float GetIntersection(const BBox &a, const BBox &b);

/**
 * Function to calculate the union of two bounding boxes.
 * @param a one of the BBox.
 * @param b the other BBox.
 * @return union area of the two bounding boxes.
 */
float GetUnion(const BBox &a, const BBox &b);

/**
 * Function to calculate the IoU (Intersection over Union) of two bounding boxes.
 * @param a one of the BBox.
 * @param b the other BBox.
 * @return IoU of the two bounding boxes.
 */
float GetIoU(const BBox &a, const BBox &b);

/**
 * Function to merge two BBox together.
 * @param a one of the BBox.
 * @param b the other BBox.
 * @return Merged bounding box.
 */
BBox MergeBoxes(const BBox &a, const BBox &b);

/**
 * Clamp BBox.
 * @param a BBox to be clamped.
 * @param b boundary BBox.
 * @return clamped BBox.
 */
BBox ClampBox(const BBox &a, const BBox &b);

/**
 * Interpolate bounding boxes.
 * @param currLeft left x coordinate.
 * @param currRight right x coordinate.
 * @param currBottom bottom y coordinate.
 * @param currTop top y coordiante.
 * @param xScaler scale ratio along width direction.
 * @param yScaler scale ratio along height direction.
 * @param currColumn current column index.
 * @param currRow current row index.
 * @param type bbox interpolation type.
 * @param bboxNorm bounding box scaled factor.
 * @return interpolated BBox.
 */
BBox InterpolateBoxes(float currLeft, float currRight, float currBottom, float currTop, float xScaler, float yScaler,
                      int currColumn, int currRow, BBoxInterpolationType type = IMAGE_INTERPOLATION,
                      float bboxNorm = 1.0);

/**
 * Scale BBox.
 * @param bbox input BBox.
 * @param xScaler scale ratio along width direction.
 * @param yScaler scale ratio along height direction.
 * @param type BBox scaling type.
 * @return scaled BBox.
 */
BBox ScaleBox(const BBox &bbox, float xScaler, float yScaler, BBoxScaleType type = NORMAL);

/**
 * Transform BBox.
 * @param bbox input BBox.
 * @param xScaler scale ratio along width direction.
 * @param yScaler scale ratio along height direction.
 * @param xOffset offset along width direction in pixels.
 * @param yOffset offset along height direction in pixels.
 * @return transformed BBox.
 */
BBox TransformBox(const BBox &bbox, float xScaler, float yScaler, float xOffset, float yOffset);

/**
 * Squarify BBox.
 * @param box input BBox.
 * @param boundary boundary BBox used for clamping.
 * @param scale scaling factor.
 * @return squarified BBox.
 */
BBox SquarifyBox(const BBox &box, const BBox &boundary, float scale);

}} // namespace cvcore::tensor_ops

#endif // CVCORE_BBOX_UTILS_H