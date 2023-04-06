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

#include "cv/tensor_ops/BBoxUtils.h"

#include <algorithm>
#include <stdexcept>

namespace cvcore { namespace tensor_ops {

namespace {

bool IsValid(const BBox &box)
{
    return box.xmin >= 0 && box.ymin >= 0 && box.xmin < box.xmax && box.ymin < box.ymax;
}

} // anonymous namespace

float GetArea(const BBox &box)
{
    if (box.xmax < box.xmin || box.ymax < box.ymin)
    {
        return 0.f;
    }
    return static_cast<float>((box.xmax - box.xmin) * (box.ymax - box.ymin));
}

float GetIntersection(const BBox &a, const BBox &b)
{
    const int lowerX = std::max(a.xmin, b.xmin);
    const int upperX = std::min(a.xmax, b.xmax);
    const int lowerY = std::max(a.ymin, b.ymin);
    const int upperY = std::min(a.ymax, b.ymax);
    const int diffX  = lowerX < upperX ? upperX - lowerX : 0;
    const int diffY  = lowerY < upperY ? upperY - lowerY : 0;
    return static_cast<float>(diffX * diffY);
}

float GetUnion(const BBox &a, const BBox &b)
{
    return GetArea(a) + GetArea(b) - GetIntersection(a, b);
}

float GetIoU(const BBox &a, const BBox &b)
{
    return GetIntersection(a, b) / GetUnion(a, b);
}

BBox MergeBoxes(const BBox &a, const BBox &b)
{
    if (!IsValid(a) || !IsValid(b))
    {
        return IsValid(a) ? a : b;
    }
    BBox res;
    res.xmin = std::min(a.xmin, b.xmin);
    res.xmax = std::max(a.xmax, b.xmax);
    res.ymin = std::min(a.ymin, b.ymin);
    res.ymax = std::max(a.ymax, b.ymax);
    return res;
}

BBox ClampBox(const BBox &a, const BBox &b)
{
    return {std::max(a.xmin, b.xmin), std::max(a.ymin, b.ymin), std::min(a.xmax, b.xmax), std::min(a.ymax, b.ymax)};
}

BBox InterpolateBoxes(float currLeft, float currRight, float currBottom, float currTop, float xScaler, float yScaler,
                      int currColumn, int currRow, BBoxInterpolationType type, float bboxNorm)
{
    BBox currBoxInfo;
    if (type == CONST_INTERPOLATION)
    {
        float centerX    = ((currColumn * xScaler + 0.5) / bboxNorm);
        float centerY    = ((currRow * yScaler + 0.5) / bboxNorm);
        float left       = (currLeft - centerX);
        float right      = (currRight + centerX);
        float top        = (currTop - centerY);
        float bottom     = (currBottom + centerY);
        currBoxInfo.xmin = left * -bboxNorm;
        currBoxInfo.xmax = right * bboxNorm;
        currBoxInfo.ymin = top * -bboxNorm;
        currBoxInfo.ymax = bottom * bboxNorm;
    }
    else if (type == IMAGE_INTERPOLATION)
    {
        int centerX      = (int)((currColumn + 0.5f) * xScaler);
        int centerY      = (int)((currRow + 0.5f) * yScaler);
        int left         = (int)(currLeft * xScaler);
        int right        = (int)(currRight * xScaler);
        int top          = (int)(currTop * yScaler);
        int bottom       = (int)(currBottom * yScaler);
        currBoxInfo.xmin = centerX - left;
        currBoxInfo.xmax = centerX + right;
        currBoxInfo.ymin = centerY - top;
        currBoxInfo.ymax = centerY + bottom;
    }
    else
    {
        throw std::runtime_error("invalid bbox interpolation type");
    }
    return currBoxInfo;
}

BBox ScaleBox(const BBox &bbox, float xScaler, float yScaler, BBoxScaleType type)
{
    BBox output;
    if (type == NORMAL)
    {
        int xMin = (int)(bbox.xmin * xScaler + 0.5f);
        int yMin = (int)(bbox.ymin * yScaler + 0.5f);
        int xMax = (int)(bbox.xmax * xScaler + 0.5f);
        int yMax = (int)(bbox.ymax * yScaler + 0.5f);
        output   = {xMin, yMin, xMax, yMax};
    }
    else if (type == CENTER)
    {
        float xCenter = (bbox.xmax + bbox.xmin) / 2.0f;
        float yCenter = (bbox.ymax + bbox.ymin) / 2.0f;

        float width  = (bbox.xmax - bbox.xmin) * xScaler;
        float height = (bbox.ymax - bbox.ymin) * yScaler;

        output = {int(xCenter - width / 2 + 0.5f), int(yCenter - height / 2 + 0.5f), int(xCenter + width / 2 + 0.5f),
                  int(yCenter + height / 2 + 0.5f)};
    }
    else
    {
        throw std::runtime_error("invalid bbox scaling type");
    }
    return output;
}

BBox TransformBox(const BBox &bbox, float xScaler, float yScaler, float xOffset, float yOffset)
{
    int xMin = (int)((bbox.xmin + xOffset) * xScaler + 0.5f);
    int yMin = (int)((bbox.ymin + yOffset) * yScaler + 0.5f);
    int xMax = (int)((bbox.xmax + xOffset) * xScaler + 0.5f);
    int yMax = (int)((bbox.ymax + yOffset) * yScaler + 0.5f);
    return {xMin, yMin, xMax, yMax};
}

BBox SquarifyBox(const BBox &box, const BBox &boundary, float scale)
{
    BBox output    = ClampBox(box, boundary);
    float updateWH = scale * std::max(output.xmax - output.xmin, output.ymax - output.ymin);
    float scaleW   = updateWH / float(output.xmax - output.xmin);
    float scaleH   = updateWH / float(output.ymax - output.ymin);
    output         = ScaleBox(output, scaleW, scaleH, CENTER);

    output   = ClampBox(output, boundary);
    int xmin = output.xmin;
    int ymin = output.ymin;
    int l    = std::min(output.xmax - output.xmin, output.ymax - output.ymin);
    return {xmin, ymin, xmin + l, ymin + l};
}

}} // namespace cvcore::tensor_ops
