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

#ifndef CVCORE_BBOX_H
#define CVCORE_BBOX_H

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace cvcore {

/**
 * A struct.
 * Structure used to store bounding box.
 */
struct BBox
{
    int xmin{0}; /**< minimum x coordinate. */
    int ymin{0}; /**< minimum y coordinate. */
    int xmax{0}; /**< maximum x coordinate. */
    int ymax{0}; /**< maximum y coordinate. */

    /**
    * Clamp a bounding box based on a restricting clamp box
    * @param Clamping bounding box (xmin, ymin, xmax, ymax)
    * @return Clamped bounding box
    */
    BBox clamp(const BBox &clampBox) const
    {
        BBox outbox;
        outbox.xmin = std::max(clampBox.xmin, xmin);
        outbox.xmax = std::min(clampBox.xmax, xmax);
        outbox.ymin = std::max(clampBox.ymin, ymin);
        outbox.ymax = std::min(clampBox.ymax, ymax);
        return outbox;
    }

    /**
    * @return Width of the bounding box
    */
    size_t getWidth() const
    {
        return xmax - xmin;
    }

    /**
    * @return Height of the bounding box
    */
    size_t getHeight() const
    {
        return ymax - ymin;
    }

    /**
     * Checks if the bounding box is valid.
     */
    bool isValid() const
    {
        return (xmin < xmax) && (ymin < ymax) && (getWidth() > 0) && (getHeight() > 0);
    }

    /**
    * Returns the center of the bounding box
    * @return X,Y coordinate tuple
    */
    std::pair<int, int> getCenter() const
    {
        int centerX = xmin + getWidth() / 2;
        int centerY = ymin + getHeight() / 2;
        return std::pair<int, int>(centerX, centerY);
    }

    /**
    * Scales bounding box based along the width and height retaining the same center.
    * @param Scale in X direction along the width
    * @param Scale in Y direction along the height
    * @return Scaled bounding box
    */
    BBox scale(float scaleW, float scaleH) const
    {
        auto center = getCenter();
        float newW  = getWidth() * scaleW;
        float newH  = getHeight() * scaleH;
        BBox outbox;
        outbox.xmin = center.first - newW / 2;
        outbox.xmax = center.first + newW / 2;
        outbox.ymin = center.second - newH / 2;
        outbox.ymax = center.second + newH / 2;

        return outbox;
    }

    /**
    * Resizes bounding box to a square bounding box based on
    * the longest edge and clamps the bounding box based on the limits provided.
    * @param Clamping bounding box (xmin, ymin, xmax, ymax)
    * @return Sqaure bounding box
    */
    BBox squarify(const BBox &clampBox) const
    {
        size_t w = getWidth();
        size_t h = getHeight();

        BBox clampedBox1 = clamp(clampBox);
        if (!clampedBox1.isValid())
        {
            throw std::range_error("Invalid bounding box generated\n");
        }
        float scaleW     = static_cast<float>(std::max(w, h)) / w;
        float scaleH     = static_cast<float>(std::max(w, h)) / h;
        BBox scaledBBox  = clampedBox1.scale(scaleW, scaleH);
        BBox clampedBox2 = scaledBBox.clamp(clampBox);
        if (!clampedBox2.isValid())
        {
            throw std::range_error("Invalid bounding box generated\n");
        }
        size_t newW      = clampedBox2.getWidth();
        size_t newH      = clampedBox2.getHeight();
        size_t minW      = std::min(newH, newW);
        clampedBox2.ymax = clampedBox2.ymin + minW;
        clampedBox2.xmax = clampedBox2.xmin + minW;
        return clampedBox2;
    }
};

} // namespace cvcore
#endif // CVCORE_BBOX_H
