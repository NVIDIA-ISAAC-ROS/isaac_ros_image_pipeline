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

#ifndef CVCORE_DBSCAN_H
#define CVCORE_DBSCAN_H

#include "cv/core/Array.h"
#include "cv/core/BBox.h"

namespace cvcore { namespace tensor_ops {

/**
 * An enum.
 * Enum type for BBox merge type.
 */
enum BBoxMergeType
{
    MAXIMUM,  /**< merge by expanding bounding boxes */
    WEIGHTED, /**< weighted by confidence scores. */
};

/**
 * DBScan implementation used for post-processing of object detection.
 */
class DBScan
{
public:
    /**
     * DBScan constructor.
     * @param pointsize size of the input array.
     * @param minPoints minimum number of neighbors within the radius.
     * @param epsilon radius of neighborhood around a point.
     */
    DBScan(int pointsSize, int minPoints, float epsilon);

    /**
     * Run DBScan cluster and return the cluster indices.
     * @param input input unclustered BBoxes array.
     * @param clusters output array containing cluster ids.
     */
    void doCluster(Array<BBox> &input, Array<int> &clusters);

    /**
     * Run DBScan cluster and return the cluster bboxes.
     * @param input input unclustered BBoxes array.
     * @param output output clustered BBoxes array.
     * @param type bbox merge type
     */
    void doClusterAndMerge(Array<BBox> &input, Array<BBox> &output, BBoxMergeType type = MAXIMUM);

    /**
     * Run DBScan cluster and return the cluster bboxes weighted on input weights.
     * @param input input unclustered BBoxes array.
     * @param weights weights needed for merging clusterd bboxes.
     * @param output output clustered BBoxes array.
     * @param type bbox merge type
     */
    void doClusterAndMerge(Array<BBox> &input, Array<float> &weights, Array<BBox> &output,
                           BBoxMergeType type = WEIGHTED);

    /**
     * Get the number of clusters.
     * @return number of clusters.
     */
    int getNumClusters() const;

private:
    int m_pointsSize;
    int m_numClusters;
    int m_minPoints;
    float m_epsilon;
    Array<int> m_clusterStates;
};

}} // namespace cvcore::tensor_ops

#endif // CVCORE_DBSCAN_H
