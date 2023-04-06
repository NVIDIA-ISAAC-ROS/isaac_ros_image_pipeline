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

#include "cv/tensor_ops/DBScan.h"
#include "cv/tensor_ops/BBoxUtils.h"

#include <stdexcept>
#include <vector>

namespace cvcore { namespace tensor_ops {

constexpr int kUnclassified = -1;
constexpr int kCorePoint    = 1;
constexpr int kBorderPoint  = 2;
constexpr int kNoise        = -2;

namespace {

float CalculateDistance(const BBox &lhs, const BBox &rhs)
{
    const float iou = GetIoU(lhs, rhs);
    return 1.0f - iou;
}

void MergeMaximumBBoxes(Array<BBox> &input, Array<int> &clusters, Array<BBox> &output)
{
    BBox tempBox = {-1, -1, -1, -1};
    // Initialize each cluster-box with a placeholder that has no cluster
    for (int i = 0; i < output.getSize(); i++)
    {
        // It's a struct so these pushes are by value
        output[i] = tempBox;
    }

    for (int i = 0; i < input.getSize(); i++)
    {
        int clusterId = clusters[i];
        if (clusterId >= 0)
        {
            // Box merging is associative & commutative
            output[clusterId] = MergeBoxes(input[i], output[clusterId]);
        }
    }
}

void MergeWeightedBBoxes(Array<BBox> &input, Array<int> &clusters, Array<float> &weights, Array<BBox> &output)
{
    int numClusters           = output.getSize();
    // centos has gcc 4.8.5 which complains about initializing variable sized arrays with {}.
    // Use std::vector for variable sized array.
    std::vector<float> xmins(numClusters, 0);
    std::vector<float> ymins(numClusters, 0);
    std::vector<float> xmaxs(numClusters, 0);
    std::vector<float> ymaxs(numClusters, 0);
    std::vector<float> scales(numClusters, 0);

    for (int i = 0; i < input.getSize(); i++)
    {
        int clusterId = clusters[i];
        if (clusterId >= 0)
        {
            xmins[clusterId] += input[i].xmin * weights[i];
            ymins[clusterId] += input[i].ymin * weights[i];
            xmaxs[clusterId] += input[i].xmax * weights[i];
            ymaxs[clusterId] += input[i].ymax * weights[i];
            scales[clusterId] += weights[i];
        }
    }

    for (int i = 0; i < numClusters; i++)
    {
        output[i] = {int(xmins[i] / scales[i] + 0.5f), int(ymins[i] / scales[i] + 0.5f),
                     int(xmaxs[i] / scales[i] + 0.5f), int(ymaxs[i] / scales[i] + 0.5f)};
    }
}

} // anonymous namespace

DBScan::DBScan(int pointsSize, int minPoints, float epsilon)
    : m_pointsSize(pointsSize)
    , m_numClusters(0)
    , m_minPoints(minPoints)
    , m_epsilon(epsilon)
    , m_clusterStates(pointsSize, true)
{
    m_clusterStates.setSize(pointsSize);
}

void DBScan::doCluster(Array<BBox> &input, Array<int> &clusters)
{
    // Reset all cluster id
    for (int i = 0; i < m_pointsSize; i++)
    {
        clusters[i]        = -1;
        m_clusterStates[i] = kUnclassified;
    }
    int nextClusterId = 0;
    for (int cIndex = 0; cIndex < m_pointsSize; cIndex++)
    {
        std::vector<int> neighbors;
        for (int neighborIndex = 0; neighborIndex < m_pointsSize; neighborIndex++)
        {
            if (neighborIndex == cIndex)
            {
                continue; // Don't look at being your own neighbor
            }
            if (CalculateDistance(input[cIndex], input[neighborIndex]) <= m_epsilon)
            {
                // nrighborIndex is in our neighborhood
                neighbors.push_back(neighborIndex);

                if (m_clusterStates[neighborIndex] == kCorePoint)
                {
                    // We are at the neighborhood of a core point, we are at least a border point
                    m_clusterStates[cIndex] = kBorderPoint;
                    // Take the first cluster number as you can
                    if (clusters[cIndex] == -1)
                    {
                        clusters[cIndex] = clusters[neighborIndex];
                    }
                }
            }
        }
        if (neighbors.size() >= m_minPoints - 1)
        {
            m_clusterStates[cIndex] = kCorePoint;
            if (clusters[cIndex] == -1)
            {
                // We're not in the neighborhood of other core points
                // So we're the core of a new cluster
                clusters[cIndex] = nextClusterId;
                nextClusterId++;
            }

            // Set all neighbors that came before us to be border points
            for (int neighborListIndex = 0;
                 neighborListIndex < neighbors.size() && neighbors[neighborListIndex] < cIndex; neighborListIndex++)
            {
                if (m_clusterStates[neighbors[neighborListIndex]] == kNoise)
                {
                    // If it was noise, now it's a border point in our cluster
                    m_clusterStates[neighbors[neighborListIndex]] = kBorderPoint;
                    // Make sure everything that's in our neighborhood is our cluster id
                    clusters[neighbors[neighborListIndex]] = clusters[cIndex];
                }
            }
        }
        else
        {
            // We are a border point, or a noise point
            if (m_clusterStates[cIndex] == kUnclassified)
            {
                m_clusterStates[cIndex] = kNoise;
                clusters[cIndex]        = -1;
            }
        }
    }

    m_numClusters = nextClusterId; // Number of clusters
}

void DBScan::doClusterAndMerge(Array<BBox> &input, Array<BBox> &output, BBoxMergeType type)
{
    Array<int> clusters(m_pointsSize, true);
    clusters.setSize(m_pointsSize);
    doCluster(input, clusters);
    output.setSize(m_numClusters);

    // merge bboxes based on different modes
    if (type == MAXIMUM)
    {
        MergeMaximumBBoxes(input, clusters, output);
    }
    else
    {
        throw std::runtime_error("Unsupported bbox merge type.");
    }
}

void DBScan::doClusterAndMerge(Array<BBox> &input, Array<float> &weights, Array<BBox> &output, BBoxMergeType type)
{
    Array<int> clusters(m_pointsSize, true);
    clusters.setSize(m_pointsSize);
    doCluster(input, clusters);
    output.setSize(m_numClusters);

    // merge type must be WEIGHTED
    if (type != WEIGHTED)
    {
        throw std::runtime_error("Bbox merge type must be WEIGHTED.");
    }
    MergeWeightedBBoxes(input, clusters, weights, output);
}

int DBScan::getNumClusters() const
{
    return m_numClusters;
}

}} // namespace cvcore::tensor_ops
