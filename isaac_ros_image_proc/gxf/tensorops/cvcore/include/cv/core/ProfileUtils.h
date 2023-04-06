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

#ifndef CVCORE_PROFILE_UTILS_H
#define CVCORE_PROFILE_UTILS_H

#include <string>

namespace cvcore {

/**
 * Export one profiling item to specified json file.
 * @param outputPath output json file path.
 * @param taskName item name showing in the output json file.
 * @param tMin minimum running time in milliseconds.
 * @param tMax maximum running time in milliseconds.
 * @param tAvg average running time in milliseconds.
 * @param isCPU whether CPU or GPU time.
 * @param iterations number of iterations.
 */
void ExportToJson(const std::string outputPath, const std::string taskName, float tMin, float tMax, float tAvg,
                  bool isCPU, int iterations = 100);

} // namespace cvcore

#endif // CVCORE_PROFILE_UTILS_H
