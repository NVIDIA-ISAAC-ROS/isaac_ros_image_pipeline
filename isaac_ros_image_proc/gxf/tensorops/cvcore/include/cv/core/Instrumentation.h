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

#ifndef CVCORE_INSTRUMENTATION_H
#define CVCORE_INSTRUMENTATION_H

#include <iostream>

namespace cvcore { namespace profiler {

/**
 * A enum class used to find out the type of profiler output required
 */
enum class ProfilerJsonOutputType : uint32_t
{
    JSON_OFF,       /**< print the aggregate values of each timer in pretty print format */
    JSON_AGGREGATE, /**< print the aggregate values of each timer in JSON format
                         along with the pretty print format. Pretty print format
                         gets printed on the terminal */
    JSON_SEPARATE   /**< print all the elapsed times for all timers along with the
                         aggregate values from JSON_AGGREGATE option */
};

/**
* Flush call to print the timer values in a file input
* @param jsonHelperType used to find out the type of profiler output required
* @return filename used to write the timer values
*/
void flush(const std::string& filename, ProfilerJsonOutputType jsonHelperType);

/**
* Flush call to print the timer values in a output stream
* @param jsonHelperType used to find out the type of profiler output required
* @return output stream used to write the timer values
*/
void flush(std::ostream& output, ProfilerJsonOutputType jsonHelperType);

/**
* Flush call to print the timer values on the terminal
* @param jsonHelperType used to find out the type of profiler output required
*/
void flush(ProfilerJsonOutputType jsonHelperType);


/**
* Clear all the profile timers
*/
void clear();

}}
#endif
