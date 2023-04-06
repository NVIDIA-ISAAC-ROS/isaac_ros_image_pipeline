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

#include "cv/core/Instrumentation.h"

#ifdef NVBENCH_ENABLE
#include <nvbench/CPU.h>
#include <nvbench/GPU.h>
#include <nvbench/VPI.h>
#endif

namespace cvcore { namespace profiler {

#ifdef NVBENCH_ENABLE
nv::bench::JsonHelper mapProfilerJsonOutputTypeToNvbenchType(ProfilerJsonOutputType jsonType)
{
    nv::bench::JsonHelper nvbenchJsonOutputType = nv::bench::JsonHelper::JSON_OFF;
    if (jsonType == ProfilerJsonOutputType::JSON_OFF)
    {
        nvbenchJsonOutputType = nv::bench::JsonHelper::JSON_OFF;
    }
    else if (jsonType == ProfilerJsonOutputType::JSON_SEPARATE)
    {
        nvbenchJsonOutputType = nv::bench::JsonHelper::JSON_SEPARATE;
    }
    else if (jsonType == ProfilerJsonOutputType::JSON_AGGREGATE)
    {
        nvbenchJsonOutputType = nv::bench::JsonHelper::JSON_AGGREGATE;
    }
    return nvbenchJsonOutputType;
}
#endif

void flush(const std::string& filename, ProfilerJsonOutputType jsonType)
{
#ifdef NVBENCH_ENABLE
    nv::bench::JsonHelper nvbenchJsonOutputType = mapProfilerJsonOutputTypeToNvbenchType(jsonType);
    if (!filename.empty())
    {
        nv::bench::Pool::instance().flushToFile(filename.c_str(), -1, INT_MAX, nvbenchJsonOutputType);
    }
    else
    {
        nv::bench::Pool::instance().flush(std::clog, -1, INT_MAX, nvbenchJsonOutputType);
    }
#else
    return;
#endif

}

void flush(std::ostream& output, ProfilerJsonOutputType jsonType)
{
#ifdef NVBENCH_ENABLE
    nv::bench::JsonHelper nvbenchJsonOutputType = mapProfilerJsonOutputTypeToNvbenchType(jsonType);
    nv::bench::Pool::instance().flush(output, -1, INT_MAX, nvbenchJsonOutputType);
#else
    return;
#endif
}

void flush(ProfilerJsonOutputType jsonType)
{
#ifdef NVBENCH_ENABLE
    nv::bench::JsonHelper nvbenchJsonOutputType = mapProfilerJsonOutputTypeToNvbenchType(jsonType);
    nv::bench::Pool::instance().flush(std::clog, -1, INT_MAX, nvbenchJsonOutputType);
#else
    return;
#endif
}

void clear()
{
#ifdef NVBENCH_ENABLE
    nv::bench::Pool::instance().clear();
#else
    return;
#endif
}

}}
