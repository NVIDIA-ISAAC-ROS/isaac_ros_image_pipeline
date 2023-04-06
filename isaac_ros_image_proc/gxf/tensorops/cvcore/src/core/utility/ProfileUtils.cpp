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

#include <nlohmann/json.hpp>

#include <cuda_runtime.h>

#include <fstream>
#include <iomanip>
#include <stdexcept>

#if defined(_MSC_VER) || defined(__WIN32)
#   include <intrin.h>
#   include <windows.h>
#   include <sysinfoapi.h>
#endif

using json = nlohmann::json;

namespace cvcore {

namespace {

#if defined(_MSC_VER) || defined(__WIN32)
std::string GetCPUName()
{
    // https://docs.microsoft.com/en-us/previous-versions/visualstudio/visual-studio-2008/hskdteyh(v=vs.90)?redirectedfrom=MSDN
    char CPUBrandString[0x40];
    int CPUInfo[4] = {-1};

    // Calling __cpuid with 0x80000000 as the InfoType argument
    // gets the number of valid extended IDs.
    __cpuid(CPUInfo, 0x80000000);
    unsigned i, nExIds = CPUInfo[0];
    memset(CPUBrandString, 0, sizeof(CPUBrandString));

    // Get the information associated with each extended ID.
    for (i=0x80000000; i<=nExIds; ++i)
    {
        __cpuid(CPUInfo, i);

        // Interpret CPU brand string and cache information.
        if  (i == 0x80000002)
            memcpy(CPUBrandString, CPUInfo, sizeof(CPUInfo));
        else if  (i == 0x80000003)
            memcpy(CPUBrandString + 16, CPUInfo, sizeof(CPUInfo));
        else if  (i == 0x80000004)
            memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
    }
    return CPUBrandString;
}
#else
std::string GetCPUName()
{
    std::ifstream cpuInfo("/proc/cpuinfo");
    if (!cpuInfo.good())
    {
        throw std::runtime_error("unable to retrieve cpu info");
    }
    std::string line;
    while (std::getline(cpuInfo, line))
    {
        int delimiterPos = line.find(':');
        if (delimiterPos != std::string::npos)
        {
            std::string key = line.substr(0, delimiterPos);
            if (key.find("model name") != std::string::npos)
            {
                std::string info = line.substr(delimiterPos + 1);
                info.erase(0, info.find_first_not_of(' '));
                return info;
            }
        }
    }
    return "CPU"; // default name if no cpu model name retrieved
}
#endif

std::string GetGPUName()
{
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp prop;
    cudaError_t error = cudaGetDeviceProperties(&prop, deviceId);
    if (error != 0)
    {
        throw std::runtime_error("unable to retrieve cuda device info");
    }
    return std::string(prop.name);
}

} // anonymous namespace

void ExportToJson(const std::string outputPath, const std::string taskName, float tMin, float tMax, float tAvg,
                  bool isCPU, int iterations = 100)
{
    std::ifstream in(outputPath);
    json jsonHandler;
    if (in.good())
    {
        in >> jsonHandler;
    }
    in.close();

    const std::string platform      = isCPU ? "CPU: " + GetCPUName() : "GPU: " + GetGPUName();
    jsonHandler[platform][taskName] = {{"iter", iterations}, {"min", tMin}, {"max", tMax}, {"avg", tAvg}};

    std::ofstream out(outputPath);
    out << std::setw(4) << jsonHandler << std::endl;
    out.close();
}

} // namespace cvcore
