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

#ifndef CORE_H
#define CORE_H

namespace cvcore {

// Enable dll imports/exports in case of windows support
#ifdef _WIN32
#ifdef CVCORE_EXPORT_SYMBOLS // Needs to be enabled in case of compiling dll
#define CVCORE_API __declspec(dllexport)  // Exports symbols when compiling the library.
#else
#define CVCORE_API __declspec(dllimport)  // Imports the symbols when linked with library.
#endif
#else
#define CVCORE_API
#endif

} // namespace cvcore
#endif // CORE_H
