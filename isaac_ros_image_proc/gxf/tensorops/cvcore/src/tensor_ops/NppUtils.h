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

#ifndef CVCORE_NPP_UTILS_H
#define CVCORE_NPP_UTILS_H

#include <nppdefs.h>

#include <cuda_runtime.h>

namespace cvcore { namespace tensor_ops {

NppStreamContext GetNppStreamContext(cudaStream_t stream);

}} // namespace cvcore::tensor_ops

#endif // CVCORE_NPP_UTILS_H
