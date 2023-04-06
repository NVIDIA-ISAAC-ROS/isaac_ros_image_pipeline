// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CVCORE_TENSORLIST_H
#define CVCORE_TENSORLIST_H

#include <type_traits>

#include "Traits.h"
#include "Array.h"
#include "Tensor.h"

namespace cvcore {

/**
 * @brief Implementation of a list of tensors of the same rank but, potentially, different dimensions
 */
template<typename TensorType>
using TensorList = typename std::enable_if<traits::is_tensor<TensorType>::value, Array<TensorType>>::type;

} // namespace cvcore

#endif // CVCORE_TENSORLIST_H
