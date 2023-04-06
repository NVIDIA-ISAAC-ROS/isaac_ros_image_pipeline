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

#ifndef CVCORE_TRAITS_H
#define CVCORE_TRAITS_H

#include <type_traits>

#include "MathTypes.h"
#include "Tensor.h"

namespace cvcore { namespace traits {

// -----------------------------------------------------------------------------
// Type Properties
// -----------------------------------------------------------------------------

template<typename TensorType>
struct is_tensor : std::false_type
{
};

template<TensorLayout TL, ChannelCount CC, ChannelType CT>
struct is_tensor<Tensor<TL, CC, CT>> : std::true_type
{
};

template<typename TensorType>
struct is_planar : std::false_type
{
    static_assert(is_tensor<TensorType>::value, "");
};

template<TensorLayout TL, ChannelCount CC, ChannelType CT>
struct is_planar<Tensor<TL, CC, CT>> : std::integral_constant<bool, TL == CHW || TL == DCHW || TL == CDHW>
{
};

template<typename TensorType>
struct is_interleaved : std::false_type
{
    static_assert(is_tensor<TensorType>::value, "");
};

template<TensorLayout TL, ChannelCount CC, ChannelType CT>
struct is_interleaved<Tensor<TL, CC, CT>> : std::integral_constant<bool, TL == HWC || TL == DHWC>
{
};

template<typename TensorType>
struct is_batch : std::false_type
{
    static_assert(is_tensor<TensorType>::value, "");
};

template<TensorLayout TL, ChannelCount CC, ChannelType CT>
struct is_batch<Tensor<TL, CC, CT>> : std::integral_constant<bool, TL == DHWC || TL == DCHW || TL == CDHW>
{
};

// -----------------------------------------------------------------------------
// Type Modifications
// -----------------------------------------------------------------------------

template<typename TensorType>
struct to_planar
{
    static_assert(is_tensor<TensorType>::value, "");

    using type = TensorType;
};

template<ChannelCount CC, ChannelType CT>
struct to_planar<Tensor<HWC, CC, CT>>
{
    using type = Tensor<CHW, CC, CT>;
};

template<ChannelCount CC, ChannelType CT>
struct to_planar<Tensor<DHWC, CC, CT>>
{
    using type = Tensor<DCHW, CC, CT>;
};

template<typename TensorType>
using to_planar_t = typename to_planar<TensorType>::type;

template<typename TensorType>
struct to_interleaved
{
    static_assert(is_tensor<TensorType>::value, "");

    using type = TensorType;
};

template<ChannelCount CC, ChannelType CT>
struct to_interleaved<Tensor<CHW, CC, CT>>
{
    using type = Tensor<HWC, CC, CT>;
};

template<ChannelCount CC, ChannelType CT>
struct to_interleaved<Tensor<DCHW, CC, CT>>
{
    using type = Tensor<DHWC, CC, CT>;
};

template<ChannelCount CC, ChannelType CT>
struct to_interleaved<Tensor<CDHW, CC, CT>>
{
    using type = Tensor<DHWC, CC, CT>;
};

template<typename TensorType>
using to_interleaved_t = typename to_interleaved<TensorType>::type;

template<typename TensorType>
struct add_batch
{
    static_assert(is_tensor<TensorType>::value, "");

    using type = TensorType;
};

template<ChannelCount CC, ChannelType CT>
struct add_batch<Tensor<CHW, CC, CT>>
{
    using type = Tensor<DCHW, CC, CT>;
};

template<ChannelCount CC, ChannelType CT>
struct add_batch<Tensor<HWC, CC, CT>>
{
    using type = Tensor<DHWC, CC, CT>;
};

template<typename TensorType>
using add_batch_t = typename add_batch<TensorType>::type;

template<typename TensorType>
struct remove_batch
{
    static_assert(is_tensor<TensorType>::value, "");

    using type = TensorType;
};

template<ChannelCount CC, ChannelType CT>
struct remove_batch<Tensor<DCHW, CC, CT>>
{
    using type = Tensor<CHW, CC, CT>;
};

template<ChannelCount CC, ChannelType CT>
struct remove_batch<Tensor<CDHW, CC, CT>>
{
    using type = Tensor<CHW, CC, CT>;
};

template<ChannelCount CC, ChannelType CT>
struct remove_batch<Tensor<DHWC, CC, CT>>
{
    using type = Tensor<HWC, CC, CT>;
};

template<typename TensorType>
using remove_batch_t = typename remove_batch<TensorType>::type;

template<typename TensorType>
struct to_c1
{
    static_assert(is_tensor<TensorType>::value, "");
};

template<TensorLayout TL, ChannelCount CC, ChannelType CT>
struct to_c1<Tensor<TL, CC, CT>>
{
    using type = Tensor<TL, C1, CT>;
};

template<typename TensorType>
using to_c1_t = typename to_c1<TensorType>::type;

template<typename TensorType>
struct to_c2
{
    static_assert(is_tensor<TensorType>::value, "");
};

template<TensorLayout TL, ChannelCount CC, ChannelType CT>
struct to_c2<Tensor<TL, CC, CT>>
{
    using type = Tensor<TL, C2, CT>;
};

template<typename TensorType>
using to_c2_t = typename to_c2<TensorType>::type;

template<typename TensorType>
struct to_c3
{
    static_assert(is_tensor<TensorType>::value, "");
};

template<TensorLayout TL, ChannelCount CC, ChannelType CT>
struct to_c3<Tensor<TL, CC, CT>>
{
    using type = Tensor<TL, C3, CT>;
};

template<typename TensorType>
using to_c3_t = typename to_c3<TensorType>::type;

template<typename TensorType>
struct to_c4
{
    static_assert(is_tensor<TensorType>::value, "");
};

template<TensorLayout TL, ChannelCount CC, ChannelType CT>
struct to_c4<Tensor<TL, CC, CT>>
{
    using type = Tensor<TL, C4, CT>;
};

template<typename TensorType>
using to_c4_t = typename to_c4<TensorType>::type;

template<typename TensorType>
struct to_cx
{
    static_assert(is_tensor<TensorType>::value, "");
};

template<TensorLayout TL, ChannelCount CC, ChannelType CT>
struct to_cx<Tensor<TL, CC, CT>>
{
    using type = Tensor<TL, CX, CT>;
};

template<typename TensorType>
using to_cx_t = typename to_cx<TensorType>::type;

template<typename TensorType>
struct to_u8
{
    static_assert(is_tensor<TensorType>::value, "");
};

template<TensorLayout TL, ChannelCount CC, ChannelType CT>
struct to_u8<Tensor<TL, CC, CT>>
{
    using type = Tensor<TL, CC, U8>;
};

template<typename TensorType>
using to_u8_t = typename to_u8<TensorType>::type;

template<typename TensorType>
struct to_u16
{
    static_assert(is_tensor<TensorType>::value, "");
};

template<TensorLayout TL, ChannelCount CC, ChannelType CT>
struct to_u16<Tensor<TL, CC, CT>>
{
    using type = Tensor<TL, CC, U16>;
};

template<typename TensorType>
using to_u16_t = typename to_u16<TensorType>::type;

template<typename TensorType>
struct to_f16
{
    static_assert(is_tensor<TensorType>::value, "");
};

template<TensorLayout TL, ChannelCount CC, ChannelType CT>
struct to_f16<Tensor<TL, CC, CT>>
{
    using type = Tensor<TL, CC, F16>;
};

template<typename TensorType>
using to_f16_t = typename to_f16<TensorType>::type;

template<typename TensorType>
struct to_f32
{
    static_assert(is_tensor<TensorType>::value, "");
};

template<TensorLayout TL, ChannelCount CC, ChannelType CT>
struct to_f32<Tensor<TL, CC, CT>>
{
    using type = Tensor<TL, CC, F32>;
};

template<typename TensorType>
using to_f32_t = typename to_f32<TensorType>::type;

template<typename TensorType>
using to_c1u8 = to_c1<to_u8_t<TensorType>>;

template<typename TensorType>
using to_c1u16 = to_c1<to_u16_t<TensorType>>;

template<typename TensorType>
using to_c1f16 = to_c1<to_f16_t<TensorType>>;

template<typename TensorType>
using to_c1f32 = to_c1<to_f32_t<TensorType>>;

template<typename TensorType>
using to_c2u8 = to_c2<to_u8_t<TensorType>>;

template<typename TensorType>
using to_c2u16 = to_c2<to_u16_t<TensorType>>;

template<typename TensorType>
using to_c2f16 = to_c2<to_f16_t<TensorType>>;

template<typename TensorType>
using to_c2f32 = to_c2<to_f32_t<TensorType>>;

template<typename TensorType>
using to_c3u8 = to_c3<to_u8_t<TensorType>>;

template<typename TensorType>
using to_c3u16 = to_c3<to_u16_t<TensorType>>;

template<typename TensorType>
using to_c3f16 = to_c3<to_f16_t<TensorType>>;

template<typename TensorType>
using to_c3f32 = to_c3<to_f32_t<TensorType>>;

template<typename TensorType>
using to_c4u8 = to_c4<to_u8_t<TensorType>>;

template<typename TensorType>
using to_c4u16 = to_c4<to_u16_t<TensorType>>;

template<typename TensorType>
using to_c4f16 = to_c4<to_f16_t<TensorType>>;

template<typename TensorType>
using to_c4f32 = to_c4<to_f32_t<TensorType>>;

template<typename TensorType>
using to_cxu8 = to_cx<to_u8_t<TensorType>>;

template<typename TensorType>
using to_cxu16 = to_cx<to_u16_t<TensorType>>;

template<typename TensorType>
using to_cxf16 = to_cx<to_f16_t<TensorType>>;

template<typename TensorType>
using to_cxf32 = to_cx<to_f32_t<TensorType>>;

template<typename TensorType>
using to_c1u8_t = to_c1_t<to_u8_t<TensorType>>;

template<typename TensorType>
using to_c1u16_t = to_c1_t<to_u16_t<TensorType>>;

template<typename TensorType>
using to_c1f16_t = to_c1_t<to_f16_t<TensorType>>;

template<typename TensorType>
using to_c1f32_t = to_c1_t<to_f32_t<TensorType>>;

template<typename TensorType>
using to_c2u8_t = to_c2_t<to_u8_t<TensorType>>;

template<typename TensorType>
using to_c2u16_t = to_c2_t<to_u16_t<TensorType>>;

template<typename TensorType>
using to_c2f16_t = to_c2_t<to_f16_t<TensorType>>;

template<typename TensorType>
using to_c2f32_t = to_c2_t<to_f32_t<TensorType>>;

template<typename TensorType>
using to_c3u8_t = to_c3_t<to_u8_t<TensorType>>;

template<typename TensorType>
using to_c3u16_t = to_c3_t<to_u16_t<TensorType>>;

template<typename TensorType>
using to_c3f16_t = to_c3_t<to_f16_t<TensorType>>;

template<typename TensorType>
using to_c3f32_t = to_c3_t<to_f32_t<TensorType>>;

template<typename TensorType>
using to_c4u8_t = to_c4_t<to_u8_t<TensorType>>;

template<typename TensorType>
using to_c4u16_t = to_c4_t<to_u16_t<TensorType>>;

template<typename TensorType>
using to_c4f16_t = to_c4_t<to_f16_t<TensorType>>;

template<typename TensorType>
using to_c4f32_t = to_c4_t<to_f32_t<TensorType>>;

template<typename TensorType>
using to_cxu8_t = to_cx_t<to_u8_t<TensorType>>;

template<typename TensorType>
using to_cxu16_t = to_cx_t<to_u16_t<TensorType>>;

template<typename TensorType>
using to_cxf16_t = to_cx_t<to_f16_t<TensorType>>;

template<typename TensorType>
using to_cxf32_t = to_cx_t<to_f32_t<TensorType>>;

template<typename T>
struct get_dim;

template<>
struct get_dim<float>
{
    static constexpr int value = 1;
};

template<>
struct get_dim<double>
{
    static constexpr int value = 1;
};

template<>
struct get_dim<Vector2d>
{
    static constexpr int value = 2;
};

template<>
struct get_dim<Vector2f>
{
    static constexpr int value = 2;
};

template<>
struct get_dim<Vector3d>
{
    static constexpr int value = 3;
};

template<>
struct get_dim<Vector3f>
{
    static constexpr int value = 3;
};
}} // namespace cvcore::traits

#endif // CVCORE_TRAITS_H
