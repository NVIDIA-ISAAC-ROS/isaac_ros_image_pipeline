// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "extensions/rectify/components/rectify_params_generator.hpp"
#include "extensions/rectify/components/stereo_extrinsics_normalizer.hpp"
#include "extensions/rectify/components/undistort_params_generator.hpp"
#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()

GXF_EXT_FACTORY_SET_INFO(0xd8d7816ec0485ad4, 0xff795a414bd445ca, "RECTIFY",
                         "Extension containing components for rectify",
                         "Isaac SDK", "2.0.0", "LICENSE");

GXF_EXT_FACTORY_ADD(0xa9ddb12454d54aeb, 0x9739263d9fd1f635,
                      nvidia::isaac::RectifyParamsGenerator, nvidia::gxf::Codelet,
                      "Generate rectification parameters.");

GXF_EXT_FACTORY_ADD(0x14726cde87cf11ee, 0xb9d10242ac120002,
                      nvidia::isaac::UndistortParamsGenerator, nvidia::gxf::Codelet,
                      "Generate undistort parameters.");

GXF_EXT_FACTORY_ADD(0xa9ddb12454d54aeb, 0x9738263d9fd1f635,
                    nvidia::isaac::StereoExtrinsicsNormalizer, nvidia::gxf::Codelet,
                    "Generate rectification undisortion post processing parameters.");

GXF_EXT_FACTORY_END()
