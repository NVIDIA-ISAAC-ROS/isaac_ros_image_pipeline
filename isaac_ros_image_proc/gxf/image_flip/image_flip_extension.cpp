// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <string>

#include "gxf/core/gxf.h"
#include "gxf/std/extension_factory_helper.hpp"
#include "image_flip.hpp"

GXF_EXT_FACTORY_BEGIN()

GXF_EXT_FACTORY_SET_INFO(0xee2bfa0599f04926, 0x9b7c4dc4ae6d63cb, "ImageFlipExtension",
                         "Image Flip GXF extension", "NVIDIA", "1.0.0", "LICENSE");

GXF_EXT_FACTORY_ADD(0x5738fb840285469e, 0x9073e6aa491d8a14, nvidia::isaac_ros::ImageFlip,
                    nvidia::gxf::Codelet, "Codelet that flips an image.");

GXF_EXT_FACTORY_END()
