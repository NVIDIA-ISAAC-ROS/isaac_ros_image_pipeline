// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#pragma once

#include "gems/core/math/pose3.hpp"
#include "gxf/multimedia/camera.hpp"

namespace nvidia {
namespace isaac {

// Converts an isaac pose object to a GXF pose object
gxf::Pose3D ConvertToGxfPose(const ::nvidia::isaac::Pose3d& pose);

// Converts GXFPose to Isaac pose
::nvidia::isaac::Pose3d ConvertToIsaacPose(const gxf::Pose3D& pose);

}  // namespace isaac
}  // namespace nvidia
