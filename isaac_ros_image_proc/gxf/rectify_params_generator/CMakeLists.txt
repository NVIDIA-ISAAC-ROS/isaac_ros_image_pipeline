# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

project(gxf_rectify_params_generator LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-fPIC -w)
endif()

# Dependencies
find_package(yaml-cpp)
find_package(GXF ${ISAAC_ROS_GXF_VERSION} MODULE REQUIRED
    multimedia
    isaac_messages
)
find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Create extension
add_library(gxf_rectify_params_generator SHARED
    extensions/rectify/rectify.cpp
    extensions/rectify/components/rectify_params_generator.cpp
    extensions/rectify/components/rectify_params_generator.hpp
    extensions/rectify/components/stereo_extrinsics_normalizer.cpp
    extensions/rectify/components/stereo_extrinsics_normalizer.hpp
)
target_include_directories(gxf_rectify_params_generator PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR})


add_library(gxf_rectify_utils STATIC
extensions/rectify/utils/camera_utils.cpp
extensions/rectify/utils/camera_utils.hpp
extensions/rectify/utils/utils.cpp
extensions/rectify/utils/utils.hpp
)
target_include_directories(gxf_rectify_utils PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(gxf_rectify_utils
  PUBLIC
    yaml-cpp
  PRIVATE
    Eigen3::Eigen
    GXF::multimedia
    GXF::isaac_messages
    ${OpenCV_LIBS}
)

target_link_libraries(gxf_rectify_params_generator
  PUBLIC
    GXF::multimedia
    GXF::isaac_messages
    ${OpenCV_LIBS}
  PRIVATE
    gxf_rectify_utils
    Eigen3::Eigen
)
