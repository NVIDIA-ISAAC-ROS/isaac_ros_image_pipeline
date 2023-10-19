# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

project(gxf_sgm_disparity LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-fPIC -w)
endif()

find_package(GXF ${ISAAC_ROS_GXF_VERSION} MODULE REQUIRED
    COMPONENTS
    isaac_messages
    multimedia
    std
)
find_package(yaml-cpp)
find_package(vpi REQUIRED)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# SGM Disparity extension
add_library(gxf_sgm SHARED
    extensions/sgm/sgm.cpp
    extensions/sgm/camera_message_compositor.cpp
    extensions/sgm/camera_message_compositor.hpp
    extensions/sgm/disparity_compositor.cpp
    extensions/sgm/disparity_compositor.hpp
    extensions/sgm/sgm_disparity.cpp
    extensions/sgm/sgm_disparity.hpp
)

# Tensorops extension
add_library(tensorops STATIC
    extensions/tensorops/components/ImageUtils.cpp
    extensions/tensorops/components/ImageUtils.hpp
)
target_include_directories(tensorops PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(gxf_vpi STATIC
    gems/vpi/constants.hpp
    gems/vpi/constants.cpp
    gems/vpi/image_wrapper.cpp
    gems/vpi/image_wrapper.hpp
)
target_include_directories(gxf_vpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(tensorops
  PUBLIC
    GXF::multimedia
    GXF::std
)

target_link_libraries(gxf_vpi
  PUBLIC
    GXF::multimedia
    GXF::std
    vpi
)

target_link_libraries(gxf_sgm
  PUBLIC
    GXF::multimedia
    GXF::std
    vpi
    yaml-cpp
  PRIVATE
    GXF::isaac_messages
    gxf_vpi
    tensorops
)
