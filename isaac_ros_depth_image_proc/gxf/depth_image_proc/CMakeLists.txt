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

project(gxf_depth_image_proc LANGUAGES C CXX CUDA)

# Dependencies
find_package(CUDAToolkit REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(GXF ${ISAAC_ROS_GXF_VERSION} MODULE REQUIRED
    COMPONENTS
    std
    cuda
    multimedia
    isaac_ros_messages
    isaac_messages
)
find_package(yaml-cpp)

# Occupancy Grid Projector extension
add_library(gxf_depth_image_proc SHARED
    depth_image_proc/depth_image_proc_extension.cpp
    depth_image_proc/depth_to_point_cloud.cpp
    depth_image_proc/depth_to_point_cloud_cuda.cu.cpp
    depth_image_proc/depth_to_point_cloud_cuda.cu.hpp
    depth_image_proc/depth_to_point_cloud.hpp
)
# Mark as CUDA files with non-standard extensions
set_source_files_properties(
  depth_image_proc/depth_to_point_cloud_cuda.cu.cpp
  depth_image_proc/depth_to_point_cloud_cuda.cu.hpp
  PROPERTIES LANGUAGE CUDA
)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)
target_link_libraries(gxf_depth_image_proc
  PUBLIC
    CUDA::cudart
    Eigen3::Eigen
    GXF::std
    GXF::cuda
    GXF::multimedia
    yaml-cpp
  PRIVATE
    GXF::isaac_ros_messages
    GXF::isaac_messages
)

# Disable -Wpedantic flag
get_target_property(target_options gxf_depth_image_proc COMPILE_OPTIONS)
list(REMOVE_ITEM target_options "-Wpedantic")
set_property(TARGET gxf_depth_image_proc PROPERTY COMPILE_OPTIONS ${target_options})