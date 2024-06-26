# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.5)
project(gxf_isaac_depth_image_proc LANGUAGES C CXX CUDA)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Dependencies
find_package(CUDAToolkit REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(yaml-cpp)

# Occupancy Grid Projector extension
ament_auto_add_library(${PROJECT_NAME} SHARED
  gxf/extensions/depth_image_proc/depth_image_proc_extension.cpp
  gxf/extensions/depth_image_proc/depth_to_point_cloud.cpp
  gxf/extensions/depth_image_proc/depth_to_point_cloud_cuda.cu.cpp
  gxf/extensions/depth_image_proc/depth_to_point_cloud_cuda.cu.hpp
  gxf/extensions/depth_image_proc/depth_to_point_cloud.hpp
)
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/gxf")
# Mark as CUDA files with non-standard extensions
set_source_files_properties(
  gxf/extensions/depth_image_proc/depth_to_point_cloud_cuda.cu.cpp
  gxf/extensions/depth_image_proc/depth_to_point_cloud_cuda.cu.hpp
  PROPERTIES LANGUAGE CUDA
)
target_link_libraries(${PROJECT_NAME}
  CUDA::cudart
  Eigen3::Eigen
  yaml-cpp
)
set_target_properties(${PROJECT_NAME} PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

# Disable -Wpedantic flag
get_target_property(target_options ${PROJECT_NAME} COMPILE_OPTIONS)
list(REMOVE_ITEM target_options "-Wpedantic")
set_property(TARGET ${PROJECT_NAME} PROPERTY COMPILE_OPTIONS ${target_options})

# Install the binary file
install(TARGETS ${PROJECT_NAME} DESTINATION share/${PROJECT_NAME}/gxf/lib)

ament_auto_package(INSTALL_TO_SHARE)