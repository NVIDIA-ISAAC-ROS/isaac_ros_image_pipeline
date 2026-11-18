# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.22.1)
project(cvcuda_conversions LANGUAGES C CXX)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Header-only interface library. It is consumed by targets that already link
# CV-CUDA (nvcv_types) and the CUDA runtime, so we deliberately do NOT put those
# imported targets in the exported link interface: doing so would force every
# downstream package to find_package(CUDAToolkit)/CV-CUDA before loading this
# package's export. We only propagate our own include directory and the ROS
# package dependencies.
add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")
# Propagate message / buffer include paths to consumers of this header-only lib.
target_link_libraries(${PROJECT_NAME} INTERFACE
  ${cuda_buffer_TARGETS}
  ${sensor_msgs_TARGETS}
  ${isaac_ros_tensor_msgs_TARGETS}
  tensor_msgs::tensor_msgs
)

install(DIRECTORY include/ DESTINATION include)
install(TARGETS ${PROJECT_NAME} EXPORT export_${PROJECT_NAME})

ament_export_targets(export_${PROJECT_NAME})
ament_export_include_directories(include)
ament_export_dependencies(cuda_buffer isaac_ros_tensor_msgs sensor_msgs tensor_msgs)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)
  find_package(CUDAToolkit REQUIRED)

  ament_add_gtest(cvcuda_conversions_test test/cvcuda_conversions_test.cpp)
  target_link_libraries(cvcuda_conversions_test
    ${PROJECT_NAME}
    nvcv_types
    CUDA::cudart
    CUDA::cuda_driver
  )
endif()

ament_auto_package()
