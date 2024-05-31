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

cmake_minimum_required(VERSION 3.22.1)
project(gxf_isaac_rectify LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-fPIC -w)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Dependencies
find_package(yaml-cpp)
find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Extension for RectifyParamsGenerator component
ament_auto_add_library(${PROJECT_NAME} SHARED
  gxf/extensions/rectify/rectify.cpp
  gxf/extensions/rectify/components/rectify_params_generator.cpp
  gxf/extensions/rectify/components/rectify_params_generator.hpp
  gxf/extensions/rectify/components/stereo_extrinsics_normalizer.cpp
  gxf/extensions/rectify/components/stereo_extrinsics_normalizer.hpp
  gxf/extensions/rectify/components/undistort_params_generator.cpp
  gxf/extensions/rectify/components/undistort_params_generator.hpp
  # utils
  gxf/extensions/rectify/utils/camera_utils.cpp
  gxf/extensions/rectify/utils/camera_utils.hpp
  gxf/extensions/rectify/utils/utils.cpp
  gxf/extensions/rectify/utils/utils.hpp
)
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/gxf")
target_link_libraries(${PROJECT_NAME}
  yaml-cpp
  Eigen3::Eigen
  ${OpenCV_LIBS}
)
set_target_properties(${PROJECT_NAME} PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

# Install the binary file
install(TARGETS ${PROJECT_NAME} DESTINATION share/${PROJECT_NAME}/gxf/lib)

ament_auto_package(INSTALL_TO_SHARE)
