# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(gxf_isaac_tensorops LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-fPIC -w)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Dependencies
find_package(CUDAToolkit)
find_package(vpi REQUIRED)

# Create extension
ament_auto_add_library(${PROJECT_NAME} SHARED
  gxf/extensions/tensorops/tensorops.cpp
  gxf/extensions/tensorops/components/BBoxGenerator.cpp
  gxf/extensions/tensorops/components/BBoxGenerator.hpp
  gxf/extensions/tensorops/components/CameraModel.cpp
  gxf/extensions/tensorops/components/CameraModel.hpp
  gxf/extensions/tensorops/components/ConvertColorFormat.cpp
  gxf/extensions/tensorops/components/ConvertColorFormat.hpp
  gxf/extensions/tensorops/components/CropAndResize.cpp
  gxf/extensions/tensorops/components/CropAndResize.hpp
  gxf/extensions/tensorops/components/Frame3D.cpp
  gxf/extensions/tensorops/components/Frame3D.hpp
  gxf/extensions/tensorops/components/ImageAdapter.cpp
  gxf/extensions/tensorops/components/ImageAdapter.hpp
  gxf/extensions/tensorops/components/ImageUtils.cpp
  gxf/extensions/tensorops/components/ImageUtils.hpp
  gxf/extensions/tensorops/components/InterleavedToPlanar.cpp
  gxf/extensions/tensorops/components/InterleavedToPlanar.hpp
  gxf/extensions/tensorops/components/Normalize.cpp
  gxf/extensions/tensorops/components/Normalize.hpp
  gxf/extensions/tensorops/components/Reshape.cpp
  gxf/extensions/tensorops/components/Reshape.hpp
  gxf/extensions/tensorops/components/Resize.cpp
  gxf/extensions/tensorops/components/Resize.hpp
  gxf/extensions/tensorops/components/TensorOperator.cpp
  gxf/extensions/tensorops/components/TensorOperator.hpp
  gxf/extensions/tensorops/components/TensorStream.cpp
  gxf/extensions/tensorops/components/TensorStream.hpp
  gxf/extensions/tensorops/components/Undistort.cpp
  gxf/extensions/tensorops/components/Undistort.hpp
  gxf/extensions/tensorops/components/detail/ImageAdapterTensorImpl.cpp
  gxf/extensions/tensorops/components/detail/ImageAdapterTensorImpl.hpp
  gxf/extensions/tensorops/components/detail/ImageAdapterVideoBufferImpl.cpp
  gxf/extensions/tensorops/components/detail/ImageAdapterVideoBufferImpl.hpp

  gxf/extensions/tensorops/core/ArithmeticOperations.cpp
  gxf/extensions/tensorops/core/BBoxUtils.cpp
  gxf/extensions/tensorops/core/ColorConversions.cpp
  gxf/extensions/tensorops/core/DBScan.cpp
  gxf/extensions/tensorops/core/Errors.cpp
  gxf/extensions/tensorops/core/Filters.cpp
  gxf/extensions/tensorops/core/FusedOperations.cpp
  gxf/extensions/tensorops/core/GeometryTransforms.cpp
  gxf/extensions/tensorops/core/IImageWarp.cpp
  gxf/extensions/tensorops/core/OneEuroFilter.cpp
  gxf/extensions/tensorops/core/TensorOperators.cpp
  gxf/extensions/tensorops/core/VPIColorConvertImpl.cpp
  gxf/extensions/tensorops/core/VPIRemapImpl.cpp
  gxf/extensions/tensorops/core/VPIResizeImpl.cpp
  gxf/extensions/tensorops/core/VPIStatusMapping.cpp
  gxf/extensions/tensorops/core/VPIStereoDisparityEstimatorImpl.cpp
  gxf/extensions/tensorops/core/VPITensorOperators.cpp
  gxf/extensions/tensorops/core/BBoxUtils.h
  gxf/extensions/tensorops/core/DBScan.h
  gxf/extensions/tensorops/core/Errors.h
  gxf/extensions/tensorops/core/Filters.h
  gxf/extensions/tensorops/core/IImageWarp.h
  gxf/extensions/tensorops/core/ITensorOperatorContext.h
  gxf/extensions/tensorops/core/ITensorOperatorStream.h
  gxf/extensions/tensorops/core/ImageUtils.h
  gxf/extensions/tensorops/core/OneEuroFilter.h
  gxf/extensions/tensorops/core/TensorOperators.h
  gxf/extensions/tensorops/core/VPIColorConvertImpl.h
  gxf/extensions/tensorops/core/VPIEnumMapping.h
  gxf/extensions/tensorops/core/VPIImageWarp.h
  gxf/extensions/tensorops/core/VPIRemapImpl.h
  gxf/extensions/tensorops/core/VPIResizeImpl.h
  gxf/extensions/tensorops/core/VPIStatusMapping.h
  gxf/extensions/tensorops/core/VPIStereoDisparityEstimatorImpl.h
  gxf/extensions/tensorops/core/VPITensorOperators.h
  gxf/extensions/tensorops/core/NppUtils.cpp
  gxf/extensions/tensorops/core/NppUtils.h
  gxf/extensions/tensorops/core/Filters.cpp
  gxf/extensions/tensorops/core/Filters.h
  gxf/extensions/tensorops/core/Array.cpp
  gxf/extensions/tensorops/core/CVError.cpp
  gxf/extensions/tensorops/core/MathTypes.cpp
  gxf/extensions/tensorops/core/Memory.cpp
  gxf/extensions/tensorops/core/Tensor.cpp
  gxf/extensions/tensorops/core/Array.h
  gxf/extensions/tensorops/core/BBox.h
  gxf/extensions/tensorops/core/CVError.h
  gxf/extensions/tensorops/core/CameraModel.h
  gxf/extensions/tensorops/core/ComputeEngine.h
  gxf/extensions/tensorops/core/Core.h
  gxf/extensions/tensorops/core/Image.h
  gxf/extensions/tensorops/core/MathTypes.h
  gxf/extensions/tensorops/core/Memory.h
  gxf/extensions/tensorops/core/Tensor.h
  gxf/extensions/tensorops/core/Traits.h
)
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/gxf")
target_link_libraries(${PROJECT_NAME}
  CUDA::cudart
  CUDA::nppc
  CUDA::nppial
  CUDA::nppicc
  CUDA::nppidei
  CUDA::nppif
  CUDA::nppig
  CUDA::nppisu
  yaml-cpp
  vpi
)
set_target_properties(${PROJECT_NAME} PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Install the binary file
install(TARGETS ${PROJECT_NAME} DESTINATION share/${PROJECT_NAME}/gxf/lib)

# Install headers
install(DIRECTORY gxf/
        DESTINATION include)
ament_export_include_directories(include)


# Embed versioning information into installed files
ament_index_get_resource(ISAAC_ROS_COMMON_CMAKE_PATH isaac_ros_common_cmake_path isaac_ros_common)
include("${ISAAC_ROS_COMMON_CMAKE_PATH}/isaac_ros_common-version-info.cmake")
generate_version_info(${PROJECT_NAME})

ament_auto_package(INSTALL_TO_SHARE)
