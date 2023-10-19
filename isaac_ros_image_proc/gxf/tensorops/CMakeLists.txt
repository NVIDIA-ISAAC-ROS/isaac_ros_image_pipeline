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

project(gxf_tensorops LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-fPIC -w)
endif()

# Dependencies
find_package(CUDAToolkit)
find_package(vpi REQUIRED)
find_package(GXF ${ISAAC_ROS_GXF_VERSION} MODULE REQUIRED
    core
    cuda
    multimedia
)

# Create extension
add_library(gxf_tensorops SHARED
  extensions/tensorops/tensorops.cpp
  extensions/tensorops/components/BBoxGenerator.cpp
  extensions/tensorops/components/BBoxGenerator.hpp
  extensions/tensorops/components/CameraModel.cpp
  extensions/tensorops/components/CameraModel.hpp
  extensions/tensorops/components/ConvertColorFormat.cpp
  extensions/tensorops/components/ConvertColorFormat.hpp
  extensions/tensorops/components/CropAndResize.cpp
  extensions/tensorops/components/CropAndResize.hpp
  extensions/tensorops/components/Frame3D.cpp
  extensions/tensorops/components/Frame3D.hpp
  extensions/tensorops/components/ImageAdapter.cpp
  extensions/tensorops/components/ImageAdapter.hpp
  extensions/tensorops/components/ImageUtils.cpp
  extensions/tensorops/components/ImageUtils.hpp
  extensions/tensorops/components/InterleavedToPlanar.cpp
  extensions/tensorops/components/InterleavedToPlanar.hpp
  extensions/tensorops/components/Normalize.cpp
  extensions/tensorops/components/Normalize.hpp
  extensions/tensorops/components/Reshape.cpp
  extensions/tensorops/components/Reshape.hpp
  extensions/tensorops/components/Resize.cpp
  extensions/tensorops/components/Resize.hpp
  extensions/tensorops/components/TensorOperator.cpp
  extensions/tensorops/components/TensorOperator.hpp
  extensions/tensorops/components/TensorStream.cpp
  extensions/tensorops/components/TensorStream.hpp
  extensions/tensorops/components/Undistort.cpp
  extensions/tensorops/components/Undistort.hpp
  extensions/tensorops/components/detail/ImageAdapterTensorImpl.cpp
  extensions/tensorops/components/detail/ImageAdapterTensorImpl.hpp
  extensions/tensorops/components/detail/ImageAdapterVideoBufferImpl.cpp
  extensions/tensorops/components/detail/ImageAdapterVideoBufferImpl.hpp

  extensions/tensorops/core/ArithmeticOperations.cpp
  extensions/tensorops/core/BBoxUtils.cpp
  extensions/tensorops/core/ColorConversions.cpp
  extensions/tensorops/core/DBScan.cpp
  extensions/tensorops/core/Errors.cpp
  extensions/tensorops/core/Filters.cpp
  extensions/tensorops/core/FusedOperations.cpp
  extensions/tensorops/core/GeometryTransforms.cpp
  extensions/tensorops/core/IImageWarp.cpp
  extensions/tensorops/core/OneEuroFilter.cpp
  extensions/tensorops/core/TensorOperators.cpp
  extensions/tensorops/core/VPIColorConvertImpl.cpp
  extensions/tensorops/core/VPIRemapImpl.cpp
  extensions/tensorops/core/VPIResizeImpl.cpp
  extensions/tensorops/core/VPIStatusMapping.cpp
  extensions/tensorops/core/VPIStereoDisparityEstimatorImpl.cpp
  extensions/tensorops/core/VPITensorOperators.cpp
  extensions/tensorops/core/BBoxUtils.h
  extensions/tensorops/core/DBScan.h
  extensions/tensorops/core/Errors.h
  extensions/tensorops/core/Filters.h
  extensions/tensorops/core/IImageWarp.h
  extensions/tensorops/core/ITensorOperatorContext.h
  extensions/tensorops/core/ITensorOperatorStream.h
  extensions/tensorops/core/ImageUtils.h
  extensions/tensorops/core/OneEuroFilter.h
  extensions/tensorops/core/TensorOperators.h
  extensions/tensorops/core/VPIColorConvertImpl.h
  extensions/tensorops/core/VPIEnumMapping.h
  extensions/tensorops/core/VPIImageWarp.h
  extensions/tensorops/core/VPIRemapImpl.h
  extensions/tensorops/core/VPIResizeImpl.h
  extensions/tensorops/core/VPIStatusMapping.h
  extensions/tensorops/core/VPIStereoDisparityEstimatorImpl.h
  extensions/tensorops/core/VPITensorOperators.h
  extensions/tensorops/core/NppUtils.cpp
  extensions/tensorops/core/NppUtils.h
  extensions/tensorops/core/Filters.cpp
  extensions/tensorops/core/Filters.h
  extensions/tensorops/core/Array.cpp
  extensions/tensorops/core/CVError.cpp
  extensions/tensorops/core/MathTypes.cpp
  extensions/tensorops/core/Memory.cpp
  extensions/tensorops/core/Tensor.cpp
  extensions/tensorops/core/Array.h
  extensions/tensorops/core/BBox.h
  extensions/tensorops/core/CVError.h
  extensions/tensorops/core/CameraModel.h
  extensions/tensorops/core/ComputeEngine.h
  extensions/tensorops/core/Core.h
  extensions/tensorops/core/Image.h
  extensions/tensorops/core/MathTypes.h
  extensions/tensorops/core/Memory.h
  extensions/tensorops/core/Tensor.h
  extensions/tensorops/core/Traits.h
)

target_include_directories(gxf_tensorops PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(gxf_tensorops
  PRIVATE
    CUDA::cudart
    CUDA::nppc
    CUDA::nppial
    CUDA::nppicc
    CUDA::nppidei
    CUDA::nppif
    CUDA::nppig
    CUDA::nppisu
    GXF::cuda
    GXF::multimedia
    yaml-cpp
    vpi
)