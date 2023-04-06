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
    cuda
    multimedia
)

# Create extension
add_library(gxf_tensorops SHARED
  CameraModel.cpp
  CameraModel.hpp
  CMakeLists.txt
  ConvertColorFormat.cpp
  ConvertColorFormat.hpp
  CropAndResize.cpp
  CropAndResize.hpp
  Frame3D.cpp
  Frame3D.hpp
  ImageAdapter.cpp
  ImageAdapter.hpp
  ImageUtils.cpp
  ImageUtils.hpp
  InterleavedToPlanar.cpp
  InterleavedToPlanar.hpp
  Normalize.cpp
  Normalize.hpp
  Reshape.cpp
  Reshape.hpp
  Resize.cpp
  Resize.hpp
  TensorOperator.cpp
  TensorOperator.hpp
  TensorOps.cpp
  TensorStream.cpp
  TensorStream.hpp
  Undistort.cpp
  Undistort.hpp
  detail/ImageAdapterTensorImpl.cpp
  detail/ImageAdapterTensorImpl.hpp
  detail/ImageAdapterVideoBufferImpl.cpp
  detail/ImageAdapterVideoBufferImpl.hpp
)

target_include_directories(gxf_tensorops PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/cvcore/include)

add_library(cvcore STATIC
  # Tensorops
  cvcore/src/tensor_ops/ArithmeticOperations.cpp
  cvcore/src/tensor_ops/BBoxUtils.cpp
  cvcore/src/tensor_ops/ColorConversions.cpp
  cvcore/src/tensor_ops/DBScan.cpp
  cvcore/src/tensor_ops/Errors.cpp
  cvcore/src/tensor_ops/Filters.cpp
  cvcore/src/tensor_ops/Filters.h
  cvcore/src/tensor_ops/FusedOperations.cpp
  cvcore/src/tensor_ops/GeometryTransforms.cpp
  cvcore/src/tensor_ops/IImageWarp.cpp
  cvcore/src/tensor_ops/NppUtils.cpp
  cvcore/src/tensor_ops/NppUtils.h
  cvcore/src/tensor_ops/OneEuroFilter.cpp
  cvcore/src/tensor_ops/TensorOperators.cpp

  # Tensorops/VPI
  cvcore/src/tensor_ops/vpi/VPIColorConvertImpl.cpp
  cvcore/src/tensor_ops/vpi/VPIColorConvertImpl.h
  cvcore/src/tensor_ops/vpi/VPIEnumMapping.h
  cvcore/src/tensor_ops/vpi/VPIImageWarp.h
  cvcore/src/tensor_ops/vpi/VPIRemapImpl.cpp
  cvcore/src/tensor_ops/vpi/VPIRemapImpl.h
  cvcore/src/tensor_ops/vpi/VPIResizeImpl.cpp
  cvcore/src/tensor_ops/vpi/VPIResizeImpl.h
  cvcore/src/tensor_ops/vpi/VPIStatusMapping.cpp
  cvcore/src/tensor_ops/vpi/VPIStatusMapping.h
  cvcore/src/tensor_ops/vpi/VPIStereoDisparityEstimatorImpl.cpp
  cvcore/src/tensor_ops/vpi/VPIStereoDisparityEstimatorImpl.h
  cvcore/src/tensor_ops/vpi/VPITensorOperators.cpp
  cvcore/src/tensor_ops/vpi/VPITensorOperators.h

  # Core
  cvcore/src/core/cvcore/Array.cpp
  cvcore/src/core/cvcore/Dummy.cu
  cvcore/src/core/cvcore/MathTypes.cpp
  cvcore/src/core/cvcore/Tensor.cpp
  cvcore/src/core/utility/CVError.cpp
  cvcore/src/core/utility/Instrumentation.cpp
  cvcore/src/core/utility/Memory.cpp
  cvcore/src/core/utility/ProfileUtils.cpp
)

target_include_directories(cvcore PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/cvcore/include)
target_compile_definitions(cvcore PRIVATE -DENABLE_VPI=1)

target_link_libraries(cvcore PUBLIC
  CUDA::cudart
  CUDA::nppc
  CUDA::nppial
  CUDA::nppicc
  CUDA::nppidei
  CUDA::nppif
  CUDA::nppig
  CUDA::nppisu
  vpi
)

target_link_libraries(gxf_tensorops
  PUBLIC
    GXF::cuda
    GXF::multimedia
    yaml-cpp
  PRIVATE
    cvcore
)
