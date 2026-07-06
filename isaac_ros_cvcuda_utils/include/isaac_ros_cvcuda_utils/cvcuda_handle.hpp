// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <stdexcept>
#include <utility>

#include "isaac_ros_cvcuda_utils/cvcuda_utilities.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor.hpp"
#include "isaac_ros_nitros/types/nitros_buffer.hpp"
#include "nvcv/Tensor.hpp"
#include "nvcv/TensorDataAccess.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace cvcuda_utils
{

using nvidia::isaac_ros::nitros::NitrosImage;
using nvidia::isaac_ros::nitros::NitrosBuffer;
using nvidia::isaac_ros::nitros::NitrosTensor;
using nvidia::isaac_ros::nitros::NitrosBufferAccessor;
using nvidia::isaac_ros::nitros::ReadHandle;
using nvidia::isaac_ros::nitros::WriteHandle;

// CV-CUDA handle wrappers for safe zero-copy integration with NITROS Image
//
// These wrappers serve two purposes:
//   1. Prevent dangling pointers: Hold shared_ptr<NitrosBuffer> to keep buffer alive
//      while CV-CUDA tensor wrapper references the GPU memory
//   2. Extend handle lifetimes: Keep ReadHandle/WriteHandle alive to maintain proper
//      CUDA event synchronization throughout CV-CUDA operations
template<typename NitrosHandleT>
class CVCUDATensorHandle
{
public:
  CVCUDATensorHandle() = default;

  CVCUDATensorHandle(
    nvcv::Tensor tensor,
    std::shared_ptr<NitrosBuffer> buffer,
    NitrosHandleT handle)
  : tensor_(std::move(tensor)), nitros_buffer_(buffer), nitros_handle_(std::move(handle))
  {}

  CVCUDATensorHandle(const CVCUDATensorHandle &) = delete;
  CVCUDATensorHandle & operator=(const CVCUDATensorHandle &) = delete;

  CVCUDATensorHandle(CVCUDATensorHandle &&) = default;
  CVCUDATensorHandle & operator=(CVCUDATensorHandle &&) = default;

  ~CVCUDATensorHandle() = default;

  nvcv::Tensor & get_tensor() {return tensor_;}
  const nvcv::Tensor & get_tensor() const {return tensor_;}

  // Get the raw data pointer from the underlying tensor
  const uint8_t * get_buffer_data_ptr() const
  {
    auto data = tensor_.exportData<nvcv::TensorDataStridedCuda>();
    if (data) {
      return reinterpret_cast<const uint8_t *>(data->basePtr());
    }
    return nullptr;
  }

  // Wrap a NitrosImage as a CV-CUDA tensor (NHWC).
  //
  // NV12 handling:
  // cvcuda::AdvCvtColor only accepts Tensor input, so a contiguous stacked layout is required.
  // It requires the Y plane (HxW) and interleaved UV plane (H/2 x W) to be contiguous with height
  // H*3/2, width W, and format FMT_Y8 (single-channel uint8). This "stacked"
  // representation is how CV-CUDA interprets NV12 in its YUV-to-RGB kernels
  // (see NVCV_COLOR_YUV2RGB_NV12 in cvcuda/Types.h).
  static CVCUDATensorHandle from_nitros_image(
    const NitrosImage & msg,
    NitrosHandleT nitros_handle,
    nvcv::ImageFormat format,
    int num_channels,
    int bytes_per_channel)
  {
    auto buffer = NitrosBufferAccessor<NitrosImage>::get_buffer(msg);

    // Creating tensor in NHWC format. Assumes nv12 image has no inter-plane padding gap
    // and pitch y == pitch uv.
    const int32_t tensor_height = (msg.encoding == kEncodingNV12) ?
      static_cast<int32_t>(msg.height * 3 / 2) :
      static_cast<int32_t>(msg.height);

    nvcv::TensorDataStridedCuda::Buffer tensor_buffer;
    tensor_buffer.strides[3] = bytes_per_channel;
    tensor_buffer.strides[2] = num_channels * tensor_buffer.strides[3];
    tensor_buffer.strides[1] = msg.step;
    tensor_buffer.strides[0] = tensor_height * tensor_buffer.strides[1];
    tensor_buffer.basePtr = const_cast<NVCVByte *>(
      reinterpret_cast<const NVCVByte *>(nitros_handle.get_ptr()));

    constexpr size_t kBatchSize{1};
    nvcv::Tensor::Requirements reqs{nvcv::Tensor::CalcRequirements(
        kBatchSize, {static_cast<int32_t>(msg.width), tensor_height}, format)};

    nvcv::TensorDataStridedCuda tensor_data{
      nvcv::TensorShape{reqs.shape, reqs.rank, reqs.layout},
      nvcv::DataType{reqs.dtype}, tensor_buffer};

    nvcv::Tensor tensor{nvcv::TensorWrapData(tensor_data)};

    return CVCUDATensorHandle(std::move(tensor), buffer, std::move(nitros_handle));
  }

private:
  nvcv::Tensor tensor_;
  std::shared_ptr<NitrosBuffer> nitros_buffer_;
  NitrosHandleT nitros_handle_;
};

inline void ComputeBufferStrides(
  size_t width, size_t height, size_t channels,
  const nvcv::TensorLayout & tensor_layout,
  const size_t bytes_per_element,
  nvcv::TensorDataStridedCuda::Buffer & buffer)
{
  if (tensor_layout == nvcv::TENSOR_HWC) {
    buffer.strides[2] = bytes_per_element;
    buffer.strides[1] = channels * buffer.strides[2];
    buffer.strides[0] = width * buffer.strides[1];
  } else if (tensor_layout == nvcv::TENSOR_CHW) {
    buffer.strides[2] = bytes_per_element;
    buffer.strides[1] = width * buffer.strides[2];
    buffer.strides[0] = height * buffer.strides[1];
  } else if (tensor_layout == nvcv::TENSOR_NHWC) {
    buffer.strides[3] = bytes_per_element;
    buffer.strides[2] = channels * buffer.strides[3];
    buffer.strides[1] = width * buffer.strides[2];
    buffer.strides[0] = height * buffer.strides[1];
  } else if (tensor_layout == nvcv::TENSOR_NCHW) {
    buffer.strides[3] = bytes_per_element;
    buffer.strides[2] = width * buffer.strides[3];
    buffer.strides[1] = height * buffer.strides[2];
    buffer.strides[0] = channels * buffer.strides[1];
  } else {
    throw std::invalid_argument("Received unexpected tensor layout!");
  }
}

// Generic function to wrap CV-CUDA tensor with NITROS handle
template<typename NitrosHandleT>
CVCUDATensorHandle<NitrosHandleT> WrapCVCUDATensor(
  const NitrosImage & msg,
  NitrosHandleT nitros_handle,
  nvcv::ImageFormat format,
  int num_channels,
  int bytes_per_element,
  size_t batch = 1,
  nvcv::TensorLayout layout = nvcv::TENSOR_NHWC)
{
  auto buffer = NitrosBufferAccessor<NitrosImage>::get_buffer(msg);

  // Creating tensor in NHWC format
  nvcv::TensorDataStridedCuda::Buffer tensor_buffer;
  ComputeBufferStrides(msg.width, msg.height, num_channels, layout, bytes_per_element,
    tensor_buffer);
  tensor_buffer.basePtr = const_cast<NVCVByte *>(
    reinterpret_cast<const NVCVByte *>(nitros_handle.get_ptr()));

  nvcv::Tensor::Requirements reqs{nvcv::Tensor::CalcRequirements(
      batch, {static_cast<int32_t>(msg.width),
        static_cast<int32_t>(msg.height)}, format)};

  nvcv::TensorDataStridedCuda tensor_data{
    nvcv::TensorShape{reqs.shape, reqs.rank, layout},
    nvcv::DataType{reqs.dtype}, tensor_buffer};

  nvcv::Tensor tensor{nvcv::TensorWrapData(tensor_data)};

  return CVCUDATensorHandle<NitrosHandleT>(std::move(tensor), buffer, std::move(nitros_handle));
}

inline void ComputeBufferStrides(
  const nvcv::TensorShape::ShapeType & shape,
  const nvcv::TensorLayout & tensor_layout,
  const size_t bytes_per_element,
  nvcv::TensorDataStridedCuda::Buffer & buffer)
{
  // Manually compute strides, we should get this from CUDA with NITROS later
  if (tensor_layout == nvcv::TENSOR_HWC || tensor_layout == nvcv::TENSOR_CHW) {
    buffer.strides[2] = bytes_per_element;
    buffer.strides[1] = shape[2] * buffer.strides[2];
    buffer.strides[0] = shape[1] * buffer.strides[1];
  } else if (tensor_layout == nvcv::TENSOR_NHWC || tensor_layout == nvcv::TENSOR_NCHW) {
    buffer.strides[3] = bytes_per_element;
    buffer.strides[2] = shape[3] * buffer.strides[3];
    buffer.strides[1] = shape[2] * buffer.strides[2];
    buffer.strides[0] = shape[1] * buffer.strides[1];
  } else {
    throw std::invalid_argument("Received unexpected tensor layout!");
  }
}

inline size_t GetBytesPerElement(const nvcv::DataType & dtype)
{
  size_t element_size = 0;
  switch (dtype) {
    case nvcv::TYPE_U8:
      return 1;
    case nvcv::TYPE_U16:
      return 2;
    case nvcv::TYPE_S8:
      return 1;
    case nvcv::TYPE_S16:
      return 2;
    case nvcv::TYPE_F16:
      return 2;
    case nvcv::TYPE_F32:
      return 4;
    case nvcv::TYPE_F64:
      return 8;
    case nvcv::TYPE_3U8:
      return 3;
    case nvcv::TYPE_4U8:
      return 4;
    case nvcv::TYPE_3F32:
      return 12;
    case nvcv::TYPE_4F32:
      return 16;
    default:
      throw std::invalid_argument("Unsupported data type: " +
        std::to_string(static_cast<int>(dtype)));
  }
  return element_size;
}

// Generic function to wrap NitrosTensor to CV-CUDA tensor with NITROS handle
template<typename NitrosHandleT>
CVCUDATensorHandle<NitrosHandleT> WrapCVCUDATensor(
  const NitrosTensor & nitros_tensor,
  NitrosHandleT nitros_handle,
  nvcv::TensorShape::ShapeType & shape,
  nvcv::DataType dtype,
  nvcv::TensorLayout tensor_layout = nvcv::TENSOR_NHWC)
{
  auto buffer = NitrosBufferAccessor<NitrosTensor>::get_buffer(nitros_tensor);
  nvcv::TensorDataStridedCuda::Buffer tensor_buffer;
  tensor_buffer.basePtr = const_cast<NVCVByte *>(
    reinterpret_cast<const NVCVByte *>(nitros_handle.get_ptr()));
  size_t bytes_per_element = GetBytesPerElement(dtype);
  ComputeBufferStrides(shape, tensor_layout, bytes_per_element, tensor_buffer);
  nvcv::TensorShape tensor_shape{shape, tensor_layout};
  nvcv::TensorDataStridedCuda data{tensor_shape, dtype, tensor_buffer};

  nvcv::Tensor tensor{nvcv::TensorWrapData(data)};
  return CVCUDATensorHandle<NitrosHandleT>(std::move(tensor), buffer, std::move(nitros_handle));
}

// Generic function to wrap NitrosTensor to CV-CUDA tensor with NITROS handle
template<typename NitrosHandleT>
CVCUDATensorHandle<NitrosHandleT> WrapCVCUDATensor(
  const NitrosTensor & nitros_tensor,
  NitrosHandleT nitros_handle,
  int32_t width,
  int32_t height,
  nvcv::ImageFormat format,
  nvcv::DataType dtype,
  int32_t batch = 1,
  nvcv::TensorLayout tensor_layout = nvcv::TENSOR_NCHW)
{
  auto buffer = NitrosBufferAccessor<NitrosTensor>::get_buffer(nitros_tensor);
  nvcv::TensorDataStridedCuda::Buffer tensor_buffer;
  tensor_buffer.basePtr = const_cast<NVCVByte *>(
    reinterpret_cast<const NVCVByte *>(nitros_handle.get_ptr()));

  nvcv::Tensor::Requirements reqs = nvcv::Tensor::CalcRequirements(batch, {width, height}, format);
  nvcv::TensorShape tensor_shape{reqs.shape, reqs.rank, tensor_layout};
  nvcv::TensorShape::ShapeType shape_data;
  for (int i = 0; i < tensor_shape.rank(); ++i) {
    shape_data[i] = tensor_shape[i];
  }
  ComputeBufferStrides(shape_data, tensor_layout, GetBytesPerElement(dtype), tensor_buffer);

  nvcv::TensorDataStridedCuda data(nvcv::TensorShape{reqs.shape, reqs.rank, tensor_layout},
    nvcv::DataType{reqs.dtype}, tensor_buffer);
  nvcv::Tensor tensor{nvcv::TensorWrapData(data)};
  return CVCUDATensorHandle<NitrosHandleT>(std::move(tensor), buffer, std::move(nitros_handle));
}

template<typename NitrosHandleT>
CVCUDATensorHandle<NitrosHandleT> WrapCVCUDATensorNV12(
  const NitrosImage & msg,
  NitrosHandleT nitros_handle)
{
  if (msg.width % 2 != 0 || msg.height % 2 != 0) {
    throw std::invalid_argument("NV12 requires even width and height");
  }
  return CVCUDATensorHandle<NitrosHandleT>::from_nitros_image(
    msg, std::move(nitros_handle), nvcv::FMT_Y8, 1, 1);
}

}  // namespace cvcuda_utils
}  // namespace isaac_ros
}  // namespace nvidia
