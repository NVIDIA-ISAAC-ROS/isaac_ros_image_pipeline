// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <vpi/Image.h>
#include <vpi/Context.h>
#include <vpi/Types.h>

#include <memory>
#include <stdexcept>
#include <utility>
#include <string>

#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "isaac_ros_nitros_disparity_image_type/nitros_disparity_image.hpp"
#include "isaac_ros_nitros/types/nitros_buffer.hpp"
#include "isaac_ros_vpi_utils/vpi_utilities.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace vpi_utils
{

using nvidia::isaac_ros::nitros::NitrosImage;
using nvidia::isaac_ros::nitros::NitrosBuffer;
using nvidia::isaac_ros::nitros::NitrosBufferAccessor;
using nvidia::isaac_ros::nitros::ReadHandle;
using nvidia::isaac_ros::nitros::WriteHandle;
using nvidia::isaac_ros::nitros::NitrosDisparityImage;

// VPI handle wrappers for safe zero-copy integration with NITROS Image
//
// These wrappers serve two purposes:
//   1. Prevent dangling pointers: Hold shared_ptr<NitrosBuffer> to keep buffer alive
//      while VPI image wrapper references the GPU memory
//   2. Extend handle lifetimes: Keep ReadHandle/WriteHandle alive to maintain proper
//      CUDA event synchronization throughout VPI operations
template<typename NitrosHandleT>
class VPIImageHandle
{
public:
  VPIImageHandle() = default;

  VPIImageHandle(
    VPIImage vpi_img,
    const VPIImageData & vpi_data,
    std::shared_ptr<NitrosBuffer> buffer,
    NitrosHandleT handle)
  : vpi_image_(vpi_img), vpi_image_data_(vpi_data), nitros_buffer_(buffer),
    nitros_handle_(std::move(handle))
  {
    if (context_ == nullptr) {
      VPIStatus status = vpiContextGetCurrent(&context_);
      if (status != VPI_SUCCESS) {
        throw std::runtime_error("vpiContextGetCurrent failed");
      }

      status = vpiContextGetFlags(context_, &backend_flags_);
      if (status != VPI_SUCCESS) {
        throw std::runtime_error("vpiContextGetFlags failed");
      }

      VPIImageFormat image_format;
      vpiImageGetFormat(vpi_image_, &image_format);
    }
  }

  VPIImageHandle(const VPIImageHandle &) = delete;
  VPIImageHandle & operator=(const VPIImageHandle &) = delete;

  VPIImageHandle(VPIImageHandle && other) noexcept
  : vpi_image_(other.vpi_image_),
    vpi_image_data_(std::move(other.vpi_image_data_)),
    nitros_buffer_(std::move(other.nitros_buffer_)),
    nitros_handle_(std::move(other.nitros_handle_)),
    backend_flags_(other.backend_flags_),
    context_(other.context_)
  {
    other.vpi_image_ = nullptr;
    other.context_ = nullptr;
  }

  VPIImageHandle & operator=(VPIImageHandle && other) noexcept
  {
    if (this != &other) {
      release();
      vpi_image_ = other.vpi_image_;
      vpi_image_data_ = std::move(other.vpi_image_data_);
      nitros_buffer_ = std::move(other.nitros_buffer_);
      nitros_handle_ = std::move(other.nitros_handle_);
      backend_flags_ = other.backend_flags_;
      context_ = other.context_;
      other.vpi_image_ = nullptr;
      other.context_ = nullptr;
    }
    return *this;
  }

  ~VPIImageHandle()
  {
    release();
  }

  void release()
  {
    if (vpi_image_) {
      vpiImageDestroy(vpi_image_);
      vpi_image_ = nullptr;
    }
  }

  bool update_data_pointer(uint8_t * data)
  {
    vpi_image_data_.buffer.pitch.planes[0].pBase = data;
    VPIStatus status = vpiImageSetWrapper(vpi_image_, &vpi_image_data_);
    if (status != VPI_SUCCESS) {
      RCLCPP_ERROR(rclcpp::get_logger("VPIImageHandle::update_data_pointer"),
        "vpiImageSetWrapper for data pointer update failed with status %d", status);
      return false;
    }
    return true;
  }

  VPIImage get_vpi_image() const {return vpi_image_;}
  VPIImageData & get_vpi_image_data() {return vpi_image_data_;}
  const VPIImageData & get_vpi_image_data() const {return vpi_image_data_;}

  // Static factory method to create VPIImageHandle from NITROS image
  static VPIImageHandle<NitrosHandleT> wrap(
    const NitrosImage & msg,
    uint64_t backend_flags,
    NitrosHandleT nitros_handle)
  {
    auto buffer = NitrosBufferAccessor<NitrosImage>::get_buffer(msg);

    // Convert encoding to VPI format
    auto vpi_format = nvidia::isaac_ros::vpi_utils::ToVpiFormat(msg.encoding);

    // Get base data pointer from buffer
    const uint8_t * base_ptr = buffer->get_data();

    VPIImageData vpi_image_data{};
    vpi_image_data.bufferType = VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;
    vpi_image_data.buffer.pitch.format = vpi_format.image_format;
    vpi_image_data.buffer.pitch.numPlanes = static_cast<uint32_t>(msg.num_planes());
    for (size_t i = 0; i < vpi_image_data.buffer.pitch.numPlanes; i++) {
      const auto & plane = msg.get_plane(i);
      vpi_image_data.buffer.pitch.planes[i].pixelType = vpi_format.pixel_type[i];
      vpi_image_data.buffer.pitch.planes[i].width = plane.width;
      vpi_image_data.buffer.pitch.planes[i].height = plane.height;
      vpi_image_data.buffer.pitch.planes[i].pitchBytes = plane.stride;
      vpi_image_data.buffer.pitch.planes[i].offsetBytes = msg.get_plane_offset(i);
      vpi_image_data.buffer.pitch.planes[i].pBase =
        const_cast<uint8_t *>(base_ptr + msg.get_plane_offset(i));
    }

    VPIImage vpi_image{nullptr};
    VPIStatus status = vpiImageCreateWrapper(&vpi_image_data, nullptr, backend_flags, &vpi_image);
    if (status != VPI_SUCCESS) {
      throw std::runtime_error("vpiImageCreateWrapper failed");
    }

    return VPIImageHandle<NitrosHandleT>(std::move(vpi_image), vpi_image_data, buffer,
      std::move(nitros_handle));
  }

  // Static factory method to create VPIImageHandle from NITROS Disparity image of type
  // nitros_disparity_image_32FC1
  static VPIImageHandle<NitrosHandleT> wrap(
    const NitrosDisparityImage & msg,
    uint64_t backend_flags,
    NitrosHandleT nitros_handle)
  {
    auto buffer = NitrosBufferAccessor<NitrosDisparityImage>::get_buffer(msg);
    // Convert encoding to VPI format
    auto vpi_format = nvidia::isaac_ros::vpi_utils::ToVpiFormat("32FC1");
    uint32_t width = msg.get_width();
    uint32_t height = msg.get_height();
    // Get base data pointer from buffer
    const uint8_t * base_ptr = buffer->get_data();

    VPIImageData vpi_image_data{};
    vpi_image_data.bufferType = VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;
    vpi_image_data.buffer.pitch.format = vpi_format.image_format;
    vpi_image_data.buffer.pitch.numPlanes = 1;
    vpi_image_data.buffer.pitch.planes[0].pixelType = VPI_PIXEL_TYPE_F32;
    vpi_image_data.buffer.pitch.planes[0].width = width;
    vpi_image_data.buffer.pitch.planes[0].height = height;
    vpi_image_data.buffer.pitch.planes[0].pitchBytes = width * sizeof(float);
    vpi_image_data.buffer.pitch.planes[0].offsetBytes = 0;
    vpi_image_data.buffer.pitch.planes[0].pBase = const_cast<uint8_t *>(base_ptr);

    VPIImage vpi_image{nullptr};
    VPIStatus status = vpiImageCreateWrapper(&vpi_image_data, nullptr, backend_flags, &vpi_image);
    if (status != VPI_SUCCESS) {
      throw std::runtime_error("vpiImageCreateWrapper failed");
    }

    return VPIImageHandle<NitrosHandleT>(std::move(vpi_image), vpi_image_data, buffer,
      std::move(nitros_handle));
  }

private:
  VPIImage vpi_image_{nullptr};
  VPIImageData vpi_image_data_{};
  std::shared_ptr<NitrosBuffer> nitros_buffer_;
  NitrosHandleT nitros_handle_;
  uint64_t backend_flags_{0};
  VPIContext context_{nullptr};
};

// Generic function to wrap VPI image with NITROS handle
// This is a convenience wrapper that delegates to VPIImageHandle<T>::wrap()
template<typename NitrosHandleT>
VPIImageHandle<NitrosHandleT> WrapVPIImage(
  const NitrosImage & msg,
  uint64_t backend_flags,
  NitrosHandleT nitros_handle)
{
  return VPIImageHandle<NitrosHandleT>::wrap(msg, backend_flags, std::move(nitros_handle));
}

template<typename NitrosHandleT>
VPIImageHandle<NitrosHandleT> WrapVPIImage(
  const NitrosDisparityImage & msg,
  uint64_t backend_flags,
  NitrosHandleT nitros_handle)
{
  return VPIImageHandle<NitrosHandleT>::wrap(msg, backend_flags, std::move(nitros_handle));
}

// Helper function to update VPI image data pointer
template<typename NitrosHandleT>
bool UpdateVPIImageDataPointer(
  VPIImageHandle<NitrosHandleT> & handle,
  uint8_t * data)
{
  return handle.update_data_pointer(data);
}

}  // namespace vpi_utils
}  // namespace isaac_ros
}  // namespace nvidia
