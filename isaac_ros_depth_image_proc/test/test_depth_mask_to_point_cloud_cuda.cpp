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

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "isaac_ros_depth_image_proc/depth_mask_to_point_cloud.cu.hpp"
#include "gtest/gtest.h"

namespace nvidia
{
namespace isaac_ros
{
namespace depth_image_proc
{
namespace
{

//! Convert a CUDA fixture failure into an exception.
//!
//! `status` is the CUDA result to check.
//!
//! Returns normally for `cudaSuccess`; otherwise throws `std::runtime_error` with CUDA's error
//! text.
void CheckCuda(cudaError_t status)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

// Gives raw cudaMalloc allocations deterministic, type-preserving test ownership.
template<typename T>
struct CudaDeleter
{
  //! Best-effort release one device allocation.
  //!
  //! `pointer` is null or a device pointer previously returned by cudaMalloc. CUDA cleanup errors
  //! are intentionally ignored because unique_ptr destruction cannot report them.
  void operator()(T * pointer) const
  {
    if (pointer != nullptr) {
      // Explicitly discard cudaFree's status because this deleter has no error channel.
      (void)cudaFree(pointer);
    }
  }
};

// Owns one typed device allocation with CudaDeleter cleanup.
template<typename T>
using DevicePointer = std::unique_ptr<T, CudaDeleter<T>>;

//! Copy host vector elements into a newly owned device allocation.
//!
//! `T` is the unchanged element representation on host and device.
//! `values` supplies a nonempty sequence of contiguous host elements copied into the allocation.
//!
//! Returns unique ownership of device storage for `values.size()` elements after the blocking copy
//! completes. The owner must outlive any kernel reading the allocation. CUDA failures throw through
//! CheckCuda().
template<typename T>
DevicePointer<T> Upload(const std::vector<T> & values)
{
  T * device = nullptr;

  // cudaMalloc writes through void**; reinterpret the typed owner address so CUDA can store an
  // aligned allocation that is subsequently accessed only as T.
  CheckCuda(cudaMalloc(reinterpret_cast<void **>(&device), values.size() * sizeof(T)));
  DevicePointer<T> owner(device);
  CheckCuda(cudaMemcpy(device, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
  return owner;
}

// Owns every device allocation required by one direct conversion launch.
class WorkspaceOwner
{
public:
  //! Allocate a complete direct-launch workspace and output buffer.
  //!
  //! `pixel_count` is positive, fits uint32, and sets every per-pixel capacity. Its byte-size
  //! products must be representable by std::size_t.
  //!
  //! On success, all fields returned by workspace() and output() are ready for a launch. CUDA or
  //! CUB failures throw after releasing any allocations completed earlier in construction. This
  //! owner must outlive all conversion work using its allocations.
  explicit WorkspaceOwner(std::size_t pixel_count)
  {
    try {
      workspace_.pixel_capacity = pixel_count;

      // cudaMalloc exposes void** for typed allocations. Reinterpret each owner address so CUDA can
      // write an aligned device pointer that is subsequently accessed only by its declared type.
      CheckCuda(cudaMalloc(
        reinterpret_cast<void **>(&workspace_.candidates_xyz), pixel_count * 3U * sizeof(float)));
      CheckCuda(cudaMalloc(
        reinterpret_cast<void **>(&workspace_.valid_flags), pixel_count * sizeof(uint32_t)));
      CheckCuda(cudaMalloc(
        reinterpret_cast<void **>(&workspace_.compact_offsets), pixel_count * sizeof(uint32_t)));
      CheckCuda(
        cudaMalloc(reinterpret_cast<void **>(&workspace_.selected_count), sizeof(uint32_t)));
      CheckCuda(CalculateScanTempStorage(
        pixel_count, &workspace_.scan_temp_storage_bytes));
      CheckCuda(cudaMalloc(&workspace_.scan_temp_storage, workspace_.scan_temp_storage_bytes));

      // Apply the same typed-owner reinterpretation to the separately allocated output buffer.
      CheckCuda(cudaMalloc(reinterpret_cast<void **>(&output_), pixel_count * 3U * sizeof(float)));
    } catch (...) {
      Release();
      throw;
    }
  }

  //! Best-effort release every workspace and output allocation.
  //!
  //! All queued work using these allocations must have completed before destruction.
  ~WorkspaceOwner()
  {
    Release();
  }

  WorkspaceOwner(const WorkspaceOwner &) = delete;
  WorkspaceOwner & operator=(const WorkspaceOwner &) = delete;

  //! Borrow the populated workspace descriptor.
  //!
  //! Returns a mutable host reference valid for this owner's lifetime. The descriptor's pointers
  //! refer to device allocations and remain owned by this object. Callers must not replace its
  //! pointers, sizes, or capacity because cleanup relies on them.
  DepthMaskToPointCloudCudaWorkspace & workspace()
  {
    return workspace_;
  }

  //! Borrow the compact XYZ output allocation.
  //!
  //! Returns a device pointer with capacity for three floats per workspace pixel. Ownership remains
  //! with this object.
  float * output()
  {
    return output_;
  }

private:
  //! Best-effort release every allocation, including partially completed construction.
  //!
  //! CUDA cleanup errors are intentionally ignored, and all owner fields are reset before return.
  void Release()
  {
    // These contiguous void casts make deliberate suppression of teardown errors explicit.
    (void)cudaFree(output_);
    (void)cudaFree(workspace_.scan_temp_storage);
    (void)cudaFree(workspace_.selected_count);
    (void)cudaFree(workspace_.compact_offsets);
    (void)cudaFree(workspace_.valid_flags);
    (void)cudaFree(workspace_.candidates_xyz);
    output_ = nullptr;
    workspace_ = {};
  }

  DepthMaskToPointCloudCudaWorkspace workspace_;
  float * output_{nullptr};
};

TEST(DepthMaskToPointCloudCuda, BackProjectsAndCompactsInRasterOrder)
{
  constexpr std::size_t kPixelCount = 4U;

  // Pixels zero and three are valid. Pixel one is masked out, while pixel two is selected but
  // rejected for non-finite depth.
  const auto depth = Upload<float>({1.0F, 2.0F, std::numeric_limits<float>::quiet_NaN(), 4.0F});
  const auto mask = Upload<uint8_t>({1U, 0U, 1U, 1U});
  WorkspaceOwner owner(kPixelCount);

  DepthMaskToPointCloudCudaInput input;
  // View the typed test allocations as bytes; the storage tags below restore their pixel types.
  input.depth = reinterpret_cast<const uint8_t *>(depth.get());
  input.mask = reinterpret_cast<const uint8_t *>(mask.get());
  input.width = 2U;
  input.height = 2U;
  input.pixel_count = kPixelCount;
  input.depth_step = 2U * sizeof(float);
  input.mask_step = 2U * sizeof(uint8_t);
  input.depth_storage_type = DepthStorageType::kFloat32Meters;
  input.mask_storage_type = MaskStorageType::kUint8;
  input.fx = 2.0F;
  input.fy = 4.0F;
  input.cx = 0.5F;
  input.cy = 0.5F;
  input.min_depth_m = 0.5F;
  input.max_depth_m = 5.0F;

  ASSERT_EQ(
    LaunchDepthMaskToPointCloudCuda(
      input, owner.workspace(), owner.output(), nullptr),
    cudaSuccess);

  // The blocking copy on the same default stream waits for all queued conversion stages.
  uint32_t selected_count = 0U;
  ASSERT_EQ(
    cudaMemcpy(
      &selected_count, owner.workspace().selected_count, sizeof(uint32_t), cudaMemcpyDeviceToHost),
    cudaSuccess);
  ASSERT_EQ(selected_count, 2U);

  std::vector<float> points(selected_count * 3U);
  ASSERT_EQ(
    cudaMemcpy(
      points.data(), owner.output(), points.size() * sizeof(float), cudaMemcpyDeviceToHost),
    cudaSuccess);

  // Pixel zero projects to (-0.25, -0.125, 1), and pixel three projects to (1, 0.5, 4).
  const std::vector<float> expected{-0.25F, -0.125F, 1.0F, 1.0F, 0.5F, 4.0F};
  EXPECT_EQ(points, expected);
}

TEST(DepthMaskToPointCloudCuda, RejectsInvalidLaunchContracts)
{
  // The scratch query requires both a positive item count and writable host size output.
  std::size_t bytes = 0U;
  EXPECT_EQ(CalculateScanTempStorage(0U, &bytes), cudaErrorInvalidValue);
  EXPECT_EQ(CalculateScanTempStorage(1U, nullptr), cudaErrorInvalidValue);

  const auto depth = Upload<uint16_t>({1000U});
  const auto mask = Upload<uint16_t>({1U});
  WorkspaceOwner owner(1U);
  DepthMaskToPointCloudCudaInput input;
  // View the typed test allocations as bytes; the storage tags below restore their pixel types.
  input.depth = reinterpret_cast<const uint8_t *>(depth.get());
  input.mask = reinterpret_cast<const uint8_t *>(mask.get());
  input.width = 1U;
  input.height = 1U;
  input.pixel_count = 1U;
  input.depth_step = sizeof(uint16_t);
  input.mask_step = sizeof(uint16_t);
  input.depth_storage_type = DepthStorageType::kUint16Millimeters;
  input.mask_storage_type = MaskStorageType::kUint16;
  input.fx = 0.0F;
  input.fy = 1.0F;
  input.min_depth_m = 0.1F;
  input.max_depth_m = 2.0F;

  // Projection requires nonzero focal lengths.
  EXPECT_EQ(
    LaunchDepthMaskToPointCloudCuda(
      input, owner.workspace(), owner.output(), nullptr),
    cudaErrorInvalidValue);

  input.fx = 1.0F;

  // The validated pixel count must match the widened image geometry.
  input.pixel_count = 2U;
  EXPECT_EQ(
    LaunchDepthMaskToPointCloudCuda(
      input, owner.workspace(), owner.output(), nullptr),
    cudaErrorInvalidValue);
  input.pixel_count = 1U;

  // The inclusive depth interval must increase from minimum to maximum.
  input.min_depth_m = 2.0F;
  input.max_depth_m = 1.0F;
  EXPECT_EQ(
    LaunchDepthMaskToPointCloudCuda(
      input, owner.workspace(), owner.output(), nullptr),
    cudaErrorInvalidValue);
  input.min_depth_m = 0.1F;
  input.max_depth_m = 2.0F;

  // Construct an out-of-domain enum tag intentionally; the cast bypasses type-safe enumerators so
  // launch validation can be exercised without relying on an implemented storage value.
  input.depth_storage_type = static_cast<DepthStorageType>(255U);
  EXPECT_EQ(
    LaunchDepthMaskToPointCloudCuda(
      input, owner.workspace(), owner.output(), nullptr),
    cudaErrorInvalidValue);
}

}  // namespace
}  // namespace depth_image_proc
}  // namespace isaac_ros
}  // namespace nvidia
