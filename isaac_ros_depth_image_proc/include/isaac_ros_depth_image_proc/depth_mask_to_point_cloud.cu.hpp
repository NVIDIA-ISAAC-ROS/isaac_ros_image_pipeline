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

#ifndef ISAAC_ROS_DEPTH_IMAGE_PROC__DEPTH_MASK_TO_POINT_CLOUD_CU_HPP_
#define ISAAC_ROS_DEPTH_IMAGE_PROC__DEPTH_MASK_TO_POINT_CLOUD_CU_HPP_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace nvidia
{
namespace isaac_ros
{
namespace depth_image_proc
{

// Enum selecting how depth pixels are stored.
enum class DepthStorageType : uint8_t
{
  // One 32-bit floating-point value in meters per pixel.
  kFloat32Meters,

  // One unsigned 16-bit value in millimeters per pixel.
  kUint16Millimeters,
};

// Enum selecting how mask pixels are stored.
enum class MaskStorageType : uint8_t
{
  // One unsigned 8-bit value per pixel.
  kUint8,

  // One unsigned 16-bit value per pixel.
  kUint16,
};

// Describes CUDA-resident depth and mask image data, layouts, and projection parameters. This
// struct does not own the allocations; callers must keep them valid until all enqueued work has
// completed.
struct DepthMaskToPointCloudCudaInput
{
  // Positive image dimensions in pixels.
  uint32_t width{0U};
  uint32_t height{0U};

  // Validated widened product of width and height.
  std::size_t pixel_count{0U};

  // Byte offsets between consecutive image rows.
  uint32_t depth_step{0U};
  uint32_t mask_step{0U};

  // Runtime pixel representations.
  DepthStorageType depth_storage_type{DepthStorageType::kFloat32Meters};
  MaskStorageType mask_storage_type{MaskStorageType::kUint8};

  // Read-only, non-null device byte pointers.
  const uint8_t * depth{nullptr};
  const uint8_t * mask{nullptr};

  // Pinhole camera intrinsics in pixels. fx and fy are nonzero focal lengths; cx and cy locate the
  // principal point.
  float fx{0.0F};
  float fy{0.0F};
  float cx{0.0F};
  float cy{0.0F};

  // Inclusive depth interval in meters. The minimum is finite and nonnegative. The maximum is
  // greater than the minimum and may be positive infinity.
  float min_depth_m{0.0F};
  float max_depth_m{std::numeric_limits<float>::infinity()};
};

// Describes caller-owned, reusable CUDA workspace allocations. This struct does not allocate or
// free memory; every pointer below must be non-null device-accessible memory with capacity for the
// documented number of elements. Workspace allocations, input images, and compact output must not
// overlap. The workspace cannot serve concurrent conversions and must remain valid until all
// conversion work has completed.
//
// The conversion first visits matching depth and mask pixels in row-major order. A pixel is valid
// when its mask is nonzero, its depth is finite and within range, and its back-projected XYZ
// coordinates are finite. Valid XYZ candidates are then packed into the output without changing
// their order.
//
// For example, assume every depth value below is valid:
//
//   mask:             depth (meters):
//   0  1  0  0        1.0  1.0  1.0  1.0
//   1  0  1  0        1.0  1.0  1.0  1.0
//   0  1  1  0        1.0  1.0  1.0  1.0
//   0  0  0  1        1.0  1.0  1.0  1.0
//
// Flattening the images in row-major order produces:
//
//   valid_flags:
//     0  1  0  0 | 1  0  1  0 | 0  1  1  0 | 0  0  0  1
//   compact_offsets:
//     0  0  1  1 | 1  2  2  3 | 3  3  4  5 | 5  5  5  5
//
// Input pixels 1, 4, 6, 9, 10, and 15 therefore write to compact output positions 0 through 5, and
// selected_count becomes 6. candidates_xyz has space for all 16 input pixels, while compact_xyz
// receives only those six selected points. scan_temp_storage is scratch memory used to calculate
// the offsets, and pixel_capacity must be at least 16.
struct DepthMaskToPointCloudCudaWorkspace
{
  // Intermediate XYZ coordinates in raster order. Allocate three floats per input pixel; pixel i
  // stores x, y, and z at indices 3*i, 3*i+1, and 3*i+2. Entries for invalid pixels are ignored
  // according to valid_flags.
  float * candidates_xyz{nullptr};

  // One 0-or-1 validity value for each input pixel.
  uint32_t * valid_flags{nullptr};

  // Exclusive prefix sums of valid_flags. compact_offsets[i] is the number of valid pixels
  // preceding input pixel i and, for a valid pixel, its output position.
  uint32_t * compact_offsets{nullptr};

  // One device-side value receiving the number of compacted output points. Its incoming value is
  // ignored, and the result becomes readable after the launch stream completes.
  uint32_t * selected_count{nullptr};

  // Scratch allocation used by CUB's exclusive scan and its allocated size. The size must be at
  // least the value queried for the actual pixel count with
  // CalculateScanTempStorage().
  void * scan_temp_storage{nullptr};
  std::size_t scan_temp_storage_bytes{0U};

  // Maximum number of input pixels covered by the per-pixel allocations.
  std::size_t pixel_capacity{0U};
};

//! Query the temporary storage required by the CUB exclusive scan.
//!
//! `pixel_count` is the positive number of input pixels to scan and must fit in uint32.
//! `bytes` points to host storage that receives the required allocation size in bytes and must not
//! be null. It is written only when the query succeeds.
//!
//! Returns `cudaSuccess` when the size was written, or a CUDA error code when an argument or CUB
//! query is invalid. The query does not enqueue device work.
cudaError_t CalculateScanTempStorage(
  const std::size_t pixel_count, std::size_t * bytes);

//! Compact valid depth pixels into device-resident XYZ triples in raster order.
//!
//! `input` contains borrowed device image pointers, layouts, pixel storage tags, validated pixel
//! count, pinhole camera intrinsics, and the inclusive depth interval. Its image allocations must
//! remain readable until `stream` completes.
//! `workspace` contains caller-owned device allocations sized for at least `input.width *
//! input.height` pixels and CUB scratch storage sized by
//! CalculateScanTempStorage(). The conversion overwrites its per-pixel arrays
//! and `selected_count`.
//! `compact_xyz` points to caller-owned device output storage for three floats per input pixel. On
//! completion, its first `3 * selected_count` floats contain packed XYZ triples in meters and
//! raster order.
//! `stream` identifies the CUDA stream on which all work is enqueued; null selects the default
//! stream.
//!
//! Returns `cudaSuccess` when all work was successfully enqueued, or a CUDA error code when
//! validation or an enqueue operation fails. Earlier stages may already be queued when a later
//! enqueue fails, so workspace and output contents may be partially modified. The call does not
//! synchronize; execution errors can surface later, and all allocations must remain valid until
//! `stream` finishes.
cudaError_t LaunchDepthMaskToPointCloudCuda(
  const DepthMaskToPointCloudCudaInput & input,
  const DepthMaskToPointCloudCudaWorkspace & workspace,
  float * compact_xyz,
  const cudaStream_t & stream);

}  // namespace depth_image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_DEPTH_IMAGE_PROC__DEPTH_MASK_TO_POINT_CLOUD_CU_HPP_
