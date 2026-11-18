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

#include "isaac_ros_depth_image_proc/depth_mask_to_point_cloud.cu.hpp"

#include <cub/device/device_scan.cuh>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace nvidia
{
namespace isaac_ros
{
namespace depth_image_proc
{
namespace
{

constexpr int kThreadsPerBlock = 256;

static_assert(
  std::numeric_limits<std::size_t>::digits >= 64,
  "Depth-mask point cloud CUDA conversion requires a 64-bit size_t");

//! Check whether a depth storage tag identifies a supported representation.
//!
//! `storage` identifies the depth pixel type and unit.
//!
//! Returns `true` for float32 meters or uint16 millimeters; otherwise returns `false`.
bool IsSupportedDepthStorage(const DepthStorageType storage)
{
  return storage == DepthStorageType::kFloat32Meters ||
         storage == DepthStorageType::kUint16Millimeters;
}

//! Check whether a mask storage tag identifies a supported representation.
//!
//! `storage` identifies the unsigned integer width of each mask pixel.
//!
//! Returns `true` for uint8 or uint16 storage; otherwise returns `false`.
bool IsSupportedMaskStorage(const MaskStorageType storage)
{
  return storage == MaskStorageType::kUint8 || storage == MaskStorageType::kUint16;
}

//! Get the number of bytes occupied by one depth pixel.
//!
//! `storage` must identify a representation accepted by `IsSupportedDepthStorage()`.
//!
//! Returns `sizeof(float)` for float32 meters or `sizeof(uint16_t)` for uint16 millimeters.
std::size_t DepthElementSize(const DepthStorageType storage)
{
  return storage == DepthStorageType::kFloat32Meters ? sizeof(float) : sizeof(uint16_t);
}

//! Get the number of bytes occupied by one mask pixel.
//!
//! `storage` must identify a representation accepted by `IsSupportedMaskStorage()`.
//!
//! Returns `sizeof(uint8_t)` or `sizeof(uint16_t)` for the selected storage.
std::size_t MaskElementSize(const MaskStorageType storage)
{
  return storage == MaskStorageType::kUint8 ? sizeof(uint8_t) : sizeof(uint16_t);
}

//! Validate all host-checkable arguments for a direct CUDA launch without dereferencing devices.
//!
//! `input` contains non-null, correctly aligned device image pointers; positive dimensions; valid
//! byte strides and storage tags; and finite intrinsics with nonzero focal lengths.
//! `input.pixel_count` is the positive widened product of the input width and height and must fit in
//! uint32. Its depth interval has a finite, nonnegative minimum and a greater maximum that may be
//! positive infinity.
//! `workspace` contains non-null caller-owned device allocations with capacity for
//! `input.pixel_count` pixels and nonzero CUB scratch storage.
//! `compact_xyz` identifies aligned caller-owned device output storage.
//!
//! Actual allocation capacities, device accessibility, and non-overlap are caller preconditions
//! that this host-only validation cannot verify.
//!
//! Returns `cudaSuccess` when every argument passes validation, otherwise
//! `cudaErrorInvalidValue`. This host function does not access device memory or enqueue work.
cudaError_t ValidateDepthMaskToPointCloudCudaArguments(
  const DepthMaskToPointCloudCudaInput & input,
  const DepthMaskToPointCloudCudaWorkspace & workspace,
  const float * compact_xyz)
{
  const std::size_t expected_pixel_count =
    static_cast<std::size_t>(input.width) * static_cast<std::size_t>(input.height);
  if (input.width == 0U || input.height == 0U ||
    input.pixel_count == 0U || input.pixel_count != expected_pixel_count ||
    input.pixel_count > std::numeric_limits<uint32_t>::max() ||
    !IsSupportedDepthStorage(input.depth_storage_type) ||
    !IsSupportedMaskStorage(input.mask_storage_type))
  {
    return cudaErrorInvalidValue;
  }

  const std::size_t depth_element_size = DepthElementSize(input.depth_storage_type);
  const std::size_t mask_element_size = MaskElementSize(input.mask_storage_type);
  // Widen width before calculating byte counts so the products use the same size_t domain as
  // allocation sizes.
  const std::size_t minimum_depth_step =
    static_cast<std::size_t>(input.width) * depth_element_size;
  const std::size_t minimum_mask_step =
    static_cast<std::size_t>(input.width) * mask_element_size;

  // Convert pointer values to uintptr_t only to verify required alignment; the integer values are
  // never dereferenced.
  if (input.depth == nullptr || input.mask == nullptr || compact_xyz == nullptr ||
    input.depth_step < minimum_depth_step || input.mask_step < minimum_mask_step ||
    input.depth_step % depth_element_size != 0U || input.mask_step % mask_element_size != 0U ||
    reinterpret_cast<std::uintptr_t>(input.depth) % depth_element_size != 0U ||
    reinterpret_cast<std::uintptr_t>(input.mask) % mask_element_size != 0U ||
    reinterpret_cast<std::uintptr_t>(compact_xyz) % alignof(float) != 0U ||
    !std::isfinite(input.fx) || !std::isfinite(input.fy) || !std::isfinite(input.cx) ||
    !std::isfinite(input.cy) || input.fx == 0.0F || input.fy == 0.0F ||
    !std::isfinite(input.min_depth_m) || std::isnan(input.max_depth_m) ||
    input.min_depth_m < 0.0F || input.max_depth_m <= input.min_depth_m ||
    workspace.candidates_xyz == nullptr || workspace.valid_flags == nullptr ||
    workspace.compact_offsets == nullptr || workspace.selected_count == nullptr ||
    workspace.scan_temp_storage == nullptr || workspace.scan_temp_storage_bytes == 0U ||
    workspace.pixel_capacity < input.pixel_count)
  {
    return cudaErrorInvalidValue;
  }

  return cudaSuccess;
}

//! Read one device-resident depth pixel and normalize it to meters.
//!
//! `row` points to the first byte of an aligned depth image row in device memory and must remain
//! readable for this device call.
//! `column` selects an in-bounds pixel within that row.
//! `storage` states whether the row contains float32 meters or uint16 millimeters and is assumed
//! valid.
//!
//! Returns the selected depth value as float32 meters. This is a device-only helper.
__device__ float GetDepthInMeters(
  const uint8_t * row, uint32_t column, DepthStorageType storage)
{
  if (storage == DepthStorageType::kFloat32Meters) {
    // Launch validation aligns the base pointer and makes each row stride element-aligned.
    // Reinterpret the row as float pixels because this representation already stores meters.
    return reinterpret_cast<const float *>(row)[column];
  }

  // The same validated alignment permits uint16 indexing. Convert the stored millimeters
  // numerically to meters after loading the requested column.
  const uint16_t depth_millimeters =
    reinterpret_cast<const uint16_t *>(row)[column];
  return static_cast<float>(depth_millimeters) * 0.001F;
}

//! Read one device-resident mask pixel as a Boolean selection.
//!
//! `row` points to the first byte of an aligned mask image row in device memory and must remain
//! readable for this device call.
//! `column` selects an in-bounds pixel within that row.
//! `storage` states whether the row contains uint8 or uint16 pixels and is assumed valid.
//!
//! Returns `true` for any nonzero value and `false` for zero. This is a device-only helper.
__device__ bool ReadMask(const uint8_t * row, uint32_t column, MaskStorageType storage)
{
  if (storage == MaskStorageType::kUint8) {
    return row[column] != 0U;
  }

  // Validated base alignment and element-aligned row strides make uint16 indexing safe and advance
  // by two bytes per column.
  return reinterpret_cast<const uint16_t *>(row)[column] != 0U;
}

//! Back-project and classify one input pixel per CUDA thread.
//!
//! `input` contains validated device image pointers whose allocations satisfy the documented
//! capacities, plus layouts, storage tags, and finite pinhole intrinsics.
//! `input` includes the validated pixel count and inclusive depth interval.
//! `candidates_xyz` points to device output storage for three floats per input pixel in raster
//! order. Slots for invalid pixels remain unspecified.
//! `valid_flags` points to device output storage for one uint32 value per input pixel; each thread
//! writes 1 for a valid selected point or 0 otherwise. It must not overlap `candidates_xyz`.
//! Neither output allocation may overlap either input image.
//!
//! The grid may contain more threads than pixels; out-of-range threads perform no memory access.
__global__ void BackProjectAndClassifyKernel(
  DepthMaskToPointCloudCudaInput input,
  float * candidates_xyz,
  uint32_t * valid_flags)
{
  // Convert this thread's block-local index into one global, row-major pixel index. blockIdx.x
  // selects the block, blockDim.x is its thread count, and threadIdx.x selects the thread within
  // that block. The validated launch dimensions keep this 32-bit calculation in range; size_t is
  // used for subsequent pointer indexing.
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;

  // The launch rounds the thread count up to a whole number of blocks. Threads beyond the final
  // image pixel must exit before accessing memory. The host supplies the already validated widened
  // pixel count so the kernel does not repeat the image-size multiplication.
  if (index >= input.pixel_count) {
    return;
  }

  // Recover the two-dimensional image coordinate from the flattened index. Integer division selects
  // the row and the remainder selects the column. Narrowing to uint32 is safe because the bounds
  // check above guarantees both coordinates fit within the uint32 image dimensions.
  const uint32_t row_index = static_cast<uint32_t>(index / input.width);
  const uint32_t column = static_cast<uint32_t>(index % input.width);

  // Locate the first byte of each row using its own byte stride rather than assuming tightly packed
  // storage. This preserves independent depth and mask padding. Widen row_index before multiplying
  // by a byte stride.
  const auto * depth_row = input.depth +
    static_cast<std::size_t>(row_index) * input.depth_step;
  const auto * mask_row = input.mask +
    static_cast<std::size_t>(row_index) * input.mask_step;

  // Normalize the depth to meters. Reject a candidate when the mask does not select it, the depth
  // is non-finite, or it lies outside the inclusive interval. Every rejected pixel writes zero for
  // the later scan.
  const float z = GetDepthInMeters(depth_row, column, input.depth_storage_type);
  if (!ReadMask(mask_row, column, input.mask_storage_type) ||
    !isfinite(z) || z < input.min_depth_m ||
    z > input.max_depth_m)
  {
    valid_flags[index] = 0U;
    return;
  }

  // Apply the pinhole-camera back-projection equations. Pixel coordinates use (column, row_index),
  // (cx, cy) is the principal point, and (fx, fy) contains focal lengths in pixels. Converting
  // large indices to the intrinsic float domain may round them, which is intentional for this
  // float32 output path.
  const float x = (static_cast<float>(column) - input.cx) * z / input.fx;
  const float y = (static_cast<float>(row_index) - input.cy) * z / input.fy;

  // Finite inputs can still overflow during projection. Reject such points so the final PointCloud2
  // remains dense and contains no infinities.
  if (!isfinite(x) || !isfinite(y)) {
    valid_flags[index] = 0U;
    return;
  }

  // Keep the candidate at its original raster index. A later exclusive scan and scatter use
  // valid_flags to remove gaps without changing point order.
  valid_flags[index] = 1U;
  candidates_xyz[index * 3U] = x;
  candidates_xyz[index * 3U + 1U] = y;
  candidates_xyz[index * 3U + 2U] = z;
}

//! Copy valid XYZ candidates into consecutive output positions with one CUDA thread per pixel.
//!
//! `candidates_xyz` points to three device-resident floats per input pixel.
//! `valid_flags` points to one device-resident 0-or-1 value per input pixel.
//! `compact_offsets` points to the exclusive prefix sum of `valid_flags`; each valid pixel uses its
//! offset as its output index.
//! `pixel_count` is the number of entries in `valid_flags` and `compact_offsets`.
//! `compact_xyz` points to device output storage large enough for three floats per valid pixel.
//! All four allocations must be device-accessible, sufficiently sized, and mutually
//! non-overlapping. Threads for invalid or out-of-range pixels perform no output write.
__global__ void ScatterCompactedKernel(
  const float * candidates_xyz,
  const uint32_t * valid_flags,
  const uint32_t * compact_offsets,
  std::size_t pixel_count,
  float * compact_xyz)
{
  // The validated launch dimensions keep this 32-bit calculation in range; size_t is used for
  // subsequent pointer indexing.
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= pixel_count || valid_flags[index] == 0U) {
    return;
  }

  const std::size_t output_index = compact_offsets[index];
  compact_xyz[output_index * 3U] = candidates_xyz[index * 3U];
  compact_xyz[output_index * 3U + 1U] = candidates_xyz[index * 3U + 1U];
  compact_xyz[output_index * 3U + 2U] = candidates_xyz[index * 3U + 2U];
}

//! Compute the number of selected output points with one CUDA thread.
//!
//! `valid_flags` points to one device-resident 0-or-1 value per input pixel.
//! `compact_offsets` points to the exclusive prefix sum of `valid_flags`.
//! `pixel_count` is the number of entries in both input arrays and must be positive.
//! `selected_count` points to a one-element device-resident uint32 allocation receiving the result
//! and must not overlap either input array.
//!
//! Thread (0, 0) adds the final exclusive offset to the final 0-or-1 flag, yielding the total.
//! Additional threads perform no write.
__global__ void SetSelectedCountKernel(
  const uint32_t * valid_flags,
  const uint32_t * compact_offsets,
  std::size_t pixel_count,
  uint32_t * selected_count)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *selected_count = compact_offsets[pixel_count - 1U] + valid_flags[pixel_count - 1U];
  }
}

}  // namespace

cudaError_t CalculateScanTempStorage(
  const std::size_t pixel_count, std::size_t * bytes)
{
  if (pixel_count == 0U || bytes == nullptr ||
    pixel_count > std::numeric_limits<uint32_t>::max())
  {
    return cudaErrorInvalidValue;
  }

  std::size_t required = 0U;
  // CUB requires typed input and output pointers to select the uint32 scan specialization. Typed
  // null pointers request only the scratch size without providing real arrays or launching a scan.
  const cudaError_t status = cub::DeviceScan::ExclusiveSum(
    nullptr,
    required,
    static_cast<const uint32_t *>(nullptr),
    static_cast<uint32_t *>(nullptr),
    pixel_count);
  if (status == cudaSuccess) {
    *bytes = required;
  }
  return status;
}

cudaError_t LaunchDepthMaskToPointCloudCuda(
  const DepthMaskToPointCloudCudaInput & input,
  const DepthMaskToPointCloudCudaWorkspace & workspace,
  float * compact_xyz,
  const cudaStream_t & stream)
{
  const cudaError_t validation_status = ValidateDepthMaskToPointCloudCudaArguments(
    input, workspace, compact_xyz);
  if (validation_status != cudaSuccess) {
    return validation_status;
  }

  // Convert before division so std::ceil sees the fractional quotient instead of an
  // integer-truncated value. The uint32 pixel limit guarantees double represents the count exactly.
  const double rounded_block_count =
    std::ceil(static_cast<double>(input.pixel_count) / kThreadsPerBlock);
  // The validated pixel limit guarantees the rounded block count fits in int.
  const int blocks = static_cast<int>(rounded_block_count);

  // Stage 1: back-project every pixel and classify it for stable compaction.
  BackProjectAndClassifyKernel<<<
    blocks,  // grid_size
    kThreadsPerBlock,  // block_size
    0,  // dynamic_shared_memory_bytes
    stream>>>(
    input, workspace.candidates_xyz, workspace.valid_flags);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }

  // Stage 2: ExclusiveSum converts each flag into the number of valid preceding pixels.
  // For a valid pixel, that count is its zero-based compact_xyz position, preserving raster order.
  // Example: flags [0, 1, 0, 1, 1] become offsets [0, 0, 1, 1, 2], so valid pixels 1, 3, and 4
  // write to compact positions 0, 1, and 2.
  // CUB accepts the scratch capacity by non-const reference, so preserve the const descriptor.
  std::size_t temp_bytes = workspace.scan_temp_storage_bytes;
  status = cub::DeviceScan::ExclusiveSum(
    workspace.scan_temp_storage,
    temp_bytes,
    workspace.valid_flags,
    workspace.compact_offsets,
    input.pixel_count,
    stream);
  if (status != cudaSuccess) {
    return status;
  }

  // Stage 3: scatter valid raster-order candidates into consecutive output slots.
  ScatterCompactedKernel << < blocks, kThreadsPerBlock, 0, stream >> > (workspace.candidates_xyz,
    workspace.valid_flags,
    workspace.compact_offsets,
    input.pixel_count,
    compact_xyz);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }

  // Stage 4: derive the selected count from the final exclusive offset and flag.
  SetSelectedCountKernel << < 1, 1, 0, stream >> > (
    workspace.valid_flags, workspace.compact_offsets, input.pixel_count, workspace.selected_count);
  return cudaGetLastError();
}

}  // namespace depth_image_proc
}  // namespace isaac_ros
}  // namespace nvidia
