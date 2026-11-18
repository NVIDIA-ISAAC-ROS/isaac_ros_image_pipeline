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

#include "isaac_ros_depth_image_proc/depth_mask_to_point_cloud.hpp"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "isaac_ros_depth_image_proc/depth_mask_to_point_cloud.cu.hpp"
#include "pointcloud_conversions/pointcloud_conversions.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace depth_image_proc
{
namespace
{

constexpr std::size_t kCoordinatesPerPoint = 3U;
constexpr std::size_t kPackedPointBytes = kCoordinatesPerPoint * sizeof(float);

//! Convert a CUDA status into the builder's exception-based error contract.
//!
//! `status` is the result of the CUDA operation being checked.
//! `operation` identifies that operation in any generated error message.
//!
//! Returns normally for `cudaSuccess`; otherwise throws `std::runtime_error` containing the CUDA
//! error text. This function performs no synchronization.
void CheckCuda(cudaError_t status, const std::string & operation)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(operation + " failed: " + cudaGetErrorString(status));
  }
}

//! Get the byte width of a supported ROS depth encoding.
//!
//! `encoding` is compared exactly against the depth encodings accepted by the CUDA path.
//!
//! Returns `sizeof(float)` for `32FC1`, `sizeof(uint16_t)` for `16UC1`, or zero for every other
//! encoding.
std::size_t DepthBytesPerPixel(const std::string & encoding)
{
  if (encoding == "32FC1") {
    return sizeof(float);
  }
  if (encoding == "16UC1") {
    return sizeof(uint16_t);
  }
  return 0U;
}

//! Get the byte width of a supported single-channel ROS mask encoding.
//!
//! `encoding` is compared exactly against the mask encodings accepted by the CUDA path.
//!
//! Returns `sizeof(uint8_t)` for `mono8` and `8UC1`, `sizeof(uint16_t)` for `mono16` and `16UC1`,
//! or zero for every other encoding.
std::size_t MaskBytesPerPixel(const std::string & encoding)
{
  if (encoding == "mono8" || encoding == "8UC1") {
    return sizeof(uint8_t);
  }
  if (encoding == "mono16" || encoding == "16UC1") {
    return sizeof(uint16_t);
  }
  return 0U;
}

//! Validate an image's buffer backend and byte order.
//!
//! `image` supplies the buffer backend name and byte order; its payload is not accessed.
//! `input_name` identifies the image in any generated error message.
//!
//! Returns normally for little-endian CPU or CUDA storage. Throws `std::invalid_argument` for every
//! other backend or for a big-endian image.
void ValidateInputStorage(
  const sensor_msgs::msg::Image & image, const std::string & input_name)
{
  const std::string backend = image.data.get_backend_type();
  if (backend != "cpu" && backend != "cuda") {
    throw std::invalid_argument(
      input_name + " image uses unsupported buffer backend '" + backend +
      "'; expected 'cpu' or 'cuda'");
  }
  if (image.is_bigendian) {
    throw std::invalid_argument("Big-endian depth and mask images are unsupported");
  }
}

//! Validate an image payload against the bytes addressable by its declared rows.
//!
//! `image` supplies positive dimensions, a byte row step, and its backing buffer.
//! `bytes_per_pixel` is the nonzero element width selected from `image.encoding`.
//! `input_name` identifies the image in any generated error message.
//!
//! Returns normally after checking row capacity, element alignment, and backing-buffer size. Throws
//! `std::invalid_argument` for invalid metadata or a truncated payload.
void ValidateImagePayload(
  const sensor_msgs::msg::Image & image,
  std::size_t bytes_per_pixel,
  const std::string & input_name)
{
  const std::size_t minimum_step = image.width * bytes_per_pixel;
  if (image.step < minimum_step || image.step % bytes_per_pixel != 0U) {
    throw std::invalid_argument(input_name + " image row step is too small or misaligned");
  }

  // Widen the row step before multiplication so uint32 arithmetic cannot wrap.
  const std::size_t required_bytes =
    static_cast<std::size_t>(image.step) * image.height;
  if (image.data.size() < required_bytes) {
    throw std::invalid_argument(input_name + " image payload is truncated");
  }
}

//! Calculate a nonzero image pixel count that fits all downstream metadata and allocations.
//!
//! `width` and `height` are the common depth and mask dimensions in pixels.
//!
//! Returns their product as `std::size_t`. Throws `std::invalid_argument` when either dimension is
//! zero and `std::overflow_error` when the packed PointCloud2 row size cannot fit in uint32.
std::size_t CheckedPixelCount(uint32_t width, uint32_t height)
{
  // Widen width before multiplication so uint32 arithmetic cannot wrap.
  const std::size_t pixel_count = static_cast<std::size_t>(width) * height;
  if (pixel_count == 0U) {
    throw std::invalid_argument("Depth image dimensions must be nonzero");
  }
  if (pixel_count > std::numeric_limits<uint32_t>::max() / kPackedPointBytes) {
    throw std::overflow_error("Depth image is too large for packed PointCloud2 row metadata");
  }
  return pixel_count;
}

// Groups float-safe pinhole intrinsics, all expressed in pixels, for the CUDA launch.
struct CameraIntrinsics
{
  float fx;
  float fy;
  float cx;
  float cy;
};

// Collects all metadata needed after one successful validation pass.
struct ValidatedBuildInputs
{
  CameraIntrinsics intrinsics;
  DepthStorageType depth_storage_type;
  MaskStorageType mask_storage_type;
  std::size_t pixel_count;
};

//! Best-effort release a typed CUDA allocation and clear its non-owning workspace field.
//!
//! `pointer` is an owning host variable containing a device pointer returned by CUDA allocation.
//! Null is accepted. The referenced variable is always reset to null.
//!
//! CUDA cleanup errors are intentionally discarded because this helper is used by noexcept
//! teardown paths.
template<typename T>
void FreeDevice(T *& pointer)
{
  if (pointer != nullptr) {
    // Explicitly discard cudaFree's status because teardown cannot propagate cleanup failures.
    (void)cudaFree(pointer);
    pointer = nullptr;
  }
}

}  // namespace

class DepthMaskToPointCloudBuilder::DepthMaskToPointCloudImpl
{
public:
  //! Validate the immutable depth interval and allocate resources shared by all builds.
  //!
  //! `min_depth_m` and `max_depth_m` supply the inclusive depth interval retained for the lifetime
  //! of this implementation. The maximum may be positive infinity.
  //!
  //! Throws `std::invalid_argument` for an invalid interval and propagates CUDA initialization
  //! failures. A partially initialized instance releases every resource before rethrowing.
  DepthMaskToPointCloudImpl(float min_depth_m, float max_depth_m)
  : min_depth_m_(min_depth_m), max_depth_m_(max_depth_m)
  {
    if (!std::isfinite(min_depth_m_) || std::isnan(max_depth_m_) || min_depth_m_ < 0.0F ||
      max_depth_m_ <= min_depth_m_)
    {
      throw std::invalid_argument("Depth-mask point-cloud range is invalid");
    }

    // Constructor failures require explicit cleanup because the implementation is not yet fully
    // constructed.
    try {
      InitializeCuda();
    } catch (...) {
      ReleaseCuda();
      throw;
    }
  }

  //! Wait for queued CUDA work and best-effort release all owned resources.
  //!
  //! Teardown may block while the stream finishes and suppresses CUDA cleanup errors.
  ~DepthMaskToPointCloudImpl()
  {
    ReleaseCuda();
  }

  //! Execute one serialized conversion using the reusable stream and workspace.
  //!
  //! `depth` is the candidate depth image in float32 meters or uint16 millimeters.
  //! `camera_info` supplies the pinhole intrinsics and optional expected image dimensions.
  //! `mask` supplies uint8 or uint16 selections with dimensions matching `depth`.
  //!
  //! Returns sole ownership of a CUDA-backed packed XYZ cloud, or null when no pixel passes
  //! filtering. Throws on invalid input, unrepresentable sizes, allocation failure, or CUDA
  //! failure.
  //!
  //! The method synchronizes the shared stream to read the selected count. The final output copy
  //! is asynchronous, with completion recorded through the buffer backend. Its mutex serializes
  //! calls because temporary uploads, workspace allocations, and the stream are reused.
  std::unique_ptr<sensor_msgs::msg::PointCloud2> Build(
    const sensor_msgs::msg::Image & depth,
    const sensor_msgs::msg::CameraInfo & camera_info,
    const sensor_msgs::msg::Image & mask)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    // Validate metadata and calculate all required capacities before touching input storage.
    const ValidatedBuildInputs validated = ValidateInputs(depth, camera_info, mask);
    EnsureWorkspace(validated.pixel_count);

    // Input handles upload CPU-backed buffers to promoted CUDA storage and borrow CUDA-backed
    // buffers directly. They order this builder-owned consumer stream after each buffer's producer
    // event; a producer's non-owning stream pointer is not safe to reuse across node or process
    // boundaries. The handles keep any promoted storage alive through the conversion.
    auto depth_handle = cuda_buffer_backend::from_input_buffer(depth.data, stream_);
    auto mask_handle = cuda_buffer_backend::from_input_buffer(mask.data, stream_);
    DepthMaskToPointCloudCudaInput input;
    input.depth = depth_handle.get_ptr();
    input.mask = mask_handle.get_ptr();
    input.width = depth.width;
    input.height = depth.height;
    input.pixel_count = validated.pixel_count;
    input.depth_step = depth.step;
    input.mask_step = mask.step;
    input.depth_storage_type = validated.depth_storage_type;
    input.mask_storage_type = validated.mask_storage_type;
    input.fx = validated.intrinsics.fx;
    input.fy = validated.intrinsics.fy;
    input.cx = validated.intrinsics.cx;
    input.cy = validated.intrinsics.cy;
    input.min_depth_m = min_depth_m_;
    input.max_depth_m = max_depth_m_;

    // Enqueue classification and compaction, then retrieve the count needed for message allocation.
    CheckCuda(LaunchDepthMaskToPointCloudCuda(
        input, workspace_, compact_xyz_, stream_),
      "launch depth-mask point-cloud conversion");
    CheckCuda(cudaMemcpyAsync(selected_count_host_,
                selected_count_device_,
                sizeof(uint32_t),
                cudaMemcpyDeviceToHost,
                stream_),
      "copy selected point count");

    // The count is needed on the host to size the compact PointCloud2 allocation.
    CheckCuda(cudaStreamSynchronize(stream_), "synchronize selected point count");

    const uint32_t selected_count = *selected_count_host_;
    if (selected_count == 0U) {
      return nullptr;
    }

    auto cloud = pointcloud_conversions::allocate_point_cloud_msg(selected_count, 1U, false);
    cloud->header = depth.header;

    // Filtering removed every non-finite XYZ candidate, so all published points are dense.
    cloud->is_dense = true;
    if (cloud->point_step != kPackedPointBytes) {
      throw std::logic_error("Point-cloud conversion did not allocate packed XYZ");
    }

    // The validated pixel-count limit guarantees every smaller selected cloud fits in row metadata.
    const std::size_t output_bytes = selected_count * kPackedPointBytes;

    // Record output readiness through the buffer backend on the stream that performs the copy.
    {
      auto output_handle = pointcloud_conversions::from_output_point_cloud_msg(*cloud, stream_);
      cuda_buffer_backend::to_buffer(
        compact_xyz_, output_bytes, output_handle, stream_, cudaMemcpyDeviceToDevice);
    }
    return cloud;
  }

private:
  //! Allocate the nonblocking stream and pinned host count used by every build.
  //!
  //! On success, `stream_` owns a CUDA stream and `selected_count_host_` owns one portable pinned
  //! uint32. Throws `std::runtime_error` when either CUDA operation fails. The caller is
  //! responsible for cleaning up any resource allocated before a later failure.
  void InitializeCuda()
  {
    CheckCuda(
      cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");

    // cudaHostAlloc writes through void**; reinterpret the typed owner address so CUDA can store a
    // suitably aligned allocation that is subsequently accessed only as uint32_t.
    CheckCuda(
      cudaHostAlloc(
        reinterpret_cast<void **>(&selected_count_host_), sizeof(uint32_t), cudaHostAllocPortable),
      "cudaHostAlloc(selected_count)");
  }

  //! Best-effort release all shared CUDA resources in reverse ownership order.
  //!
  //! The method first waits for `stream_`, then releases workspaces, pinned host storage, and the
  //! stream. It is noexcept and intentionally suppresses CUDA cleanup errors. Every owner field is
  //! reset even when CUDA reports a failure.
  void ReleaseCuda() noexcept
  {
    if (stream_ != nullptr) {
      // Teardown cannot propagate synchronization failures, but it still attempts to quiesce work.
      (void)cudaStreamSynchronize(stream_);
    }
    FreeWorkspace();
    if (selected_count_host_ != nullptr) {
      // Pinned-host cleanup is best effort in this noexcept teardown path.
      (void)cudaFreeHost(selected_count_host_);
      selected_count_host_ = nullptr;
    }
    if (stream_ != nullptr) {
      // Stream destruction errors cannot be reported from teardown.
      (void)cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
  }

  //! Validate all host-side input contracts and derive CUDA launch metadata.
  //!
  //! `depth` supplies its backend, dimensions, layout, payload, and depth encoding.
  //! `camera_info` supplies optional matching dimensions and the pinhole K matrix.
  //! `mask` supplies its backend, dimensions, layout, payload, and mask encoding.
  //!
  //! Returns storage tags, pixel count, and float-safe intrinsics.
  //! Throws `std::invalid_argument` for unsupported or inconsistent inputs and
  //! `std::overflow_error` for unrepresentable sizes. The payload bytes are not accessed.
  ValidatedBuildInputs ValidateInputs(
    const sensor_msgs::msg::Image & depth,
    const sensor_msgs::msg::CameraInfo & camera_info,
    const sensor_msgs::msg::Image & mask) const
  {
    ValidatedBuildInputs validated{};
    ValidateInputStorage(depth, "Depth");
    ValidateInputStorage(mask, "Mask");

    const std::size_t depth_bytes_per_pixel = DepthBytesPerPixel(depth.encoding);
    const std::size_t mask_bytes_per_pixel = MaskBytesPerPixel(mask.encoding);
    if (depth_bytes_per_pixel == 0U || mask_bytes_per_pixel == 0U) {
      throw std::invalid_argument("Depth or mask encoding is unsupported");
    }

    if (depth.width != mask.width || depth.height != mask.height) {
      throw std::invalid_argument("Depth and mask dimensions do not match");
    }

    validated.pixel_count = CheckedPixelCount(depth.width, depth.height);
    ValidateImagePayload(depth, depth_bytes_per_pixel, "Depth");
    ValidateImagePayload(mask, mask_bytes_per_pixel, "Mask");

    if (camera_info.width != 0U && camera_info.width != depth.width) {
      throw std::invalid_argument("CameraInfo width does not match depth");
    }
    if (camera_info.height != 0U && camera_info.height != depth.height) {
      throw std::invalid_argument("CameraInfo height does not match depth");
    }

    const double fx = camera_info.k[0];
    const double fy = camera_info.k[4];
    const double cx = camera_info.k[2];
    const double cy = camera_info.k[5];
    const double min_float = static_cast<double>(std::numeric_limits<float>::lowest());
    const double max_float = static_cast<double>(std::numeric_limits<float>::max());
    const auto is_float_representable = [min_float, max_float](double value) {
        return std::isfinite(value) && value >= min_float && value <= max_float;
      };
    if (!is_float_representable(fx) || !is_float_representable(fy) ||
      !is_float_representable(cx) || !is_float_representable(cy))
    {
      throw std::invalid_argument("CameraInfo projection values must be finite floats");
    }

    validated.intrinsics = CameraIntrinsics{
      static_cast<float>(fx),
      static_cast<float>(fy),
      static_cast<float>(cx),
      static_cast<float>(cy)};
    if (validated.intrinsics.fx == 0.0F || validated.intrinsics.fy == 0.0F) {
      throw std::invalid_argument("CameraInfo fx and fy must be nonzero floats");
    }

    validated.depth_storage_type = depth.encoding == "32FC1" ?
      DepthStorageType::kFloat32Meters : DepthStorageType::kUint16Millimeters;
    validated.mask_storage_type = mask_bytes_per_pixel == sizeof(uint8_t) ?
      MaskStorageType::kUint8 : MaskStorageType::kUint16;
    return validated;
  }

  //! Ensure every reusable device allocation can hold one conversion request.
  //!
  //! `pixel_count` is the positive validated number of input pixels and determines every allocation
  //! capacity.
  //!
  //! Returns without CUDA work when the current capacity is sufficient. Otherwise discards the old
  //! workspace and allocates all arrays for `pixel_count`. Throws on a CUB query or allocation
  //! failure. Query failures retain the old workspace; allocation failures leave it empty rather
  //! than partially initialized.
  void EnsureWorkspace(std::size_t pixel_count)
  {
    if (pixel_count <= workspace_.pixel_capacity) {
      return;
    }

    // ValidateInputs already bounded the packed XYZ size; the scalar arrays are smaller.
    const std::size_t candidate_bytes = pixel_count * kPackedPointBytes;
    const std::size_t scalar_bytes = pixel_count * sizeof(uint32_t);
    std::size_t scan_bytes = 0U;
    CheckCuda(
      CalculateScanTempStorage(pixel_count, &scan_bytes),
      "query depth-mask scan workspace");
    if (scan_bytes == 0U) {
      throw std::runtime_error("CUDA scan workspace query returned zero bytes");
    }

    FreeWorkspace();

    const char * allocation_operation = "allocate point candidates";
    cudaError_t allocation_status = cudaMalloc(
      reinterpret_cast<void **>(&workspace_.candidates_xyz), candidate_bytes);
    if (allocation_status == cudaSuccess) {
      allocation_operation = "allocate point validity flags";
      allocation_status =
        cudaMalloc(reinterpret_cast<void **>(&workspace_.valid_flags), scalar_bytes);
    }
    if (allocation_status == cudaSuccess) {
      allocation_operation = "allocate point compact offsets";
      allocation_status =
        cudaMalloc(reinterpret_cast<void **>(&workspace_.compact_offsets), scalar_bytes);
    }
    if (allocation_status == cudaSuccess) {
      allocation_operation = "allocate selected point count";
      allocation_status =
        cudaMalloc(reinterpret_cast<void **>(&selected_count_device_), sizeof(uint32_t));
    }
    if (allocation_status == cudaSuccess) {
      allocation_operation = "allocate compact XYZ workspace";
      allocation_status = cudaMalloc(reinterpret_cast<void **>(&compact_xyz_), candidate_bytes);
    }
    if (allocation_status == cudaSuccess) {
      allocation_operation = "allocate scan workspace";
      allocation_status = cudaMalloc(&workspace_.scan_temp_storage, scan_bytes);
    }
    if (allocation_status != cudaSuccess) {
      FreeWorkspace();
      CheckCuda(allocation_status, allocation_operation);
    }

    workspace_.selected_count = selected_count_device_;
    workspace_.scan_temp_storage_bytes = scan_bytes;
    workspace_.pixel_capacity = pixel_count;
  }

  //! Best-effort release all per-image device allocations.
  //!
  //! The method accepts a complete or partially initialized workspace, suppresses CUDA cleanup
  //! errors, and resets all pointers, sizes, and capacity to their empty values.
  void FreeWorkspace() noexcept
  {
    FreeDevice(workspace_.candidates_xyz);
    FreeDevice(workspace_.valid_flags);
    FreeDevice(workspace_.compact_offsets);
    FreeDevice(workspace_.scan_temp_storage);
    FreeDevice(selected_count_device_);
    FreeDevice(compact_xyz_);
    workspace_ = {};
  }

  // Immutable inclusive depth interval copied from the public constructor.
  float min_depth_m_;
  float max_depth_m_;

  // Serializes all access to the reusable stream, uploads, and workspace allocations.
  std::mutex mutex_;

  // Nonblocking stream shared by uploads, kernels, scans, and output copies.
  cudaStream_t stream_{nullptr};

  // Reusable device-side compaction arrays and CUB scratch allocation.
  DepthMaskToPointCloudCudaWorkspace workspace_;

  // Device count written by CUDA and its one-element pinned-host readback destination.
  uint32_t * selected_count_device_{nullptr};
  uint32_t * selected_count_host_{nullptr};

  // Reusable device output with capacity for three floats per workspace pixel.
  float * compact_xyz_{nullptr};
};

DepthMaskToPointCloudBuilder::DepthMaskToPointCloudBuilder(
  float min_depth_m, float max_depth_m)
: impl_(std::make_unique<DepthMaskToPointCloudImpl>(min_depth_m, max_depth_m))
{
}

DepthMaskToPointCloudBuilder::~DepthMaskToPointCloudBuilder() = default;

std::unique_ptr<sensor_msgs::msg::PointCloud2> DepthMaskToPointCloudBuilder::Build(
  const sensor_msgs::msg::Image & depth,
  const sensor_msgs::msg::CameraInfo & camera_info,
  const sensor_msgs::msg::Image & mask)
{
  return impl_->Build(depth, camera_info, mask);
}

bool DepthMaskToPointCloudBuilder::IsSupportedDepthEncoding(const std::string & encoding)
{
  return DepthBytesPerPixel(encoding) != 0U;
}

bool DepthMaskToPointCloudBuilder::IsSupportedMaskEncoding(const std::string & encoding)
{
  return MaskBytesPerPixel(encoding) != 0U;
}

}  // namespace depth_image_proc
}  // namespace isaac_ros
}  // namespace nvidia
