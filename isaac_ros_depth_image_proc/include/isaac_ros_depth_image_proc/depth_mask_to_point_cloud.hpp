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

#ifndef ISAAC_ROS_DEPTH_IMAGE_PROC__DEPTH_MASK_TO_POINT_CLOUD_HPP_
#define ISAAC_ROS_DEPTH_IMAGE_PROC__DEPTH_MASK_TO_POINT_CLOUD_HPP_

#include <limits>
#include <memory>
#include <string>

#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace depth_image_proc
{

//! Back-project selected depth pixels into a packed, CUDA-backed XYZ PointCloud2.
//!
//! The builder accepts float32 depth in meters or uint16 depth in millimeters and uint8 or uint16
//! masks. A nonzero mask value selects a pixel. CPU image buffers are promoted to CUDA storage,
//! while CUDA image buffers remain device-resident. Other negotiated backends are rejected before
//! their storage is accessed.
//!
//! Each output point contains three contiguous float32 fields, x, y, and z, and selected points
//! retain depth-image raster order. Invalid metadata, storage, calibration, options, or CUDA
//! operations raise an exception. Concurrent calls to Build() are safe but execute serially because
//! they reuse one CUDA stream and workspace.
class DepthMaskToPointCloudBuilder
{
public:
  //! Create a builder with reusable CUDA stream and compaction workspace state.
  //!
  //! `min_depth_m` is the finite, nonnegative inclusive minimum accepted depth in meters.
  //! `max_depth_m` is the inclusive maximum, must exceed `min_depth_m`, and may be positive
  //! infinity to disable the upper bound.
  //!
  //! Throws `std::invalid_argument` when the depth interval is invalid and `std::runtime_error`
  //! when CUDA resource initialization fails.
  explicit DepthMaskToPointCloudBuilder(
    float min_depth_m = 0.0F,
    float max_depth_m = std::numeric_limits<float>::infinity());

  //! Wait for outstanding work and release the builder's CUDA resources.
  //!
  //! Destruction may block while the private stream finishes and suppresses CUDA cleanup errors.
  //! Clouds returned by earlier Build() calls own their buffers and remain valid.
  ~DepthMaskToPointCloudBuilder();

  DepthMaskToPointCloudBuilder(const DepthMaskToPointCloudBuilder &) = delete;
  DepthMaskToPointCloudBuilder & operator=(const DepthMaskToPointCloudBuilder &) = delete;

  //! Build a filtered point cloud from valid depth pixels in raster order.
  //!
  //! `depth` is a borrowed CPU- or CUDA-backed image with `32FC1` values in meters or `16UC1`
  //! values in millimeters.
  //! `camera_info` is borrowed calibration for `depth`; zero image dimensions mean unspecified
  //! dimensions, while nonzero dimensions must match `depth`. Its K matrix supplies fx, fy, cx, and
  //! cy. The builder does not compare its timestamp or frame with either image.
  //! `mask` is a borrowed CPU- or CUDA-backed image with `mono8`, `8UC1`, `mono16`, or `16UC1`
  //! values and dimensions matching `depth`. Any nonzero value selects its depth pixel.
  //!
  //! Both images must have positive matching dimensions, little-endian elements, aligned row steps,
  //! and payloads containing every declared row. Their packed output row size must fit in uint32.
  //!
  //! Returns sole ownership of a CUDA-backed, dense, one-row cloud whose 12-byte points contain
  //! float32 x, y, and z coordinates in meters. Points retain raster order, and the full depth
  //! header is copied. Returns null when no pixel passes the nonzero-mask, finite-depth,
  //! inclusive-range, and finite-projection filters.
  //!
  //! CPU uploads and selected-point count retrieval complete before return. The final output copy
  //! may still be running. The cloud can be published immediately; consumers must acquire a buffer
  //! read handle on their own stream to order access after the recorded completion event.
  //!
  //! Throws `std::invalid_argument` for invalid messages, `std::overflow_error` for unrepresentable
  //! sizes, `std::logic_error` for an unexpected output layout, and runtime exceptions for
  //! allocation or CUDA failures observed during this call. Asynchronous CUDA failures may surface
  //! when consumers wait for completion. Inputs are borrowed, unmodified, and need to remain valid
  //! only for this call.
  std::unique_ptr<sensor_msgs::msg::PointCloud2> Build(
    const sensor_msgs::msg::Image & depth,
    const sensor_msgs::msg::CameraInfo & camera_info,
    const sensor_msgs::msg::Image & mask);

  //! Check whether a depth encoding has a CUDA conversion path.
  //!
  //! `encoding` is compared exactly against the supported ROS encoding names.
  //!
  //! Returns `true` for `32FC1` and `16UC1`; otherwise returns `false`.
  static bool IsSupportedDepthEncoding(const std::string & encoding);

  //! Check whether a single-channel mask encoding has a CUDA conversion path.
  //!
  //! `encoding` is compared exactly against the supported ROS encoding names.
  //!
  //! Returns `true` for `mono8`, `8UC1`, `mono16`, and `16UC1`; otherwise returns `false`.
  static bool IsSupportedMaskEncoding(const std::string & encoding);

private:
  class DepthMaskToPointCloudImpl;

  // Owns all CUDA resources and mutable workspace state.
  std::unique_ptr<DepthMaskToPointCloudImpl> impl_;
};

}  // namespace depth_image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_DEPTH_IMAGE_PROC__DEPTH_MASK_TO_POINT_CLOUD_HPP_
