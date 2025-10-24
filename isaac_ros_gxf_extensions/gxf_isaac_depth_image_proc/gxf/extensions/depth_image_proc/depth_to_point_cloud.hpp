// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NVIDIA_ISAAC_ROS_EXTENSIONS_DEPTH_TO_POINT_CLOUD_HPP_
#define NVIDIA_ISAAC_ROS_EXTENSIONS_DEPTH_TO_POINT_CLOUD_HPP_

#include "gxf/multimedia/camera.hpp"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"

#include "extensions/depth_image_proc/depth_to_point_cloud_cuda.cu.hpp"
#include "extensions/messages/camera_message.hpp"
#include "messages/point_cloud_message.hpp"

namespace nvidia {
namespace isaac_ros {
namespace depth_image_proc {
// GXF codelet that subscribes to depth image and an optional rgb image,
// and publishes a pointcloud. The pointcloud is colorized is the
// optional rgb image input is provided
class DepthToPointCloud : public gxf::Codelet {
 public:
  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;
  gxf_result_t registerInterface(gxf::Registrar * registrar) override;

 private:
  // memory pool
  gxf::Parameter<gxf::Handle<gxf::Allocator>> allocator_;
  // Data receiver to get depth image data
  gxf::Parameter<gxf::Handle<gxf::Receiver>> depth_receiver_;
  // Data receiver to get rgb image data
  gxf::Parameter<gxf::Handle<gxf::Receiver>> image_receiver_;
  // Data transmitter to send the point_cloud data
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> point_cloud_transmitter_;
  // Paramter to limit the number of pixels converted to points
  gxf::Parameter<int32_t> skip_;

  bool colorize_point_cloud_;

  DepthToPointCloudNodeCUDA cloud_compute_;

  /**
   * @brief Validates whether the input depth message is of the expected format
   *
   * @param depth_message The input depth message
   */
  gxf::Expected<void> validateDepthMessage(const nvidia::isaac::CameraMessageParts& depth_message);

  /**
   * @brief Validates whether the input rgb message is of the expected format
   *
   * @param depth_message The depth message
   * @param depth_message The rgb image message
   */
  gxf::Expected<void> validateImageMessage(
    const nvidia::isaac::CameraMessageParts& depth_message,
    const nvidia::isaac::CameraMessageParts& image_message);

  /**
   * @brief Creates a PointCloudProperties struct used in the cuda kernel
   *
   * @param depth_message The input depth message
   * @param skip Paramter to limit the number of pixels converted to points
   */
  PointCloudProperties createPointCloudProperties(
    const nvidia::isaac::CameraMessageParts & depth_img,
    const int skip);

  /**
   * @brief Creates a DepthProperties struct used in the cuda kernel
   *
   * @param depth_message The input depth message
   */
  DepthProperties createDepthProperties(const nvidia::isaac::CameraMessageParts & depth_img);
};

}  // namespace depth_image_proc
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // NVIDIA_ISAAC_ROS_EXTENSIONS_DEPTH_TO_POINT_CLOUD_HPP_
