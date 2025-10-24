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

#include "isaac_ros_depth_image_proc/align_depth_to_color_node.cu.hpp"

#include <algorithm>
#include <limits>

#include <sensor_msgs/msg/camera_info.hpp>

namespace nvidia
{
namespace isaac_ros
{
namespace depth_image_proc
{

namespace
{

constexpr size_t kBlockSize = 256;

}

// Main alignment kernel with all optimizations
__global__ void AlignDepthToColorKernel(
    const float* __restrict__ depth_data,
    unsigned int* __restrict__ z_buffer_uint,  // For atomicMin
    int depth_w,
    int depth_h,
    int color_w,
    int color_h,
    float inv_fx_d, float inv_fy_d, float cx_d, float cy_d,  // Precomputed inverses
    float fx_c, float fy_c, float cx_c, float cy_c,
    const float* __restrict__ transform_3x4)  // 3x4 transform (12 floats)
{
    // Grid-stride loop
    // Grid stride loop is recommended for the GPU.
    // https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    const int total_depth_pixels = depth_w * depth_h;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_depth_pixels;
         idx += blockDim.x * gridDim.x) {

        const int u = idx % depth_w;
        const int v = idx / depth_w;
        const float z = depth_data[idx];

        // Immediate discard for invalid depth (optimization 6)
        if (!isfinite(z) || z <= 0.0f) {
            continue;
        }

        // Unproject to depth camera coordinates (optimization 4 - precomputed inverses)
        const float x_d = (static_cast<float>(u) - cx_d) * z * inv_fx_d;
        const float y_d = (static_cast<float>(v) - cy_d) * z * inv_fy_d;

        // Transform to color camera frame (optimization 4 - compact 3x4 form)
        const float x_c = fmaf(transform_3x4[0], x_d, fmaf(transform_3x4[1], y_d, fmaf(transform_3x4[2], z, transform_3x4[3])));
        const float y_c = fmaf(transform_3x4[4], x_d, fmaf(transform_3x4[5], y_d, fmaf(transform_3x4[6], z, transform_3x4[7])));
        const float z_c = fmaf(transform_3x4[8], x_d, fmaf(transform_3x4[9], y_d, fmaf(transform_3x4[10], z, transform_3x4[11])));

        // Safety check - behind camera (optimization 6)
        if (z_c <= 0.0f) {
            continue;
        }

        // Project to color image
        const float inv_z_c = 1.0f / z_c;  // Single divide
        const int u_c = static_cast<int>(fmaf(fx_c * x_c, inv_z_c, cx_c));
        const int v_c = static_cast<int>(fmaf(fy_c * y_c, inv_z_c, cy_c));

        // Bounds check
        if (u_c < 0 || u_c >= color_w || v_c < 0 || v_c >= color_h) {
            continue;
        }

        // Z-buffer with atomicMin (optimization 2)
        const int color_idx = v_c * color_w + u_c;
        const unsigned int z_c_uint = __float_as_uint(z_c);
        atomicMin(&z_buffer_uint[color_idx], z_c_uint);
    }
}

// Second pass: copy z-buffer to output
__global__ void CopyZBufferToOutputKernel(
    const unsigned int* __restrict__ z_buffer_uint,
    float* __restrict__ aligned_depth,
    int total_pixels)
{
    // Grid-stride loop
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_pixels;
         idx += blockDim.x * gridDim.x) {

        const unsigned int z_uint = z_buffer_uint[idx];
        const float z_float = __uint_as_float(z_uint);

        // Only write finite depths (skip inf initialization)
        aligned_depth[idx] = isfinite(z_float) ? z_float : 0.0f;
    }
}

__global__ void InitZBufferKernel(unsigned int* __restrict__ buffer, unsigned int value, int n)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
        buffer[idx] = value;
    }
}

cudaError_t AlignDepthToColor(
    const float* depth_data,
    float* aligned_depth,
    const sensor_msgs::msg::CameraInfo & depth_camera_info,
    const sensor_msgs::msg::CameraInfo & color_camera_info,
    const double* color_pose_depth,
    cudaStream_t stream,
    float* gpu_time_ms)
{
    // CUDA event timing
    cudaEvent_t start, stop;
    bool measure_time = (gpu_time_ms != nullptr);
    if (measure_time) {
        CUDA_RETURN_IF_ERROR(cudaEventCreate(&start));
        CUDA_RETURN_IF_ERROR(cudaEventCreate(&stop));
        CUDA_RETURN_IF_ERROR(cudaEventRecord(start, stream));
    }

    const float fx_d = static_cast<float>(depth_camera_info.k[0]);
    const float fy_d = static_cast<float>(depth_camera_info.k[4]);
    const float cx_d = static_cast<float>(depth_camera_info.k[2]);
    const float cy_d = static_cast<float>(depth_camera_info.k[5]);
    const float fx_c = static_cast<float>(color_camera_info.k[0]);
    const float fy_c = static_cast<float>(color_camera_info.k[4]);
    const float cx_c = static_cast<float>(color_camera_info.k[2]);
    const float cy_c = static_cast<float>(color_camera_info.k[5]);

    // Precompute inverses (optimization 4)
    const float inv_fx_d = 1.0f / fx_d;
    const float inv_fy_d = 1.0f / fy_d;

    // Convert 4x4 transform to 3x4 float (optimization 4)
    float transform_3x4[12];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            transform_3x4[i * 4 + j] = static_cast<float>(color_pose_depth[i * 4 + j]);
        }
    }

    // Using original depth image dimensions
    const int original_depth_w = static_cast<int>(depth_camera_info.width);
    const int original_depth_h = static_cast<int>(depth_camera_info.height);
    const size_t original_depth_pixels = static_cast<size_t>(original_depth_w) * original_depth_h;
    const size_t color_pixels = static_cast<size_t>(color_camera_info.width) * color_camera_info.height;

    // Allocate other GPU buffers
    unsigned int * z_buffer_uint;
    float * gpu_transform;
    CUDA_RETURN_IF_ERROR(
        cudaMallocAsync(&z_buffer_uint, color_pixels * sizeof(unsigned int), stream));
    CUDA_RETURN_IF_ERROR(cudaMallocAsync(&gpu_transform, 12 * sizeof(float), stream));

    // Upload transform matrix
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
        gpu_transform, transform_3x4, 12 * sizeof(float), cudaMemcpyHostToDevice, stream));

    // Initialize z-buffer to +inf (as uint) - fix host/device issue
    const float inf_float = std::numeric_limits<float>::infinity();
    unsigned int inf_uint;
    memcpy(&inf_uint, &inf_float, sizeof(float));  // Host-side bit conversion

    const int color_blocks = (color_pixels + kBlockSize - 1) / kBlockSize;
    InitZBufferKernel<<<color_blocks, kBlockSize, 0, stream>>>(
        z_buffer_uint, inf_uint, color_pixels);

    // Step 2: Depth alignment
    const int align_blocks = (original_depth_pixels + kBlockSize - 1) / kBlockSize;
    AlignDepthToColorKernel<<<align_blocks, kBlockSize, 0, stream>>>(
        depth_data,
        z_buffer_uint,
        original_depth_w, original_depth_h,
        color_camera_info.width, color_camera_info.height,
        inv_fx_d, inv_fy_d, cx_d, cy_d,
        fx_c, fy_c, cx_c, cy_c,
        gpu_transform
    );

    // Step 3: Copy z-buffer to output
    CopyZBufferToOutputKernel<<<color_blocks, kBlockSize, 0, stream>>>(
        z_buffer_uint, aligned_depth, color_pixels);

    CUDA_RETURN_IF_ERROR(cudaFreeAsync(z_buffer_uint, stream));
    CUDA_RETURN_IF_ERROR(cudaFreeAsync(gpu_transform, stream));

    // End timing
    if (measure_time) {
        CUDA_RETURN_IF_ERROR(cudaEventRecord(stop, stream));
        CUDA_RETURN_IF_ERROR(cudaEventSynchronize(stop));
        CUDA_RETURN_IF_ERROR(cudaEventElapsedTime(gpu_time_ms, start, stop));
        CUDA_RETURN_IF_ERROR(cudaEventDestroy(start));
        CUDA_RETURN_IF_ERROR(cudaEventDestroy(stop));
    }
    return cudaSuccess;
}

}  // namespace depth_image_proc
}  // namespace isaac_ros
}  // namespace nvidia