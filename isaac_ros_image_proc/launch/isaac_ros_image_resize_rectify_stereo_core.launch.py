# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any, Dict

from isaac_ros_examples import IsaacROSLaunchFragment
from launch_ros.descriptions import ComposableNode


class IsaacROSResizeRectifyStereoLaunchFragment(IsaacROSLaunchFragment):

    @staticmethod
    def get_composable_nodes(interface_specs: Dict[str, Any]) -> Dict[str, ComposableNode]:
        return {
            'resize_left_node': ComposableNode(
                name='resize_left',
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::ResizeNode',
                parameters=[{
                    'output_width': interface_specs['image_resolution']['width'],
                    'output_height': interface_specs['image_resolution']['height'],
                    'keep_aspect_ratio': True
                }],
                remappings=[
                    ('camera_info', '/left/camera_info'),
                    ('image', '/left/image_raw'),
                    ('resize/camera_info', '/left/camera_info_resize'),
                    ('resize/image', '/left/image_resize')]
            ),

            'resize_right_node': ComposableNode(
                name='resize_right',
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::ResizeNode',
                parameters=[{
                    'output_width': interface_specs['image_resolution']['width'],
                    'output_height': interface_specs['image_resolution']['height'],
                    'keep_aspect_ratio': True
                }],
                remappings=[
                    ('camera_info', '/right/camera_info'),
                    ('image', '/right/image_raw'),
                    ('resize/camera_info', '/right/camera_info_resize'),
                    ('resize/image', '/right/image_resize')]
            ),

            'rectify_left_node': ComposableNode(
                name='rectify_left',
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::RectifyNode',
                parameters=[{
                    'output_width': interface_specs['image_resolution']['width'],
                    'output_height': interface_specs['image_resolution']['height'],
                }],
                remappings=[
                    ('image_raw', '/left/image_resize'),
                    ('camera_info', '/left/camera_info_resize'),
                    ('image_rect', '/left/image_rect'),
                    ('camera_info_rect', '/left/camera_info_rect')]
            ),

            'rectify_right_node': ComposableNode(
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::RectifyNode',
                name='rectify_right',
                parameters=[{
                    'output_width': interface_specs['image_resolution']['width'],
                    'output_height': interface_specs['image_resolution']['height'],
                }],
                remappings=[
                    ('image_raw', '/right/image_resize'),
                    ('camera_info', '/right/camera_info_resize'),
                    ('image_rect', '/right/image_rect'),
                    ('camera_info_rect', '/right/camera_info_rect')]
            )
        }
