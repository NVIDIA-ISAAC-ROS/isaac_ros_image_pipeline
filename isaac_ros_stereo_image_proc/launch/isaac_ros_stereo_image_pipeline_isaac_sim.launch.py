# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os

from ament_index_python.packages import get_package_share_directory
import launch
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    disparity_node = ComposableNode(
        name='disparity',
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::DisparityNode',
        parameters=[{
                'backends': 'CUDA',
                'max_disparity': 64.0,
        }],
        remappings=[('left/image_rect', 'front_stereo_camera/left_rgb/image_raw'),
                    ('left/camera_info', 'front_stereo_camera/left_rgb/camerainfo'),
                    ('right/image_rect', 'front_stereo_camera/right_rgb/image_raw'),
                    ('right/camera_info', 'front_stereo_camera/right_rgb/camerainfo')
                    ]
    )

    pointcloud_node = ComposableNode(
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::PointCloudNode',
        parameters=[{
                'use_color': True,
                'unit_scaling': 1.0
        }],
        remappings=[('left/image_rect_color', 'front_stereo_camera/left_rgb/image_raw'),
                    ('left/camera_info', 'front_stereo_camera/left_rgb/camerainfo'),
                    ('right/camera_info', 'front_stereo_camera/right_rgb/camerainfo')
                    ]
    )

    container = ComposableNodeContainer(
        name='disparity_container',
        namespace='disparity',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[disparity_node, pointcloud_node],
        output='screen'
    )

    rviz_config_path = os.path.join(get_package_share_directory(
        'isaac_ros_stereo_image_proc'), 'config', 'isaac_ros_stereo_image_proc_isaac_sim.rviz')

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen'
    )

    return (launch.LaunchDescription([container, rviz_node]))
