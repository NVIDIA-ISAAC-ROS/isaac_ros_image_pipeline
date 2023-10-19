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
        name='disparity_node',
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::DisparityNode',
        parameters=[{
                'backends': 'CUDA',
                'max_disparity': 64.0,
        }],
        remappings=[('/left/image_rect', '/left/image_rect_color'),
                    ('/right/image_rect', '/right/image_rect_color')]
        )

    pointcloud_node = ComposableNode(
        name='pointcloud_node',
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::PointCloudNode',
        parameters=[{
                'use_color': True,
        }])

    # RealSense
    realsense_config_file_path = os.path.join(
        get_package_share_directory('isaac_ros_stereo_image_proc'),
        'config', 'realsense.yaml'
    )

    realsense_node = ComposableNode(
        package='realsense2_camera',
        plugin='realsense2_camera::RealSenseNodeFactory',
        parameters=[realsense_config_file_path],
        remappings=[('/infra1/image_rect_raw', '/left/image_rect'),
                    ('/infra1/camera_info', '/left/camera_info'),
                    ('/infra2/image_rect_raw', '/right/image_rect'),
                    ('/infra2/camera_info', '/right/camera_info')
                    ]
    )

    image_format_converter_node_left = ComposableNode(
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        name='image_format_node_left',
        parameters=[{
                'encoding_desired': 'bgr8',
        }],
        remappings=[
            ('image_raw', 'left/image_rect'),
            ('image', 'left/image_rect_color')]
    )

    image_format_converter_node_right = ComposableNode(
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        name='image_format_node_right',
        parameters=[{
                'encoding_desired': 'rgb8',
        }],
        remappings=[
            ('image_raw', 'right/image_rect'),
            ('image', 'right/image_rect_color')]
    )

    container = ComposableNodeContainer(
        name='disparity_container',
        namespace='disparity',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[disparity_node, pointcloud_node, realsense_node,
                                      image_format_converter_node_left,
                                      image_format_converter_node_right],
        output='screen'
    )

    rviz_config_path = os.path.join(get_package_share_directory(
        'isaac_ros_stereo_image_proc'), 'config', 'isaac_ros_stereo_image_proc_realsense.rviz')

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen')

    return (launch.LaunchDescription([container, rviz]))
