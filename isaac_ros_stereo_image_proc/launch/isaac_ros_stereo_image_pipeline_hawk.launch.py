# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            'module_id',
            default_value='2',
            description='Index specifying the stereo camera module to use.'),
    ]
    module_id = LaunchConfiguration('module_id')

    argus_stereo_node = ComposableNode(
        name='argus_stereo',
        package='isaac_ros_argus_camera',
        plugin='nvidia::isaac_ros::argus::ArgusStereoNode',
        parameters=[{'module_id': module_id}],
    )

    left_resize_node = ComposableNode(
        name='left_resize_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'output_width': 960,
            'output_height': 576,
        }],
        remappings=[
            ('image', 'left/image_raw'),
            ('camera_info', 'left/camera_info'),
            ('resize/image', 'left/image_resize'),
            ('resize/camera_info', 'left/camera_info_resize')
        ]
    )

    right_resize_node = ComposableNode(
        name='right_resize_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'output_width': 960,
            'output_height': 576,
        }],
        remappings=[
            ('image', 'right/image_raw'),
            ('camera_info', 'right/camera_info'),
            ('resize/image', 'right/image_resize'),
            ('resize/camera_info', 'right/camera_info_resize')
        ]
    )

    left_rectify_node = ComposableNode(
        name='left_rectify_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::RectifyNode',
        parameters=[{
            'output_width': 960,
            'output_height': 576,
        }],
        remappings=[
            ('image_raw', '/left/image_resize'),
            ('camera_info', '/left/camera_info_resize'),
            ('image_rect', '/left/image_rect'),
            ('camera_info_rect', '/left/camera_info')
        ]
    )

    right_rectify_node = ComposableNode(
        name='right_rectify_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::RectifyNode',
        parameters=[{
            'output_width': 960,
            'output_height': 576,
        }],
        remappings=[
            ('image_raw', '/right/image_resize'),
            ('camera_info', '/right/camera_info_resize'),
            ('image_rect', '/right/image_rect'),
            ('camera_info_rect', '/right/camera_info')
        ]
    )

    disparity_node = ComposableNode(
        name='disparity_node',
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::DisparityNode',
        parameters=[{
                'backends': 'CUDA',
                'max_disparity': 64.0,
        }],
        remappings=[
            ('/left/camera_info', '/left/camera_info_rect'),
            ('/right/camera_info', '/right/camera_info_rect'),
        ],
    )

    pointcloud_node = ComposableNode(
        name='pointcloud_node',
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::PointCloudNode',
        parameters=[{
                'use_color': True,
        }],
        remappings=[
            ('/left/image_rect_color', '/left/image_rect'),
            ('/left/camera_info', '/left/camera_info_rect'),
            ('/right/camera_info', '/right/camera_info_rect'),
        ],
    )

    container = ComposableNodeContainer(
        name='disparity_container',
        namespace='disparity',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            left_resize_node,
            right_resize_node,
            left_rectify_node,
            right_rectify_node,
            argus_stereo_node,
            disparity_node,
            pointcloud_node
        ],
        output='screen',
        arguments=['--ros-args', '--log-level', 'info',
                   '--log-level', 'color_format_convert:=info',
                   '--log-level', 'NitrosImage:=info',
                   '--log-level', 'NitrosNode:=info'
                   ],
    )

    final_launch_description = launch_args + [container]
    return (launch.LaunchDescription(final_launch_description))
