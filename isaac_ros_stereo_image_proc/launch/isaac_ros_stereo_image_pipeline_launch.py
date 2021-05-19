# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import launch
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    disparity_node = ComposableNode(
        name='disparity',
        package='isaac_ros_stereo_image_proc',
        plugin='isaac_ros::stereo_image_proc::DisparityNode',
        parameters=[{
                'backends': 'CUDA',
                'window_size': 5,
                'max_disparity': 64,
        }])

    pointcloud_node = ComposableNode(
        package='isaac_ros_stereo_image_proc',
        plugin='isaac_ros::stereo_image_proc::PointCloudNode',
        parameters=[{
                'use_color': False,
        }],
        remappings=[('/left/image_rect_color', '/left/image_rect')])

    realsense_camera_node = Node(
        package='realsense2_camera',
        node_executable='realsense2_camera_node',
        parameters=[{
                'infra_height': 270,
                'infra_width': 480,
                'enable_color': False,
                'enable_depth': False,
        }],
        remappings=[('/infra1/image_rect_raw', '/left/image_rect'),
                    ('/infra1/camera_info', '/left/camera_info'),
                    ('/infra2/image_rect_raw', '/right/image_rect'),
                    ('/infra2/camera_info', '/right/camera_info')])

    container = ComposableNodeContainer(
        name='disparity_container',
        namespace='disparity',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[disparity_node, pointcloud_node],
        output='screen'
    )

    return (launch.LaunchDescription([container, realsense_camera_node]))
