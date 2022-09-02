# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    image_flip_node = ComposableNode(
        name='image_flip_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFlipNode',
    )
    image_flip_container = ComposableNodeContainer(
        name='argus_mono_container',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[image_flip_node],
        namespace='',
        output='screen',
        arguments=['--ros-args', '--log-level', 'info'],
    )
    return launch.LaunchDescription([image_flip_container])
