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
import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


class IsaacROSPointCloudXyzrgbLaunchFragment(IsaacROSLaunchFragment):
    @staticmethod
    def get_composable_nodes(interface_specs: Dict[str, Any]) -> Dict[str, ComposableNode]:
        return {
            'point_cloud_xyzrgb_node': ComposableNode(
                package='isaac_ros_depth_image_proc',
                plugin='nvidia::isaac_ros::depth_image_proc::PointCloudXyzrgbNode',
                name='point_cloud_xyzrgb',
                namespace='',
                remappings=[
                    ('depth_registered/image_rect', '/zed2i/zed_node/depth/depth_registered'),
                    ('rgb/image_rect_color', 'zed2i/zed_node/rgb/image_rect_color'),
                    ('rgb/camera_info', '/zed2i/zed_node/rgb/camera_info'),
                    ('points', '/points')
                ],
                parameters=[{
                    'queue_size': 10
                }]
            )
        }


def generate_launch_description():
    # Pass an empty dict for interface_specs
    interface_specs = {}
    
    point_cloud_xyzrgb_container = ComposableNodeContainer(
        package='rclcpp_components',
        name='point_cloud_xyzrgb_container',
        namespace='',
        executable='component_container_mt',
        composable_node_descriptions=IsaacROSPointCloudXyzrgbLaunchFragment
        .get_composable_nodes(interface_specs).values(),
        output='screen'
    )
    
    # Convert dict_values to list before concatenation
    launch_actions = list(IsaacROSPointCloudXyzrgbLaunchFragment.get_launch_actions().values())
    
    return launch.LaunchDescription(
        [point_cloud_xyzrgb_container] + launch_actions)