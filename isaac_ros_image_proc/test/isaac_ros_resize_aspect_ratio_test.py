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

"""Aspect ratio test for the Isaac ROS Resize node."""

import os
import pathlib
import time

from isaac_ros_test import IsaacROSBaseTest, JSONConversion
import launch_ros
import pytest
import rclpy
from sensor_msgs.msg import CameraInfo, Image

USE_RELATIVE_SCALE = True
SCALE_HEIGHT = 2.0
SCALE_WIDTH = 3.0
HEIGHT = 20
WIDTH = 30
BACKENDS = 'CUDA'


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with all ROS 2 nodes for testing."""
    composable_nodes = [
        launch_ros.descriptions.ComposableNode(
            package='isaac_ros_image_proc',
            plugin='isaac_ros::image_proc::ResizeNode',
            name='resize_node',
            namespace=IsaacROSResizeAspectRatioTest.generate_namespace(),
            parameters=[{
                    'use_relative_scale': USE_RELATIVE_SCALE,
                    'scale_height': SCALE_HEIGHT,
                    'scale_width': SCALE_WIDTH,
                    'height': HEIGHT,
                    'width': WIDTH,
                    'backends': BACKENDS,
            }])]

    resize_container = launch_ros.actions.ComposableNodeContainer(
        package='rclcpp_components',
        name='resize_container',
        namespace='',
        executable='component_container',
        composable_node_descriptions=composable_nodes,
        output='screen'
    )

    return IsaacROSResizeAspectRatioTest.generate_test_description([resize_container])


class IsaacROSResizeAspectRatioTest(IsaacROSBaseTest):
    """Validate resizing with different aspect ratio."""

    filepath = pathlib.Path(os.path.dirname(__file__))

    @IsaacROSBaseTest.for_each_test_case(subfolder='resize')
    def test_resize_image_forwarded(self, test_folder) -> None:
        """Expect the image to be resized into a different aspect ratio."""
        self.generate_namespace_lookup(['image', 'camera_info', 'resized/image'])
        received_messages = {}

        resized_image_sub, = self.create_logging_subscribers(
            subscription_requests=[('resized/image', Image)],
            received_messages=received_messages
        )

        image_pub = self.node.create_publisher(
            Image, self.namespaces['image'], self.DEFAULT_QOS)
        camera_info_pub = self.node.create_publisher(
            CameraInfo, self.namespaces['camera_info'], self.DEFAULT_QOS)

        try:
            image = JSONConversion.load_image_from_json(
                test_folder / 'image.json')
            camera_info = JSONConversion.load_camera_info_from_json(
                test_folder / 'camera_info.json')

            # Wait at most TIMEOUT seconds for subscriber to respond
            TIMEOUT = 2
            end_time = time.time() + TIMEOUT

            done = False
            while time.time() < end_time:
                # Synchronize timestamps on both messages
                timestamp = self.node.get_clock().now().to_msg()
                image.header.stamp = timestamp
                camera_info.header.stamp = timestamp

                # Publish test case over both topics
                image_pub.publish(image)
                camera_info_pub.publish(camera_info)

                rclpy.spin_once(self.node, timeout_sec=0.1)

                # If we have received a message on the output topic, break
                if 'resized/image' in received_messages:
                    done = True
                    break

            self.assertTrue(done, "Didn't receive output on resized/image topic!")

            resized_image = received_messages['resized/image']

            # Make sure that the output image size is set to desired dimensions
            desired_height = image.height * SCALE_HEIGHT if USE_RELATIVE_SCALE else HEIGHT
            desired_width = image.width * SCALE_WIDTH if USE_RELATIVE_SCALE else WIDTH
            self.assertEqual(resized_image.height, desired_height,
                             f'Height is not {desired_height}!')
            self.assertEqual(resized_image.width, desired_width, f'Width is not {desired_width}!')

        finally:
            self.node.destroy_subscription(resized_image_sub)
            self.node.destroy_publisher(image_pub)
            self.node.destroy_publisher(camera_info_pub)
