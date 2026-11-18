# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

OUTPUT_HEIGHT = 920
OUTPUT_WIDTH = 1380


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with all ROS 2 nodes for testing."""
    composable_nodes = [
        launch_ros.descriptions.ComposableNode(
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::ResizeNode',
            name='resize_node',
            namespace=IsaacROSResizeAspectRatioTest.generate_namespace(),
            parameters=[{
                'output_height': OUTPUT_HEIGHT,
                'output_width': OUTPUT_WIDTH,
                'keep_aspect_ratio': False,
            }])]

    resize_container = launch_ros.actions.ComposableNodeContainer(
        package='rclcpp_components',
        name='resize_container',
        namespace='',
        executable='component_container_mt',
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
        self.generate_namespace_lookup(['image', 'camera_info', 'resize/image'])
        received_messages = {}

        resized_image_sub, = self.create_logging_subscribers(
            subscription_requests=[('resize/image', Image)],
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
            self.assertNotEqual(image.width * OUTPUT_HEIGHT, image.height * OUTPUT_WIDTH,
                                'Test output must have a different aspect ratio from the input')

            # Wait at most TIMEOUT seconds for subscriber to respond
            TIMEOUT = 30
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
                if 'resize/image' in received_messages:
                    done = True
                    break

            self.assertTrue(done, "Didn't receive output on resize/image topic!")

            resized_image = received_messages['resize/image']

            # Make sure that the output image size is set to desired dimensions
            self.assertEqual(resized_image.height, OUTPUT_HEIGHT,
                             f'Height is not {OUTPUT_HEIGHT}!')
            self.assertEqual(resized_image.width, OUTPUT_WIDTH,
                             f'Width is not {OUTPUT_WIDTH}!')

        finally:
            self.node.destroy_subscription(resized_image_sub)
            self.node.destroy_publisher(image_pub)
            self.node.destroy_publisher(camera_info_pub)
