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

"""Edge case test for the Isaac ROS Resize node."""

import os
import pathlib
import time

from isaac_ros_test import IsaacROSBaseTest, JSONConversion
import launch_ros
import pytest
from rcl_interfaces.msg import Log
import rclpy
from rclpy.logging import LoggingSeverity
from sensor_msgs.msg import CameraInfo, Image

USE_RELATIVE_SCALE = True
SCALE_HEIGHT = -2.0
SCALE_WIDTH = -3.0
HEIGHT = -20
WIDTH = -20


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with all ROS 2 nodes for testing."""
    composable_nodes = [
        launch_ros.descriptions.ComposableNode(
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::ResizeNode',
            name='resize_node',
            namespace=IsaacROSResizeInvalidTest.generate_namespace(),
            parameters=[{
                'output_height': -20,
                'output_width': -20,
            }])]

    resize_container = launch_ros.actions.ComposableNodeContainer(
        name='resize_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=composable_nodes,
        output='screen'
    )

    return IsaacROSResizeInvalidTest.generate_test_description([resize_container])


class IsaacROSResizeInvalidTest(IsaacROSBaseTest):
    """Validate error-catching behavior with invalid numbers."""

    filepath = pathlib.Path(os.path.dirname(__file__))

    @IsaacROSBaseTest.for_each_test_case(subfolder='resize')
    def test_resize_invalid(self, test_folder) -> None:
        """Expect the node to log an error when given invalid input."""
        self.generate_namespace_lookup(['image', 'camera_info', 'resize/image'])
        received_messages = {}

        resized_image_sub, rosout_sub = self.create_logging_subscribers(
            subscription_requests=[(self.namespaces['resize/image'], Image), ('/rosout', Log)],
            use_namespace_lookup=False,
            received_messages=received_messages,
            accept_multiple_messages=True,
            qos_profile=self.DEFAULT_BUFFER_QOS
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

            while time.time() < end_time:
                # Synchronize timestamps on both messages
                timestamp = self.node.get_clock().now().to_msg()
                image.header.stamp = timestamp
                camera_info.header.stamp = timestamp

                # Publish test case over both topics
                image_pub.publish(image)
                camera_info_pub.publish(camera_info)

                rclpy.spin_once(self.node, timeout_sec=0.1)

            self.assertIn('/rosout', received_messages, "Didn't receive output on /rosout topic!")

            # Make sure that at least one output log message is a non-empty error
            self.assertTrue(any([
                LoggingSeverity(rosout.level) == LoggingSeverity.ERROR and len(rosout.msg) > 0
                for rosout in received_messages['/rosout']]),
                'No message with non-empty message and Error severity!')

            # Make sure no output image was received in the error case
            self.assertEqual(
                len(received_messages[self.namespaces['resize/image']]), 0,
                'Resized image was received despite error!')

        finally:
            self.node.destroy_subscription(resized_image_sub)
            self.node.destroy_subscription(rosout_sub)
            self.node.destroy_subscription(rosout_sub)
            self.node.destroy_publisher(image_pub)
            self.node.destroy_publisher(camera_info_pub)
