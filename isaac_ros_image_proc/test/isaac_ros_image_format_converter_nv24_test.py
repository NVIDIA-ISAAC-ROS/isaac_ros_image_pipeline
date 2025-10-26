# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""NV24 test for the Isaac ROS Image Format Converter node."""

import time

import cv2
from cv_bridge import CvBridge
from isaac_ros_test import IsaacROSBaseTest
import launch_ros
import numpy as np
import pytest
import rclpy
from sensor_msgs.msg import Image

ENCODING_DESIRED = 'bgr8'
HEIGHT = 300
WIDTH = 300


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with all ROS 2 nodes for testing."""
    composable_nodes_1 = launch_ros.descriptions.ComposableNode(
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
            name='image_format_node_nv24',
            namespace=IsaacROSFormatNV24Test.generate_namespace(),
            parameters=[{
                'encoding_desired': 'nv24',
                'image_width': WIDTH,
                'image_height': HEIGHT
            }],
            remappings=[('image', 'image_raw2')],
        )

    composable_nodes_2 = launch_ros.descriptions.ComposableNode(
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
            name='image_format_node_bgr8',
            namespace=IsaacROSFormatNV24Test.generate_namespace(),
            parameters=[{
                'encoding_desired': ENCODING_DESIRED,
                'image_width': WIDTH,
                'image_height': HEIGHT
            }],
            remappings=[('image_raw', 'image_raw2')],
        )

    format_container = launch_ros.actions.ComposableNodeContainer(
        name='format_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            composable_nodes_1,
            composable_nodes_2
        ],
        arguments=['--ros-args', '--log-level', 'info',
                   '--log-level', 'isaac_ros_test.image_format_node_bgr8:=debug'],
        output='screen'
    )

    return IsaacROSFormatNV24Test.generate_test_description([format_container])


class IsaacROSFormatNV24Test(IsaacROSBaseTest):
    """Vaidate format conversion to the nv24 format."""

    def test_nv24_to_bgr_conversion(self) -> None:
        """Expect the node to convert nv24 input images into the bgr8 format."""
        self.generate_namespace_lookup(['image_raw', 'image'])
        received_messages = {}

        image_sub, = self.create_logging_subscribers(
            subscription_requests=[('image', Image)],
            received_messages=received_messages
        )

        image_raw_pub = self.node.create_publisher(
            Image, self.generate_namespace('image_raw'), self.DEFAULT_QOS)

        try:
            # Generate an input image in RGB encoding
            cv_image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
            cv_image[:] = (255, 0, 0)  # Full red

            image_raw = CvBridge().cv2_to_imgmsg(cv_image)
            image_raw.encoding = 'bgr8'  # Set image encoding explicitly

            # Wait at most TIMEOUT seconds for subscriber to respond
            TIMEOUT = 2
            end_time = time.time() + TIMEOUT

            done = False
            while time.time() < end_time:
                # Update timestamp on message
                timestamp = self.node.get_clock().now().to_msg()
                image_raw.header.stamp = timestamp

                # Publish test case over single topic
                image_raw_pub.publish(image_raw)

                rclpy.spin_once(self.node, timeout_sec=0.1)

                # If we have received a message on the output topic, break
                if 'image' in received_messages:
                    done = True
                    break

            self.assertTrue(done, "Didn't receive output on image topic!")

            image = received_messages['image']

            # Make sure that the output image has the desired encoding
            self.assertEqual(image.encoding, ENCODING_DESIRED, 'Incorrect output encoding!')

            # Make sure output image pixels match OpenCV result
            image_actual = CvBridge().imgmsg_to_cv2(image)
            image_expected = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            self.assertImagesEqual(image_actual, image_expected)

        finally:
            self.node.destroy_subscription(image_sub)
            self.node.destroy_publisher(image_raw_pub)
