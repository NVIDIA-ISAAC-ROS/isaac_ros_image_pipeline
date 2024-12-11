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

import os
import pathlib
import time

from cv_bridge import CvBridge
from isaac_ros_test import IsaacROSBaseTest
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import numpy as np

import pytest
import rclpy

from sensor_msgs.msg import Image


INPUT_IMAGE_WIDTH = 100
INPUT_IMAGE_HEIGHT = 100
OUTPUT_IMAGE_HEIGHT = 200
OUTPUT_IMAGE_WIDTH = 200
ENCODING = 'bgr8'


@pytest.mark.rostest
def generate_test_description():
    pad_node = ComposableNode(
        name='pad_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::PadNode',
        namespace=IsaacROSImageProcPadNodeTest.generate_namespace(),
        parameters=[{
            'output_image_width': OUTPUT_IMAGE_WIDTH,
            'output_image_height': OUTPUT_IMAGE_HEIGHT,
            'padding_type': 'BOTTOM_LEFT'
        }])

    return IsaacROSImageProcPadNodeTest.generate_test_description([
        ComposableNodeContainer(
            name='pad_node_container',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=[pad_node],
            namespace=IsaacROSImageProcPadNodeTest.generate_namespace(),
            output='screen',
            arguments=['--ros-args', '--log-level', 'info',
                       '--log-level', 'isaac_ros_test.encoder:=debug'],
        )
    ])


class IsaacROSImageProcPadNodeTest(IsaacROSBaseTest):
    filepath = pathlib.Path(os.path.dirname(__file__))

    def test_image_pad(self):
        """
        Test Image Padding feature.

        Test that the PadNode is correctly padding the image based on
        the given padding type.
        """
        TIMEOUT = 300
        received_messages = {}
        RED_EXPECTED_VAL = 1
        GREEN_EXPECTED_VAL = 2
        BLUE_EXPECTED_VAL = 3

        self.generate_namespace_lookup(['image', 'padded_image'])

        image_pub = self.node.create_publisher(
            Image, self.namespaces['image'], self.DEFAULT_QOS)

        subs = self.create_logging_subscribers(
            [('padded_image', Image)], received_messages)

        try:
            # Create white image
            cv_image = np.zeros((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, 3), np.uint8)
            cv_image[:] = (BLUE_EXPECTED_VAL, GREEN_EXPECTED_VAL, RED_EXPECTED_VAL)
            image = CvBridge().cv2_to_imgmsg(cv_image)
            image.encoding = ENCODING

            end_time = time.time() + TIMEOUT
            done = False

            while time.time() < end_time:
                image_pub.publish(image)
                rclpy.spin_once(self.node, timeout_sec=(0.1))
                if 'padded_image' in received_messages:
                    done = True
                    break
            self.assertTrue(done, 'Appropriate output not received')
            padded_image = received_messages['padded_image']
            self.assertEqual(padded_image.encoding, ENCODING,
                             f'Encoding different received encoding:{padded_image.encoding}')
            self.assertEqual(OUTPUT_IMAGE_HEIGHT, padded_image.height,
                             f'Height different received height:{padded_image.height}')
            self.assertEqual(OUTPUT_IMAGE_WIDTH, padded_image.width,
                             f'Width different received width:{padded_image.width}')

            padded_image_cv2 = CvBridge().imgmsg_to_cv2(padded_image, 'bgr8')
            data_matched = False
            if ((padded_image_cv2[0:100, 100:, :] == cv_image[:, :, :]).all()):
                data_matched = True
            self.assertTrue(data_matched,
                            "The data didn't get copied correctly to top right corner.")
        finally:
            self.node.destroy_subscription(subs)
            self.node.destroy_publisher(image_pub)
