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

"""mono8 mask encoding test for the Isaac ROS Alpha Blend node."""

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


ALPHA = 0.7
DIMENSION_WIDTH = 120
DIMENSION_HEIGHT = 100
MASK_VAL = 200
IMAGE_BLUE = 0
IMAGE_GREEN = 100
IMAGE_RED = 200
EXPECTED_BLUE = 60
EXPECTED_GREEN = 130
EXPECTED_RED = 200


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with all ROS 2 nodes for testing."""
    composable_nodes = [
        ComposableNode(
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::AlphaBlendNode',
            name='alpha_blend_node',
            namespace=IsaacROSAlphaBlendMonoTest.generate_namespace(),
            parameters=[{
                'alpha': ALPHA
            }])]

    blend_container = ComposableNodeContainer(
        name='blend_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=composable_nodes,
        output='screen'
    )

    return IsaacROSAlphaBlendMonoTest.generate_test_description([blend_container])


class IsaacROSAlphaBlendMonoTest(IsaacROSBaseTest):
    filepath = pathlib.Path(os.path.dirname(__file__))

    def test_alpha_blend_mono(self):
        """
        Test alpha blending feature for mono8 mask.

        Test that the AlphaBlendNode is correctly blending the image and mask.
        Given:
        - Alpha is 0.7
        - Mask is a mono8 image with all values 200
        - Image is a bgr8 image with blue values 0, green values 100, and red values 200
        Given that alpha is 0.7, that the mask is a mono8 image with all values 200, and
        that the image is a bgr8 image with blue values 0, green values 100, and red values
        200, the output should be a bgr8 image with:
        - Blue values 0.7 * 0 + 0.3 * 200 = 0 + 60 = 60.
        - Green values 0.7 * 100 + 0.3 * 200 = 70 + 60 = 130.
        - Red values 0.7 * 200 + 0.3 * 200 = 140 + 60 = 200.
        """
        TIMEOUT = 300
        received_messages = {}

        self.generate_namespace_lookup(['image_input', 'mask_input', 'blended_image'])

        mask_pub = self.node.create_publisher(
            Image, self.namespaces['mask_input'], self.DEFAULT_QOS)

        image_pub = self.node.create_publisher(
            Image, self.namespaces['image_input'], self.DEFAULT_QOS)

        subs = self.create_logging_subscribers(
            [('blended_image', Image)], received_messages)

        try:
            # Create white image mask
            cv_mask = np.zeros((DIMENSION_HEIGHT, DIMENSION_WIDTH, 1), np.uint8)
            cv_mask[:] = MASK_VAL
            mask = CvBridge().cv2_to_imgmsg(cv_mask)
            mask.encoding = 'mono8'

            # Create bgr image
            cv_image = np.zeros((DIMENSION_HEIGHT, DIMENSION_WIDTH, 3), np.uint8)
            cv_image[:, :, 0] = IMAGE_BLUE
            cv_image[:, :, 1] = IMAGE_GREEN
            cv_image[:, :, 2] = IMAGE_RED
            image = CvBridge().cv2_to_imgmsg(cv_image)
            image.encoding = 'bgr8'

            end_time = time.time() + TIMEOUT
            done = False

            while time.time() < end_time:
                mask_pub.publish(mask)
                image_pub.publish(image)
                rclpy.spin_once(self.node, timeout_sec=(0.1))
                if 'blended_image' in received_messages:
                    done = True
                    break
            self.assertTrue(done, 'Appropriate output not received')
            image = received_messages['blended_image']
            self.assertEqual(image.height, DIMENSION_HEIGHT)
            self.assertEqual(image.width, DIMENSION_WIDTH)
            self.assertEqual(image.encoding, 'bgr8')

            blended_image = np.frombuffer(image.data, np.uint8)
            blended_image = blended_image.reshape(DIMENSION_HEIGHT,
                                                  DIMENSION_WIDTH,
                                                  3)
            self.assertTrue(
                np.all(blended_image[:, :, 0] == EXPECTED_BLUE) and
                np.all(blended_image[:, :, 1] == EXPECTED_GREEN) and
                np.all(blended_image[:, :, 2] == EXPECTED_RED))
        finally:
            self.node.destroy_subscription(subs)
            self.node.destroy_publisher(image_pub)
