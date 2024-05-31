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

"""Proof-of-Life test for the Isaac ROS Resize node."""

import os
import pathlib
import time

from cv_bridge import CvBridge
from isaac_ros_test import IsaacROSBaseTest, JSONConversion
import launch_ros
import numpy as np
import pytest
import rclpy
from sensor_msgs.msg import CameraInfo, Image

WIDTH = 100
HEIGHT = 100


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with all ROS 2 nodes for testing."""
    composable_nodes = [
        launch_ros.descriptions.ComposableNode(
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::CropNode',
            name='crop_node',
            namespace=IsaacROSCropTest.generate_namespace(),
            parameters=[{
                'input_width': 300,
                'input_height': 300,
                'crop_height': WIDTH,
                'crop_width': HEIGHT,
                'crop_mode': 'TOPRIGHT'
            }])]

    crop_container = launch_ros.actions.ComposableNodeContainer(
        package='rclcpp_components',
        name='crop_container',
        namespace='',
        executable='component_container',
        composable_node_descriptions=composable_nodes,
        parameters=[{'encoding_desired': 'rgb8'}],
        output='screen',
        arguments=['--ros-args', '--log-level', 'info',
                   '--log-level', 'isaac_ros_test.crop_node:=debug'],
    )

    return IsaacROSCropTest.generate_test_description([crop_container])


class IsaacROSCropTest(IsaacROSBaseTest):
    """Validate image cropping in top right location."""

    filepath = pathlib.Path(os.path.dirname(__file__))

    @IsaacROSBaseTest.for_each_test_case(subfolder='crop')
    def test_resize_typical(self, test_folder) -> None:
        """Expect the node to output images with correctly cropped dimensions."""
        self.generate_namespace_lookup([
            'image',
            'camera_info',
            'crop/image',
            'crop/camera_info'
        ])
        received_messages = {}

        croppedd_image_sub = self.create_logging_subscribers(
            subscription_requests=[('crop/image', Image)],
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
                if 'crop/image' in received_messages:
                    done = True
                    break

            self.assertTrue(
                done, "Didn't receive output on crop/image topic!")

            cropped_image = received_messages['crop/image']

            # Make sure that the output image size is set to desired dimensions
            desired_height = HEIGHT
            desired_width = WIDTH
            self.assertEqual(cropped_image.height, desired_height,
                             f'Height is not {desired_height}!')
            self.assertEqual(cropped_image.width, desired_width,
                             f'Width is not {desired_width}!')

            blue_image_cv = np.full(
                (desired_width, desired_height, 3), (255, 0, 0), np.uint8)

            cropped_image_cv = CvBridge().imgmsg_to_cv2(
                cropped_image, desired_encoding='bgr8')

            self.assertImagesEqual(
                cropped_image_cv, blue_image_cv, threshold_fraction=0)

        finally:
            self.node.destroy_subscription(croppedd_image_sub)
            self.node.destroy_publisher(image_pub)
            self.node.destroy_publisher(camera_info_pub)
