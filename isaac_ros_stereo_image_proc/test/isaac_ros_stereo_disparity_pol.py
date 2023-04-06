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

import os
import pathlib
import time

from isaac_ros_test import IsaacROSBaseTest, JSONConversion

from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

import pytest
import rclpy

from sensor_msgs.msg import CameraInfo, Image
from stereo_msgs.msg import DisparityImage

MAX_DISPARITY = 64.0


@pytest.mark.rostest
def generate_test_description():
    disparity_node = ComposableNode(
        name='disparity',
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::DisparityNode',
        namespace=IsaacROSDisparityTest.generate_namespace(),
        parameters=[{
                'backends': 'CUDA',
                'max_disparity': MAX_DISPARITY,
        }],
        remappings=[('disparity', 'disparity')]
    )

    container = ComposableNodeContainer(
        name='disparity_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[disparity_node],
        output='screen',
        arguments=['--ros-args', '--log-level', 'info']
    )
    return IsaacROSDisparityTest.generate_test_description([container])


class IsaacROSDisparityTest(IsaacROSBaseTest):
    filepath = pathlib.Path(os.path.dirname(__file__))

    @IsaacROSBaseTest.for_each_test_case()
    def test_image_disparity(self, test_folder):
        TIMEOUT = 10
        received_messages = {}

        self.generate_namespace_lookup(['left/image_rect', 'right/image_rect',
                                        'left/camera_info', 'right/camera_info',
                                        'disparity'])

        subs = self.create_logging_subscribers(
            [('disparity', DisparityImage)], received_messages)

        image_left_pub = self.node.create_publisher(
            Image, self.namespaces['left/image_rect'], self.DEFAULT_QOS
        )
        image_right_pub = self.node.create_publisher(
            Image, self.namespaces['right/image_rect'], self.DEFAULT_QOS
        )
        camera_info_right = self.node.create_publisher(
            CameraInfo, self.namespaces['right/camera_info'], self.DEFAULT_QOS
        )
        camera_info_left = self.node.create_publisher(
            CameraInfo, self.namespaces['left/camera_info'], self.DEFAULT_QOS
        )

        try:
            image_left = JSONConversion.load_image_from_json(test_folder / 'image_left.json')
            image_right = JSONConversion.load_image_from_json(test_folder / 'image_right.json')
            camera_info = JSONConversion.load_camera_info_from_json(
                test_folder / 'camera_info.json')
            end_time = time.time() + TIMEOUT
            done = False

            while time.time() < end_time:
                image_left_pub.publish(image_left)
                image_right_pub.publish(image_right)
                camera_info_left.publish(camera_info)
                camera_info_right.publish(camera_info)
                rclpy.spin_once(self.node, timeout_sec=0.1)

                if 'disparity' in received_messages:
                    done = True
                    break
            self.assertTrue(done, 'Didnt recieve output on disparity topic')

            disparity = received_messages['disparity']
            self.assertEqual(disparity.image.height, camera_info.height)
            self.assertEqual(disparity.image.width, camera_info.width)
            self.assertAlmostEqual(disparity.f, -0.3678634)
            self.assertAlmostEqual(disparity.t, 434.9440002)
            self.assertAlmostEqual(disparity.min_disparity, 0.0)
            self.assertAlmostEqual(disparity.max_disparity, MAX_DISPARITY)

        finally:
            [self.node.destroy_subscription(sub) for sub in subs]
            self.node.destroy_publisher(image_left_pub)
            self.node.destroy_publisher(image_right_pub)
            self.node.destroy_publisher(camera_info_right)
            self.node.destroy_publisher(camera_info_left)
