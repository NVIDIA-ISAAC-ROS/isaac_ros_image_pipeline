# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""TimeStamp match test for the Isaac ROS Rectify node."""

import os
import pathlib
import time

import cv2
# Need to load sklearn first on Jetson.
from isaac_ros_test import IsaacROSBaseTest, JSONConversion
import launch_ros
import pytest
import rclpy
from sensor_msgs.msg import CameraInfo, Image

VISUALIZE = False


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with all ROS 2 nodes for testing."""
    composable_nodes = [
        launch_ros.descriptions.ComposableNode(
            name='left_rectify_node',
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::RectifyNode',
            namespace=IsaacROSRectifyOSSTest.generate_namespace(),
            parameters=[{
                'output_width': 1920,
                'output_height': 1200
            }],
            remappings=[
                ('image_raw', 'left/image_raw'),
                ('camera_info', 'left/camera_info'),
                ('image_rect', 'left/image_rect'),
                ('camera_info_rect', 'left/camera_info_rect')
            ]
        ),
        launch_ros.descriptions.ComposableNode(
            name='right_rectify_node',
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::RectifyNode',
            namespace=IsaacROSRectifyOSSTest.generate_namespace(),
            parameters=[{
                'output_width': 1920,
                'output_height': 1200,
            }],
            remappings=[
                ('image_raw', 'right/image_raw'),
                ('camera_info', 'right/camera_info'),
                ('image_rect', 'right/image_rect'),
                ('camera_info_rect', 'right/camera_info_rect')
            ]
        ),
        launch_ros.descriptions.ComposableNode(
            name='left_oss_rectify_node',
            package='image_proc',
            plugin='image_proc::RectifyNode',
            namespace=IsaacROSRectifyOSSTest.generate_namespace(),
            parameters=[{
                'width': 1920,
                'height': 1200,
                'use_scale': False
            }],
            remappings=[
                ('image', 'left/image_raw'),
                ('camera_info', 'left/camera_info'),
                ('image_rect', 'left/image_rect_oss')
            ]
        ),
        launch_ros.descriptions.ComposableNode(
            name='right_oss_rectify_node',
            package='image_proc',
            plugin='image_proc::RectifyNode',
            namespace=IsaacROSRectifyOSSTest.generate_namespace(),
            parameters=[{
                'width': 1920,
                'height': 1200,
                'use_scale': False
            }],
            remappings=[
                ('image', 'right/image_raw'),
                ('camera_info', 'right/camera_info'),
                ('image_rect', 'right/image_rect_oss')
            ]
        )]

    rectify_container = launch_ros.actions.ComposableNodeContainer(
        name='rectify_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=composable_nodes,
        output='screen',
        arguments=['--ros-args', '--log-level', 'info'],
    )

    return IsaacROSRectifyOSSTest.generate_test_description([rectify_container])


class IsaacROSRectifyOSSTest(IsaacROSBaseTest):
    """Validate that input timestamps match output timestamps."""

    filepath = pathlib.Path(os.path.dirname(__file__))

    @IsaacROSBaseTest.for_each_test_case(subfolder='rectify_stereo')
    def test_rectify_chessboard(self, test_folder) -> None:
        """Expect the output of OSS rectify and Isaac ROS rectify to be the similar."""
        self.generate_namespace_lookup(['left/image_raw', 'left/camera_info',
                                        'right/image_raw', 'right/camera_info',
                                        'left/image_rect', 'left/image_rect_oss',
                                        'right/image_rect', 'right/image_rect_oss'])
        left_image_raw_pub = self.node.create_publisher(
            Image, self.namespaces['left/image_raw'], self.DEFAULT_QOS)
        left_camera_info_pub = self.node.create_publisher(
            CameraInfo, self.namespaces['left/camera_info'], self.DEFAULT_QOS)
        right_image_raw_pub = self.node.create_publisher(
            Image, self.namespaces['right/image_raw'], self.DEFAULT_QOS)
        right_camera_info_pub = self.node.create_publisher(
            CameraInfo, self.namespaces['right/camera_info'], self.DEFAULT_QOS)

        received_messages = {}
        left_image_rect_sub, right_image_rect_sub, \
            left_oss_image_rect_sub, right_oss_image_rect_sub = \
            self.create_logging_subscribers([
                ('left/image_rect', Image),
                ('right/image_rect', Image),
                ('left/image_rect_oss', Image),
                ('right/image_rect_oss', Image),
            ], received_messages, accept_multiple_messages=True)
        try:
            image_raw_left, _ = JSONConversion.load_chessboard_image_from_json(
                test_folder / 'image_raw_left.json')
            camera_info_left = JSONConversion.load_camera_info_from_json(
                test_folder / 'camera_info_left.json')
            image_raw_right, _ = JSONConversion.load_chessboard_image_from_json(
                test_folder / 'image_raw_right.json')
            camera_info_right = JSONConversion.load_camera_info_from_json(
                test_folder / 'camera_info_right.json')

            # Wait at most TIMEOUT seconds for subscriber to respond
            TIMEOUT = 50
            end_time = time.time() + TIMEOUT

            done = False
            output_topics = ['left/image_rect', 'right/image_rect',
                             'left/image_rect_oss', 'right/image_rect_oss']

            # Synchronize timestamps on both messages
            timestamp = self.node.get_clock().now().to_msg()
            image_raw_left.header.stamp = timestamp
            camera_info_left.header.stamp = timestamp
            image_raw_right.header.stamp = timestamp
            camera_info_right.header.stamp = timestamp

            while time.time() < end_time:
                # Publish test case
                left_camera_info_pub.publish(camera_info_left)
                right_camera_info_pub.publish(camera_info_right)
                left_image_raw_pub.publish(image_raw_left)
                right_image_raw_pub.publish(image_raw_right)

                rclpy.spin_once(self.node, timeout_sec=0.2)
                # If we have received a message on the output topic, break
                if all(len(received_messages.get(topic, [])) > 0 for topic in output_topics):
                    done = True
                    break

            self.assertTrue(done,
                            "Didn't receive output on each topic!\n"
                            f'Expected messages on {len(output_topics)} topics:\n\t'
                            + '\n\t'.join(output_topics)
                            + '\nReceived messages on '
                            + f'{len(received_messages.keys())} topics: \n\t'
                            + '\n\t'.join(f'{key}: {len(received_messages[key])}'
                                          for key in received_messages.keys()))
            image_left_rect = \
                self.bridge.imgmsg_to_cv2(received_messages['left/image_rect'][0])
            image_right_rect = \
                self.bridge.imgmsg_to_cv2(received_messages['right/image_rect'][0])
            image_left_rect_oss = \
                self.bridge.imgmsg_to_cv2(received_messages['left/image_rect_oss'][0])
            image_right_rect_oss = \
                self.bridge.imgmsg_to_cv2(received_messages['right/image_rect_oss'][0])
            if (VISUALIZE):
                cv2.imwrite('image_left_rect.png', image_left_rect)
                cv2.imwrite('image_right_rect.png', image_right_rect)
                cv2.imwrite('image_left_rect_oss.png', image_left_rect_oss)
                cv2.imwrite('image_right_rect_oss.png', image_right_rect_oss)
                cv2.imwrite('image_left_diff.png',
                            cv2.absdiff(image_left_rect, image_left_rect_oss))
                cv2.imwrite('image_right_diff.png',
                            cv2.absdiff(image_right_rect, image_right_rect_oss))
            self.assertImagesEqual(image_left_rect, image_left_rect_oss, 0.00001)
            self.assertImagesEqual(image_right_rect, image_right_rect_oss, 0.00001)
        finally:
            self.assertTrue(self.node.destroy_subscription(left_image_rect_sub))
            self.assertTrue(self.node.destroy_subscription(right_image_rect_sub))
            self.assertTrue(self.node.destroy_subscription(left_oss_image_rect_sub))
            self.assertTrue(self.node.destroy_subscription(right_oss_image_rect_sub))
            self.assertTrue(self.node.destroy_publisher(left_image_raw_pub))
            self.assertTrue(self.node.destroy_publisher(left_camera_info_pub))
            self.assertTrue(self.node.destroy_publisher(right_image_raw_pub))
            self.assertTrue(self.node.destroy_publisher(right_camera_info_pub))
