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

"""Proof-of-Life test for the Isaac ROS Rectify node."""

import os
import pathlib
import time

# Need to load sklearn first on Jetson.
from sklearn.linear_model import LinearRegression
import cv2  # noqa: I100
from isaac_ros_test import IsaacROSBaseTest, JSONConversion
import launch_ros
import pytest
import rclpy
from sensor_msgs.msg import CameraInfo, Image


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with all ROS 2 nodes for testing."""
    composable_nodes = [
        launch_ros.descriptions.ComposableNode(
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::RectifyNode',
            name='rectify_node',
            namespace=IsaacROSRectifyTest.generate_namespace(),
            remappings=[('image', 'image_raw')],
            parameters=[{
                'output_width': 1280,
                'output_height': 800,
            }]
        )]

    rectify_container = launch_ros.actions.ComposableNodeContainer(
        name='rectify_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=composable_nodes,
        output='screen',
        arguments=['--ros-args', '--log-level', 'info'],
    )

    return IsaacROSRectifyTest.generate_test_description([rectify_container])


class IsaacROSRectifyTest(IsaacROSBaseTest):
    """Validate image rectification with chessboard images."""

    filepath = pathlib.Path(os.path.dirname(__file__))

    @IsaacROSBaseTest.for_each_test_case(subfolder='rectify')
    def test_rectify_chessboard(self, test_folder) -> None:
        """Expect the node to rectify chessboard images."""
        self.generate_namespace_lookup(['image_raw', 'camera_info', 'image_rect'])

        image_raw_pub = self.node.create_publisher(
            Image, self.namespaces['image_raw'], self.DEFAULT_QOS)
        camera_info_pub = self.node.create_publisher(
            CameraInfo, self.namespaces['camera_info'], self.DEFAULT_QOS)

        received_messages = {}
        image_rect_sub, = self.create_logging_subscribers(
            [('image_rect', Image)], received_messages)

        try:
            image_raw, chessboard_dims = JSONConversion.load_chessboard_image_from_json(
                test_folder / 'image_raw.json')
            camera_info = JSONConversion.load_camera_info_from_json(
                test_folder / 'camera_info.json')

            # Wait at most TIMEOUT seconds for subscriber to respond
            TIMEOUT = 2
            end_time = time.time() + TIMEOUT

            done = False
            while time.time() < end_time:
                # Synchronize timestamps on both messages
                timestamp = self.node.get_clock().now().to_msg()
                image_raw.header.stamp = timestamp
                camera_info.header.stamp = timestamp

                # Publish test case over both topics
                image_raw_pub.publish(image_raw)
                camera_info_pub.publish(camera_info)

                rclpy.spin_once(self.node, timeout_sec=0.1)

                # If we have received a message on the output topic, break
                if 'image_rect' in received_messages:
                    done = True
                    break

            self.assertTrue(done, "Didn't receive output on image_rect topic!")

            # Collect received image
            image_rect = self.bridge.imgmsg_to_cv2(received_messages['image_rect'])

            # Convert to grayscale and find chessboard corners in rectified image
            image_rect_gray = cv2.cvtColor(image_rect, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(image_rect_gray, chessboard_dims, None)
            self.assertTrue(ret, "Couldn't find chessboard corners in output image!")

            # Make sure that each row of chessboard in rectified image is close to a straight line
            # Since the chessboard image is initially very distorted, this check confirms that the
            # image was actually rectified
            R_SQUARED_THRESHOLD = 0.81  # Allows R = 0.9
            for i in range(chessboard_dims[1]):
                row = corners[chessboard_dims[0]*i:chessboard_dims[0]*(i+1)]
                x, y = row[:, :, 0], row[:, :, 1]
                reg = LinearRegression().fit(x, y)
                self.assertGreaterEqual(
                    reg.score(x, y), R_SQUARED_THRESHOLD,
                    f'Rectified chessboard corners for row {i} were not in a straight line!')

        finally:
            self.assertTrue(self.node.destroy_subscription(image_rect_sub))
            self.assertTrue(self.node.destroy_publisher(image_raw_pub))
            self.assertTrue(self.node.destroy_publisher(camera_info_pub))
