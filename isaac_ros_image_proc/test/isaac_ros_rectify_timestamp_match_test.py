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

"""TimeStamp match test for the Isaac ROS Rectify node."""

import os
import pathlib
import time

# Need to load sklearn first on Jetson.
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
        executable='component_container_mt',
        composable_node_descriptions=composable_nodes,
        output='screen',
        arguments=['--ros-args', '--log-level', 'info'],
    )

    return IsaacROSRectifyTest.generate_test_description([rectify_container])


class IsaacROSRectifyTest(IsaacROSBaseTest):
    """Validate that input timestamps match output timestamps."""

    filepath = pathlib.Path(os.path.dirname(__file__))
    input_timestamps = []

    @IsaacROSBaseTest.for_each_test_case(subfolder='rectify')
    def test_rectify_chessboard(self, test_folder) -> None:
        """
        Expect the node to accept matched time stamps(image and camera_info).

        And generate matched time stamps outputs(image and camera_info).
        """
        self.generate_namespace_lookup(['image_raw', 'camera_info',
                                        'image_rect', 'camera_info_rect'])

        image_raw_pub = self.node.create_publisher(
            Image, self.namespaces['image_raw'], self.DEFAULT_QOS)
        camera_info_pub = self.node.create_publisher(
            CameraInfo, self.namespaces['camera_info'], self.DEFAULT_QOS)

        received_messages = []
        self.create_exact_time_sync_logging_subscribers(
            [('image_rect', Image), ('camera_info_rect', CameraInfo)],
            received_messages, accept_multiple_messages=True)

        try:
            image_raw = JSONConversion.load_image_from_json(
                test_folder / 'image_raw.json')
            camera_info = JSONConversion.load_camera_info_from_json(
                test_folder / 'camera_info.json')

            # Wait at most TIMEOUT seconds for subscriber to respond
            TIMEOUT = 10
            end_time = time.time() + TIMEOUT

            done = False

            while time.time() < end_time:
                # Synchronize timestamps on both input messages
                timestamp = self.node.get_clock().now().to_msg()
                image_raw.header.stamp = timestamp
                camera_info.header.stamp = timestamp
                self.input_timestamps.append(timestamp)

                # Publish test case over both topics
                image_raw_pub.publish(image_raw)
                camera_info_pub.publish(camera_info)

                rclpy.spin_once(self.node, timeout_sec=0.5)

                # If we have received at least one synchronized message output, break
                if len(received_messages) > 0:
                    done = True
                    break

            self.assertTrue(done, "Didn't receive time synced output!")
            for received_message in received_messages:
                self.assertTrue(received_message[0].header.stamp ==
                                received_message[1].header.stamp,
                                'Timestamps are not synced!')
                self.assertTrue(received_message[0].header.stamp in self.input_timestamps,
                                'Rectified image timestamp not in list of input timestamps!')
                self.assertTrue(received_message[1].header.stamp in self.input_timestamps,
                                'Rectified camera info timestamp not in list of \
                                input timestamps!')
            self.assertTrue(done, "Didn't receive time synced output!")

        finally:
            self.assertTrue(self.node.destroy_publisher(image_raw_pub))
            self.assertTrue(self.node.destroy_publisher(camera_info_pub))
