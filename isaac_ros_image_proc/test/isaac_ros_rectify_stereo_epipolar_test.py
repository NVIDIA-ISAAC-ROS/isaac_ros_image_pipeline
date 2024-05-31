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

"""Epipolar line validation test for the Isaac ROS Rectify node."""

import os
import pathlib
import random
import time

import cv2  # noqa: I100
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
            namespace=IsaacROSStereoRectifyEpipolarTest.generate_namespace(),
            parameters=[{
                'output_width': 1920,
                'output_height': 1200
            }],
            remappings=[
                ('image_raw', 'left_image_raw'),
                ('camera_info', 'left_camera_info'),
                ('image_rect', 'left_image_rect')
            ]
        ),
        launch_ros.descriptions.ComposableNode(
            name='right_rectify_node',
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::RectifyNode',
            namespace=IsaacROSStereoRectifyEpipolarTest.generate_namespace(),
            parameters=[{
                'output_width': 1920,
                'output_height': 1200,
            }],
            remappings=[
                ('image_raw', 'right_image_raw'),
                ('camera_info', 'right_camera_info'),
                ('image_rect', 'right_image_rect')
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

    return IsaacROSStereoRectifyEpipolarTest.generate_test_description([rectify_container])


class IsaacROSStereoRectifyEpipolarTest(IsaacROSBaseTest):
    """Validate stereo image rectification with chessboard images."""

    filepath = pathlib.Path(os.path.dirname(__file__))

    @IsaacROSBaseTest.for_each_test_case(subfolder='rectify_stereo')
    def test_rectify_chessboard(self, test_folder) -> None:
        """Expect the node to rectify chessboard images."""
        self.generate_namespace_lookup(['left_image_raw', 'left_camera_info',
                                        'right_image_raw', 'right_camera_info'])
        left_image_raw_pub = self.node.create_publisher(
            Image, self.namespaces['left_image_raw'], self.DEFAULT_QOS)
        left_camera_info_pub = self.node.create_publisher(
            CameraInfo, self.namespaces['left_camera_info'], self.DEFAULT_QOS)
        right_image_raw_pub = self.node.create_publisher(
            Image, self.namespaces['right_image_raw'], self.DEFAULT_QOS)
        right_camera_info_pub = self.node.create_publisher(
            CameraInfo, self.namespaces['right_camera_info'], self.DEFAULT_QOS)

        received_messages = []
        self.create_exact_time_sync_logging_subscribers(
                [('left_image_rect', Image), ('right_image_rect', Image)], received_messages,
                accept_multiple_messages=True)

        image_raw_left, left_chessboard_dims = JSONConversion.load_chessboard_image_from_json(
            test_folder / 'image_raw_left.json')
        camera_info_left = JSONConversion.load_camera_info_from_json(
            test_folder / 'camera_info_left.json')
        image_raw_right, right_chessboard_dims = JSONConversion.load_chessboard_image_from_json(
            test_folder / 'image_raw_right.json')
        camera_info_right = JSONConversion.load_camera_info_from_json(
            test_folder / 'camera_info_right.json')

        # Wait at most TIMEOUT seconds for subscriber to respond
        TIMEOUT = 20
        end_time = time.time() + TIMEOUT

        done = False
        # Synchronize timestamps on both messages
        timestamp = self.node.get_clock().now().to_msg()
        image_raw_left.header.stamp = timestamp
        camera_info_left.header.stamp = timestamp
        image_raw_right.header.stamp = timestamp
        camera_info_right.header.stamp = timestamp

        while time.time() < end_time:
            # Publish test case
            left_image_raw_pub.publish(image_raw_left)
            left_camera_info_pub.publish(camera_info_left)
            right_image_raw_pub.publish(image_raw_right)
            right_camera_info_pub.publish(camera_info_right)

            rclpy.spin_once(self.node, timeout_sec=0.2)

            # If we have received a message on the output topic, break
            if len(received_messages) > 0:
                done = True
                break

        self.assertTrue(done, "Didn't receive output on image_rect topics!")

        for received_message in received_messages:
            # Collect received image
            left_image_rect = self.bridge.imgmsg_to_cv2(received_message[0])
            right_image_rect = self.bridge.imgmsg_to_cv2(received_message[1])
            # Convert to grayscale and find chessboard corners in rectified image
            left_image_rect_gray = cv2.cvtColor(left_image_rect, cv2.COLOR_BGR2GRAY)
            left_ret, left_corners = cv2.findChessboardCorners(left_image_rect_gray,
                                                               left_chessboard_dims, None)
            self.assertTrue(left_ret, "Couldn't find chessboard corners in output left image!")
            right_image_rect_gray = cv2.cvtColor(right_image_rect, cv2.COLOR_BGR2GRAY)
            right_ret, right_corners = cv2.findChessboardCorners(right_image_rect_gray,
                                                                 right_chessboard_dims, None)
            self.assertTrue(right_ret, "Couldn't find chessboard corners in output right image!")
            # Extract the x and y coordinates of the corners in left_corners and right_corners
            x_coords_left = [c[0][0] for c in left_corners]
            y_coords_left = [c[0][1] for c in left_corners]
            x_coords_right = [c[0][0] for c in right_corners]
            y_coords_right = [c[0][1] for c in right_corners]
            if (VISUALIZE):
                # Draw lines of the same color at the avergae row value for all corners
                # in the left and right image
                cv2.drawChessboardCorners(left_image_rect, left_chessboard_dims,
                                          left_corners, left_ret)
                cv2.drawChessboardCorners(right_image_rect, right_chessboard_dims,
                                          right_corners, right_ret)
                # Draw randomly colored lines connecting the corresponding corners
                for i in range(min(len(left_corners), len(right_corners))):
                    average_y = (y_coords_left[i]+y_coords_right[i])/2
                    pt1 = (0, int(average_y))
                    pt2 = (left_image_rect.shape[1], int(average_y))
                    random_color = (random.randint(0, 255),
                                    random.randint(0, 255),
                                    random.randint(0, 255))
                    cv2.line(left_image_rect, pt1, pt2, random_color, 1)
                    cv2.line(right_image_rect, pt1, pt2, random_color, 1)
                cv2.imwrite('left_image_rect.png', left_image_rect)
                cv2.imwrite('right_image_rect.png', right_image_rect)

            """
            Test 1:
            Check if the row values of the same corners of the chessboard
            in the left and right images are within threshold
            """
            # Compute the differences in row values
            row_diffs = [abs(y_coords_left[i] - y_coords_right[i])
                         for i in range(min(len(left_corners), len(right_corners)))]
            # Allows test pass if same features are within of 4 pixels in both images
            CORNER_ROW_DIFF_THRESHOLD = 4
            if (VISUALIZE):
                print('CORNER_ROW_DIFF_THRESHOLD :')
                print(CORNER_ROW_DIFF_THRESHOLD)
                print('row_diffs :')
                print(row_diffs)
            self.assertFalse(
                any(diff > CORNER_ROW_DIFF_THRESHOLD for diff in row_diffs),
                'Difference between corners row values in left and right images'
                'are not within threshold')

            """
            Test 2:
            Check if the slopes of the lines connecting each corresponding corners in
            the left and right images are within threshold from the mean.
            This test checks if these lines(epipolar lines) are parallel
            """
            # Compute the slopes of lines between corresponding corners
            slopes = [(y_coords_right[i]-y_coords_left[i])/(x_coords_right[i]-x_coords_left[i])
                      for i in range(min(len(left_corners), len(right_corners)))]
            # Create a list of difference betwen the slope and average slope
            mean_slope_diffs = slopes-(sum(slopes) / len(slopes))
            EPIPOLAR_LINES_SLOPE_DIFF_THRESHOLD = 0.005
            if (VISUALIZE):
                print('EPIPOLAR_LINES_SLOPE_DIFF_THRESHOLD :')
                print(EPIPOLAR_LINES_SLOPE_DIFF_THRESHOLD)
                print('mean_slope_diffs :')
                print(mean_slope_diffs)
            self.assertFalse(
                any(mean_slope_diff > EPIPOLAR_LINES_SLOPE_DIFF_THRESHOLD
                    for mean_slope_diff in mean_slope_diffs),
                'Epipolar lines are not parallel!')
