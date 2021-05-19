# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Proof-of-Life test for the Isaac ROS Rectify node."""

import os
import pathlib
import time

import cv2
from isaac_ros_test import IsaacROSBaseTest, JSONConversion
import launch_ros
import pytest
import rclpy
from sensor_msgs.msg import CameraInfo, Image
from sklearn.linear_model import LinearRegression


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with all ROS2 nodes for testing."""
    composable_nodes = [
        launch_ros.descriptions.ComposableNode(
            package='isaac_ros_image_proc',
            plugin='isaac_ros::image_proc::RectifyNode',
            name='rectify_node',
            namespace=IsaacROSRectifyTest.generate_namespace(),
            remappings=[('image', 'image_raw')]
        )]

    rectify_container = launch_ros.actions.ComposableNodeContainer(
        name='rectify_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=composable_nodes,
        output='screen'
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

            # Publish test case over both topics
            image_raw_pub.publish(image_raw)
            camera_info_pub.publish(camera_info)

            # Wait at most TIMEOUT seconds for subscriber to respond
            TIMEOUT = 2
            end_time = time.time() + TIMEOUT

            done = False
            while time.time() < end_time:
                rclpy.spin_once(self.node, timeout_sec=TIMEOUT)

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
