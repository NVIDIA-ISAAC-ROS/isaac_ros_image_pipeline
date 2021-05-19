# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Grayscale conversion test for the Isaac ROS Image Format Converter node."""

import time

import cv2
from cv_bridge import CvBridge
from isaac_ros_test import IsaacROSBaseTest
import launch_ros
import numpy as np
import pytest
import rclpy
from sensor_msgs.msg import CameraInfo, Image

ENCODING_DESIRED = 'mono8'
BACKENDS = 'CUDA'


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with all ROS2 nodes for testing."""
    composable_nodes = [
        launch_ros.descriptions.ComposableNode(
            package='isaac_ros_image_proc',
            plugin='isaac_ros::image_proc::ImageFormatConverterNode',
            name='image_format_node',
            namespace=IsaacROSFormatMono8Test.generate_namespace(),
            parameters=[{
                    'encoding_desired': ENCODING_DESIRED,
                    'backends': BACKENDS
            }])]

    format_container = launch_ros.actions.ComposableNodeContainer(
        name='format_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=composable_nodes,
        output='screen'
    )

    return IsaacROSFormatMono8Test.generate_test_description([format_container])


class IsaacROSFormatMono8Test(IsaacROSBaseTest):
    """Vaidate format conversion to the mono8 format."""

    def test_rgba_to_mono_conversion(self) -> None:
        """Expect the node to convert rgba8 input images into the mono8 format."""
        self.generate_namespace_lookup(['image_raw', 'image'])
        received_messages = {}

        image_sub, = self.create_logging_subscribers(
            subscription_requests=[('image', Image)],
            received_messages=received_messages
        )

        image_raw_pub = self.node.create_publisher(
            Image, self.generate_namespace('image_raw'), self.DEFAULT_QOS)

        try:
            # Generate an input image in RGBA encoding
            cv_image = np.zeros((300, 300, 4), np.uint8)
            cv_image[:] = (255, 0, 0, 200)  # Full red, partial opacity

            timestamp = self.node.get_clock().now().to_msg()
            image_raw = CvBridge().cv2_to_imgmsg(cv_image)
            image_raw.header.stamp = timestamp
            image_raw.encoding = 'rgba8'  # Set image encoding explicitly
            camera_info = CameraInfo()
            camera_info.header.stamp = timestamp
            # Arbitrary valid K matrix that will satisfy calibration check
            camera_info.k = np.eye(3).reshape(9)

            # Publish test case over single topic
            image_raw_pub.publish(image_raw)

            # Wait at most TIMEOUT seconds for subscriber to respond
            TIMEOUT = 2
            end_time = time.time() + TIMEOUT

            done = False
            while time.time() < end_time:
                rclpy.spin_once(self.node, timeout_sec=TIMEOUT)

                # If we have received a message on the output topic, break
                if 'image' in received_messages:
                    done = True
                    break

            self.assertTrue(done, "Didn't receive output on image topic!")

            image = received_messages['image']

            # Make sure that the output image has the desired encoding
            self.assertEqual(image.encoding, ENCODING_DESIRED, 'Incorrect output encoding!')

            # Make sure output image pixels match OpenCV result
            image_mono_actual = CvBridge().imgmsg_to_cv2(image)
            image_mono_expected = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2GRAY)
            self.assertImagesEqual(image_mono_actual, image_mono_expected)

        finally:
            self.node.destroy_subscription(image_sub)
            self.node.destroy_publisher(image_raw_pub)
