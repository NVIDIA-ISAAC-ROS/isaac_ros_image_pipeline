# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Edge case test for the Isaac ROS Resize node."""

import os
import pathlib
import time

from isaac_ros_test import IsaacROSBaseTest, JSONConversion
import launch_ros
import pytest
from rcl_interfaces.msg import Log
import rclpy
from rclpy.logging import LoggingSeverity
from sensor_msgs.msg import CameraInfo, Image

USE_RELATIVE_SCALE = True
SCALE_HEIGHT = -2.0
SCALE_WIDTH = -3.0
HEIGHT = -20
WIDTH = -20
BACKENDS = 'CUDA'


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with all ROS2 nodes for testing."""
    composable_nodes = [
        launch_ros.descriptions.ComposableNode(
            package='isaac_ros_image_proc',
            plugin='isaac_ros::image_proc::ResizeNode',
            name='resize_node',
            namespace=IsaacROSResizeInvalidTest.generate_namespace(),
            parameters=[{
                    'use_relative_scale': USE_RELATIVE_SCALE,
                    'scale_height': SCALE_HEIGHT,
                    'scale_width': SCALE_WIDTH,
                    'height': HEIGHT,
                    'width': WIDTH,
                    'backends': BACKENDS,
            }])]

    resize_container = launch_ros.actions.ComposableNodeContainer(
        name='resize_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=composable_nodes,
        output='screen'
    )

    return IsaacROSResizeInvalidTest.generate_test_description([resize_container])


class IsaacROSResizeInvalidTest(IsaacROSBaseTest):
    """Validate error-catching behavior with invalid numbers."""

    filepath = pathlib.Path(os.path.dirname(__file__))

    @IsaacROSBaseTest.for_each_test_case(subfolder='resize')
    def test_resize_invalid(self, test_folder) -> None:
        """Expect the node to log an error when given invalid input."""
        self.generate_namespace_lookup(['image', 'camera_info', 'resized/image'])
        received_messages = {}

        resized_image_sub, rosout_sub = self.create_logging_subscribers(
            subscription_requests=[(self.namespaces['resized/image'], Image), ('/rosout', Log)],
            use_namespace_lookup=False,
            received_messages=received_messages,
            accept_multiple_messages=True
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

            while time.time() < end_time:
                # Synchronize timestamps on both messages
                timestamp = self.node.get_clock().now().to_msg()
                image.header.stamp = timestamp
                camera_info.header.stamp = timestamp

                # Publish test case over both topics
                image_pub.publish(image)
                camera_info_pub.publish(camera_info)

                rclpy.spin_once(self.node, timeout_sec=0.1)

            self.assertIn('/rosout', received_messages, "Didn't receive output on /rosout topic!")

            # Make sure that at least one output log message is a non-empty error
            self.assertTrue(any([
                LoggingSeverity(rosout.level) == LoggingSeverity.ERROR and len(rosout.msg) > 0
                for rosout in received_messages['/rosout']]),
                'No message with non-empty message and Error severity!')

            # Make sure no output image was received in the error case
            self.assertEqual(
                len(received_messages[self.namespaces['resized/image']]), 0,
                'Resized image was received despite error!')

        finally:
            self.node.destroy_subscription(resized_image_sub)
            self.node.destroy_subscription(rosout_sub)
            self.node.destroy_subscription(rosout_sub)
            self.node.destroy_publisher(image_pub)
            self.node.destroy_publisher(camera_info_pub)
