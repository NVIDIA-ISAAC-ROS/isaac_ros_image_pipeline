# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Proof-of-Life test for the Isaac ROS Resize node."""

import os
import pathlib
import time

from isaac_ros_test import IsaacROSBaseTest, JSONConversion
import launch_ros
import pytest
import rclpy
from sensor_msgs.msg import CameraInfo, Image

USE_RELATIVE_SCALE = False
SCALE_HEIGHT = 2.0
SCALE_WIDTH = 2.0
HEIGHT = 20
WIDTH = 20
BACKENDS = 'CUDA'


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with all ROS2 nodes for testing."""
    composable_nodes = [
        launch_ros.descriptions.ComposableNode(
            package='isaac_ros_image_proc',
            plugin='isaac_ros::image_proc::ResizeNode',
            name='resize_node',
            namespace=IsaacROSResizeTest.generate_namespace(),
            parameters=[{
                    'use_relative_scale': USE_RELATIVE_SCALE,
                    'scale_height': SCALE_HEIGHT,
                    'scale_width': SCALE_WIDTH,
                    'height': HEIGHT,
                    'width': WIDTH,
                    'backends': BACKENDS,
            }])]

    resize_container = launch_ros.actions.ComposableNodeContainer(
        package='rclcpp_components',
        name='resize_container',
        namespace='',
        executable='component_container',
        composable_node_descriptions=composable_nodes,
        output='screen'
    )

    return IsaacROSResizeTest.generate_test_description([resize_container])


class IsaacROSResizeTest(IsaacROSBaseTest):
    """Validate image resizing in typical case."""

    filepath = pathlib.Path(os.path.dirname(__file__))

    @IsaacROSBaseTest.for_each_test_case(subfolder='resize')
    def test_resize_typical(self, test_folder) -> None:
        """Expect the node to output images with correctly resized dimensions."""
        self.generate_namespace_lookup(['image', 'camera_info', 'resized/image'])
        received_messages = {}

        resized_image_sub = self.create_logging_subscribers(
            subscription_requests=[('resized/image', Image)],
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

            # Publish test case over both topics
            image_pub.publish(image)
            camera_info_pub.publish(camera_info)

            # Wait at most TIMEOUT seconds for subscriber to respond
            TIMEOUT = 2
            end_time = time.time() + TIMEOUT

            done = False
            while time.time() < end_time:
                rclpy.spin_once(self.node, timeout_sec=TIMEOUT)

                # If we have received a message on the output topic, break
                if 'resized/image' in received_messages:
                    done = True
                    break

            self.assertTrue(done, "Didn't receive output on resized/image topic!")

            resized_image = received_messages['resized/image']

            # Make sure that the output image size is set to desired dimensions
            desired_height = image.height * SCALE_HEIGHT if USE_RELATIVE_SCALE else HEIGHT
            desired_width = image.width * SCALE_WIDTH if USE_RELATIVE_SCALE else WIDTH
            self.assertEqual(resized_image.height, desired_height,
                             f'Height is not {desired_height}!')
            self.assertEqual(resized_image.width, desired_width, f'Width is not {desired_width}!')

        finally:
            self.node.destroy_subscription(resized_image_sub)
            self.node.destroy_publisher(image_pub)
            self.node.destroy_publisher(camera_info_pub)
