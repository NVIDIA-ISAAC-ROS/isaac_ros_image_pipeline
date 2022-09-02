# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""3-channel color conversion test for the Isaac ROS Image Format Converter node."""

import os
import pathlib
import time

import cv2
from cv_bridge import CvBridge
from isaac_ros_test import IsaacROSBaseTest, JSONConversion
import launch_ros
import pytest
import rclpy
from sensor_msgs.msg import Image


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with all ROS2 nodes for testing."""
    composable_nodes = [
        launch_ros.descriptions.ComposableNode(
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::ImageFlipNode',
            name='image_flip_node',
            namespace=IsaacROSImageFlipTest.generate_namespace(),
            parameters=[{
                    'flip_mode': 'VERTICAL',
            }])]

    format_container = launch_ros.actions.ComposableNodeContainer(
        name='format_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=composable_nodes,
        output='screen'
    )

    return IsaacROSImageFlipTest.generate_test_description([format_container])


class IsaacROSImageFlipTest(IsaacROSBaseTest):
    """Vaidate image flipping."""

    filepath = pathlib.Path(os.path.dirname(__file__))

    @IsaacROSBaseTest.for_each_test_case(subfolder='image_flip')
    def test_image_flip_vertical(self, test_folder) -> None:
        """Expect the node to flip input images vertically."""
        self.generate_namespace_lookup(['image', 'image_flipped'])
        received_messages = {}

        image_sub, = self.create_logging_subscribers(
            subscription_requests=[('image_flipped', Image)],
            received_messages=received_messages
        )

        image_pub = self.node.create_publisher(
            Image, self.generate_namespace('image'), self.DEFAULT_QOS)

        try:
            image = JSONConversion.load_image_from_json(
                test_folder / 'image.json')
            # Wait at most TIMEOUT seconds for subscriber to respond
            TIMEOUT = 2
            end_time = time.time() + TIMEOUT

            done = False
            while time.time() < end_time:
                timestamp = self.node.get_clock().now().to_msg()
                image.header.stamp = timestamp

                image_pub.publish(image)
                rclpy.spin_once(self.node, timeout_sec=0.1)

                # If we have received a message on the output topic, break
                if 'image_flipped' in received_messages:
                    done = True
                    break

            self.assertTrue(done, "Didn't receive output on image_flipped topic!")

            flipped_image = received_messages['image_flipped']
            flipped_image_cv = CvBridge().imgmsg_to_cv2(flipped_image, desired_encoding='bgr8')
            groundtruth_image_cv = cv2.imread(str(test_folder / 'flipped_image_vertical.png'))
            self.assertImagesEqual(
                flipped_image_cv, groundtruth_image_cv, threshold_fraction=0)

        finally:
            self.node.destroy_subscription(image_sub)
            self.node.destroy_publisher(image_pub)
