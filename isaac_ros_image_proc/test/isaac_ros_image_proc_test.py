# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Proof-of-Life test for the precompiled Isaac ROS Image Proc executable."""

import os
import pathlib
import time

import cv2
from isaac_ros_test import IsaacROSBaseTest, JSONConversion
import launch_ros
import pytest
import rclpy
from sensor_msgs.msg import CameraInfo, Image


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with all ROS2 nodes for testing."""
    nodes = [
        launch_ros.actions.Node(
            package='isaac_ros_image_proc',
            executable='isaac_ros_image_proc',
            name='isaac_ros_image_proc',
            namespace=IsaacROSImageProcTest.generate_namespace()
        )]

    return IsaacROSImageProcTest.generate_test_description(nodes)


class IsaacROSImageProcTest(IsaacROSBaseTest):
    """Validate standard image proc results with sample images."""

    filepath = pathlib.Path(os.path.dirname(__file__))

    @IsaacROSBaseTest.for_each_test_case(subfolder='image_proc')
    def test_image_proc(self, test_folder) -> None:
        """Expect the node to forward images from input to all output topics."""
        self.generate_namespace_lookup([
            'image_raw', 'camera_info',
            'image_mono', 'image_rect',
            'image_color', 'image_rect_color'
        ])

        image_raw_pub = self.node.create_publisher(
            Image, self.namespaces['image_raw'], self.DEFAULT_QOS)
        camera_info_pub = self.node.create_publisher(
            CameraInfo, self.namespaces['camera_info'], self.DEFAULT_QOS)

        received_messages = {}
        image_mono_sub, image_rect_sub, image_color_sub, image_rect_color_sub = \
            self.create_logging_subscribers([
                ('image_mono', Image),
                ('image_rect', Image),
                ('image_color', Image),
                ('image_rect_color', Image),
            ], received_messages)

        try:
            timestamp = self.node.get_clock().now().to_msg()
            image_raw = JSONConversion.load_image_from_json(test_folder / 'image_raw.json')
            image_raw.header.stamp = timestamp
            camera_info = JSONConversion.load_camera_info_from_json(
                test_folder / 'camera_info.json')
            camera_info.header.stamp = timestamp

            # Publish test case over both topics
            image_raw_pub.publish(image_raw)
            camera_info_pub.publish(camera_info)

            # Wait at most TIMEOUT seconds for subscriber to respond
            TIMEOUT = 2
            end_time = time.time() + TIMEOUT

            done = False
            output_topics = ['image_mono', 'image_rect', 'image_color', 'image_rect_color']
            while time.time() < end_time:
                rclpy.spin_once(self.node, timeout_sec=TIMEOUT)

                # If we have received exactly one message on each output topic, break
                if all([topic in received_messages for topic in output_topics]):
                    done = True
                    break

            self.assertTrue(done,
                            "Didn't receive output on each topic!\n"
                            'Expected messages on:\n'
                            + '\n'.join(output_topics)
                            + '\nReceived messages on: \n'
                            + '\n'.join(received_messages.keys()))

            # Collect received images and compare to baseline
            image_mono_actual = self.bridge.imgmsg_to_cv2(received_messages['image_mono'])
            image_rect_actual = self.bridge.imgmsg_to_cv2(received_messages['image_rect'])
            image_color_actual = self.bridge.imgmsg_to_cv2(received_messages['image_color'])
            image_rect_color_actual = self.bridge.imgmsg_to_cv2(
                received_messages['image_rect_color'])

            image_mono_expected = cv2.imread(
                str(test_folder / 'image_mono.jpg'), cv2.IMREAD_GRAYSCALE)
            image_rect_expected = cv2.imread(
                str(test_folder / 'image_rect.jpg'), cv2.IMREAD_GRAYSCALE)
            image_color_expected = cv2.imread(str(test_folder / 'image_color.jpg'))
            image_rect_color_expected = cv2.imread(str(test_folder / 'image_rect_color.jpg'))

            self.assertImagesEqual(image_mono_actual, image_mono_expected)
            self.assertImagesEqual(image_rect_actual, image_rect_expected)
            self.assertImagesEqual(image_color_actual, image_color_expected)
            self.assertImagesEqual(image_rect_color_actual, image_rect_color_expected)

        finally:
            self.assertTrue(self.node.destroy_subscription(image_mono_sub))
            self.assertTrue(self.node.destroy_subscription(image_rect_sub))
            self.assertTrue(self.node.destroy_subscription(image_color_sub))
            self.assertTrue(self.node.destroy_subscription(image_rect_color_sub))
            self.assertTrue(self.node.destroy_publisher(image_raw_pub))
            self.assertTrue(self.node.destroy_publisher(camera_info_pub))
