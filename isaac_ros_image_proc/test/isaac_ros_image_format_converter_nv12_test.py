# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""NV12 format conversion tests for Isaac ROS Image Format Converter, validated vs OpenCV."""

import time

import cv2
from isaac_ros_test import IsaacROSBaseTest
import launch_ros
import numpy as np
import pytest
import rclpy
from sensor_msgs.msg import Image

HEIGHT = 300
WIDTH = 300


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with all ROS 2 nodes for testing."""
    namespace = IsaacROSFormatNV12Test.generate_namespace()
    composable_nodes = [
        launch_ros.descriptions.ComposableNode(
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
            name='nv12_to_rgb8',
            namespace=namespace,
            parameters=[{
                'encoding_desired': 'rgb8',
                'image_width': WIDTH,
                'image_height': HEIGHT,
                # Default is 'bt601'; state it explicitly here to document
                # the pairing with the OpenCV ground truth below
                # (cv2.COLOR_YUV2RGB_NV12 uses BT.601 limited range).
                'yuv_color_spec': 'bt601',
            }],
            remappings=[
                ('image_raw', 'image_raw'),
                ('image', 'image_rgb8'),
            ],
        ),
        launch_ros.descriptions.ComposableNode(
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
            name='nv12_to_mono8',
            namespace=namespace,
            parameters=[{
                'encoding_desired': 'mono8',
                'image_width': WIDTH,
                'image_height': HEIGHT,
            }],
            remappings=[
                ('image_raw', 'image_raw'),
                ('image', 'image_mono8'),
            ],
        ),
        launch_ros.descriptions.ComposableNode(
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
            name='nv12_to_rgb8_bt709',
            namespace=namespace,
            parameters=[{
                'encoding_desired': 'rgb8',
                'image_width': WIDTH,
                'image_height': HEIGHT,
                # Non-default spec: exercises the yuv_color_spec plumbing.
                # CV-CUDA AdvCvtColor only supports limited-range BT.709.
                'yuv_color_spec': 'bt709',
            }],
            remappings=[
                ('image_raw', 'image_raw_bt709'),
                ('image', 'image_rgb8_bt709'),
            ],
        ),
    ]

    format_container = launch_ros.actions.ComposableNodeContainer(
        name='format_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=composable_nodes,
        output='screen'
    )

    return IsaacROSFormatNV12Test.generate_test_description([format_container])


class IsaacROSFormatNV12Test(IsaacROSBaseTest):
    """Validate NV12 format conversions (RGB8, MONO8)."""

    @staticmethod
    def _create_nv12_image(rgb_image):
        """Convert an RGB image to NV12 and pack into a sensor_msgs/Image."""
        yuv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YUV_I420)
        h, w = rgb_image.shape[:2]

        y_plane = yuv[:h, :]
        u_plane = yuv[h:h + h // 4].reshape(h // 2, w // 2)
        v_plane = yuv[h + h // 4:].reshape(h // 2, w // 2)
        uv_interleaved = np.empty((h // 2, w), dtype=np.uint8)
        uv_interleaved[:, 0::2] = u_plane
        uv_interleaved[:, 1::2] = v_plane

        nv12 = np.vstack([y_plane, uv_interleaved])

        msg = Image()
        msg.height = h
        msg.width = w
        msg.encoding = 'nv12'
        msg.step = w
        msg.is_bigendian = 0
        msg.data = nv12.tobytes()
        return msg

    def _run_nv12_conversion(self, output_topic, expected_encoding, cv_code, out_channels):
        """Publish NV12, receive on output_topic, compare to OpenCV ground truth."""
        self.generate_namespace_lookup(['image_raw', output_topic])
        received_messages = {}

        image_sub, = self.create_logging_subscribers(
            subscription_requests=[(output_topic, Image)],
            received_messages=received_messages)

        image_raw_pub = self.node.create_publisher(
            Image, self.generate_namespace('image_raw'), self.DEFAULT_QOS)

        try:
            rgb_image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
            rgb_image[:, :] = (255, 0, 0)  # Pure red in RGB

            nv12_msg = self._create_nv12_image(rgb_image)

            # Wait at most TIMEOUT seconds for subscriber to respond
            TIMEOUT = 30
            end_time = time.time() + TIMEOUT
            while (image_raw_pub.get_subscription_count() == 0 and
                   time.time() < end_time):
                rclpy.spin_once(self.node, timeout_sec=0.02)

            done = False
            while time.time() < end_time:
                nv12_msg.header.stamp = self.node.get_clock().now().to_msg()
                image_raw_pub.publish(nv12_msg)
                rclpy.spin_once(self.node, timeout_sec=0.02)
                if output_topic in received_messages:
                    done = True
                    break

            self.assertTrue(done, f"Didn't receive output on {output_topic}!")

            output = received_messages[output_topic]
            self.assertEqual(output.encoding, expected_encoding)
            self.assertEqual(output.height, HEIGHT)
            self.assertEqual(output.width, WIDTH)

            if out_channels == 1:
                shape = (HEIGHT, WIDTH)
            else:
                shape = (HEIGHT, WIDTH, out_channels)
            image_actual = np.frombuffer(output.data, dtype=np.uint8).reshape(shape)

            nv12_data = np.frombuffer(nv12_msg.data, dtype=np.uint8)
            nv12_array = nv12_data.reshape(HEIGHT * 3 // 2, WIDTH)
            image_expected = cv2.cvtColor(nv12_array, cv_code)

            self.assertImagesEqual(image_actual, image_expected)
        finally:
            self.node.destroy_subscription(image_sub)
            self.node.destroy_publisher(image_raw_pub)

    def test_nv12_to_rgb8_conversion(self) -> None:
        """Expect the node to convert nv12 input to rgb8."""
        self._run_nv12_conversion(
            'image_rgb8', 'rgb8', cv2.COLOR_YUV2RGB_NV12, 3)

    def test_nv12_to_mono8_conversion(self) -> None:
        """Expect the node to convert nv12 input to mono8."""
        self._run_nv12_conversion(
            'image_mono8', 'mono8', cv2.COLOR_YUV2GRAY_NV12, 1)

    @staticmethod
    def _rgb_to_ycbcr_bt709_legal(rgb_image: np.ndarray) -> np.ndarray:
        """RGB (HxWx3 uint8) -> 8-bit BT.709 limited-range YCbCr (HxWx3 uint8)."""
        # Recommendation ITU-R BT.709-6 (06/2015), "Parameter values for the
        # HDTV standards for production and international programme exchange":
        #   https://www.itu.int/rec/R-REC-BT.709                              (catalog)
        #   https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.709-6-201506-I!!PDF-E.pdf  (PDF)
        # Section 3 "Signal format" defines the RGB -> Y'CbCr signal chain:
        #   Item 3.2 - luma:   E_Y' = 0.2126*R + 0.7152*G + 0.0722*B
        #                      (i.e. Kr = 0.2126, Kb = 0.0722, Kg = 1 - Kr - Kb)
        #   Item 3.3 - chroma analogue: E_Cb = (B - E_Y') / 1.8556,
        #                               E_Cr = (R - E_Y') / 1.5748
        #                      where 1.8556 = 2 * (1 - Kb), 1.5748 = 2 * (1 - Kr)
        #   Item 3.4 - 8-bit quantization to "studio" (legal) range:
        #     D_Y'  = INT(219 * E_Y'  + 16),   nominal [16, 235]
        #     D_Cb' = INT(224 * E_Cb  + 128),  nominal [16, 240]
        #     D_Cr' = INT(224 * E_Cr  + 128),  nominal [16, 240]
        kr, kg, kb = 0.2126, 0.7152, 0.0722
        rgb = rgb_image.astype(np.float64) / 255.0
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        y_full = kr * r + kg * g + kb * b
        cb_full = (b - y_full) / (2.0 * (1.0 - kb))
        cr_full = (r - y_full) / (2.0 * (1.0 - kr))
        y = np.clip(np.round(16.0 + 219.0 * y_full), 16, 235)
        cb = np.clip(np.round(128.0 + 224.0 * cb_full), 16, 240)
        cr = np.clip(np.round(128.0 + 224.0 * cr_full), 16, 240)
        return np.stack([y, cb, cr], axis=-1).astype(np.uint8)

    @staticmethod
    def _pack_nv12(ycbcr: np.ndarray) -> np.ndarray:
        """
        Pack a YCbCr (HxWx3 uint8) image into NV12 byte layout ((H*3/2)xW uint8).

        NV12 is a Y plane (H rows at full resolution) followed by one
        interleaved CbCr plane (H/2 rows, Cb and Cr samples alternating
        per column) at 4:2:0 chroma resolution.
        """
        h, w = ycbcr.shape[:2]

        def subsample_420(plane):
            # Average each 2x2 block to one chroma sample (4:2:0 downsample).
            return np.round(
                plane.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))).astype(np.uint8)

        y_plane = ycbcr[..., 0]
        cb_sub = subsample_420(ycbcr[..., 1])
        cr_sub = subsample_420(ycbcr[..., 2])
        uv = np.empty((h // 2, w), dtype=np.uint8)
        uv[:, 0::2] = cb_sub
        uv[:, 1::2] = cr_sub
        return np.vstack([y_plane, uv])

    @staticmethod
    def _encode_nv12_bt709_legal(rgb_image: np.ndarray) -> Image:
        """Encode an RGB image (HxWx3 uint8) to a BT.709 limited-range NV12 sensor_msgs/Image."""
        h, w = rgb_image.shape[:2]
        ycbcr = IsaacROSFormatNV12Test._rgb_to_ycbcr_bt709_legal(rgb_image)
        nv12 = IsaacROSFormatNV12Test._pack_nv12(ycbcr)

        msg = Image()
        msg.height = h
        msg.width = w
        msg.encoding = 'nv12'
        msg.step = w
        msg.is_bigendian = 0
        msg.data = nv12.tobytes()
        return msg

    def test_nv12_to_rgb8_bt709(self) -> None:
        """Round-trip BT.709-limited-range NV12 through the node and expect the original RGB."""
        input_topic = 'image_raw_bt709'
        output_topic = 'image_rgb8_bt709'
        self.generate_namespace_lookup([input_topic, output_topic])
        received_messages = {}

        image_sub, = self.create_logging_subscribers(
            subscription_requests=[(output_topic, Image)],
            received_messages=received_messages)

        image_raw_pub = self.node.create_publisher(
            Image, self.generate_namespace(input_topic), self.DEFAULT_QOS)

        try:
            # Use a multi-color grid rather than a saturated primary so the
            # matrix/range difference from BT.601 actually shows up.
            rgb_image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
            rgb_image[:HEIGHT // 2, :WIDTH // 2] = (200, 150, 100)
            rgb_image[:HEIGHT // 2, WIDTH // 2:] = (50, 200, 75)
            rgb_image[HEIGHT // 2:, :WIDTH // 2] = (25, 25, 180)
            rgb_image[HEIGHT // 2:, WIDTH // 2:] = (220, 220, 220)

            nv12_msg = self._encode_nv12_bt709_legal(rgb_image)

            TIMEOUT = 5
            end_time = time.time() + TIMEOUT
            while (image_raw_pub.get_subscription_count() == 0 and
                   time.time() < end_time):
                rclpy.spin_once(self.node, timeout_sec=0.02)

            done = False
            while time.time() < end_time:
                nv12_msg.header.stamp = self.node.get_clock().now().to_msg()
                image_raw_pub.publish(nv12_msg)
                rclpy.spin_once(self.node, timeout_sec=0.02)
                if output_topic in received_messages:
                    done = True
                    break

            self.assertTrue(done, f"Didn't receive output on {output_topic}!")

            output = received_messages[output_topic]
            self.assertEqual(output.encoding, 'rgb8')
            image_actual = np.frombuffer(
                output.data, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)

            self.assertImagesEqual(image_actual, rgb_image)
        finally:
            self.node.destroy_subscription(image_sub)
            self.node.destroy_publisher(image_raw_pub)
