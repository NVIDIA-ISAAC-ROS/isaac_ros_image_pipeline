# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import pathlib
import time

import cv2
from cv_bridge import CvBridge
from isaac_ros_test import IsaacROSBaseTest, JSONConversion
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

import numpy as np
import pytest
import rclpy

from sensor_msgs.msg import CameraInfo, Image

PASS_THRESH = 0.003
WRITE_IMAGES = False


@pytest.mark.rostest
def generate_test_description():
    align_depth_to_color_node = ComposableNode(
        name='align_depth_to_color_node',
        package='isaac_ros_depth_image_proc',
        plugin='nvidia::isaac_ros::depth_image_proc::AlignDepthToColorNode',
        namespace=IsaacROSDepthImageProcAlignDepthToColorTest.generate_namespace(),
        parameters=[
            {'enable_performance_logging': True}
        ]
    )

    container = ComposableNodeContainer(
        name='align_depth_to_color_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[align_depth_to_color_node],
        output='screen'
    )

    transform_publishers = []

    # 1. world -> camera_1_link
    transform_publishers.append(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='world_to_camera_link',
        arguments=[
            '-1.15538', '-0.104895', '0.283619',
            '-0.158406', '0.312276', '0.531243', '0.771474',
            'world', 'camera_1_link'
        ]
    ))

    # 2. camera_1_link -> camera_1_infra1_frame (identity)
    transform_publishers.append(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_link_to_infra1_frame',
        arguments=[
            '0.0', '0.0', '0.0',
            '0.0', '0.0', '0.0', '1.0',
            'camera_1_link', 'camera_1_infra1_frame'
        ]
    ))

    # 3. camera_1_link -> camera_1_aligned_depth_to_infra1_frame (identity)
    transform_publishers.append(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_link_to_aligned_depth_frame',
        arguments=[
            '0.0', '0.0', '0.0',
            '0.0', '0.0', '0.0', '1.0',
            'camera_1_link', 'camera_1_aligned_depth_to_infra1_frame'
        ]
    ))

    # 4. camera_1_aligned_depth_to_infra1_frame -> camera_1_infra1_optical_frame
    transform_publishers.append(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='aligned_depth_to_infra1_optical',
        arguments=[
            '0.0', '0.0', '0.0',
            '-0.5', '0.4999999999999999', '-0.5', '0.5000000000000001',
            'camera_1_aligned_depth_to_infra1_frame', 'camera_1_infra1_optical_frame'
        ]
    ))

    # 5. camera_1_link -> camera_1_infra2_frame
    transform_publishers.append(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_link_to_infra2_frame',
        arguments=[
            '0.0', '-0.049997858703136444', '0.0',
            '0.0', '0.0', '0.0', '1.0',
            'camera_1_link', 'camera_1_infra2_frame'
        ]
    ))

    # 6. camera_1_infra2_frame -> camera_1_infra2_optical_frame
    transform_publishers.append(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='infra2_frame_to_optical',
        arguments=[
            '0.0', '0.0', '0.0',
            '-0.5', '0.4999999999999999', '-0.5', '0.5000000000000001',
            'camera_1_infra2_frame', 'camera_1_infra2_optical_frame'
        ]
    ))

    # 7. camera_1_link -> camera_1_color_frame
    transform_publishers.append(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_link_to_color_frame',
        arguments=[
            '-0.00021917633421253413', '0.014781120233237743', '-8.353878365596756e-05',
            '-0.0038682855665683746', '0.0017180306604132056', '0.005697885062545538',
            '0.99997478723526',
            'camera_1_link', 'camera_1_color_frame'
        ]
    ))

    # 8. camera_1_color_frame -> camera_1_color_optical_frame
    transform_publishers.append(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='color_frame_to_optical',
        arguments=[
            '0.0', '0.0', '0.0',
            '-0.5', '0.4999999999999999', '-0.5', '0.5000000000000001',
            'camera_1_color_frame', 'camera_1_color_optical_frame'
        ]
    ))

    all_nodes = [container] + transform_publishers

    return IsaacROSDepthImageProcAlignDepthToColorTest.generate_test_description(all_nodes)


class IsaacROSDepthImageProcAlignDepthToColorTest(IsaacROSBaseTest):
    filepath = pathlib.Path(os.path.dirname(__file__))

    @IsaacROSBaseTest.for_each_test_case()
    def test_align_depth_to_color(self, test_folder):
        TIMEOUT = 20
        received_messages = {}

        self.generate_namespace_lookup(
            ['aligned_depth', 'depth_image', 'camera_info_depth', 'camera_info_color'])

        subs = self.create_logging_subscribers(
            [('aligned_depth', Image)], received_messages)

        # Create publishers
        depth_pub = self.node.create_publisher(
            Image, self.namespaces['depth_image'], self.DEFAULT_QOS
        )
        depth_info_pub = self.node.create_publisher(
            CameraInfo, self.namespaces['camera_info_depth'], self.DEFAULT_QOS
        )
        color_info_pub = self.node.create_publisher(
            CameraInfo, self.namespaces['camera_info_color'], self.DEFAULT_QOS
        )
        try:
            # Load depth data from existing .npy files in test case
            depth_file = test_folder / 'non_aligned_depth.npy'
            expected_aligned_file = test_folder / 'aligned_depth_expected.npy'

            raw_depth_data = np.load(depth_file).astype(np.float32)
            expected_aligned_depth = np.load(expected_aligned_file).astype(np.float32)

            # Load camera info from JSON files using JSONConversion
            depth_camera_info_path = test_folder / 'depth_camera_info.json'
            color_camera_info_path = test_folder / 'rgb_camera_info.json'
            depth_camera_info = JSONConversion.load_camera_info_from_json(depth_camera_info_path)
            color_camera_info = JSONConversion.load_camera_info_from_json(color_camera_info_path)

            # Update frame IDs to match our test setup
            depth_camera_info.header.frame_id = 'camera_1_infra1_optical_frame'
            color_camera_info.header.frame_id = 'camera_1_color_optical_frame'

            for i in range(1000):

                # Create depth image message from loaded data
                cv_bridge = CvBridge()
                depth_image = cv_bridge.cv2_to_imgmsg(raw_depth_data, '32FC1')
                depth_image.header.frame_id = 'camera_1_infra1_optical_frame'

                end_time = time.time() + TIMEOUT
                done = False
                while time.time() < end_time:
                    # Update timestamps
                    current_time = self.node.get_clock().now().to_msg()
                    depth_image.header.stamp = current_time
                    depth_camera_info.header.stamp = current_time
                    color_camera_info.header.stamp = current_time

                    # Publish all inputs
                    depth_pub.publish(depth_image)
                    depth_info_pub.publish(depth_camera_info)
                    color_info_pub.publish(color_camera_info)

                    rclpy.spin_once(self.node, timeout_sec=0.1)

                    if 'aligned_depth' in received_messages:
                        done = True
                        # print("Received aligned depth output!")
                        break
                    time.sleep(0.1)

                if i < 999:
                    del received_messages['aligned_depth']

            self.assertTrue(done, 'Appropiate output not received')

            # Process the received aligned depth
            aligned_depth_msg = received_messages['aligned_depth']
            aligned_depth_output = cv_bridge.imgmsg_to_cv2(aligned_depth_msg)

            # Optionally save the aligned depth output to a file
            np.save('aligned_depth_output_from_node.npy', aligned_depth_output)
            # lets save it as a png as well
            depth_normalized = cv2.normalize(aligned_depth_output, None, 0, 255, cv2.NORM_MINMAX)
            depth_8bit = np.uint8(depth_normalized)

            depth_colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_VIRIDIS)

            # Save the colorized depth image
            if WRITE_IMAGES:
                cv2.imwrite('aligned_depth_output_colorized.png', depth_colored)

            # Calculate mean absolute error where both have valid values
            expected_aligned_depth = expected_aligned_depth.squeeze()
            aligned_depth_output = aligned_depth_output.squeeze()
            valid_mask = (aligned_depth_output > 0) & (expected_aligned_depth > 0)
            output_valid = aligned_depth_output[valid_mask]
            expected_valid = expected_aligned_depth[valid_mask]

            # also write the unaligned depth image to cv2
            unaligned_depth_output = cv_bridge.imgmsg_to_cv2(depth_image)
            unaligned_depth_normalized = cv2.normalize(
                unaligned_depth_output, None, 0, 255, cv2.NORM_MINMAX)
            unaligned_depth_8bit = np.uint8(unaligned_depth_normalized)
            unaligned_depth_colored = cv2.applyColorMap(unaligned_depth_8bit, cv2.COLORMAP_VIRIDIS)
            if WRITE_IMAGES:
                cv2.imwrite(
                    'unaligned_depth_output_colorized.png',
                    unaligned_depth_colored)

            # Assert MAE is less than PASS_THRESH
            mae = np.mean(np.abs(output_valid - expected_valid))
            self.assertLess(mae, PASS_THRESH, f'Mean error too high: {mae:.4f}m')

        finally:
            [self.node.destroy_subscription(sub) for sub in subs]
            self.node.destroy_publisher(depth_pub)
            self.node.destroy_publisher(depth_info_pub)
            self.node.destroy_publisher(color_info_pub)
