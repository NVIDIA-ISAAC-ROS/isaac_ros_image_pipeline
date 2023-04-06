# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from isaac_ros_test import IsaacROSBaseTest, JSONConversion

from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

import numpy as np
import pytest
import rclpy

from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2

USE_COLOR = True
AVOID_PADDING = True


@pytest.mark.rostest
def generate_test_description():

    disparity_node = ComposableNode(
        name='disparity_node',
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::DisparityNode',
        namespace=IsaacROSStereoPipelineComparisonTest.generate_namespace(),
        parameters=[{
                'backends': 'CUDA',
                'max_disparity': 64.0
        }],
        remappings=[('disparity', 'isaac_ros/disparity')]
    )

    point_cloud_node = ComposableNode(
        name='point_cloud_node',
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::PointCloudNode',
        namespace=IsaacROSStereoPipelineComparisonTest.generate_namespace(),
        parameters=[{'use_color': USE_COLOR, }],
        remappings=[('disparity', 'isaac_ros/disparity'),
                    ('points2', 'isaac_ros/points2'),
                    ('left/image_rect_color', 'left/image_rect')])

    container = ComposableNodeContainer(
        name='point_cloud_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[disparity_node, point_cloud_node],
        output='screen'
    )

    # Reference nodes
    reference_disparity_node = Node(
        package='stereo_image_proc',
        executable='disparity_node',
        name='reference_disparity_node',
        namespace=IsaacROSStereoPipelineComparisonTest.generate_namespace(),
        output='screen',
        remappings=[('disparity', 'reference/disparity')]
    )

    reference_point_cloud_node = Node(
        package='stereo_image_proc',
        executable='point_cloud_node',
        name='reference_point_cloud_node',
        namespace=IsaacROSStereoPipelineComparisonTest.generate_namespace(),
        output='screen',
        parameters=[{
            'use_color': USE_COLOR,
            'use_system_default_qos': True,
            'avoid_point_cloud_padding': AVOID_PADDING
        }],
        remappings=[('disparity', 'reference/disparity'),
                    ('points2', 'reference/points2'),
                    ('left/image_rect_color', 'left/image_rect')])

    return IsaacROSStereoPipelineComparisonTest.generate_test_description([
        container, reference_disparity_node, reference_point_cloud_node])


class IsaacROSStereoPipelineComparisonTest(IsaacROSBaseTest):
    filepath = pathlib.Path(os.path.dirname(__file__))

    @IsaacROSBaseTest.for_each_test_case()
    def test_comparison(self, test_folder):
        TIMEOUT = 20
        received_messages = {}

        self.generate_namespace_lookup(['left/image_rect', 'right/image_rect',
                                        'left/camera_info', 'right/camera_info',
                                        'isaac_ros/points2', 'reference/points2'])
        isaac_ros_sub, ref_sub = self.create_logging_subscribers(
            [('isaac_ros/points2', PointCloud2),
             ('reference/points2', PointCloud2)],
            received_messages, accept_multiple_messages=True
        )
        image_left_pub = self.node.create_publisher(
            Image, self.namespaces['left/image_rect'], self.DEFAULT_QOS
        )
        image_right_pub = self.node.create_publisher(
            Image, self.namespaces['right/image_rect'], self.DEFAULT_QOS
        )
        camera_info_right = self.node.create_publisher(
            CameraInfo, self.namespaces['right/camera_info'], self.DEFAULT_QOS
        )
        camera_info_left = self.node.create_publisher(
            CameraInfo, self.namespaces['left/camera_info'], self.DEFAULT_QOS
        )

        try:
            image_left = JSONConversion.load_image_from_json(test_folder / 'image_left.json')
            image_right = JSONConversion.load_image_from_json(test_folder / 'image_right.json')
            camera_info = JSONConversion.load_camera_info_from_json(
                test_folder / 'camera_info.json')

            image_left.header.frame_id = 'world'
            image_right.header = image_left.header
            camera_info.header = image_left.header

            end_time = time.time() + TIMEOUT
            done = False

            while time.time() < end_time:
                image_left_pub.publish(image_left)
                image_right_pub.publish(image_right)
                camera_info_left.publish(camera_info)
                camera_info_right.publish(camera_info)

                rclpy.spin_once(self.node, timeout_sec=(0.1))

                if all([len(messages) >= 1 for messages in received_messages.values()]):
                    done = True
                    break

            self.assertTrue(done)

            FIELD_NAMES = ['x', 'y', 'z'] + (['rgb'] if USE_COLOR else [])

            for isaac_ros_msg, ref_msg in zip(
                    received_messages['isaac_ros/points2'],
                    received_messages['reference/points2']
            ):
                # Points returned as generators for memory efficiency
                isaac_ros_pts = point_cloud2.read_points(
                    isaac_ros_msg, field_names=FIELD_NAMES, skip_nans=False)
                ref_pts = point_cloud2.read_points(
                    ref_msg, field_names=FIELD_NAMES, skip_nans=False)

                xyz_error = 0
                rgb_error = 0
                n = 0
                for isaac_ros_pt, ref_pt in zip(isaac_ros_pts, ref_pts):
                    # Unpack array to avoid 0-d and type inference issues
                    isaac_ros_pt = np.array(
                        [isaac_ros_pt[0], isaac_ros_pt[1], isaac_ros_pt[2], isaac_ros_pt[3]])
                    ref_pt = np.array([ref_pt[0], ref_pt[1], ref_pt[2], ref_pt[3]])

                    if np.any(np.isnan(isaac_ros_pt)) or np.any(np.isnan(ref_pt)):
                        continue

                    xyz_error += np.sum(np.square(isaac_ros_pt[:3] - ref_pt[:3]))

                    if USE_COLOR:
                        rgb_error += np.square(isaac_ros_pt[3] - ref_pt[3])

                    n += 1

                self.assertGreater(n, 0)
                # MSE error
                xyz_error = xyz_error / n
                self.node.get_logger().info(f'XYZ Error: {xyz_error}')
                self.node.get_logger().info(f'RGB Error: {rgb_error}')

                XYZ_ERROR_THRESHOLD = 10  # Units in mm
                self.assertLess(xyz_error, XYZ_ERROR_THRESHOLD)

                # Point coloring is determined purely by input image, so should be exactly the same
                self.assertEqual(rgb_error, 0)
        finally:
            self.node.destroy_subscription(isaac_ros_sub)
            self.node.destroy_subscription(ref_sub)
            self.node.destroy_publisher(image_left_pub)
            self.node.destroy_publisher(image_right_pub)
            self.node.destroy_publisher(camera_info_right)
            self.node.destroy_publisher(camera_info_left)
