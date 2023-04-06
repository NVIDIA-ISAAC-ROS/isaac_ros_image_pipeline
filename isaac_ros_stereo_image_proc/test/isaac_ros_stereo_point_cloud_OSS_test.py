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

import cv2
from cv_bridge import CvBridge
from isaac_ros_test import IsaacROSBaseTest, JSONConversion

from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

import numpy as np
import pytest
import rclpy

from sensor_msgs.msg import CameraInfo, Image
from sensor_msgs.msg import PointCloud2
from stereo_msgs.msg import DisparityImage


@pytest.mark.rostest
def generate_test_description():
    pointcloud_node = ComposableNode(
        name='point_cloud',
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::PointCloudNode',
        namespace=IsaacROSPointCloudOSSTest.generate_namespace(),
        parameters=[{
            'use_color': True,
        }],)

    container = ComposableNodeContainer(
        name='point_cloud_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[pointcloud_node],
        output='screen',
    )

    pointcloud_node_xyz = ComposableNode(
        name='point_cloud_node_xyz',
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::PointCloudNode',
        namespace=IsaacROSPointCloudOSSTest.generate_namespace(),
        parameters=[{
            'use_color': False,
        }],
        remappings=[
            ('points2', 'xyz/points2'),
        ])

    container_xyz = ComposableNodeContainer(
        name='point_cloud_xyz_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[pointcloud_node_xyz],
        output='screen',
    )

    return IsaacROSPointCloudOSSTest.generate_test_description([container, container_xyz])


class IsaacROSPointCloudOSSTest(IsaacROSBaseTest):
    filepath = pathlib.Path(os.path.dirname(__file__))

    @IsaacROSBaseTest.for_each_test_case()
    def test_comparison(self, test_folder):
        TIMEOUT = 20
        received_messages = {}

        self.generate_namespace_lookup(['left/image_rect_color', 'disparity',
                                        'left/camera_info', 'right/camera_info',
                                        'points2', 'xyz/points2'])

        rgb_sub, xyz_sub = \
            self.create_logging_subscribers([
                ('points2', PointCloud2),
                ('xyz/points2', PointCloud2),
            ], received_messages,
                accept_multiple_messages=True)

        image_left_pub = self.node.create_publisher(
            Image, self.namespaces['left/image_rect_color'], self.DEFAULT_QOS
        )
        disparity_pub = self.node.create_publisher(
            DisparityImage, self.namespaces['disparity'], self.DEFAULT_QOS
        )
        camera_info_left = self.node.create_publisher(
            CameraInfo, self.namespaces['left/camera_info'], self.DEFAULT_QOS
        )
        camera_info_right = self.node.create_publisher(
            CameraInfo, self.namespaces['right/camera_info'], self.DEFAULT_QOS
        )

        try:
            image_left = JSONConversion.load_image_from_json(
                test_folder / 'image_left.json')

            disparity_image = DisparityImage()
            disp_img = cv2.imread(os.path.join(
                self.filepath, 'test_cases', 'stereo_images_chair', 'test_disparity.png'),
                cv2.IMREAD_UNCHANGED).astype(np.float32)
            disparity_image.image = CvBridge().cv2_to_imgmsg(disp_img, '32FC1')
            disparity_image.min_disparity = np.min(disp_img).astype(np.float)

            camera_info = JSONConversion.load_camera_info_from_json(
                test_folder / 'camera_info.json')

            end_time = time.time() + TIMEOUT
            done = False

            while time.time() < end_time:
                image_left_pub.publish(image_left)
                disparity_pub.publish(disparity_image)
                camera_info_left.publish(camera_info)
                camera_info_right.publish(camera_info)

                rclpy.spin_once(self.node, timeout_sec=(0.1))

                if all([
                    len(messages) >= 1
                    for messages in received_messages.values()
                ]):
                    done = True
                    break
            self.assertTrue(done)
            for rgb_msg in received_messages['points2']:
                self.assertEqual(rgb_msg.point_step, 16)

            for xyz_msg in received_messages['xyz/points2']:
                self.assertEqual(xyz_msg.point_step, 12)
        finally:
            self.node.destroy_subscription(rgb_sub)
            self.node.destroy_subscription(xyz_sub)
            self.node.destroy_publisher(image_left_pub)
            self.node.destroy_publisher(disparity_pub)
            self.node.destroy_publisher(camera_info_right)
            self.node.destroy_publisher(camera_info_left)
