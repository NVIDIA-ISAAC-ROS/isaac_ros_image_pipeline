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

from cv_bridge import CvBridge
from isaac_ros_test import IsaacROSBaseTest, JSONConversion
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

import numpy as np
import pytest
import rclpy

from sensor_msgs.msg import CameraInfo, Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


@pytest.mark.rostest
def generate_test_description():
    point_cloud_xyzrgb_node = ComposableNode(
        name='point_cloud_xyzrgb_node',
        package='isaac_ros_depth_image_proc',
        plugin='nvidia::isaac_ros::depth_image_proc::PointCloudXyzrgbNode',
        namespace=IsaacROSDepthImageProcPointCloudXyzrgbOSSTest.generate_namespace(),
        remappings=[('points', 'isaac_ros/points')],
        parameters=[{
                'skip': 1
            }])

    # Reference Node
    reference_point_cloud_xyzrgb_node = ComposableNode(
        name='reference_point_cloud_xyzrgb_node',
        package='depth_image_proc',
        plugin='depth_image_proc::PointCloudXyzrgbNode',
        namespace=IsaacROSDepthImageProcPointCloudXyzrgbOSSTest.generate_namespace(),
        remappings=[('points', 'reference/points')])

    container = ComposableNodeContainer(
        name='point_cloud_xyzrgb_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[point_cloud_xyzrgb_node,
                                      reference_point_cloud_xyzrgb_node],
        output='screen'
    )

    return IsaacROSDepthImageProcPointCloudXyzrgbOSSTest.generate_test_description(
        [container])


class IsaacROSDepthImageProcPointCloudXyzrgbOSSTest(IsaacROSBaseTest):
    filepath = pathlib.Path(os.path.dirname(__file__))

    @IsaacROSBaseTest.for_each_test_case()
    def test_xyzrgb_point_cloud(self, test_folder):
        TIMEOUT = 10
        received_messages = {}

        self.generate_namespace_lookup(['depth_registered/image_rect',
                                        'rgb/image_rect_color', 'rgb/camera_info',
                                        'isaac_ros/points', 'reference/points'])

        subs = \
            self.create_logging_subscribers([
                ('isaac_ros/points', PointCloud2),
                ('reference/points', PointCloud2),
            ], received_messages,
                accept_multiple_messages=True,
                # sensor qos required for depth_image_proc
                qos_profile=rclpy.qos.qos_profile_sensor_data)

        depth_pub = self.node.create_publisher(
            Image, self.namespaces['depth_registered/image_rect'], self.DEFAULT_QOS
        )
        rgb_image_pub = self.node.create_publisher(
            Image, self.namespaces['rgb/image_rect_color'], self.DEFAULT_QOS
        )
        cam_info_pub = self.node.create_publisher(
            CameraInfo, self.namespaces['rgb/camera_info'], self.DEFAULT_QOS
        )

        try:
            raw_depth = np.load(str(test_folder) + '/depth.npy').astype(np.float32)
            depth_image = CvBridge().cv2_to_imgmsg(raw_depth, '32FC1')
            depth_image.header.frame_id = 'left_cam'
            rgb_image = JSONConversion.load_image_from_json(
                test_folder / 'image_left.json')
            camera_info = JSONConversion.load_camera_info_from_json(
                test_folder / 'camera_info.json')

            end_time = time.time() + TIMEOUT
            done = False

            while time.time() < end_time:
                depth_pub.publish(depth_image)
                cam_info_pub.publish(camera_info)
                rgb_image_pub.publish(rgb_image)
                rclpy.spin_once(self.node, timeout_sec=0.1)
                if all(
                    len(messages) >= 1
                    for messages in received_messages.values()
                ):
                    done = True
                    break
            self.assertTrue(done, 'Appropiate output not received')

            FIELD_NAMES = ['x', 'y', 'z', 'rgb']
            for isaac_ros_msg, ref_msg in zip(
                    received_messages['isaac_ros/points'],
                    received_messages['reference/points']
            ):
                # Points returned as generators for memory efficiency
                isaac_ros_pts = point_cloud2.read_points(
                    isaac_ros_msg, field_names=FIELD_NAMES, skip_nans=False)
                ref_pts = point_cloud2.read_points(
                    ref_msg, field_names=FIELD_NAMES, skip_nans=False)

                xyz_error = 0
                rgb_error = 0
                n = 0
                nan_count_isaac_ros = 0
                nan_count_ref = 0
                for isaac_ros_pt, ref_pt in zip(isaac_ros_pts, ref_pts):
                    # Unpack array to avoid 0-d and type inference issues
                    isaac_ros_pt = np.array(
                        [isaac_ros_pt[0], isaac_ros_pt[1], isaac_ros_pt[2], isaac_ros_pt[3]])
                    ref_pt = np.array(
                        [ref_pt[0], ref_pt[1], ref_pt[2], ref_pt[3]])

                    if np.any(np.isnan(isaac_ros_pt)) or np.any(np.isnan(ref_pt)):
                        if (np.any(np.isnan(isaac_ros_pt))):
                            nan_count_isaac_ros += 1
                        if (np.any(np.isnan(ref_pt))):
                            nan_count_ref += 1
                        continue
                    xyz_error += np.sum(
                        np.square(isaac_ros_pt[:3] - ref_pt[:3]))

                    rgb_error += (isaac_ros_pt[3] - ref_pt[3])**2

                    n += 1
                self.assertGreater(n, 0)
                # MSE error
                xyz_error = xyz_error / n
                self.node.get_logger().info(f'XYZ Error: {xyz_error}')
                self.node.get_logger().info(f'RGB Error: {rgb_error}')
                self.node.get_logger().info(
                    f'nan_count_isaac_ros: {nan_count_isaac_ros}')
                self.node.get_logger().info(f'nan_count_ref: {nan_count_ref}')

                XYZ_ERROR_THRESHOLD = 1e-6  # Units in mm
                self.assertLess(xyz_error, XYZ_ERROR_THRESHOLD)

                # Point coloring is determined purely by input image, so should be exactly the same
                self.assertEqual(rgb_error, 0)

        finally:
            [self.node.destroy_subscription(sub) for sub in subs]
            self.node.destroy_publisher(depth_pub)
            self.node.destroy_publisher(cam_info_pub)
