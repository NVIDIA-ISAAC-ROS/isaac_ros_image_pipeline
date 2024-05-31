# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

SKIP = 7


@pytest.mark.rostest
def generate_test_description():
    point_cloud_xyz_node = ComposableNode(
        name='point_cloud_xyz_node',
        package='isaac_ros_depth_image_proc',
        plugin='nvidia::isaac_ros::depth_image_proc::PointCloudXyzNode',
        namespace=IsaacROSDepthImageProcPointCloudXyzTest.generate_namespace(),
        parameters=[{
            'skip': SKIP
        }])

    container = ComposableNodeContainer(
        name='point_cloud_xyz_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[point_cloud_xyz_node],
        output='screen'
    )

    return IsaacROSDepthImageProcPointCloudXyzTest.generate_test_description([container])


class IsaacROSDepthImageProcPointCloudXyzTest(IsaacROSBaseTest):
    filepath = pathlib.Path(os.path.dirname(__file__))

    @IsaacROSBaseTest.for_each_test_case()
    def test_xyz_point_cloud(self, test_folder):
        TIMEOUT = 10
        received_messages = {}

        self.generate_namespace_lookup(['image_rect', 'camera_info',
                                        'points'])

        subs = self.create_logging_subscribers(
            [('points', PointCloud2)], received_messages)

        depth_pub = self.node.create_publisher(
            Image, self.namespaces['image_rect'], self.DEFAULT_QOS
        )
        depth_cam_info_pub = self.node.create_publisher(
            CameraInfo, self.namespaces['camera_info'], self.DEFAULT_QOS
        )

        try:
            raw_depth = np.load(str(test_folder) + '/depth.npy').astype(np.float32)
            depth_image = CvBridge().cv2_to_imgmsg(raw_depth, '32FC1')
            print(len(depth_image.data))
            depth_image.header.frame_id = 'left_cam'

            camera_info = JSONConversion.load_camera_info_from_json(
                test_folder / 'camera_info.json')

            end_time = time.time() + TIMEOUT
            done = False

            while time.time() < end_time:
                depth_pub.publish(depth_image)
                depth_cam_info_pub.publish(camera_info)

                rclpy.spin_once(self.node, timeout_sec=0.1)

                if 'points' in received_messages:
                    done = True
                    break
            self.assertTrue(done, 'Appropiate output not received')

            FIELD_NAMES = ['x', 'y', 'z']
            isaac_ros_pts = point_cloud2.read_points(
                    received_messages['points'], field_names=FIELD_NAMES, skip_nans=False)
            self.assertEqual(len(isaac_ros_pts), int(depth_image.height*depth_image.width/SKIP),
                             'Incorrect number of points in PointCloud!')

        finally:
            [self.node.destroy_subscription(sub) for sub in subs]
            self.node.destroy_publisher(depth_pub)
            self.node.destroy_publisher(depth_cam_info_pub)
