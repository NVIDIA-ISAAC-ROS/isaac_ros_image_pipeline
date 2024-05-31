# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from isaac_ros_test import IsaacROSBaseTest
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

import numpy as np
import pytest
import rclpy

from sensor_msgs.msg import Image

PASS_THRESH = 0.0001


@pytest.mark.rostest
def generate_test_description():
    convert_metric_node = ComposableNode(
        name='convert_metric_node',
        package='isaac_ros_depth_image_proc',
        plugin='nvidia::isaac_ros::depth_image_proc::ConvertMetricNode',
        namespace=IsaacROSDepthImageProcConvertMetricTest.generate_namespace())

    container = ComposableNodeContainer(
        name='convert_metric_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[convert_metric_node],
        output='screen'
    )

    return IsaacROSDepthImageProcConvertMetricTest.generate_test_description(
        [container])


class IsaacROSDepthImageProcConvertMetricTest(IsaacROSBaseTest):
    filepath = pathlib.Path(os.path.dirname(__file__))

    @IsaacROSBaseTest.for_each_test_case()
    def test_convert_metric(self, test_folder):
        TIMEOUT = 10
        received_messages = {}

        self.generate_namespace_lookup(['image', 'image_raw'])

        subs = self.create_logging_subscribers(
            [('image', Image)], received_messages)

        depth_pub = self.node.create_publisher(
            Image, self.namespaces['image_raw'], self.DEFAULT_QOS
        )

        try:
            # time.sleep(10)
            raw_depth_uint16 = np.load(str(test_folder) + '/depth.npy').astype(np.uint16)
            cv_bridge = CvBridge()
            depth_image = cv_bridge.cv2_to_imgmsg(raw_depth_uint16, 'mono16')
            depth_image.header.frame_id = 'left_cam'

            end_time = time.time() + TIMEOUT
            done = False

            while time.time() < end_time:
                depth_pub.publish(depth_image)
                rclpy.spin_once(self.node, timeout_sec=0.1)
                if 'image' in received_messages:
                    done = True
                    break
            self.assertTrue(done, 'Appropiate output not received')

            depth_float32_msg = received_messages['image']
            depth_float32 = cv_bridge.imgmsg_to_cv2(depth_float32_msg)
            # Check if the output type is float32
            self.assertEqual(depth_float32_msg.encoding, '32FC1', 'output is not float32')
            # Check if the dimension of the images are the same
            self.assertEqual(raw_depth_uint16.shape[0], depth_float32.shape[0],
                             'Image widths are different')
            self.assertEqual(raw_depth_uint16.shape[1], depth_float32.shape[1],
                             'Image heights are different')
            # Check if the output is correct
            # Convert original depth from mm to meters and compare the output
            raw_depth_float32 = raw_depth_uint16.astype(np.float32) / 1000
            difference = np.linalg.norm(raw_depth_float32 - depth_float32)
            self.assertLess(difference, PASS_THRESH)

        finally:
            [self.node.destroy_subscription(sub) for sub in subs]
            self.node.destroy_publisher(depth_pub)
