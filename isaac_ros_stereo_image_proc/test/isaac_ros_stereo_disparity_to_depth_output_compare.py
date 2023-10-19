# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from sensor_msgs.msg import Image
from stereo_msgs.msg import DisparityImage

# Generate test data for isaac_ros_depth_image_proc_point_cloud_xyz_pol.py
SAVE_DEPTH = False


@pytest.mark.rostest
def generate_test_description():
    disparity_to_depth_node = ComposableNode(
        name='disparity_to_depth',
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::DisparityToDepthNode',
        namespace=IsaacROSDisparityToDepthComparisonTest.generate_namespace()
    )

    container = ComposableNodeContainer(
        name='disparity_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[disparity_to_depth_node],
        output='screen',
        arguments=['--ros-args', '--log-level', 'info']
    )
    return IsaacROSDisparityToDepthComparisonTest.generate_test_description([container])


class IsaacROSDisparityToDepthComparisonTest(IsaacROSBaseTest):
    """Compare the output depth against the values manually calculated."""

    filepath = pathlib.Path(os.path.dirname(__file__))

    @IsaacROSBaseTest.for_each_test_case()
    def test_image_disparity(self, test_folder):
        TIMEOUT = 10
        received_messages = {}

        self.generate_namespace_lookup(['depth', 'disparity'])

        subs = self.create_logging_subscribers(
            [('depth', Image)], received_messages)

        disparity_pub = self.node.create_publisher(
            DisparityImage, self.namespaces['disparity'], self.DEFAULT_QOS
        )

        try:
            disparity_image = JSONConversion.load_disparity_image_from_json(
                test_folder / 'disparity.json')
            end_time = time.time() + TIMEOUT
            done = False

            while time.time() < end_time:
                disparity_pub.publish(disparity_image)
                rclpy.spin_once(self.node, timeout_sec=1.0)
                if 'depth' in received_messages:
                    done = True
                    break
            self.assertTrue(done, 'Didnt recieve output on disparity topic')

            depth = received_messages['depth']
            # Calculate ground truth depth using depth = focal_length*baseline/disparity
            # https://github.com/ros2/common_interfaces/blob/humble/stereo_msgs/msg/DisparityImage.msg#L10
            disparity_image_json = JSONConversion.load_from_json(
                test_folder / 'disparity.json')
            raw_disparity = np.load(str(test_folder)+'/'+disparity_image_json['image']
                                    ).astype(np.float32)
            # Baseline is assumed to be equal to -disparity_image_json['f']
            ground_truth_depth = (-disparity_image_json['f']
                                  * disparity_image_json['t']) / raw_disparity
            # Some values in ground_truth_depth is inf deu to division by 0
            # Replace these inf with 0
            ground_truth_depth[ground_truth_depth == np.inf] = 0
            # Check ground truth against value received from Isaac ROS DisparityToDepthNode
            self.assertImagesEqual(
                ground_truth_depth, CvBridge().imgmsg_to_cv2(depth), 0.00001)
            if(SAVE_DEPTH):
                np.save(str(test_folder)+'/depth.npy', ground_truth_depth)

        finally:
            [self.node.destroy_subscription(sub) for sub in subs]
            self.node.destroy_publisher(disparity_pub)
