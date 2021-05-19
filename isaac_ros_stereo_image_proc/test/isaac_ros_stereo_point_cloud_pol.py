# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


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
        plugin='isaac_ros::stereo_image_proc::PointCloudNode',
        namespace=IsaacROSPointCloudTest.generate_namespace(),
        parameters=[{
            'use_color': True,
        }])

    container = ComposableNodeContainer(
        name='point_cloud_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[pointcloud_node],
        output='screen'
    )

    return IsaacROSPointCloudTest.generate_test_description([container])


class IsaacROSPointCloudTest(IsaacROSBaseTest):
    filepath = pathlib.Path(os.path.dirname(__file__))

    @IsaacROSBaseTest.for_each_test_case()
    def test_rgbxyz_point_cloud(self, test_folder):
        TIMEOUT = 10
        received_messages = {}

        self.generate_namespace_lookup(['left/image_rect_color', 'disparity',
                                        'left/camera_info', 'right/camera_info',
                                        'points2'])

        subs = self.create_logging_subscribers(
            [('points2', PointCloud2)], received_messages)

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

                rclpy.spin_once(self.node, timeout_sec=0.1)

                if 'points2' in received_messages:
                    done = True
                    break

            self.assertTrue(done, 'Appropiate output not received')

        finally:
            [self.node.destroy_subscription(sub) for sub in subs]
            self.node.destroy_publisher(image_left_pub)
            self.node.destroy_publisher(disparity_pub)
            self.node.destroy_publisher(camera_info_right)
            self.node.destroy_publisher(camera_info_left)
