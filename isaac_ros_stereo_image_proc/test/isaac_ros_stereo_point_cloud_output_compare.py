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

from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

import numpy as np
import pytest
import rclpy

from sensor_msgs.msg import CameraInfo, Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from stereo_msgs.msg import DisparityImage

USE_COLOR = True
AVOID_PADDING = False


@pytest.mark.rostest
def generate_test_description():
    pointcloud_node = ComposableNode(
        name='point_cloud',
        package='isaac_ros_stereo_image_proc',
        plugin='isaac_ros::stereo_image_proc::PointCloudNode',
        namespace=IsaacROSPointCloudComparisonTest.generate_namespace(),
        parameters=[{
            'use_color': USE_COLOR,
        }],
        remappings=[
            ('points2', 'isaac_ros/points2'),
        ])

    container = ComposableNodeContainer(
        name='point_cloud_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[pointcloud_node],
        output='screen',
    )

    # Reference Node
    ref = Node(
        package='stereo_image_proc',
        executable='point_cloud_node',
        name='point_cloud_node_proc',
        namespace=IsaacROSPointCloudComparisonTest.generate_namespace(),
        output='screen',
        parameters=[{
            'use_color': USE_COLOR,
            'use_system_default_qos': True,
            'avoid_point_cloud_padding': AVOID_PADDING
        }],
        remappings=[
            ('points2', 'proc/points2'),
        ]
    )

    return IsaacROSPointCloudComparisonTest.generate_test_description([container, ref])


class IsaacROSPointCloudComparisonTest(IsaacROSBaseTest):
    filepath = pathlib.Path(os.path.dirname(__file__))

    @IsaacROSBaseTest.for_each_test_case()
    def test_comparison(self, test_folder):
        TIMEOUT = 20
        received_messages = {}

        self.generate_namespace_lookup(['left/image_rect_color', 'disparity',
                                        'left/camera_info', 'right/camera_info',
                                        'isaac_ros/points2', 'proc/points2'])
        isaac_ros_sub, ref_sub = \
            self.create_logging_subscribers([
                ('isaac_ros/points2', PointCloud2),
                ('proc/points2', PointCloud2),
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

            for isaac_ros_msg, ref_msg in \
                    zip(received_messages['isaac_ros/points2'], received_messages['proc/points2']):
                if USE_COLOR:
                    isaac_ros_pts = point_cloud2.read_points(
                        isaac_ros_msg, field_names=('x', 'y', 'z', 'rgb'), skip_nans=True)
                    ref_pts = point_cloud2.read_points(
                        ref_msg, field_names=('x', 'y', 'z', 'rgb'), skip_nans=True)
                else:
                    isaac_ros_pts = point_cloud2.read_points(
                        isaac_ros_msg, field_names=('x', 'y', 'z'), skip_nans=True)
                    ref_pts = point_cloud2.read_points(
                        ref_msg, field_names=('x', 'y', 'z'), skip_nans=True)
                xyz_error = 0
                rgb_error = 0
                n = 0
                for (isaac_ros_pt, ref_pt) in zip(isaac_ros_pts, ref_pts):
                    xyz_error += (isaac_ros_pt[0] - ref_pt[0])**2
                    xyz_error += (isaac_ros_pt[1] - ref_pt[1])**2
                    xyz_error += (isaac_ros_pt[2] - ref_pt[2])**2

                    if USE_COLOR:
                        rgb_error += (isaac_ros_pt[3] - ref_pt[3])**2

                    n += 1
                self.assertGreater(n, 0)
                # MSE error
                xyz_error = xyz_error / n
                self.node.get_logger().info(f'XYZ Error: {xyz_error}')
                self.node.get_logger().info(f'RGB Error: {rgb_error}')
                self.assertLess(xyz_error, 1e-6)
                self.assertEqual(rgb_error, 0)
        finally:
            self.node.destroy_subscription(isaac_ros_sub)
            self.node.destroy_subscription(ref_sub)
            self.node.destroy_publisher(image_left_pub)
            self.node.destroy_publisher(disparity_pub)
            self.node.destroy_publisher(camera_info_right)
            self.node.destroy_publisher(camera_info_left)
