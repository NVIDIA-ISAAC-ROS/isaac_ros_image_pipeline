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

"""Tool to generate test cases for the isaac_ros_image_proc ROS 2 package."""

import os
from pathlib import Path
import time
from typing import Tuple

import cv2
from cv_bridge import CvBridge
from isaac_ros_test import JSONConversion
import numpy as np
import rclpy
from sensor_msgs.msg import CameraInfo, Image


class TestCaseGenerator:
    """Class for test case generation utilities."""

    def __init__(self, test_cases_filepath: Path) -> None:
        """
        Create a TestCaseGenerator.

        Parameters
        ----------
        test_cases_filepath : Path
            The path to the folder in which test cases should be populated.

        """
        rclpy.init()
        self.node = rclpy.create_node('generate_test_cases')
        self.bridge = CvBridge()
        self.test_cases_filepath = test_cases_filepath

    def shutdown(self) -> None:
        """Shut down the test case generator tool."""
        rclpy.shutdown()

    def remove_all_test_cases(self) -> None:
        """Remove all test cases in this generator's test cases folder."""
        for test_case in self.test_cases_filepath.iterdir():
            for file in test_case.iterdir():
                file.unlink()
            test_case.rmdir()

    def add_test_case(
            self, name: str, image: np.ndarray,
            encoding: str = 'bgr8', camera_info: CameraInfo = None) -> None:
        """
        Add a test case to this generator's test cases folder.

        Parameters
        ----------
        name : str
            The name of the folder for this test case
        image : np.ndarray
            The input image for this test case, as NumPy array
        encoding : str, optional
            The image encoding to use, by default 'bgr8'
        camera_info : CameraInfo, optional
            The camera calibration parameters to use, by default None.
            If None, a default set of parameters is loaded from file.

        """
        (self.test_cases_filepath / name).mkdir(exist_ok=True)
        image_msg = self.bridge.cv2_to_imgmsg(image)
        image_msg.encoding = encoding

        if camera_info is None:
            # If no camera_info was specified, load default values from the test_cases/ folder
            camera_info = JSONConversion.load_camera_info_from_json(
                self.test_cases_filepath.parent / 'camera_info.json')

        JSONConversion.save_camera_info_to_json(
            camera_info, self.test_cases_filepath / name / 'camera_info.json')
        JSONConversion.save_image_to_json(
            image_msg, self.test_cases_filepath / name / 'image_raw.json')

        self.generate_ground_truth(image_msg, camera_info, self.test_cases_filepath / name)

    def add_grid_test_case(self,
                           name: str,
                           width: int = 640, height: int = 480,
                           horizontal_lines: int = 10, vertical_lines: int = 10,
                           background_color: Tuple[int, int, int] = (0, 0, 0),
                           line_color: Tuple[int, int, int] = (255, 255, 255),
                           line_thickness: int = 7
                           ) -> None:
        """
        Add a grid-based test case.

        Parameters
        ----------
        name : str
            The name of this test case
        width : int, optional
            The width of the grid image, by default 640
        height : int, optional
            The height of the grid image, by default 480
        horizontal_lines : int, optional
            The number of horizontal lines to draw, by default 10
        vertical_lines : int, optional
            The number of vertical lines to draw, by default 10
        background_color : Tuple[int, int, int], optional
            The background color of the grid in BGR, by default (0, 0, 0)
        line_color : Tuple[int, int, int], optional
            The line color of the grid in BGR, by default (255, 255, 255)
        line_thickness : int, optional
            The thickness of the lines in pixels, by default 7

        """
        # Create 3-channel image that will be distorted by the processor
        distorted_image = np.zeros((height, width, 3), np.uint8)
        distorted_image[:, :] = background_color

        # Add horizontal lines
        for y in np.linspace(0, height, num=horizontal_lines):
            y = int(y)
            cv2.line(distorted_image, (0, y), (width, y), line_color, line_thickness)

        # Add vertical lines
        for x in np.linspace(0, width, num=vertical_lines):
            x = int(x)
            cv2.line(distorted_image, (x, 0), (x, height), line_color, line_thickness)

        self.add_test_case(name, distorted_image)

    def add_image_file_test_case(self, name: str, image_filepath: Path) -> None:
        """
        Add an image file-based test case.

        Parameters
        ----------
        name : str
            The name of this test case
        image_filepath : Path
            The path to the image file to use

        """
        image = cv2.imread(str(image_filepath))
        self.add_test_case(name, image)

    def generate_ground_truth(
            self, image_raw: Image, camera_info: CameraInfo, test_folder: Path) -> None:
        """
        Publish the test input and receive the test output to save as ground truth.

        Parameters
        ----------
        image_raw : Image
            The input ROS 2 Image message to send
        camera_info : CameraInfo
            The input ROS 2 Camera Info message to send
        test_folder : Path
            The test case folder to save the ground truth outputs to

        """
        QOS = 10  # Default Quality of Service queue length

        image_raw_pub = self.node.create_publisher(
            Image, 'image_raw', QOS)
        camera_info_pub = self.node.create_publisher(
            CameraInfo, 'camera_info', QOS)

        received_images = {}
        image_mono_sub = self.node.create_subscription(
            Image, 'image_mono', lambda msg: received_images.update({'image_mono': msg}), QOS)
        image_rect_sub = self.node.create_subscription(
            Image, 'image_rect', lambda msg: received_images.update({'image_rect': msg}), QOS)
        image_color_sub = self.node.create_subscription(
            Image, 'image_color', lambda msg: received_images.update({'image_color': msg}), QOS)
        image_rect_color_sub = self.node.create_subscription(
            Image, 'image_rect_color',
            lambda msg: received_images.update({'image_rect_color': msg}), QOS)

        try:
            # Publish test case over both topics
            image_raw_pub.publish(image_raw)
            camera_info_pub.publish(camera_info)

            # Wait at most TIMEOUT seconds to receive ground truth images
            TIMEOUT = 2
            end_time = time.time() + TIMEOUT

            done = False
            output_topics = ['image_mono', 'image_rect', 'image_color', 'image_rect_color']
            while time.time() < end_time:
                rclpy.spin_once(self.node, timeout_sec=TIMEOUT)

                # If we have received exactly one message on each output topic, break
                if all([topic in received_images for topic in output_topics]):
                    done = True
                    break
            assert done, "Didn't receive output messages on all subscribers! " \
                'Make sure image_proc is running!'

            for topic in output_topics:
                cv2.imwrite(str(test_folder / f'{topic}.jpg'),
                            self.bridge.imgmsg_to_cv2(received_images[topic]))

        finally:
            self.node.destroy_subscription(image_mono_sub)
            self.node.destroy_subscription(image_rect_sub)
            self.node.destroy_subscription(image_color_sub)
            self.node.destroy_subscription(image_rect_color_sub)
            self.node.destroy_publisher(image_raw_pub)
            self.node.destroy_publisher(camera_info_pub)


if __name__ == '__main__':
    gen = TestCaseGenerator(Path(os.path.dirname(__file__)) / 'test_cases')

    gen.remove_all_test_cases()
    gen.add_grid_test_case('white_grid')
    gen.add_grid_test_case('nvidia_green_grid', line_color=(0, 185, 118))  # NVIDIA green
    gen.add_grid_test_case('dense_grid', horizontal_lines=50,
                           vertical_lines=50, line_thickness=2)
    gen.add_image_file_test_case('nvidia_icon', Path(
        os.path.dirname(__file__)) / 'test_cases' / 'NVIDIAprofile.png')
    gen.shutdown()
