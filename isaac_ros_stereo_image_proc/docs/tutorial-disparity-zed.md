# Tutorial With Zed

<div align="center"><img src="../../resources/zed_sgm.png"/></div>

## Overview

This tutorial demonstrates how to perform depth-camera based reconstruction using a [Zed](https://www.stereolabs.com/) camera and [disparity_node](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_pipeline/blob/main/isaac_ros_stereo_image_proc/src/disparity_node.cpp).

> **Note**: This tutorial requires a compatible Zed camera from the list available [here](https://gitlab-master.nvidia.com/isaac_ros/nvidia-isaac-ros/-/blob/dev/profile/zed-setup.md#camera-compatibility).

1. Complete the [zed setup tutorial](https://gitlab-master.nvidia.com/isaac_ros/nvidia-isaac-ros/-/blob/dev//profile/zed-setup.md).

2. Open a new terminal and launch the Docker container using the `run_dev.sh` script:

    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
      ./scripts/run_dev.sh
    ```

3. Build and source the workspace:

    ```bash
    cd /workspaces/isaac_ros-dev && \
      colcon build --symlink-install && \
      source install/setup.bash
    ```

    > **Note**: If you are not using the zed2i camera, modify the `camera_model` variable in the [launch file](../launch/isaac_ros_stereo_image_pipeline_zed.launch.py#72) to `zed`, `zed2`, `zed2i`, `zedm`, `zedx` or `zedxm`. Also change the [Fixed Frame](http://wiki.ros.org/rviz/UserGuide#The_Fixed_Frame) in rviz to `camera_model`+`_left_camera_optical_frame`.

4. Run the launch file, which launches the example, and wait for 10 seconds.

    ```bash
    ros2 launch isaac_ros_stereo_image_proc isaac_ros_stereo_image_pipeline_zed.launch.py
    ```
