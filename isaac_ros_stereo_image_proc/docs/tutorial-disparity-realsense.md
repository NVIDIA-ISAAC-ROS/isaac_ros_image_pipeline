# Tutorial With RealSense

This tutorial demonstrates how to perform depth-camera based reconstruction using a [Realsense](https://www.intel.com/content/www/us/en/architecture-and-technology/realsense-overview.html) camera and [disparity_node](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_pipeline/blob/main/isaac_ros_stereo_image_proc/src/disparity_node.cpp).

> **Note**: This tutorial requires a compatible RealSense camera from the list available [here](https://github.com/NVIDIA-ISAAC-ROS/.github/blob/main/profile/realsense-setup.md#camera-compatibility).

1. Complete the [RealSense setup tutorial](https://github.com/NVIDIA-ISAAC-ROS/.github/blob/main/profile/realsense-setup.md).

2. Clone the `isaac_ros_image_pipeline`:

    ```bash
    cd ~/workspaces/isaac_ros-dev/src
    ```

    ```bash
    https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_pipeline.git
    ```

3. Open a new terminal and launch the Docker container using the `run_dev.sh` script:

    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
      ./scripts/run_dev.sh
    ```

4. Build and source the workspace:

    ```bash
    cd /workspaces/isaac_ros-dev && \
      colcon build --symlink-install && \
      source install/setup.bash
    ```

5. Run the launch file, which launches the example, and wait for 10 seconds.

    ```bash
    ros2 launch isaac_ros_stereo_image_proc isaac_ros_stereo_image_pipeline.launch.py
    ```

Here is a screenshot of the result from running the example:

<div align="center"><img src="../../resources/realsense_example.png"/></div>
