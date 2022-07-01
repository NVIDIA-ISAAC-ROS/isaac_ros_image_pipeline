## Tutorial - Stereo Image Pipeline
1. Complete the first three steps in the [quick start](../../README.md#quickstart) to set up your development environment.

2. Connect a compatible Realsense camera (ex: D435, D455) to your host machine.

3. Inside the container, build and source the workspace:  
    ```bash
    cd /workspaces/isaac_ros-dev && \
      colcon build --symlink-install && \
      source install/setup.bash
    ```
4. (Optional) Run tests to verify complete and correct installation:  
    ```bash
    colcon test --executor sequential
    ```
5. Spin up the stereo image pipeline and Realsense camera node with the launchfile:  
    ```bash
    ros2 launch isaac_ros_stereo_image_proc isaac_ros_stereo_image_pipeline.launch.py
    ```
