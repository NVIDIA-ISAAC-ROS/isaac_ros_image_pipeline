# Tutorial with Isaac Sim

This tutorial demonstrates how to perform depth-camera based reconstruction using the `disparity_node` and stereo image pairs streamed from Isaac Sim.

Last validated with [Isaac Sim 2022.2.1](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/release_notes.html#id1)

1. Complete steps 1-4 listed under [Quickstart section](../../README.md#quickstart) in the main README.
2. Install and launch Isaac Sim following the steps in the [Isaac ROS Isaac Sim Setup Guide](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/isaac-sim-sil-setup.md).
3. Open the Isaac ROS Common USD scene (using the *Content* tab) located at:

    ```text
    http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2022.2.1/Isaac/Samples/ROS2/Scenario/carter_warehouse_apriltags_worker.usd
    ```
   And wait for it to load completely.
4. Go to the *Stage* tab and select `/World/Carter_ROS/ROS_Cameras/ros2_create_camera_right_info`, then in *Property* tab *-> OmniGraph Node -> Inputs -> stereoOffset X* change `0` to `-175.92`.
    <div align="center"><img src="../../resources/Isaac_sim_set_stereo_offset.png" width="500px"/></div>
5. Enable the right camera for a stereo image pair. Go to the *Stage* tab and select `/World/Carter_ROS/ROS_Cameras/enable_camera_right`, then tick the *Condition* checkbox.
    <div align="center"><img src="../../resources/Isaac_sim_enable_stereo.png" width="500px"/></div>
6. Press **Play** to start publishing data from the Isaac Sim application.
    <div align="center"><img src="../../resources/Isaac_sim_play.png" width="800px"/></div>
7. In a separate terminal, start the `isaac_ros_stereo_image_proc` graph using the launch files:

    ```bash
    ros2 launch isaac_ros_stereo_image_proc isaac_ros_stereo_image_pipeline_isaac_sim.launch.py
    ```

    You should see a RViz window, as shown below:
    <div align="center"><img src="../../resources/Rviz.png" width="800px"/></div>
