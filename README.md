# `isaac_ros_image_pipeline`

## Overview
This metapackage offers similar functionality as the standard, CPU-based [`image_pipeline` metapackage](http://wiki.ros.org/image_pipeline), but does so by leveraging the Jetson platform's specialized computer vision hardware. Considerable effort has been made to ensure that replacing `image_pipeline` with `isaac_ros_image_pipeline` on a Jetson device is as painless a transition as possible.

## System Requirements
This Isaac ROS package is designed and tested to be compatible with ROS2 Foxy on Jetson hardware.
### Jetson
- AGX Xavier or Xavier NX
- JetPack 4.6

### x86_64
- CUDA 10.2/11.2 supported discrete GPU
- VPI 1.1.11
- Ubuntu 18.04+

### Docker
Precompiled ROS2 Foxy packages are not available for JetPack 4.6 (based on Ubuntu 18.04 Bionic). You can either manually compile ROS2 Foxy and required dependent packages from source or use the Isaac ROS development Docker image from [Isaac ROS Common](https://github.com/NVIDIA-AI-IOT/isaac_ros_common) based on images from [jetson-containers](https://github.com/dusty-nv/jetson-containers). The Docker images support both Jetson and x86_64 platfroms. The x86_64 docker image includes VPI Debian packages for CUDA 11.2.

Run the following script in `isaac_ros_common` to build the image and launch the container:

`$ scripts/run_dev.sh <optional_path>`

You can either provide an optional path to mirror in your host ROS workspace with Isaac ROS packages, which will be made available in the container as `/workspaces/isaac_ros-dev`, or you can setup a new workspace in the container.

### Package Dependencies
- [isaac_ros_common](https://github.com/NVIDIA-AI-IOT/isaac_ros_common)
- [image_common](https://github.com/ros-perception/image_common.git)
- [vision_cv](https://github.com/ros-perception/vision_opencv.git)
- [OpenCV 4.5+](https://opencv.org/)
- [scikit-learn](https://scikit-learn.org/stable/install.html)
- [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page)

**Note:** `isaac_ros_common' is used for running tests and/or creating a development container. It also contains VPI Debian packages that can be installed natively on a development machine without a container.

## Quickstart
1. Create a ROS2 workspace if one is not already prepared:  
`mkdir -p your_ws/src`  
**Note:** The workspace can have any name; the quickstart assumes you name it `your_ws`.
2. Clone this metapackage repository to `your_ws/src/isaac_ros_image_pipeline`. Check that you have [Git LFS](https://git-lfs.github.com/) installed before cloning to pull down all large files.  
`cd your_ws/src && git clone https://github.com/NVIDIA-AI-IOT/isaac_ros_image_pipeline`
3. Build and source the workspace:  
`cd your_ws && colcon build --symlink-install && source install/setup.bash`
4. (Optional) Run tests to verify complete and correct installation:  
`colcon test`
5. Start `isaac_ros_image_proc` using the prebuilt executable:  
`ros2 run isaac_ros_image_proc isaac_ros_image_proc`
6. In a separate terminal, spin up a **calibrated** camera publisher to `/image_raw` and `/camera_info` using any package(for example, `v4l2_camera`):  
`ros2 run v4l2_camera v4l2_camera_node`
7. Observe the rectified image output in grayscale and color on `/image_rect` and `/image_rect_color`, respectively:  
`ros2 run image_view image_view --ros-args -r image:=image_rect`  
`ros2 run image_view image_view --ros-args -r image:=image_rect_color`

### Replacing `image_pipeline` with `isaac_ros_image_pipeline`
1. Add a dependency on `isaac_ros_image_pipeline` to `your_package/package.xml` and `your_package/CMakeLists.txt`. If all desired packages under an existing `image_pipeline` dependency have Isaac ROS alternatives (see **Supported Packages**), then the original `image_pipeline` dependency may be removed entirely.
2. Change the package and plugin names in any `*.launch.py` launch files to use `[package name]` and `isaac_ros::image_proc::[component_name]` respectively. For a list of all packages, see **Supported Packages**. For a list of all ROS2 Components made available, see the per-package detailed documentation below.

## Supported Packages
At this time, the packages under the standard `image_pipeline` have the following support:

| Existing Package     | Isaac ROS Alternative             |
| -------------------- | --------------------------------- |
| `image_pipeline`     | See `isaac_ros_image_pipeline`    |
| `image_proc`         | See `isaac_ros_image_proc`        |
| `stereo_image_proc`  | See `isaac_ros_stereo_image_proc` |
| `depth_image_proc`   | On roadmap                        |
| `camera_calibration` | Continue using existing package   |
| `image_publisher`    | Continue using existing package   |
| `image_view`         | Continue using existing package   |
| `image_rotate`       | Continue using existing package   |

See also:
- `isaac_ros_apriltag`: Accelerated ROS2 wrapper for Apriltag detection
- `isaac_ros_common`: Utilities for robust ROS2 testing, in conjunction with `launch_test`

## Tutorial - Stereo Image Pipeline
1. Connect a compatible Realsense camera (D435, D455) to your host machine.
2. Build and source the workspace:  
`cd your_ws && colcon build --symlink-install && source install/setup.bash`
3. Spin up the stereo image pipeline and Realsense camera node with the launchfile:  
`ros2 launch isaac_ros_stereo_image_proc isaac_ros_stereo_image_pipeline_launch.py`

**Note** For best performance on Jetson, ensure that power settings are configured appropriately ([Power Management for Jetson](https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/power_management_jetson_xavier.html#wwpID0EUHA)).

# ROS2 Package API
## `isaac_ros_image_proc`
### Overview
The `isaac_ros_image_proc` package offers functionality for rectifying/undistorting images from a monocular camera setup, resizing the image, and changing the image format. It largely replaces the `image_proc` package, though the image format conversion facility also functions as a way to replace the CPU-based image format conversion in `cv_bridge`.

### Available Components

| Component                  | Topics Subscribed                                   | Topics Published                                                  | Parameters                                                                                                                                                                                                                                                                                                                                                                                                          |
| -------------------------- | --------------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ImageFormatConverterNode` | `image_raw`, `camera_info`: The input camera stream | `image`: The converted image                                      | `backends`: The VPI backend to use, which is CUDA by default (options: "CPU", "CUDA", "VIC") <br> `encoding_desired`: Target encoding to convert to. Note: VIC does not support RGB8 and BGR8 for either input or output encoding.                                                                                                                                                                                                                  |                                  |
| `RectifyNode`              | `image`, `camera_info`: The input camera stream     | `image_rect`: The rectified image                                 | `interpolation`: The VPI interpolation scheme to use during undistortion, which is Catmull-Rom Spline by default <br> `backends`: The VPI backend to use, which is CUDA by default  (options: "CUDA", "VIC")                                                                                                                                                                                                        |
| `ResizeNode`               | `image`, `camera_info`: The input camera stream     | `resized/image`, `resized/camera_info`: The resized camera stream | `use_relative_scale`: Whether to scale in a relative fashion, which is true by default <br> `scale_height`: The fraction to relatively scale height by <br> `scale_width`: The fraction to relatively scale width by <br> `height`: The absolute height to resize to <br> `width`: The absolute width to resize to <br> `backends`: The VPI backend to use, which is CUDA by default(options: "CPU", "CUDA", "VIC") |

## `isaac_ros_stereo_image_proc`
### Overview
The `isaac_ros_stereo_image_proc` package offers functionality for handling image pairs from a binocular/stereo camera setup, calculating the disparity between the two images, and producing a point cloud with depth information. It largely replaces the `stereo_image_proc` package.

### Available Components

| Component        | Topics Subscribed                                                                                                                                                                                              | Topics Published                                   | Parameters                                                                                                                                                                                                                                                                                       |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `DisparityNode`  | `left/image_rect`, `left/camera_info`: The left camera stream <br> `right/image_rect`, `right/camera_info`: The right camera stream                                                                            | `disparity`: The disparity between the two cameras | `max_disparity`: The maximum value for disparity per pixel, which is 64 by default. With TEGRA backend, this value must be 256. <br> `window_size`: The window size for SGM, which is 5 by default  <br> `backends`: The VPI backend to use, which is CUDA by default (options: "CUDA", "TEGRA") |
| `PointCloudNode` | `left/image_rect_color`: The coloring for the point cloud <br> `left/camera_info`: The left camera info <br> `right/camera_info`: The right camera info <br> `disparity` The disparity between the two cameras | `points2`: The output point cloud                  | `queue_size`: The length of the subscription queues, which is `rmw_qos_profile_default.depth` by default <br> `use_color`: Whether or not the output point cloud should have color. The default value is true. <br> `unit_scaling`: The amount to scale the xyz points by                        |


# References
[1] D. Scharstein, H. Hirschmüller, Y. Kitajima, G. Krathwohl, N. Nesic, X. Wang, and P. Westling. [High-resolution stereo datasets with subpixel-accurate ground truth](http://www.cs.middlebury.edu/~schar/papers/datasets-gcpr2014.pdf). In German Conference on Pattern Recognition (GCPR 2014), Münster, Germany, September 2014.