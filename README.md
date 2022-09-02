# Isaac ROS Image Pipeline

<div align="center"><img src="resources/100_right.JPG" width="300"/><img src="resources/300_right_hallway2_rect.png" width="300"/><img src="resources/300_right_hallway2_gray_rect.png" width="300"/></div>

## Overview
This metapackage offers similar functionality as the standard, CPU-based [`image_pipeline` metapackage](http://wiki.ros.org/image_pipeline), but does so by leveraging NVIDIA GPUs and the Jetson platform's specialized computer vision hardware. Considerable effort has been made to ensure that replacing `image_pipeline` with `isaac_ros_image_pipeline` on a Jetson device is as painless a transition as possible.


## Table of Contents
- [Isaac ROS Image Pipeline](#isaac-ros-image-pipeline)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Latest Update](#latest-update)
  - [Performance](#performance)
  - [Supported Platforms](#supported-platforms)
    - [Docker](#docker)
    - [Package Dependencies](#package-dependencies)
  - [Quickstart](#quickstart)
    - [Replacing `image_pipeline` with `isaac_ros_image_pipeline`](#replacing-image_pipeline-with-isaac_ros_image_pipeline)
  - [Supported Packages](#supported-packages)
- [ROS2 Package API](#ros2-package-api)
  - [`isaac_ros_image_proc`](#isaac_ros_image_proc)
    - [Overview](#overview-1)
    - [Available Components](#available-components)
  - [`isaac_ros_stereo_image_proc`](#isaac_ros_stereo_image_proc)
    - [Overview](#overview-2)
    - [Available Components](#available-components-1)
  - [Troubleshooting](#troubleshooting)
    - [RealSense camera issue with `99-realsense-libusb.rules`](#realsense-camera-issue-with-99-realsense-libusbrules)
      - [Symptom](#symptom)
      - [Solution](#solution)
  - [Updates](#updates)
- [References](#references)

## Latest Update
Update 2022-08-31: Image flip support and update to be compatible with JetPack 5.0.2

## Performance
The following are the benchmark performance results of Image_Proc Nodes and pipeline in this package, by supported platform:

| Pipeline              | AGX Orin | AGX Xavier | x86_64 w/ RTX 3060 Ti |
| --------------------- | :------: | :--------: | :-------------------: |
| Disparity Node (540p) | 166 fps  |   80 fps   |        424 fps        |
| Rectify Node (1080p)  | 193 fps  |  127 fps   |          --           |


## Supported Platforms
This package is designed and tested to be compatible with ROS2 Humble running on [Jetson](https://developer.nvidia.com/embedded-computing) or an x86_64 system with an NVIDIA GPU.

> **Note**: Versions of ROS2 earlier than Humble are **not** supported. This package depends on specific ROS2 implementation features that were only introduced beginning with the Humble release.


| Platform | Hardware                                                                                                                                                                                                | Software                                                                                                             | Notes                                                                                                                                                                                   |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Jetson   | [Jetson Orin](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)<br/>[Jetson Xavier](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-agx-xavier/) | [JetPack 5.0.2](https://developer.nvidia.com/embedded/jetpack)                                                       | For best performance, ensure that [power settings](https://docs.nvidia.com/jetson/archives/r34.1/DeveloperGuide/text/SD/PlatformPowerAndPerformance.html) are configured appropriately. |
| x86_64   | NVIDIA GPU                                                                                                                                                                                              | [Ubuntu 20.04+](https://releases.ubuntu.com/20.04/) <br> [CUDA 11.6.1+](https://developer.nvidia.com/cuda-downloads) |

### Docker
To simplify development, we strongly recommend leveraging the Isaac ROS Dev Docker images by following [these steps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/dev-env-setup.md). This will streamline your development environment setup with the correct versions of dependencies on both Jetson and x86_64 platforms.

> **Note:** All Isaac ROS Quickstarts, tutorials, and examples have been designed with the Isaac ROS Docker images as a prerequisite.
> 
### Package Dependencies
- [isaac_ros_common](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common)
- [image_common](https://github.com/ros-perception/image_common.git)
- [vision_cv](https://github.com/ros-perception/vision_opencv.git)
- [OpenCV 4.5+](https://opencv.org/)
- [scikit-learn](https://scikit-learn.org/stable/install.html)
- [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page)

## Quickstart
1. Set up your development environment by following the instructions [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/dev-env-setup.md).  
2. Clone this repository and its dependencies under `~/workspaces/isaac_ros-dev/src`. 
    ```bash
    cd ~/workspaces/isaac_ros-dev/src
    ```

    ```bash
    git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common
    ``` 
    
    ```bash
    git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros
    ```

    ```bash
    git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_pipeline
    ```
3. Launch the Docker container using the `run_dev.sh` script:
    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
      ./scripts/run_dev.sh
    ```
4. Inside the container, build and source the workspace:  
    ```bash
    cd /workspaces/isaac_ros-dev && \
      colcon build --symlink-install && \
      source install/setup.bash
    ```
5. (Optional) Run tests to verify complete and correct installation:  
    ```bash
    colcon test --executor sequential
    ```
6. Start `isaac_ros_image_proc` using the prebuilt executable (Using Realsense camera as an example):  
    ```bash
    ros2 run isaac_ros_image_proc isaac_ros_image_proc --ros-args -r /image_raw:=/camera/color/image_raw --ros-args -r /camera_info:=/camera/color/camera_info
    ```
7. In a separate terminal, spin up a **calibrated** camera publisher to `/image_raw` and `/camera_info` using any package(for example, `realsense2_camera`):  
    ```bash
    ros2 launch realsense2_camera rs_launch.py
    ```
8. Observe the image output in grayscale and color on `/image_mono` and `/image_rect_color`, respectively:  
    ```bash
    ros2 run image_view image_view --ros-args -r image:=image_mono  
    ros2 run image_view image_view --ros-args -r image:=image_rect_color  
    ```
> **Note:** To build the RealSense camera package for Humble, please refer to the section [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/troubleshooting.md#realsense-driver-doesnt-work-with-ros2-humble).
> 
> Other supported cameras can be found [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_argus_camera/blob/main/README.md#reference-cameras).
> 
> For camera calibration, please refer to [this guide](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/camera-calibration.md).

### Replacing `image_pipeline` with `isaac_ros_image_pipeline`
1. Add a dependency on `isaac_ros_image_pipeline` to `your_package/package.xml` and `your_package/CMakeLists.txt`. If all desired packages under an existing `image_pipeline` dependency have Isaac ROS alternatives (see **Supported Packages**), then the original `image_pipeline` dependency may be removed entirely.
2. Change the package and plugin names in any `*.launch.py` launch files to use `[package name]` and `nvidia::isaac_ros::image_proc::[component_name]` respectively. For a list of all packages, see **Supported Packages**. For a list of all ROS2 Components made available, see the per-package detailed documentation below.

## Supported Packages
At this time, the packages under the standard `image_pipeline` have the following support:

| Existing Package     | Isaac ROS Alternative             |
| -------------------- | --------------------------------- |
| `image_pipeline`     | See `isaac_ros_image_pipeline`    |
| `image_proc`         | See `isaac_ros_image_proc`        |
| `stereo_image_proc`  | See `isaac_ros_stereo_image_proc` |
| `depth_image_proc`   | Continue using existing package   |
| `camera_calibration` | Continue using existing package   |
| `image_publisher`    | Continue using existing package   |
| `image_view`         | Continue using existing package   |
| `image_rotate`       | Continue using existing package   |

# ROS2 Package API
## `isaac_ros_image_proc`
### Overview
The `isaac_ros_image_proc` package offers functionality for rectifying/undistorting images from a monocular camera setup, resizing the image, and changing the image format. It largely replaces the `image_proc` package, though the image format conversion facility also functions as a way to replace the CPU-based image format conversion in `cv_bridge`. The rectify node can also resize the image; if resizing is not needed, specify the output width/height same as input.

### Available Components

| Component                  | Topics Subscribed                                   | Topics Published                                                | Parameters                                                                                                                                                                                |
| -------------------------- | --------------------------------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ImageFormatConverterNode` | `image_raw`, `camera_info`: The input camera stream | `image`: The converted image                                    | `encoding_desired`: Target encoding to convert to.<br>                                                                                                                                    |  |
| `RectifyNode`              | `image_raw`, `camera_info`: The input camera stream | `image_rect`, `camera_info_rect`: The rectified camera stream   | `output_height`: The absolute height to resize to <br> `output_width`: The absolute width to resize to <br>  `keep_aspect_ratio`: The flag to keep the aspect_ratio when set to true <br> |  |
| `ResizeNode`               | `image`, `camera_info`: The input camera stream     | `resize/image`, `resize/camera_info`: The resized camera stream | `output_height`: The absolute height to resize to <br> `output_width`: The absolute width to resize to <br>                                                                               |
| `ImageFlipNode`            | `image`: The input image data                       | `image_flipped`: The flipped image                              |    `flip_mode`: Supports 3 modes - `HORIZONTAL`, `VERTICAL`, and `BOTH`                                                                                  |

**Limitation:** Image proc nodes require even number dimensions for images.

## `isaac_ros_stereo_image_proc`
### Overview
The `isaac_ros_stereo_image_proc` package offers functionality for handling image pairs from a binocular/stereo camera setup, calculating the disparity between the two images, and producing a point cloud with depth information. It largely replaces the `stereo_image_proc` package.

### Available Components

| Component        | Topics Subscribed                                                                                                                                                                                              | Topics Published                                   | Parameters                                                                                                                                                                                                                                  |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DisparityNode`  | `left/image_rect`, `left/camera_info`: The left camera stream <br> `right/image_rect`, `right/camera_info`: The right camera stream                                                                            | `disparity`: The disparity between the two cameras | `max_disparity`: The maximum value for disparity per pixel, which is 64 by default. With ORIN backend, this value must be 128 or 256. <br> `backends`: The VPI backend to use, which is CUDA by default (options: "CUDA", "XAVIER", "ORIN") |
| `PointCloudNode` | `left/image_rect_color`: The coloring for the point cloud <br> `left/camera_info`: The left camera info <br> `right/camera_info`: The right camera info <br> `disparity` The disparity between the two cameras | `points2`: The output point cloud                  | `use_color`: Whether or not the output point cloud should have color. The default value is false. <br> `unit_scaling`: The amount to scale the xyz points by                                                                                |

## Troubleshooting
### RealSense camera issue with `99-realsense-libusb.rules`
Some RealSense camera users have experienced [issues](https://github.com/IntelRealSense/realsense-ros/issues/1408) with libusb rules.
#### Symptom
```
admin@workstation:/workspaces/isaac_ros-dev$  ros2 launch realsense2_camera rs_launch.py
[INFO] [launch]: All log files can be found below /home/admin/.ros/log/2021-10-11-20-13-00-110633-UBUNTU-piyush-3480
[INFO] [launch]: Default logging verbosity is set to INFO
[INFO] [realsense2_camera_node-1]: process started with pid [3482]
[realsense2_camera_node-1] [INFO] [1633983180.596460523] [RealSenseCameraNode]: RealSense ROS v3.2.2
[realsense2_camera_node-1] [INFO] [1633983180.596526058] [RealSenseCameraNode]: Built with LibRealSense v2.48.0
[realsense2_camera_node-1] [INFO] [1633983180.596543343] [RealSenseCameraNode]: Running with LibRealSense v2.48.0
[realsense2_camera_node-1]  11/10 20:13:00,624 ERROR [139993561417472] (handle-libusb.h:51) failed to open usb interface: 0, error: RS2_USB_STATUS_NO_DEVICE
[realsense2_camera_node-1] [WARN] [1633983180.626359282] [RealSenseCameraNode]: Device 1/1 failed with exception: failed to set power state
[realsense2_camera_node-1] [ERROR] [1633983180.626456541] [RealSenseCameraNode]: The requested device with  is NOT found. Will Try again.
[realsense2_camera_node-1]  11/10 20:13:00,624 ERROR [139993586595584] (sensor.cpp:517) acquire_power failed: failed to set power state
[realsense2_camera_node-1]  11/10 20:13:00,626 WARNING [139993586595584] (rs.cpp:306) null pointer passed for argument "device"
```
#### Solution
1. Check if the `99-realsense-libusb.rules` file exists in `/etc/udev/rules.d/`
2. If not, disconnect the camera, copy this [file](https://github.com/IntelRealSense/librealsense/blob/master/config/99-realsense-libusb.rules) to `/etc/udev/rules.d/`, then reconnect the camera. 

## Updates

| Date       | Changes                                                                                                                                                    |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2022-08-31 | Image flip support and update to be compatible with JetPack 5.0.2                                                                                                                 |
| 2022-06-30 | Migrated to NITROS based implementation                                                                                                                    |
| 2021-10-20 | Migrated to [NVIDIA-ISAAC-ROS](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_pipeline). Fixed handling of extrinsics in Rectify and Disparity nodes. |
| 2021-08-11 | Initial release to [NVIDIA-AI-IOT](https://github.com/NVIDIA-AI-IOT/isaac_ros_image_pipeline)                                                              |

# References
[1] D. Scharstein, H. Hirschmüller, Y. Kitajima, G. Krathwohl, N. Nesic, X. Wang, and P. Westling. [High-resolution stereo datasets with subpixel-accurate ground truth](http://www.cs.middlebury.edu/~schar/papers/datasets-gcpr2014.pdf). In German Conference on Pattern Recognition (GCPR 2014), Münster, Germany, September 2014.
