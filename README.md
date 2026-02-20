# Isaac ROS Image Pipeline

NVIDIA-accelerated Image Pipeline.

<div align="center"><img src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-4.2/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_image_pipeline/100_right.jpg/" width="300px"/>
<img src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-4.2/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_image_pipeline/300_right_hallway2_rect.png/" width="300px"/></div>
<div align="center"><img src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-4.2/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_image_pipeline/300_right_hallway2_gray_rect.png/" width="300px"/></div>

## Overview

Isaac ROS Image Pipeline is a metapackage of functionality for image
processing. Camera output often needs pre-processing to meet the input
requirements of multiple different perception functions. This can
include cropping, resizing, mirroring, correcting for lens distortion,
and color space conversion. For stereo cameras, additional processing is
required to produce disparity between left + right images and a point
cloud for depth perception.

This package is accelerated using the GPU and specialized hardware
engines for image computation, replacing the CPU-based
[image_pipeline metapackage](https://docs.ros.org/en/rolling/p/image_pipeline).
Considerable effort has been made to ensure that replacing
`image_pipeline` with `isaac_ros_image_pipeline` on a Jetson or GPU
is as painless a transition as possible.

> [!Note]
> Some image pre-processing functions use specialized
> hardware engines, which offload the GPU to make more compute
> available for other tasks.
<div align="center"><a class="reference internal image-reference" href="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-4.2/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_image_pipeline/isaac_ros_image_pipeline_nodegraph.png/"><img alt="image" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-4.2/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_image_pipeline/isaac_ros_image_pipeline_nodegraph.png/" width="800px"/></a></div>

Rectify corrects for lens distortion from the received camera sensor
message. The rectified image is resized to the input resolution for
disparity, using a crop before resizing to maintain image aspect ratio.
The image is color space converted to YUV from RGB using the luma
channel (the Y in YUV) to compute disparity using
[SGM](https://en.wikipedia.org/wiki/Semi-global_matching). This
common graph of nodes can be performed without the CPU processing a
single pixel using `isaac_ros_image_pipeline`; in comparison, using
`image_pipeline`, the CPU would process each pixel ~3 times.

The Isaac ROS Image Pipeline metapackage offloads the CPU from common
image processing tasks so it can perform robotics functions best suited
for the CPU.

## Isaac ROS NITROS Acceleration

This package is powered by [NVIDIA Isaac Transport for ROS (NITROS)](https://developer.nvidia.com/blog/improve-perception-performance-for-ros-2-applications-with-nvidia-isaac-transport-for-ros/), which leverages type adaptation and negotiation to optimize message formats and dramatically accelerate communication between participating nodes.

## Performance

| Sample Graph<br/><br/>                                                                                                                                                                             | Input Size<br/><br/>   | AGX Thor T5000<br/><br/>                                                                                                                                                 | AGX Thor T4000<br/><br/>                                                                                                                                                   | DGX Spark<br/><br/>                                                                                                                                                        | x86_64 w/ RTX 5090<br/><br/>                                                                                                                                              |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Rectify Node](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.2/benchmarks/isaac_ros_image_proc_benchmark/scripts/isaac_ros_rectify_node.py)<br/><br/>                     | 1080p<br/><br/>        | [1550 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.2/results/isaac_ros_rectify_node-agx_thor.json)<br/><br/><br/>1.6 ms @ 30Hz<br/><br/>  | [768 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.2/results/isaac_ros_rectify_node-thor-t4000.json)<br/><br/><br/>4.8 ms @ 30Hz<br/><br/>   | [2500 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.2/results/isaac_ros_rectify_node-dgx_spark.json)<br/><br/><br/>2.3 ms @ 30Hz<br/><br/>   | [2500 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.2/results/isaac_ros_rectify_node-x86-5090.json)<br/><br/><br/>2.9 ms @ 30Hz<br/><br/>   |
| [Stereo Disparity Node](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.2/benchmarks/isaac_ros_stereo_image_proc_benchmark/scripts/isaac_ros_disparity_node.py)<br/><br/>   | 1080p<br/><br/>        | [217 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.2/results/isaac_ros_disparity_node-agx_thor.json)<br/><br/><br/>5.7 ms @ 30Hz<br/><br/> | [196 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.2/results/isaac_ros_disparity_node-thor-t4000.json)<br/><br/><br/>20 ms @ 30Hz<br/><br/>  | [249 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.2/results/isaac_ros_disparity_node-dgx_spark.json)<br/><br/><br/>5.4 ms @ 30Hz<br/><br/>  | [747 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.2/results/isaac_ros_disparity_node-x86-5090.json)<br/><br/><br/>4.0 ms @ 30Hz<br/><br/>  |
| [Stereo Disparity Graph](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.2/benchmarks/isaac_ros_stereo_image_proc_benchmark/scripts/isaac_ros_disparity_graph.py)<br/><br/> | 1080p<br/><br/>        | [203 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.2/results/isaac_ros_disparity_graph-agx_thor.json)<br/><br/><br/>10 ms @ 30Hz<br/><br/> | [185 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.2/results/isaac_ros_disparity_graph-thor-t4000.json)<br/><br/><br/>26 ms @ 30Hz<br/><br/> | [183 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.2/results/isaac_ros_disparity_graph-dgx_spark.json)<br/><br/><br/>9.0 ms @ 30Hz<br/><br/> | [600 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.2/results/isaac_ros_disparity_graph-x86-5090.json)<br/><br/><br/>6.9 ms @ 30Hz<br/><br/> |

---

## Documentation

Please visit the [Isaac ROS Documentation](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_pipeline/index.html) to learn how to use this repository.

---

## Packages

* [`isaac_ros_depth_image_proc`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_pipeline/isaac_ros_depth_image_proc/index.html)
  * [API](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_pipeline/isaac_ros_depth_image_proc/index.html#api)
* [`isaac_ros_image_pipeline`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_pipeline/isaac_ros_image_pipeline/index.html)
  * [Replacing `image_pipeline` with `isaac_ros_image_pipeline`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_pipeline/isaac_ros_image_pipeline/index.html#replacing-image-pipeline-with-isaac-ros-image-pipeline)
* [`isaac_ros_image_proc`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_pipeline/isaac_ros_image_proc/index.html)
  * [Quickstart](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_pipeline/isaac_ros_image_proc/index.html#quickstart)
* [`isaac_ros_stereo_image_proc`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_pipeline/isaac_ros_stereo_image_proc/index.html)
  * [Quickstart](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_pipeline/isaac_ros_stereo_image_proc/index.html#quickstart)
  * [Try More Examples](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_pipeline/isaac_ros_stereo_image_proc/index.html#try-more-examples)
  * [API](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_image_pipeline/isaac_ros_stereo_image_proc/index.html#api)

## Latest

Update 2026-02-19: Support for DGX Spark and JetPack 7.1
