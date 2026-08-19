# Isaac ROS Image Pipeline

NVIDIA-accelerated Image Pipeline.

<div align="center"><img src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-4.6/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_image_pipeline/100_right.jpg/" width="300px"/>
<img src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-4.6/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_image_pipeline/300_right_hallway2_rect.png/" width="300px"/></div>
<div align="center"><img src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-4.6/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_image_pipeline/300_right_hallway2_gray_rect.png/" width="300px"/></div>

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
<div align="center"><a class="reference internal image-reference" href="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-4.6/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_image_pipeline/isaac_ros_image_pipeline_nodegraph.png/"><img alt="image" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-4.6/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_image_pipeline/isaac_ros_image_pipeline_nodegraph.png/" width="800px"/></a></div>

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

| Sample Graph<br/><br/>                                                                                                                                                                             | Input Size<br/><br/>   | AGX Thor T5000<br/><br/>                                                                                                                                                  | AGX Thor T4000<br/><br/>                                                                                                                                                   | AGX Orin<br/><br/>                                                                                                                                                        | Orin Nano Super 8GB<br/><br/>                                                                                                                                              | DGX Spark<br/><br/>                                                                                                                                                        | x86_64 w/ RTX 5090<br/><br/>                                                                                                                                              | x86_64 w/ RTX 5070<br/><br/>                                                                                                                                                 |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Rectify Node](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.6/benchmarks/isaac_ros_image_proc_benchmark/scripts/isaac_ros_rectify_node.py)<br/><br/>                     | 1080p<br/><br/>        | [1690 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.6/results/isaac_ros_rectify_node-agx_thor.json)<br/><br/><br/>0.48 ms @ 30Hz<br/><br/>  | [1140 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.6/results/isaac_ros_rectify_node-thor-t4000.json)<br/><br/><br/>0.52 ms @ 30Hz<br/><br/> | [886 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.6/results/isaac_ros_rectify_node-agx_orin.json)<br/><br/><br/>0.51 ms @ 30Hz<br/><br/>   | [296 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.6/results/isaac_ros_rectify_node-orin_nano.json)<br/><br/><br/>1.4 ms @ 30Hz<br/><br/>    | [2500 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.6/results/isaac_ros_rectify_node-dgx_spark.json)<br/><br/><br/>0.15 ms @ 30Hz<br/><br/>  | [2490 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.6/results/isaac_ros_rectify_node-x86-5090.json)<br/><br/><br/>0.43 ms @ 30Hz<br/><br/>  | [1300 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.6/results/isaac_ros_rectify_node-x86-rtx5070.json)<br/><br/><br/>0.46 ms @ 30Hz<br/><br/>  |
| [Stereo Disparity Node](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.6/benchmarks/isaac_ros_stereo_image_proc_benchmark/scripts/isaac_ros_disparity_node.py)<br/><br/>   | 1080p<br/><br/>        | [187 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.6/results/isaac_ros_disparity_node-agx_thor.json)<br/><br/><br/>5.8 ms @ 30Hz<br/><br/>  | [128 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.6/results/isaac_ros_disparity_node-thor-t4000.json)<br/><br/><br/>27 ms @ 30Hz<br/><br/>  | [86.5 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.6/results/isaac_ros_disparity_node-agx_orin.json)<br/><br/><br/>25 ms @ 30Hz<br/><br/>  | [45.5 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.6/results/isaac_ros_disparity_node-orin_nano.json)<br/><br/><br/>22 ms @ 30Hz<br/><br/>  | [198 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.6/results/isaac_ros_disparity_node-dgx_spark.json)<br/><br/><br/>5.0 ms @ 30Hz<br/><br/>  | [478 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.6/results/isaac_ros_disparity_node-x86-5090.json)<br/><br/><br/>2.4 ms @ 30Hz<br/><br/>  | [281 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.6/results/isaac_ros_disparity_node-x86-rtx5070.json)<br/><br/><br/>2.9 ms @ 30Hz<br/><br/>  |
| [Stereo Disparity Graph](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.6/benchmarks/isaac_ros_stereo_image_proc_benchmark/scripts/isaac_ros_disparity_graph.py)<br/><br/> | 1080p<br/><br/>        | [168 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.6/results/isaac_ros_disparity_graph-agx_thor.json)<br/><br/><br/>6.1 ms @ 30Hz<br/><br/> | [115 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.6/results/isaac_ros_disparity_graph-thor-t4000.json)<br/><br/><br/>30 ms @ 30Hz<br/><br/> | [83.0 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.6/results/isaac_ros_disparity_graph-agx_orin.json)<br/><br/><br/>25 ms @ 30Hz<br/><br/> | [41.2 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.6/results/isaac_ros_disparity_graph-orin_nano.json)<br/><br/><br/>24 ms @ 30Hz<br/><br/> | [174 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.6/results/isaac_ros_disparity_graph-dgx_spark.json)<br/><br/><br/>6.0 ms @ 30Hz<br/><br/> | [413 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.6/results/isaac_ros_disparity_graph-x86-5090.json)<br/><br/><br/>2.8 ms @ 30Hz<br/><br/> | [237 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.6/results/isaac_ros_disparity_graph-x86-rtx5070.json)<br/><br/><br/>3.2 ms @ 30Hz<br/><br/> |

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

Update 2026-08-18: Compatibility and integration updates for the Isaac ROS 4.6.0 release
