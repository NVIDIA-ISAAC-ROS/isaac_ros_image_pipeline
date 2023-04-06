// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
#include "CameraModel.hpp"
#include "ConvertColorFormat.hpp"
#include "CropAndResize.hpp"
#include "Frame3D.hpp"
#include "ImageAdapter.hpp"
#include "InterleavedToPlanar.hpp"
#include "Normalize.hpp"
#include "Reshape.hpp"
#include "Resize.hpp"
#include "TensorOperator.hpp"
#include "TensorStream.hpp"
#include "Undistort.hpp"

#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()
GXF_EXT_FACTORY_SET_INFO(0x6eae64ff97a94d9b, 0xb324f85e6a98a75a, "NvCvTensorOpsExtension",
                         "Generic CVCORE tensor_ops interfaces", "Nvidia_Gxf", "3.1.0", "LICENSE");

GXF_EXT_FACTORY_ADD(0xd073a92344ba4b81, 0xbd0f18f4996048e9, nvidia::cvcore::tensor_ops::CameraModel,
                    nvidia::gxf::Component,
                    "Construct Camera distortion model / Camera intrinsic compatible with CVCORE");

GXF_EXT_FACTORY_ADD(0x6c9419223e4b4c2b, 0x899a4d65279c6507, nvidia::cvcore::tensor_ops::Frame3D, nvidia::gxf::Component,
                    "Construct Camera extrinsic compatible with CVCORE");

GXF_EXT_FACTORY_ADD(0xd94385e5b35b4634, 0x9adb0d214a3865f6, nvidia::cvcore::tensor_ops::TensorStream,
                    nvidia::gxf::Component, "Wrapper of CVCORE ITensorOperatorStream/ITensorOperatorContext");

GXF_EXT_FACTORY_ADD(0xd0c4ddad486a4a91, 0xb69c8a5304b205ef, nvidia::cvcore::tensor_ops::ImageAdapter,
                    nvidia::gxf::Component, "Utility component for conversion between message and cvcore image type");

GXF_EXT_FACTORY_ADD(0xadebc792bd0b4a56, 0x99c1405fd2ea0727, nvidia::cvcore::tensor_ops::StreamUndistort,
                    nvidia::gxf::Codelet, "Codelet for stream image undistortion in tensor_ops");

GXF_EXT_FACTORY_ADD(0xa58141ac7eca4ea5, 0x9b545446fe379a11, nvidia::cvcore::tensor_ops::Resize, nvidia::gxf::Codelet,
                    "Codelet for image resizing in tensor_ops");

GXF_EXT_FACTORY_ADD(0xeb8b5f5b36d44b48, 0x81f959fd28e6f677, nvidia::cvcore::tensor_ops::StreamResize,
                    nvidia::gxf::Codelet, "Codelet for stream image resizing in tensor_ops");

GXF_EXT_FACTORY_ADD(0x4a7ff422de3841bc, 0x9e743ac10d9294b6, nvidia::cvcore::tensor_ops::CropAndResize,
                    nvidia::gxf::Codelet, "Codelet for crop and resizing operation in tensor_ops");

GXF_EXT_FACTORY_ADD(0x7018f0b9034c462b, 0xa9fbaf7ee012974f, nvidia::cvcore::tensor_ops::Normalize, nvidia::gxf::Codelet,
                    "Codelet for image normalization in tensor_ops");

GXF_EXT_FACTORY_ADD(0x269d4237f3c3479d, 0xbcca9ecc44c71a70, nvidia::cvcore::tensor_ops::InterleavedToPlanar,
                    nvidia::gxf::Codelet, "Codelet for convert interleaved image to planar image in tensor_ops");

GXF_EXT_FACTORY_ADD(0xfc4d7b4d8fcc4daa, 0xa286056e0fcafa78, nvidia::cvcore::tensor_ops::ConvertColorFormat,
                    nvidia::gxf::Codelet, "Codelet for image color conversion in tensor_ops");

GXF_EXT_FACTORY_ADD(0x5ab4a4d8f7a34552, 0xa90be52660b076fd, nvidia::cvcore::tensor_ops::StreamConvertColorFormat,
                    nvidia::gxf::Codelet, "Codelet for stream image color conversion in tensor_ops");

GXF_EXT_FACTORY_ADD(0x26789b7d5a8d4e84, 0x86b845ec5f4cd12a, nvidia::cvcore::tensor_ops::Reshape, nvidia::gxf::Codelet,
                    "Codelet for image reshape in tensor_ops");
GXF_EXT_FACTORY_END()
