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
#include "ImageAdapterTensorImpl.hpp"

namespace nvidia {
namespace cvcore {
namespace tensor_ops {
namespace detail {

gxf::Expected<std::tuple<size_t, size_t, size_t>> GetHWCIndices(const ::cvcore::ImageType type) {
  switch (type) {
    case ::cvcore::ImageType::Y_U8:
    case ::cvcore::ImageType::Y_U16:
    case ::cvcore::ImageType::Y_F32:
    case ::cvcore::ImageType::RGB_U8:
    case ::cvcore::ImageType::BGR_U8:
    case ::cvcore::ImageType::RGB_U16:
    case ::cvcore::ImageType::BGR_U16:
    case ::cvcore::ImageType::RGB_F32:
    case ::cvcore::ImageType::BGR_F32: {
      return std::make_tuple(0, 1, 2);
    }
    case ::cvcore::ImageType::PLANAR_RGB_U8:
    case ::cvcore::ImageType::PLANAR_BGR_U8:
    case ::cvcore::ImageType::PLANAR_RGB_U16:
    case ::cvcore::ImageType::PLANAR_BGR_U16:
    case ::cvcore::ImageType::PLANAR_RGB_F32:
    case ::cvcore::ImageType::PLANAR_BGR_F32: {
      return std::make_tuple(1, 2, 0);
    }
    default: {
      GXF_LOG_ERROR("invalid image type.");
      return gxf::Unexpected{GXF_FAILURE};
    }
  }
}

gxf::Expected<gxf::PrimitiveType> GetPrimitiveType(const ::cvcore::ImageType image_type) {
  switch (image_type) {
    case ::cvcore::ImageType::Y_U8:
    case ::cvcore::ImageType::RGB_U8:
    case ::cvcore::ImageType::BGR_U8:
    case ::cvcore::ImageType::PLANAR_RGB_U8:
    case ::cvcore::ImageType::PLANAR_BGR_U8: {
      return gxf::PrimitiveType::kUnsigned8;
    }
    case ::cvcore::ImageType::Y_U16:
    case ::cvcore::ImageType::RGB_U16:
    case ::cvcore::ImageType::BGR_U16:
    case ::cvcore::ImageType::PLANAR_RGB_U16:
    case ::cvcore::ImageType::PLANAR_BGR_U16: {
      return gxf::PrimitiveType::kUnsigned16;
    }
    case ::cvcore::ImageType::Y_F32:
    case ::cvcore::ImageType::RGB_F32:
    case ::cvcore::ImageType::BGR_F32:
    case ::cvcore::ImageType::PLANAR_RGB_F32:
    case ::cvcore::ImageType::PLANAR_BGR_F32: {
      return gxf::PrimitiveType::kFloat32;
    }
    default: {
      GXF_LOG_ERROR("invalid image type.");
      return gxf::Unexpected{GXF_FAILURE};
    }
  }
}

gxf::Expected<ImageInfo> GetTensorInfo(gxf::Handle<gxf::Tensor> tensor, const ::cvcore::ImageType type) {
  const auto& shape       = tensor->shape();
  const auto rank         = tensor->rank();
  const auto storage_type = tensor->storage_type();

  if (rank != 3) {
    GXF_LOG_ERROR("unexpected tensor shape.");
    return gxf::Unexpected{GXF_FAILURE};
  }

  const auto indices = GetHWCIndices(type);
  if (!indices) {
    return gxf::Unexpected{GXF_FAILURE};
  }
  const size_t width  = shape.dimension(std::get<1>(indices.value()));
  const size_t height = shape.dimension(std::get<0>(indices.value()));

  return ImageInfo{type, width, height, storage_type != gxf::MemoryStorageType::kDevice};
}

} // namespace detail
} // namespace tensor_ops
} // namespace cvcore
} // namespace nvidia
