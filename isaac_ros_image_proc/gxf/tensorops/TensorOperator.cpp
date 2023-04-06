// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "TensorOperator.hpp"

#include "gxf/std/timestamp.hpp"

namespace nvidia {
namespace cvcore {
namespace tensor_ops {

namespace detail {

// Function to bind a cuda stream with cid into downstream message
gxf_result_t BindCudaStream(gxf::Entity& message, gxf_uid_t cid) {
  if (cid == kNullUid) {
    GXF_LOG_ERROR("stream_cid is null");
    return GXF_FAILURE;
  }
  auto output_stream_id = message.add<gxf::CudaStreamId>("stream");
  if (!output_stream_id) {
    GXF_LOG_ERROR("failed to add cudastreamid.");
    return GXF_FAILURE;
  }
  output_stream_id.value()->stream_cid = cid;
  return GXF_SUCCESS;
}

// Function to record a new cuda event
gxf_result_t RecordCudaEvent(gxf::Entity& message, gxf::Handle<gxf::CudaStream>& stream) {
  // Create a new event
  cudaEvent_t cuda_event;
  cudaEventCreateWithFlags(&cuda_event, 0);
  gxf::CudaEvent event;
  auto ret = event.initWithEvent(cuda_event, stream->dev_id(), [](auto) {});
  if (!ret) {
    GXF_LOG_ERROR("failed to init cuda event");
    return GXF_FAILURE;
  }
  // Record the event
  // Can define []() { GXF_LOG_DEBUG("tensorops event synced"); } as callback func for debug purpose
  ret = stream->record(event.event().value(),
                       [event = cuda_event, entity = message.clone().value()](auto) { cudaEventDestroy(event); });
  if (!ret) {
    GXF_LOG_ERROR("record event failed");
    return ret.error();
  }
  return GXF_SUCCESS;
}

template<typename T>
gxf_result_t RerouteMessage(gxf::Entity& output, gxf::Entity& input,
                            std::function<gxf_result_t(gxf::Handle<T>, gxf::Handle<T>)> func,
                            const char* name = nullptr) {
  auto maybe_component = input.get<T>();
  if (maybe_component) {
    auto output_component = output.add<T>(name != nullptr ? name : maybe_component.value().name());
    if (!output_component) {
      GXF_LOG_ERROR("add output component failed.");
      return output_component.error();
    }
    return func(output_component.value(), maybe_component.value());
  }
  return GXF_SUCCESS;
}

} // namespace detail

gxf_result_t TensorOperator::inferOutputInfo(gxf::Entity& input) {
  const char* input_name = input_name_.try_get() ? input_name_.try_get().value().c_str() : nullptr;
  auto input_info        = input_adapter_.get()->GetImageInfo(input, input_name);
  if (!input_info) {
    return input_info.error();
  }
  input_info_      = input_info.value();
  auto output_info = doInferOutputInfo(input);
  if (!output_info) {
    return output_info.error();
  }
  output_info_ = output_info.value();
  return GXF_SUCCESS;
}

gxf_result_t TensorOperator::updateCameraMessage(gxf::Handle<gxf::CameraModel>& output,
                                                 gxf::Handle<gxf::CameraModel>& input) {
  return doUpdateCameraMessage(output, input);
}

gxf_result_t TensorOperator::execute(gxf::Entity& output, gxf::Entity& input, cudaStream_t stream) {
  const char* output_name = output_name_.try_get() ? output_name_.try_get().value().c_str() : nullptr;
  const char* input_name  = input_name_.try_get() ? input_name_.try_get().value().c_str() : nullptr;
  return doExecute(output, input, stream, output_name, input_name);
}

gxf_result_t TensorOperator::start() {
  // Allocate cuda stream using stream pool if necessary
  if (stream_pool_.try_get()) {
    auto stream = stream_pool_.try_get().value()->allocateStream();
    if (!stream) {
      GXF_LOG_ERROR("allocating stream failed.");
      return GXF_FAILURE;
    }
    cuda_stream_ptr_ = std::move(stream.value());
    if (!cuda_stream_ptr_->stream()) {
      GXF_LOG_ERROR("allocated stream is not initialized.");
      return GXF_FAILURE;
    }
  }
  return GXF_SUCCESS;
}

gxf_result_t TensorOperator::tick() {
  // Receiving the data
  auto input_message = receiver_->receive();
  // Check received message for errors
  if (!input_message) {
    return input_message.error();
  }
  // Infer output ImageInfo and if it's no-op
  auto error = inferOutputInfo(input_message.value());
  if (error != GXF_SUCCESS) {
    return error;
  }
  // Re-direct the input message if no-op is needed
  if (no_op_) {
    transmitter_->publish(input_message.value());
    return GXF_SUCCESS;
  }
  // Create output message
  gxf::Expected<gxf::Entity> output_message = gxf::Entity::New(context());
  if (!output_message) {
    return output_message.error();
  }
  // Pass through timestamp if presented in input message
  error =
    detail::RerouteMessage<gxf::Timestamp>(output_message.value(), input_message.value(),
                                           [](gxf::Handle<gxf::Timestamp> output, gxf::Handle<gxf::Timestamp> input) {
                                             *output = *input;
                                             return GXF_SUCCESS;
                                           });
  if (error != GXF_SUCCESS) {
    return error;
  }
  // Pass through cudaStreamId or create a new cuda stream for NPP backend only
  cudaStream_t cuda_stream = 0; // default stream
  if (!stream_.try_get()) {
    // Allocate new CudaStream if StreamPool attached
    if (stream_pool_.try_get()) {
      cuda_stream = cuda_stream_ptr_->stream().value();
      if (detail::BindCudaStream(output_message.value(), cuda_stream_ptr_.cid()) != GXF_SUCCESS) {
        return GXF_FAILURE;
      }
    }
    auto input_stream_id = input_message.value().get<gxf::CudaStreamId>();
    if (input_stream_id) {
      auto stream =
        gxf::Handle<gxf::CudaStream>::Create(input_stream_id.value().context(), input_stream_id.value()->stream_cid);
      if (!stream) {
        GXF_LOG_ERROR("create cudastream from cid failed.");
        return GXF_FAILURE;
      }
      if (stream_pool_.try_get()) {
        // sync upstreaming input cuda stream
        if (!stream.value()->syncStream()) {
          GXF_LOG_ERROR("sync stream failed.");
          return GXF_FAILURE;
        }
      } else {
        cuda_stream = stream.value()->stream().value();
        if (detail::BindCudaStream(output_message.value(), stream.value().cid()) != GXF_SUCCESS) {
          return GXF_FAILURE;
        }
        cuda_stream_ptr_ = stream.value();
      }
    }
  }
  // Execute the operation
  error = execute(output_message.value(), input_message.value(), cuda_stream);
  if (error != GXF_SUCCESS) {
    GXF_LOG_ERROR("operation failed.");
    return GXF_FAILURE;
  }
  // Record the cuda event if necessary
  if (!cuda_stream_ptr_.is_null()) {
    // record on both input/output stream
    if (detail::RecordCudaEvent(input_message.value(), cuda_stream_ptr_) != GXF_SUCCESS) {
      return GXF_FAILURE;
    }
    if (detail::RecordCudaEvent(output_message.value(), cuda_stream_ptr_) != GXF_SUCCESS) {
      return GXF_FAILURE;
    }
  }
  // Update output camera message if necessary
  error = detail::RerouteMessage<gxf::CameraModel>(
    output_message.value(), input_message.value(),
    [this](gxf::Handle<gxf::CameraModel> output, gxf::Handle<gxf::CameraModel> input) {
      return updateCameraMessage(output, input);
    },
    "camera");
  if (error != GXF_SUCCESS) {
    return error;
  }
  // Pass through pose3d message if necessary
  error = detail::RerouteMessage<gxf::Pose3D>(
    output_message.value(), input_message.value(),
    [](gxf::Handle<gxf::Pose3D> output, gxf::Handle<gxf::Pose3D> input) {
      *output = *input;
      return GXF_SUCCESS;
    },
    "pose");
  if (error != GXF_SUCCESS) {
    return error;
  }
  // Send the processed data
  transmitter_->publish(output_message.value());

  return GXF_SUCCESS;
}

} // namespace tensor_ops
} // namespace cvcore
} // namespace nvidia
