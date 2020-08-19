// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "bias_softmax.h"
#include "core/providers/common.h"
#include "core/providers/cuda/cudnn_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                                \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      BiasSoftmax,                                                              \
      kMSDomain,                                                                \
      1,                                                                        \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      BiasSoftmax<T>);                                                              

template <typename T>
Status BiasSoftmax<T>::ComputeInternal(OpKernelContext* ctx) const {

  typedef typename ToCudaType<T>::MappedType CudaT;

  auto X_data = reinterpret_cast<const CudaT*>(ctx->Input<Tensor>(0));
  auto B_data = reinterpret_cast<const CudaT*>(ctx->Input<Tensor>(1));
  auto Y_data = reinterpret_cast<CudaT*>(ctx->Output<Tensor>(0));

  const TensorShape& X_shape{ctx->Input<Tensor>(0)->Shape()};
  const TensorShape& B_shape{ctx->Input<Tensor>(1)->Shape()};

  const int64_t softmax_axis = HandleNegativeAxis(softmax_axis_, X_shape.NumDimensions());
  int64_t N = X_shape.SizeToDimension(softmax_axis);
  int64_t D = X_shape.SizeFromDimension(softmax_axis);

  const int64_t broadcast_axis = HandleNegativeAxis(broadcast_axis_, X_shape.NumDimensions());
  int64_t broadcast_size = N - X_shape.SizeToDimension(broadcast_axis);

  if (D == input_shape[softmax_axis] && D <= 1024 && D * sizeof(T) <= 4096) {
    // if thread block still fits within SM registers at high occupancy
    dispatch_bias_softmax_forward(Y_data, X_data, B_data, D, N, D, broadcast_size);
  }
  else {
    // cuda DNN fallaback
    std::vector<int64_t> dims({N, 1, 1, D});
    const auto alpha = Consts<CudaT>::One;
    const auto beta = Consts<CudaT>::Zero;
    CudnnTensor input_tensor;
    CudnnTensor output_tensor;
    ORT_RETURN_IF_ERROR(input_tensor.Set(dims, CudnnTensor::GetDataType<CudaT>()));
    ORT_RETURN_IF_ERROR(output_tensor.Set(dims, CudnnTensor::GetDataType<CudaT>()));
    CUDNN_RETURN_IF_ERROR(cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, input_tensor, X_data, &beta, output_tensor, Y_data));
  }

  return Status::OK();
}

#define SPECIALIZED_COMPUTE(T) \
  REGISTER_KERNEL_TYPED(T)     \
  template Status BiasSoftmax<T>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_COMPUTE(float)
SPECIALIZED_COMPUTE(double)
SPECIALIZED_COMPUTE(MLFloat16)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
