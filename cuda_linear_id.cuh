#ifndef _CUDA_LINEAR_ID_CUH_
#define _CUDA_LINEAR_ID_CUH_

#include "index_left_fold.hpp"

template <unsigned dim>
__forceinline__ __device__ unsigned cuda_linear_id();

template <>
__forceinline__ __device__ unsigned cuda_linear_id<1>() {
  return index_left_fold(gridDim.x,  blockIdx.x,
                         blockDim.x, threadIdx.x);
}

template <>
__forceinline__ __device__ unsigned cuda_linear_id<2>() {
  return index_left_fold(gridDim.y,  blockIdx.y,
                         gridDim.x,  blockIdx.x,
                         blockDim.y, threadIdx.y,
                         blockDim.x, threadIdx.x);
}

template <>
__forceinline__ __device__ unsigned cuda_linear_id<3>() {
  return index_left_fold(gridDim.z,  blockIdx.z,
                         gridDim.y,  blockIdx.y,
                         gridDim.x,  blockIdx.x,
                         blockDim.z, threadIdx.z,
                         blockDim.y, threadIdx.y,
                         blockDim.x, threadIdx.x);
}

#endif // _CUDA_LINEAR_ID_CUH_
