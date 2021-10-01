#ifndef _CUDA_GLOBAL_LINEAR_ID_CUH_
#define _CUDA_GLOBAL_LINEAR_ID_CUH_

#include "index_left_fold.hpp"

template <unsigned dims>
__forceinline__ __device__ unsigned cuda_global_linear_id();

template <>
__forceinline__ __device__ unsigned cuda_global_linear_id<1>() {
  return index_left_fold(gridDim.x,  blockIdx.x,
                         blockDim.x, threadIdx.x);
}

template <>
__forceinline__ __device__ unsigned cuda_global_linear_id<2>() {
  return index_left_fold(gridDim.y,  blockIdx.y,
                         gridDim.x,  blockIdx.x,
                         blockDim.y, threadIdx.y,
                         blockDim.x, threadIdx.x);
}

template <>
__forceinline__ __device__ unsigned cuda_global_linear_id<3>() {
  return index_left_fold(gridDim.z,  blockIdx.z,
                         gridDim.y,  blockIdx.y,
                         gridDim.x,  blockIdx.x,
                         blockDim.z, threadIdx.z,
                         blockDim.y, threadIdx.y,
                         blockDim.x, threadIdx.x);
}

template <unsigned dims>
__forceinline__ __device__ ei cuda_size_index();

template <>
__forceinline__ __device__ ei cuda_size_index<1>() {
  return size_index_left_fold(ei{gridDim.x,  blockIdx.x},
                              ei{blockDim.x, threadIdx.x});
}

template <>
__forceinline__ __device__ ei cuda_size_index<2>() {
  return size_index_left_fold(ei{gridDim.y,  blockIdx.y},
                              ei{gridDim.x,  blockIdx.x},
                              ei{blockDim.y, threadIdx.y},
                              ei{blockDim.x, threadIdx.x});
}

template <>
__forceinline__ __device__ ei cuda_size_index<3>() {
  return size_index_left_fold(ei{gridDim.z,  blockIdx.z},
                              ei{gridDim.y,  blockIdx.y},
                              ei{gridDim.x,  blockIdx.x},
                              ei{blockDim.z, threadIdx.z},
                              ei{blockDim.y, threadIdx.y},
                              ei{blockDim.x, threadIdx.x});
}

#endif // _CUDA_GLOBAL_LINEAR_ID_CUH_
