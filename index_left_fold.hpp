#ifndef _INDEX_LEFT_FOLD_HPP_
#define _INDEX_LEFT_FOLD_HPP_

#ifdef __NVCC__
constexpr __forceinline__ __device__ __host__ unsigned
#else
constexpr unsigned
#endif
index_left_fold(const unsigned extent, const unsigned index) {
  return index;
}

template <typename T, typename ...Ts>
#ifdef __NVCC__
constexpr __forceinline__ __device__ __host__ unsigned
#else
constexpr unsigned
#endif
index_left_fold(const T e1, const T i1, const T e2, const T i2, Ts ...xs) {
  return index_left_fold(e1*e2, i1*e2+i2, xs...);
}

#endif // _INDEX_LEFT_FOLD_HPP_
