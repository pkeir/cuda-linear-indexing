#ifndef _INDEX_LEFT_FOLD_HPP_
#define _INDEX_LEFT_FOLD_HPP_

// index_left_fold: returns a linear index given a pack of extent/index pairs

#ifdef __NVCC__
__forceinline__ __device__ __host__
#endif
constexpr unsigned
index_left_fold(const unsigned extent, const unsigned index) {
  return index;
}

template <typename T, typename ...Ts>
#ifdef __NVCC__
__forceinline__ __device__ __host__
#endif
constexpr unsigned
index_left_fold(const T e1, const T i1, const T e2, const T i2, Ts ...xs) {
  return index_left_fold(e1*e2, i1*e2+i2, xs...);
}

// size_index_left_fold: returns a size + linear index pair, in an ei object

struct ei
{
  unsigned extent, index;

#ifdef __NVCC__
__forceinline__ __device__ __host__
#endif
  constexpr
  ei operator+(ei x) { return {extent * x.extent, index * x.extent + x.index}; }

#ifdef __NVCC__
__forceinline__ __device__ __host__
#endif
  constexpr bool operator==(const ei& x) const {
    return extent==x.extent && index==x.index;
  }
};

template <typename ...Ts>
#ifdef __NVCC__
__forceinline__ __device__ __host__
#endif
constexpr ei size_index_left_fold(Ts ...xs) { return (ei{1,0} + ... + xs); }

#endif // _INDEX_LEFT_FOLD_HPP_
