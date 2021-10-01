#ifdef __NVCC__
#include "cuda_linear_index.cuh"
#endif // __NVCC__
#include "index_left_fold.hpp"
#include <algorithm>
#include <vector>
#include <random>
#include <array>
#include <cassert>

// nvcc -x cu -std=c++17 main.cpp
// g++        -std=c++17 main.cpp
// clang++    -std=c++17 main.cpp

#ifdef __NVCC__
__global__ void incr(unsigned *p) {
  p[cuda_linear_index<3>()]++;
  p[cuda_size_index<3>().index]++;
}
template <unsigned dims>
__global__ void incrT(unsigned *p) {
  p[cuda_linear_index<dims>()]++;
  p[cuda_size_index<dims>().index]++;
}

template <typename T>
bool cuda_all_updated(std::array<T,6>& ex, T* d_x,
                      std::vector<T>& h_x, const unsigned nbytes)
{
  cudaMemset(d_x, 0, nbytes);

      incr<<<dim3{ex[0],ex[1],ex[2]},dim3{ex[3],ex[4],ex[5]}>>>(d_x); // 0 > 2
  incrT<3><<<dim3{ex[0],ex[1],ex[2]},dim3{ex[3],ex[4],ex[5]}>>>(d_x); // 2 > 4
  incrT<2><<<dim3{ex[0],ex[1]*ex[2]},dim3{ex[3],ex[4]*ex[5]}>>>(d_x); // 4 > 6
  incrT<1><<<dim3{ex[0]*ex[1]*ex[2]},dim3{ex[3]*ex[4]*ex[5]}>>>(d_x); // 6 > 8

  cudaMemcpy(h_x.data(), d_x, nbytes, cudaMemcpyDeviceToHost);

  return h_x == std::vector<T>(h_x.size(), 8); // 8
}

template <typename T>
bool cuda_tests()
{
  using std::array; using std::accumulate; using std::begin; using std::end;
  using std::multiplies; using std::random_device; using std::mt19937;
  using std::shuffle; using std::vector;

  array<T,6> ex{2,3,4,5,6,7}; // arbitrary extents of a 6D array
  const T sz = accumulate(begin(ex), end(ex), 1, multiplies<>{});
  const T nbytes = sz * sizeof(T);

  mt19937 g(random_device{}());

  shuffle(ex.begin(), ex.end(), g);

  vector<T> h_x(sz);
  T *d_x;
  cudaMalloc(&d_x, nbytes);

  bool b1 = cuda_all_updated(ex, d_x, h_x, nbytes);

  cudaFree(d_x);
  return b1;
}
#endif // __NVCC__

template <typename T>
bool cpu_test()
{
  T arr3[2][4][6]{};
  arr3[1][3][5] = 1;
  T *p1 = reinterpret_cast<T *>(arr3);
  T offset1 = index_left_fold(2,1, 4,3, 6,5);
  static_assert(47==index_left_fold(2,1, 4,3, 6,5));
  static_assert(ei{48,47}==size_index_left_fold(ei{2,1}, ei{4,3}, ei{6,5}));
  p1[offset1]++;

  T arr4[2][4][6][8]{};
  arr4[1][3][5][7] = 3;
  T *p2 = reinterpret_cast<T *>(arr4);
  T offset2 = index_left_fold(2,1, 4,3, 6,5, 8,7);
  static_assert(383==index_left_fold(2,1, 4,3, 6,5, 8,7));
  static_assert(ei{384,383}==
                size_index_left_fold(ei{2,1}, ei{4,3}, ei{6,5}, ei{8,7}));
  p2[offset2]++;

  return arr3[1][3][5]==2 && arr4[1][3][5][7]==4;
}

int main(int argc, char *argv[])
{
#ifdef __NVCC__
  assert(cuda_tests<unsigned>());
#endif // __NVCC__

  assert(cpu_test<unsigned>());

  static_assert(41==index_left_fold(3,0, 2,0, 1,0, 43,41),"");
  static_assert(42==index_left_fold(3,0, 2,0, 1,0, 43,42),"");

  return 0;
}
