#ifdef __NVCC__
#include "cuda_linear_id.cuh"
#endif // __NVCC__
#include "index_left_fold.hpp"
#include <algorithm>
#include <vector>
#include <random>
#include <array>
#include <cassert>

// nvcc -x cu main.cpp
// g++ main.cpp

#ifdef __NVCC__
__global__ void incr(unsigned *p) { p[cuda_linear_id<3>()]++; }
template <unsigned dim>
__global__ void incrT(unsigned *p) { p[cuda_linear_id<dim>()]++; }

template <typename T>
bool cuda_test()
{
  using std::array; using std::accumulate; using std::begin; using std::end;
  using std::multiplies; using std::random_device; using std::mt19937;
  using std::shuffle; using std::vector;

  array<T,6> ex{2,3,4,5,6,7}; // arbitrary extents of a 6D array
  const T sz = accumulate(begin(ex), end(ex), 1, multiplies<>{});
  const T nbytes = sz * sizeof(T);

  mt19937 g(random_device{}());

  shuffle(ex.begin(), ex.end(), g);

  T *d_x;
  cudaMalloc(&d_x, nbytes);
  cudaMemset(d_x, 0, nbytes);

      incr<<<dim3{ex[0],ex[1],ex[2]},dim3{ex[3],ex[4],ex[5]}>>>(d_x); // 0 > 1
  incrT<3><<<dim3{ex[0],ex[1],ex[2]},dim3{ex[3],ex[4],ex[5]}>>>(d_x); // 1 > 2
  incrT<2><<<dim3{ex[0],ex[1]*ex[2]},dim3{ex[3],ex[4]*ex[5]}>>>(d_x); // 2 > 3
  incrT<1><<<dim3{ex[0]*ex[1]*ex[2]},dim3{ex[3]*ex[4]*ex[5]}>>>(d_x); // 3 > 4

  vector<T> h_x(sz);
  cudaMemcpy(h_x.data(), d_x, nbytes, cudaMemcpyDeviceToHost);

  cudaFree(d_x);

  return h_x == std::vector<T>(h_x.size(), 4); // 4
}
#endif // __NVCC__

template <typename T>
bool cpu_test()
{
  T arr3[2][4][6]{};
  arr3[1][3][5] = 1;
  T *p1 = reinterpret_cast<T *>(arr3);
  T offset1 = index_left_fold(2,1, 4,3, 6,5);
  p1[offset1]++;

  T arr4[2][4][6][8]{};
  arr4[1][3][5][7] = 3;
  T *p2 = reinterpret_cast<T *>(arr4);
  T offset2 = index_left_fold(2,1, 4,3, 6,5, 8,7);
  p2[offset2]++;

  return arr3[1][3][5]==2 && arr4[1][3][5][7]==4;
}

int main(int argc, char *argv[])
{
#ifdef __NVCC__
  assert(cuda_test<unsigned>());
#endif // __NVCC__

  assert(cpu_test<unsigned>());

  static_assert(42==index_left_fold(3,0, 2,0, 1,0, 43,42),"");

  return 0;
}
